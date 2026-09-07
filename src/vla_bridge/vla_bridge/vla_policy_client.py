"""Queries a running openpi policy server once per control tick with the
current observation (joint state, gripper state, both camera images,
language prompt) and applies the returned action to the robot -- either the
real UR5e (robot_backend: 'rtde') or Isaac Sim (robot_backend: 'isaac_sim',
talks to ../../../isaac/pick_place_scene_bridge.py over ROS2 topics). Same
dual-backend shape as potato_drill_ws's scan_controller.py/drill_controller.py,
so this node works against Isaac Sim first and the real UR5e later with only
a launch parameter change.

Needs openpi's client package on the Python path for WebsocketClientPolicy
-- from your openpi checkout: `pip install -e packages/openpi-client` (see
openpi's own README for the exact path/instructions; it's a small package
with no heavy deps, safe to install into the ROS2 Python environment).

The policy server itself (the actual GPU-heavy pi0 model) runs separately --
see openpi's own `scripts/serve_policy.py` -- started with your fine-tuned
`pi0_ur5e_pick_place` checkpoint before launching this node.

Written and reasoned about WITHOUT the ability to run this pipeline in the
environment this was authored in -- treat as a solid first draft, not
verified to run.
"""
import csv

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from vla_bridge.robot_interface import UR5eInterface
from vla_bridge.isaac_robot_interface import IsaacSimRobotInterface


class VLAPolicyClient(Node):
    def __init__(self):
        super().__init__('vla_policy_client')

        # See main()'s MultiThreadedExecutor for why this exists: the
        # isaac_sim backend's move_joints polls joint_state (a subscription
        # on this same node, created inside IsaacSimRobotInterface) from
        # inside _run_step's own timer callback, so that subscription needs
        # to run concurrently with _run_step, not queued behind it.
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('robot_backend', 'rtde')  # 'rtde' (real UR5e) or 'isaac_sim'
        self.declare_parameter('policy_host', 'localhost')
        self.declare_parameter('policy_port', 8000)
        self.declare_parameter('prompt', 'pick up the cube and place it in the target zone')
        self.declare_parameter('control_hz', 10.0)
        # ADJUST: real-hardware camera topics -- match your actual camera driver's topic names.
        self.declare_parameter('base_image_topic', '/camera/base/image_raw')
        self.declare_parameter('wrist_image_topic', '/camera/wrist/image_raw')
        self.declare_parameter('max_steps', 300)  # safety cap -- stop after this many control ticks regardless of task completion
        # Residual RL (see isaac/residual_rl_train_env.py / train_residual_policy.py,
        # README Phase 4): 'false' (default) = pi0's action is applied as-is,
        # unchanged from the rest of this node. 'true' = a small trained
        # correction is added to pi0's joint targets each step.
        self.declare_parameter('use_residual_policy', False)
        self.declare_parameter('residual_model_path', '')
        self.declare_parameter('residual_scale', 0.05)  # must match the value used in train_residual_policy.py

        # Offline evaluation (isaac_sim backend only -- needs the ground-truth
        # cube position pick_place_scene_bridge.py publishes on
        # /vla/eval/cube_position; no real-hardware equivalent exists). Runs
        # n_trials episodes back-to-back (requesting a scene reset between
        # each via /vla/eval/reset) and logs one row per trial to
        # results_csv_path, in the same (trial, success, steps, xy_error_mm)
        # shape hybrid_pick_place_demo.py and openvla_pick_place_demo.py log,
        # so all three pipelines' results can be compared directly.
        self.declare_parameter('eval_mode', False)
        self.declare_parameter('n_trials', 20)
        self.declare_parameter('success_xy_tolerance_m', 0.03)  # matches hybrid_pick_place_demo.py / residual_rl_train_env.py
        self.declare_parameter('holding_gripper_threshold', 0.5)  # matches residual_rl_train_env.py
        self.declare_parameter('lifted_z_threshold', 0.03)  # matches residual_rl_train_env.py
        self.declare_parameter('eval_target_position', [0.45, 0.30, 0.0])  # must match isaac/pick_place_scene.py's PLACE_TARGET_POSITION
        self.declare_parameter('results_csv_path', 'vla_eval_results.csv')

        self.prompt = self.get_parameter('prompt').value
        self.control_period_s = 1.0 / self.get_parameter('control_hz').value
        self.max_steps = self.get_parameter('max_steps').value
        self.residual_scale = self.get_parameter('residual_scale').value

        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.policy = WebsocketClientPolicy(
            host=self.get_parameter('policy_host').value,
            port=self.get_parameter('policy_port').value,
        )

        if self.get_parameter('use_residual_policy').value:
            from stable_baselines3 import PPO  # lazy import: only needed when use_residual_policy is true
            residual_model_path = self.get_parameter('residual_model_path').value
            if not residual_model_path:
                raise ValueError("use_residual_policy is true but residual_model_path is empty")
            self.residual_policy = PPO.load(residual_model_path)
            self.get_logger().info(f'use_residual_policy=true, loaded {residual_model_path}')
        else:
            self.residual_policy = None

        self.backend = self.get_parameter('robot_backend').value
        if self.backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(self, callback_group=self._cb_group)
            image_topics = ('/vla/base_image', '/vla/wrist_image')
        else:
            self.robot = UR5eInterface(self.get_parameter('robot_ip').value)
            image_topics = (
                self.get_parameter('base_image_topic').value,
                self.get_parameter('wrist_image_topic').value,
            )

        self.eval_mode = self.get_parameter('eval_mode').value
        if self.eval_mode and self.backend != 'isaac_sim':
            self.get_logger().warn(
                'eval_mode requires robot_backend=isaac_sim (needs ground-truth cube '
                'position from pick_place_scene_bridge.py) -- disabling eval_mode.')
            self.eval_mode = False
        if self.eval_mode:
            self.n_trials = self.get_parameter('n_trials').value
            self.success_xy_tolerance_m = self.get_parameter('success_xy_tolerance_m').value
            self.holding_gripper_threshold = self.get_parameter('holding_gripper_threshold').value
            self.lifted_z_threshold = self.get_parameter('lifted_z_threshold').value
            self.eval_target_position = np.array(self.get_parameter('eval_target_position').value, dtype=float)
            self.results_csv_path = self.get_parameter('results_csv_path').value
            self._trial_idx = 0
            self._trial_successes = 0
            self._was_holding = False
            self._latest_cube_position = None
            self._eval_reset_pub = self.create_publisher(Empty, '/vla/eval/reset', 10)
            self.create_subscription(
                Point, '/vla/eval/cube_position', self._on_cube_position, 10,
                callback_group=self._cb_group)
            with open(self.results_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['trial', 'success', 'steps', 'xy_error_mm'])
            self.get_logger().info(
                f'eval_mode=true: running {self.n_trials} trials, logging to {self.results_csv_path}')

        self.cv_bridge = CvBridge()
        self._base_image = None
        self._wrist_image = None
        self.create_subscription(
            Image, image_topics[0], self._on_base_image, 10, callback_group=self._cb_group)
        self.create_subscription(
            Image, image_topics[1], self._on_wrist_image, 10, callback_group=self._cb_group)

        # ADJUST: neither backend currently reports the gripper's true
        # current position back to this node (UR5eInterface's real gripper
        # driver isn't wired up -- see robot_interface.py's module
        # docstring -- and pick_place_scene_bridge.py doesn't publish it
        # separately from joint state yet). Tracked here as the last
        # commanded value instead of a true sensor reading; close this gap
        # before trusting gripper-state-dependent policy behavior.
        self._last_commanded_gripper = 0.0

        self._step_count = 0
        self._timer = self.create_timer(
            self.control_period_s, self._run_step, callback_group=self._cb_group)

    def _on_base_image(self, msg: Image):
        self._base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _on_wrist_image(self, msg: Image):
        self._wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _on_cube_position(self, msg: Point):
        self._latest_cube_position = np.array([msg.x, msg.y, msg.z])

    def _run_step(self):
        if self._step_count >= self.max_steps:
            if self.eval_mode:
                self._finish_trial(success=False)
                return
            self.get_logger().info(f'max_steps ({self.max_steps}) reached, stopping.')
            self.robot.stop()
            return
        if self._base_image is None or self._wrist_image is None:
            self.get_logger().warn('waiting for camera images...', throttle_duration_sec=2.0)
            return

        joints = self.robot.get_joint_positions()
        if joints is None:
            self.get_logger().warn('waiting for joint state...', throttle_duration_sec=2.0)
            return

        obs = {
            "joints": np.asarray(joints[:6], dtype=np.float32),
            "gripper": np.array([self._last_commanded_gripper], dtype=np.float32),
            "base_rgb": self._base_image,
            "wrist_rgb": self._wrist_image,
            "prompt": self.prompt,
        }
        result = self.policy.infer(obs)
        # openpi's UR5eOutputs/AbsoluteActions transforms (server-side) already
        # convert the model's predicted deltas back to absolute joint
        # positions -- what comes back here is a ready-to-execute action
        # (chunk_len, 7) or (7,), NOT something this client needs to
        # de-delta itself.
        action_chunk = np.asarray(result["actions"])
        action = action_chunk[0] if action_chunk.ndim == 2 else action_chunk

        target_joints = action[:6]
        target_gripper = float(np.clip(action[6], 0.0, 1.0))

        if self.residual_policy is not None:
            # Same 14-dim observation residual_rl_train_env.py trains
            # against: proprioception + pi0's own proposed action -- no
            # object-pose info, so this works identically in sim and here.
            residual_obs = np.concatenate([obs["joints"], obs["gripper"], action]).astype(np.float32)
            residual_action, _ = self.residual_policy.predict(residual_obs, deterministic=True)
            target_joints = target_joints + self.residual_scale * np.clip(residual_action, -1.0, 1.0)

        self.robot.move_joints(target_joints)
        self.robot.set_gripper(target_gripper)
        self._last_commanded_gripper = target_gripper

        self._step_count += 1

        if self.eval_mode and self._latest_cube_position is not None:
            is_holding = (target_gripper >= self.holding_gripper_threshold
                          and self._latest_cube_position[2] > self.lifted_z_threshold)
            xy_error_mm = 1000.0 * float(np.linalg.norm(
                self._latest_cube_position[:2] - self.eval_target_position[:2]))
            released_here = self._was_holding and not is_holding
            self._was_holding = is_holding
            if released_here and xy_error_mm <= self.success_xy_tolerance_m * 1000.0:
                self._finish_trial(success=True, xy_error_mm=xy_error_mm)
                return

        self.get_logger().info(
            f'step {self._step_count}/{self.max_steps}: gripper={target_gripper:.2f}',
            throttle_duration_sec=1.0)

    def _finish_trial(self, success, xy_error_mm=-1.0):
        """Logs one trial's outcome, then either wraps up the eval run (all
        n_trials done) or requests a scene reset and starts the next trial."""
        self._trial_idx += 1
        self._trial_successes += int(success)
        with open(self.results_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([self._trial_idx, success, self._step_count, xy_error_mm])
        self.get_logger().info(
            f'trial {self._trial_idx}/{self.n_trials}: '
            f'{"SUCCESS" if success else "FAILURE"} (steps={self._step_count}, xy_error_mm={xy_error_mm:.1f})')

        if self._trial_idx >= self.n_trials:
            self.get_logger().info(
                f'{self._trial_successes}/{self.n_trials} trials succeeded -- eval complete, stopping.')
            self.robot.stop()
            self._timer.cancel()
            return

        self._eval_reset_pub.publish(Empty())
        self._step_count = 0
        self._was_holding = False


def main():
    rclpy.init()
    node = VLAPolicyClient()
    # MultiThreadedExecutor (paired with the ReentrantCallbackGroup set up in
    # __init__), not rclpy.spin(node)'s default single-threaded one: the
    # isaac_sim backend's move_joints blocks _run_step in a poll loop
    # waiting on joint_state, which only refreshes via this same node's own
    # subscription callback. A single-threaded executor can't service that
    # subscription while _run_step is still running, so the poll would
    # never see fresh data and every move would just time out. Only matters
    # for robot_backend=isaac_sim -- the real UR5e path talks to RTDE
    # directly, no ROS2 subscription involved, so rclpy.spin(node) was fine
    # there.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
