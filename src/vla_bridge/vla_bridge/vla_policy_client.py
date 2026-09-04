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
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vla_bridge.robot_interface import UR5eInterface
from vla_bridge.isaac_robot_interface import IsaacSimRobotInterface


class VLAPolicyClient(Node):
    def __init__(self):
        super().__init__('vla_policy_client')

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

        self.prompt = self.get_parameter('prompt').value
        self.control_period_s = 1.0 / self.get_parameter('control_hz').value
        self.max_steps = self.get_parameter('max_steps').value

        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.policy = WebsocketClientPolicy(
            host=self.get_parameter('policy_host').value,
            port=self.get_parameter('policy_port').value,
        )

        self.backend = self.get_parameter('robot_backend').value
        if self.backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(self)
            image_topics = ('/vla/base_image', '/vla/wrist_image')
        else:
            self.robot = UR5eInterface(self.get_parameter('robot_ip').value)
            image_topics = (
                self.get_parameter('base_image_topic').value,
                self.get_parameter('wrist_image_topic').value,
            )

        self.cv_bridge = CvBridge()
        self._base_image = None
        self._wrist_image = None
        self.create_subscription(Image, image_topics[0], self._on_base_image, 10)
        self.create_subscription(Image, image_topics[1], self._on_wrist_image, 10)

        # ADJUST: neither backend currently reports the gripper's true
        # current position back to this node (UR5eInterface's real gripper
        # driver isn't wired up -- see robot_interface.py's module
        # docstring -- and pick_place_scene_bridge.py doesn't publish it
        # separately from joint state yet). Tracked here as the last
        # commanded value instead of a true sensor reading; close this gap
        # before trusting gripper-state-dependent policy behavior.
        self._last_commanded_gripper = 0.0

        self._step_count = 0
        self.create_timer(self.control_period_s, self._run_step)

    def _on_base_image(self, msg: Image):
        self._base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _on_wrist_image(self, msg: Image):
        self._wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def _run_step(self):
        if self._step_count >= self.max_steps:
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

        self.robot.move_joints(target_joints)
        self.robot.set_gripper(target_gripper)
        self._last_commanded_gripper = target_gripper

        self._step_count += 1
        self.get_logger().info(
            f'step {self._step_count}/{self.max_steps}: gripper={target_gripper:.2f}',
            throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = VLAPolicyClient()
    try:
        rclpy.spin(node)
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
