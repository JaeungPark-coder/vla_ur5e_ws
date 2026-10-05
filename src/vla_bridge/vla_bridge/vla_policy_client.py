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
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from vla_bridge.gripper_state import grasp_disagreement
from vla_bridge.robot_interface import UR5eInterface
from vla_bridge.isaac_robot_interface import IsaacSimRobotInterface
from vla_bridge.gripper_command import GripperHysteresis
from vla_bridge.feedforward import chunk_velocities


class VLAPolicyClient(Node):
    def __init__(self):
        super().__init__('vla_policy_client')

        # See main()'s MultiThreadedExecutor for why this exists: the
        # isaac_sim backend's move_joints polls joint_state (a subscription
        # on this same node, created inside IsaacSimRobotInterface) from
        # inside _run_step's own timer callback, so that subscription needs
        # to run concurrently with _run_step, not queued behind it.
        self._cb_group = ReentrantCallbackGroup()
        # The control loop gets its OWN mutually-exclusive group so it cannot
        # overlap itself. ReentrantCallbackGroup does not merely let
        # *different* callbacks run concurrently -- rclpy's
        # ReentrantCallbackGroup.can_execute() returns True unconditionally,
        # and Executor._make_handler clears the entity's _executor_event as
        # soon as the timer is *taken* (which resets it), before awaiting the
        # callback. So with a MultiThreadedExecutor the same timer is
        # re-dispatched while the previous _run_step is still running.
        #
        # That matters here because _run_step blocks for far longer than its
        # period: control_hz defaults to 10 (100 ms) while a single step does
        # a policy.infer() round-trip plus a move_joints() that waits for the
        # arm. Overlapping runs would publish conflicting joint targets, race
        # on _step_count/_last_commanded_gripper, and -- worst -- call
        # policy.infer() concurrently on one WebsocketClientPolicy, which is a
        # single connection and not safe to share.
        #
        # Subscriptions stay in the reentrant group, so they still refresh
        # while _run_step blocks; the two groups run concurrently under the
        # MultiThreadedExecutor. Only self-overlap is forbidden.
        self._control_cb_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('robot_backend', 'rtde')  # 'rtde' (real UR5e) or 'isaac_sim'
        self.declare_parameter('policy_host', 'localhost')
        self.declare_parameter('policy_port', 8000)
        self.declare_parameter('prompt', 'pick up the cube and place it in the target zone')
        # 2026-09-23: was 10.0. collect_demos.py records at CONTROL_HZ=60.0
        # (isaac/scripted_pick_place.py) and the model learns per-tick joint
        # DELTAS at that rate (openpi_integration/train_config_snippet.py's
        # DeltaActions mask); this node re-queries and applies only
        # action_chunk[0] once per control tick (see _run_step below), so
        # serving at 10Hz replayed those deltas ~6x too slowly. Matching the
        # recording rate here only fixes the arithmetic, not whether a real
        # UR5e + a policy.infer() round-trip can actually keep to 60Hz --
        # confirm that live before trusting it; if it can't, the fix is
        # re-recording demonstrations at a rate serving can sustain, not
        # loosening this back down.
        self.declare_parameter('control_hz', 60.0)
        # ADJUST: real-hardware camera topics -- match your actual camera driver's topic names.
        self.declare_parameter('base_image_topic', '/camera/base/image_raw')
        self.declare_parameter('wrist_image_topic', '/camera/wrist/image_raw')
        # 2026-09-23: was 'none'. Only consulted when robot_backend='rtde'
        # (the isaac_sim backend always reads the simulated joint state
        # instead, unaffected by this default either way) -- but on real
        # hardware, 'none' means robot_interface.py's placeholder relay,
        # gripper state reported as NOT MEASURED, so this node feeds the
        # policy the value it just commanded instead of what the gripper
        # actually did (a grasp that failed to close then looks identical
        # to one that worked -- see the runtime warning a few hundred lines
        # down, which already flagged this; this just makes the default
        # match the warning instead of contradicting it). 'robotiq_socket':
        # a real Robotiq over the UR controller's Socket ADI interface (port
        # gripper_socket_port at robot_ip) -- see robotiq_socket_gripper.py.
        # Nothing in this project has run against real hardware yet (every
        # rtde-path file's own docstring says so), so there is no working
        # real-hardware launch this default change can break; it only means
        # the FIRST real launch fails loudly at startup (if no Robotiq is
        # actually reachable at robot_ip:gripper_socket_port) instead of
        # silently collecting unmeasured-gripper data. Pass --gripper_driver
        # none explicitly if you deliberately want the old placeholder
        # relay (e.g. bring-up with the gripper not yet wired).
        self.declare_parameter('gripper_driver', 'robotiq_socket')
        self.declare_parameter('gripper_socket_port', 63352)
        # 2026-10-05: was 300. Demonstrations are ~1044 actions long (one action
        # per 60 Hz sim tick, 17 s -- see isaac/collect_demos.py), so a cap of 300
        # ended every trial at ~29% of the task and logged it FAILURE no matter how
        # good the policy was. Counted in ACTIONS applied (execute_horizon per
        # inference), not inferences.
        self.declare_parameter('max_steps', 1800)  # safety cap -- stop after this many actions regardless of task completion
        # 2026-10-05: the dataset's gripper action is the MEASURED position one tick
        # later (~0.7 while holding the cube) but the demonstrator COMMANDED 1.0.
        # gripper_hysteresis turns the policy output into a close/open decision, on
        # the guess that echoing ~0.7 leaves ~0 squeeze force. A replay check in the
        # simulator (isaac/check_action_replay.py) found no benefit: raw 10/10,
        # continuous x1/0.7 scaling 6/6, hysteresis 8/10 (small n; the difference may
        # be chance). Recorded values only -- a trained policy's noisy output may differ.
        # So it defaults to OFF (raw clipped value, the original behaviour); the
        # option stays for experiments. Thresholds below only matter when it is on.
        self.declare_parameter('gripper_hysteresis', False)
        self.declare_parameter('gripper_close_threshold', 0.45)
        self.declare_parameter('gripper_open_threshold', 0.25)
        # 2026-10-05: apply this many actions of each inferred chunk before
        # re-inferring. 1 = the original behaviour (re-infer every action). Every
        # action in a chunk is absolute and relative to the state at inference
        # time (AbsoluteActions), so open-loop execution of the first K is valid.
        # Forced to 1 when use_residual_policy is on (its observation is per-action).
        self.declare_parameter('execute_horizon', 1)
        # 2026-10-05: isaac_sim backend only. true = one action <-> exactly one sim
        # tick (the bridge must be started with --lockstep), so inference latency no
        # longer changes how much sim time an action covers. false = the original
        # free-running bridge. See isaac/lockstep_protocol.py for why.
        self.declare_parameter('lockstep', False)
        # 2026-10-05: lockstep only. Send a joint-VELOCITY target with every position
        # target, taken from the difference of consecutive rows of the policy's chunk
        # (see feedforward.py). The expert's RMPflow commands carried one every tick;
        # positions alone made the arm lag the demonstrated motion by ~0.30 rad RMS in
        # offline replay, with the velocity target ~0.0002 rad.
        self.declare_parameter('velocity_feedforward', False)
        # Smoothing/clipping of that feed-forward (feedforward.py): the plain difference of
        # consecutive rows amplifies row noise, which made offline replay diverge. 11 = ~0.18 s;
        # same tracking as 1 on clean rows (7/7 placed both), better at 0.01 rad row noise.
        self.declare_parameter('velocity_smoothing_window', 11)
        self.declare_parameter('velocity_clip_rad_s', 3.0)
        self.declare_parameter('action_dt_s', 1.0 / 60.0)  # time between consecutive chunk rows = one sim tick (the data's rate; NOT the dataset's fps=30 label)
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
        # 2026-09-23: was 0.03 ("matches residual_rl_train_env.py" -- true,
        # but residual_rl_train_env.py's own 0.03 was itself a stray
        # duplicate of isaac/pick_place_scene.py's LIFT_Z_THRESHOLD, now
        # fixed to import that instead of redeclaring it). LIFT_Z_THRESHOLD
        # is 0.08 -- this file cannot import it directly (it must run with
        # no Isaac Sim on the real robot, and pick_place_scene.py pulls in
        # isaacsim at import time), so the value is copied here by hand.
        # Keep this in sync with pick_place_scene.LIFT_Z_THRESHOLD if that
        # ever changes -- nothing enforces the two matching automatically.
        # Without this, an episode collect_demos.py would reject as "never
        # lifted" (1-3cm) was instead being logged SUCCESS by eval_mode's
        # results_csv, defeating the three-pipeline comparison this csv
        # schema exists for (see the class docstring above).
        self.declare_parameter('lifted_z_threshold', 0.08)
        self.declare_parameter('eval_target_position', [0.45, 0.30, 0.0])  # must match isaac/pick_place_scene.py's PLACE_TARGET_POSITION
        self.declare_parameter('results_csv_path', 'vla_eval_results.csv')

        self.prompt = self.get_parameter('prompt').value
        self.control_period_s = 1.0 / self.get_parameter('control_hz').value
        self.max_steps = self.get_parameter('max_steps').value
        self.execute_horizon = max(1, int(self.get_parameter('execute_horizon').value))
        self.lockstep = bool(self.get_parameter('lockstep').value)
        self.velocity_feedforward = bool(self.get_parameter('velocity_feedforward').value)
        self.action_dt_s = float(self.get_parameter('action_dt_s').value)
        self.velocity_smoothing_window = int(self.get_parameter('velocity_smoothing_window').value)
        self.velocity_clip_rad_s = float(self.get_parameter('velocity_clip_rad_s').value)
        self._gripper_filter = None
        if self.get_parameter('gripper_hysteresis').value:
            self._gripper_filter = GripperHysteresis(
                close_above=self.get_parameter('gripper_close_threshold').value,
                open_below=self.get_parameter('gripper_open_threshold').value)
        self._base_ack = 0   # lockstep: request id the latest images belong to
        self._wrist_ack = 0
        self._base_epoch = 0  # lockstep: scene-reset count the latest images belong to
        self._wrist_epoch = 0
        self._min_epoch = 0   # lockstep: after requesting a reset, only observations from a later epoch count
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
        if self.residual_policy is not None and self.execute_horizon != 1:
            self.get_logger().warn('use_residual_policy needs a fresh observation per action; forcing execute_horizon=1')
            self.execute_horizon = 1

        self.backend = self.get_parameter('robot_backend').value
        if self.backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, callback_group=self._cb_group, lockstep=self.lockstep)
            image_topics = ('/vla/base_image', '/vla/wrist_image')
        else:
            if self.lockstep:
                self.get_logger().warn('lockstep only applies to robot_backend=isaac_sim -- ignoring.')
                self.lockstep = False
            robot_ip = self.get_parameter('robot_ip').value
            gripper_driver_name = self.get_parameter('gripper_driver').value
            gripper_driver = None
            if gripper_driver_name == 'robotiq_socket':
                from vla_bridge.robotiq_socket_gripper import RobotiqSocketGripper
                gripper_driver = RobotiqSocketGripper(
                    robot_ip, port=self.get_parameter('gripper_socket_port').value)
                gripper_driver.activate()
                self.get_logger().info(f'gripper_driver=robotiq_socket, activated at {robot_ip}')
            elif gripper_driver_name != 'none':
                raise ValueError(
                    f"unknown gripper_driver {gripper_driver_name!r} -- expected "
                    "'none' or 'robotiq_socket'")
            # servo_time must match this node's control period: servoJ is
            # told how long each streamed target is meant to govern, so a
            # mismatch either starves the controller or overruns the next tick.
            self.robot = UR5eInterface(
                robot_ip, servo_time=self.control_period_s, gripper_driver=gripper_driver)
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

        # Gripper proprioception comes from robot.get_gripper_state() now,
        # which says whether anyone actually measured it. Falling back to the
        # last command is still possible (no driver on the real arm, no
        # joint_state yet in sim) but it is announced once rather than
        # passing silently -- see vla_bridge/gripper_state.py.
        self._warned_unmeasured_gripper = False
        self._last_commanded_gripper = 0.0

        self._step_count = 0
        self._timer = self.create_timer(
            self.control_period_s, self._run_step, callback_group=self._control_cb_group)

    def _on_base_image(self, msg: Image):
        self._base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self._base_ack = int(msg.header.stamp.sec)
        self._base_epoch = int(msg.header.stamp.nanosec)

    def _on_wrist_image(self, msg: Image):
        self._wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        self._wrist_ack = int(msg.header.stamp.sec)
        self._wrist_epoch = int(msg.header.stamp.nanosec)

    def _observation_is_current(self):
        """Lockstep: true once joint_state AND both images carry the id of the
        last request sent -- i.e. all three are from the tick that request
        produced, not an earlier one still in flight."""
        wanted = self.robot.last_request_id()
        return (self.robot.acked_request_id() >= wanted
                and self._base_ack >= wanted and self._wrist_ack >= wanted
                and self.robot.acked_epoch() >= self._min_epoch
                and self._base_epoch >= self._min_epoch and self._wrist_epoch >= self._min_epoch)

    def _gripper_command(self, policy_output):
        if self._gripper_filter is not None:
            return self._gripper_filter.update(policy_output)
        return float(np.clip(policy_output, 0.0, 1.0))

    def _on_cube_position(self, msg: Point):
        self._latest_cube_position = np.array([msg.x, msg.y, msg.z])

    def _run_step(self):
        if self._step_count >= self.max_steps:
            if self.eval_mode:
                self._finish_trial(success=False)
                return
            self.get_logger().info(f'max_steps ({self.max_steps}) reached, stopping.')
            self.robot.stop()
            self._timer.cancel()  # was: logged again on every timer tick, ~60 lines/s forever
            return
        if self._base_image is None or self._wrist_image is None:
            self.get_logger().warn('waiting for camera images...', throttle_duration_sec=2.0)
            return
        if self.lockstep and not self._observation_is_current():
            self.get_logger().warn(
                'lockstep: waiting for the observation produced by the last request...',
                throttle_duration_sec=2.0)
            return

        joints = self.robot.get_joint_positions()
        if joints is None:
            self.get_logger().warn('waiting for joint state...', throttle_duration_sec=2.0)
            return

        gripper_state = self.robot.get_gripper_state()
        if not gripper_state.measured and not self._warned_unmeasured_gripper:
            self._warned_unmeasured_gripper = True
            self.get_logger().warn(
                f'gripper state is not measured ({gripper_state.source}). The policy is '
                f'being fed the value it commanded, which cannot disagree with itself -- '
                f'so a grasp that failed to close looks exactly like one that worked, '
                f'both to the policy and to the eval metric. Note the demonstrations were '
                f'recorded with the MEASURED position, so this is also a train/serve '
                f'mismatch.')

        obs = {
            "joints": np.asarray(joints[:6], dtype=np.float32),
            "gripper": np.array([gripper_state.position], dtype=np.float32),
            "base_rgb": self._base_image,
            "wrist_rgb": self._wrist_image,
            "prompt": self.prompt,
        }
        result = self.policy.infer(obs)
        # openpi's UR5eOutputs/AbsoluteActions transforms (server-side) already
        # convert the model's predicted deltas back to absolute joint
        # positions -- what comes back here is a ready-to-execute action
        # (chunk_len, 7) or (7,), NOT something this client needs to
        # de-delta itself. Every row is absolute and relative to the state at
        # inference time, so applying the first `execute_horizon` rows in a row
        # (open loop) is valid; execute_horizon=1 is the original behaviour.
        action_chunk = np.asarray(result["actions"])
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]

        n_apply = max(1, min(self.execute_horizon, len(action_chunk)))
        velocities = (chunk_velocities(action_chunk[:, :6], self.action_dt_s,
                                       window=self.velocity_smoothing_window,
                                       max_speed=self.velocity_clip_rad_s)
                      if (self.lockstep and self.velocity_feedforward) else None)
        for i in range(n_apply):
            action = action_chunk[i]
            target_joints = action[:6]
            target_gripper = self._gripper_command(float(action[6]))

            if self.residual_policy is not None:
                # Same 14-dim observation residual_rl_train_env.py trains
                # against: proprioception + pi0's own proposed action -- no
                # object-pose info, so this works identically in sim and here.
                residual_obs = np.concatenate([obs["joints"], obs["gripper"], action]).astype(np.float32)
                residual_action, _ = self.residual_policy.predict(residual_obs, deterministic=True)
                target_joints = target_joints + self.residual_scale * np.clip(residual_action, -1.0, 1.0)

            if self.lockstep:
                # The bridge applies the gripper command on the same tick as the
                # joint target, so it has to be queued first (see
                # IsaacSimRobotInterface.move_joints).
                self.robot.set_gripper(target_gripper)
                # Only the last tick of the executed block needs images: the policy looks at
                # the observation at the START of the next chunk and nothing in between.
                acked = self.robot.move_joints(
                    target_joints, joint_velocities=None if velocities is None else velocities[i],
                    publish_images=(i == n_apply - 1 or self._step_count + 1 >= self.max_steps))
                if not acked:
                    self.get_logger().error(
                        'lockstep: no acknowledgement from the bridge -- is pick_place_scene_bridge.py '
                        'running with --lockstep (or VLA_BRIDGE_LOCKSTEP=1)? Both sides must match.',
                        throttle_duration_sec=5.0)
            else:
                self.robot.move_joints(target_joints)
                self.robot.set_gripper(target_gripper)
            self._last_commanded_gripper = target_gripper
            self._step_count += 1
            if self._step_count >= self.max_steps:
                break

        # The observation that the command echo made impossible: the gripper
        # is somewhere other than where it was sent. On a closing command
        # that means it stopped on something -- the object, or nothing while
        # stalling -- and either way it is worth seeing.
        settled = self.robot.get_gripper_state()
        disagreement = grasp_disagreement(target_gripper, settled)
        if disagreement:
            self.get_logger().info(
                f'gripper commanded {target_gripper:.2f}, measured '
                f'{settled.position:.2f} (off by {disagreement:.2f})'
                + ('' if settled.object_detected is None else
                   f', object {"detected" if settled.object_detected else "NOT detected"}'),
                throttle_duration_sec=1.0)

        if self.eval_mode and self._latest_cube_position is not None:
            # Scored on where the gripper IS, not where it was told to go.
            # Using the command here counted a grasp that closed on nothing
            # as a hold, so a run could report success having never picked
            # anything up. Falls back to the command when nothing measured
            # it, which is the old behaviour and is warned about above.
            holding_position = settled.position if settled.measured else target_gripper
            is_holding = (holding_position >= self.holding_gripper_threshold
                          and self._latest_cube_position[2] > self.lifted_z_threshold)
            xy_error_mm = 1000.0 * float(np.linalg.norm(
                self._latest_cube_position[:2] - self.eval_target_position[:2]))
            released_here = self._was_holding and not is_holding
            self._was_holding = is_holding
            if released_here and xy_error_mm <= self.success_xy_tolerance_m * 1000.0:
                self._finish_trial(success=True, xy_error_mm=xy_error_mm)
                return

        self.get_logger().info(
            f'step {self._step_count}/{self.max_steps}: '
            f'gripper commanded {target_gripper:.2f}, {settled.describe()}',
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

        if self.lockstep:
            # Until the bridge has really reset it keeps re-publishing the pre-reset observation;
            # only accept one from a later epoch.
            self._min_epoch = max(self.robot.acked_epoch(), self._base_epoch, self._wrist_epoch) + 1
        self._eval_reset_pub.publish(Empty())
        self._step_count = 0
        if self._gripper_filter is not None:
            self._gripper_filter.reset()
        self._was_holding = False
        # Reused as the "not ready yet" gate: without this, _run_step keeps
        # scoring against the previous trial's stale cache while the bridge
        # is synchronously blocked inside scene.reset(), which can log a
        # false SUCCESS before this trial has done anything.
        self._base_image = None
        self._wrist_image = None
        self._latest_cube_position = None


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
