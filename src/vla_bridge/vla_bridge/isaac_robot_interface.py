"""ROS2-facing robot backend for Isaac Sim, matching robot_interface.
UR5eInterface's method surface (move_joints, get_joint_positions,
gripper_open/close, stop, close) so vla_policy_client.py runs against either
backend unchanged -- same dual-backend pattern
potato_drill_ws/src/potato_scan/potato_scan/isaac_robot_interface.py uses.

Talks to isaac/pick_place_scene_bridge.py (a separate standalone script run
inside Isaac Sim itself, analogous to potato_scan's isaac_scene.py) over
plain ROS2 topics:

  publish   sensor_msgs/JointState  -> joint_target_topic
            (the sim applies this as a direct joint-position target --
            no Cartesian IK needed at inference time, since openpi's UR5e
            policy already outputs joint-space actions; RMPflow in
            pick_place_scene.py is only used to GENERATE the scripted demo
            data in collect_demos.py, not at policy inference time)
  publish   std_msgs/Float32        -> gripper_target_topic (0=open..1=closed)
  subscribe sensor_msgs/JointState  <- joint_state_topic (current arm joint positions)

Written and reasoned about WITHOUT the ability to run this pipeline in the
environment this was authored in -- treat as a solid first draft.
"""
import numpy as np

from vla_bridge.gripper_state import GripperState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


class IsaacSimRobotInterface:
    def __init__(self, node, joint_target_topic='/vla/joint_target',
                 gripper_target_topic='/vla/gripper_target',
                 joint_state_topic='/vla/joint_state',
                 settle_timeout_s=3.0, joint_tolerance_rad=0.02,
                 callback_group=None):
        """callback_group: pass a ReentrantCallbackGroup shared with the
        calling node's timer, spun via a MultiThreadedExecutor -- move_joints
        below blocks polling get_joint_positions() from inside that timer
        callback, and the joint_state subscription that actually refreshes
        it needs to run concurrently with that poll, not queued behind it.
        A single-threaded executor (or leaving this on the node's default
        MutuallyExclusiveCallbackGroup) would never service this
        subscription while the timer callback is still running, so the
        polled value would never change and every move would just time out
        -- see vla_policy_client.py's main()."""
        self.node = node
        self.settle_timeout_s = settle_timeout_s
        self.joint_tolerance_rad = joint_tolerance_rad

        self.joint_target_pub = node.create_publisher(JointState, joint_target_topic, 10)
        self.gripper_target_pub = node.create_publisher(Float32, gripper_target_topic, 10)
        self._last_commanded_gripper = 0.0

        self._latest_joint_state = None
        node.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, 10, callback_group=callback_group)

    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    def get_joint_positions(self):
        if self._latest_joint_state is None:
            return None
        # pick_place_scene_bridge.py's publish_observation() appends the
        # gripper position as a 7th element after the 6 arm joints -- slice
        # it off here so this matches UR5eInterface.get_joint_positions()'s
        # 6-dim contract (both the settle-check below and vla_policy_client's
        # move_joints() calls compare this against a 6-dim target).
        return np.array(self._latest_joint_state.position[:6])

    def get_tcp_pose(self):
        # Not used by the VLA client (which works in joint space throughout)
        # -- present only for interface-shape parity with robot_interface.UR5eInterface.
        return None, None

    def move_joints(self, joint_positions, speed=None, acceleration=None):
        """Publishes the target and polls the joint-state subscription
        until it settles within tolerance or settle_timeout_s elapses.
        `speed`/`acceleration` accepted for interface compatibility but
        unused -- the sim's target-tracking rate is fixed in
        pick_place_scene_bridge.py.
        """
        import time
        msg = JointState()
        msg.position = [float(p) for p in np.asarray(joint_positions, dtype=float)]
        self.joint_target_pub.publish(msg)

        target = np.asarray(joint_positions, dtype=float)
        t0 = time.time()
        while time.time() - t0 < self.settle_timeout_s:
            time.sleep(0.02)
            current = self.get_joint_positions()
            if current is not None and np.max(np.abs(current - target)) <= self.joint_tolerance_rad:
                return True
        return False

    def set_gripper(self, position: float):
        self._last_commanded_gripper = float(position)
        self.gripper_target_pub.publish(Float32(data=float(position)))

    def get_gripper_state(self):
        """Where the gripper actually is, read back from the simulator.

        pick_place_scene_bridge appends it to joint_state as a 7th element,
        and the value is GripperController.get_normalized_position() -- a
        reading of the drive joint, already on the 0..1 scale set_gripper
        uses. get_joint_positions() slices it off to keep its 6-dim arm
        contract; this is where it gets used instead of thrown away.

        object_detected stays None: the simulator has no equivalent of the
        Robotiq gOBJ status byte unless a fingertip contact sensor is added
        to the scene, and inferring contact from position alone would be a
        guess dressed as a measurement.
        """
        if self._latest_joint_state is None:
            return GripperState.from_command(
                self._last_commanded_gripper, 'no joint_state received yet')
        position = self._latest_joint_state.position
        if len(position) < 7:
            return GripperState.from_command(
                self._last_commanded_gripper,
                f'joint_state has {len(position)} values, expected 7 with the gripper last')
        return GripperState(position=float(position[6]), measured=True,
                            object_detected=None, source='sim joint_state[6]')

    def gripper_open(self):
        self.set_gripper(0.0)

    def gripper_close(self):
        self.set_gripper(1.0)

    def stop(self):
        pass  # no in-flight trajectory queue to cancel with a joint-target interface

    def close(self):
        pass
