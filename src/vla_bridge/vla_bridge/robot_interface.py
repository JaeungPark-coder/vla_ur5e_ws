"""Thin wrapper around ur_rtde for the VLA policy client -- adapted from
potato_drill_ws/src/potato_scan/potato_scan/robot_interface.py's
UR5eInterface (same move_to_pose/get_tcp_pose/stop/close contract), with the
drill-specific methods (drill_on/drill_off/force_drill) dropped -- this
project drives a gripper, not a drill -- and gripper_open/gripper_close/
set_gripper added instead.

ADJUST: gripper control below is a placeholder using the same tool digital
output the potato_scan project used for its drill relay -- correct only if
your gripper is wired as a simple on/off relay on that pin. A Robotiq 2F-85
(the gripper assumed in isaac/isaac_sim_common.py) is more commonly
integrated via its own URCap / Modbus driver with continuous position
control -- if that's your setup, replace set_gripper()'s body with the
appropriate Robotiq driver call (e.g. via `pymodbus` or the URCap's
RTDE-exposed registers) instead of setToolDigitalOut.
"""
import numpy as np
import rtde_control
import rtde_receive
import rtde_io


class UR5eInterface:
    def __init__(self, robot_ip, speed=0.25, acceleration=0.5, gripper_output_pin=0,
                 servo_time=0.1, servo_lookahead_time=0.1, servo_gain=300):
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
        self.gripper_output_pin = gripper_output_pin
        # servoJ parameters -- see move_joints. servo_time should match the
        # control period of whatever is streaming targets.
        self.servo_time = servo_time
        self.servo_lookahead_time = servo_lookahead_time
        self.servo_gain = servo_gain
        self.control = rtde_control.RTDEControlInterface(robot_ip)
        self.receive = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.io = rtde_io.RTDEIOInterface(robot_ip)

    def move_to_pose(self, position, rotvec, speed=None, acceleration=None):
        """Returns True on success, False on a target the controller
        rejects as unreachable (rather than raising)."""
        pose = list(np.asarray(position, dtype=float)) + list(np.asarray(rotvec, dtype=float))
        try:
            result = self.control.moveL(pose, speed or self.speed, acceleration or self.acceleration)
        except RuntimeError:
            return False
        return result is not False

    def move_joints(self, joint_positions, speed=None, acceleration=None):
        """Stream a joint-space target for a policy running at a fixed rate.

        Uses servoJ, NOT moveJ. moveJ is a blocking point-to-point move: it
        plans a full trapezoidal accel/cruise/decel profile to the target and
        only returns once the arm is there. For the small per-tick deltas a
        VLA policy emits that is wrong twice over -- the arm re-accelerates
        from rest every tick, so motion comes out stuttering rather than
        continuous; and each call takes longer than the control period (at
        speed=0.25 rad/s, acceleration=0.5 rad/s^2 even a 0.02 rad step needs
        ~0.4 s against a 100 ms budget at control_hz=10), which pushes the
        whole control loop behind.

        servoJ is the real-time streaming counterpart: it returns immediately
        and the controller keeps tracking the most recent target, so each new
        command refines the motion instead of restarting it. `stop()` calls
        servoStop() to end the servo mode cleanly.

        `speed`/`acceleration` are accepted for interface compatibility and
        ignored -- servoJ's own signature takes them but ur_rtde documents
        them as unused; tracking is shaped by lookahead_time and gain.

        ADJUST: servo_time should match the caller's control period
        (vla_policy_client's control_hz). lookahead_time (0.03-0.2 s) smooths
        the trajectory, gain (100-2000) sets how hard the controller pulls
        toward the target -- raise gain for tighter tracking, lower it if the
        arm feels harsh. Tune these on the real robot at low speed.
        """
        try:
            result = self.control.servoJ(
                list(np.asarray(joint_positions, dtype=float)),
                0.0, 0.0,  # speed/acceleration: unused by servoJ
                self.servo_time, self.servo_lookahead_time, self.servo_gain)
        except RuntimeError:
            return False
        return result is not False

    def move_joints_blocking(self, joint_positions, speed=None, acceleration=None):
        """Point-to-point joint move that returns only once the arm arrives.

        Kept for one-off repositioning (homing, moving to a start pose) --
        anything that is a discrete move rather than a streamed one. Do not
        use it inside a policy control loop; see move_joints."""
        try:
            result = self.control.moveJ(
                list(np.asarray(joint_positions, dtype=float)),
                speed or self.speed, acceleration or self.acceleration)
        except RuntimeError:
            return False
        return result is not False

    def get_tcp_pose(self):
        pose = self.receive.getActualTCPPose()
        return np.array(pose[:3]), np.array(pose[3:])

    def get_joint_positions(self):
        return np.array(self.receive.getActualQ())

    def set_gripper(self, position: float):
        """position: 0.0 (open) .. 1.0 (closed). ADJUST -- see module
        docstring; this placeholder only supports a binary relay (position
        >= 0.5 -> closed)."""
        self.io.setToolDigitalOut(self.gripper_output_pin, position >= 0.5)

    def gripper_open(self):
        self.set_gripper(0.0)

    def gripper_close(self):
        self.set_gripper(1.0)

    def stop(self):
        """End servo mode and halt. servoStop() is the counterpart to
        move_joints' servoJ; stopL alone leaves the servo loop running."""
        try:
            self.control.servoStop()
        except RuntimeError:
            pass
        self.control.stopL()

    def close(self):
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.io.disconnect()
