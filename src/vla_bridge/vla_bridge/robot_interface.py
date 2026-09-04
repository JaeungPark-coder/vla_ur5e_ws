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
    def __init__(self, robot_ip, speed=0.25, acceleration=0.5, gripper_output_pin=0):
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
        self.gripper_output_pin = gripper_output_pin
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
        """Direct joint-space move -- used when applying a VLA policy's
        joint-space action chunk (openpi's UR5e contract logs/predicts
        joint positions, not Cartesian poses -- see
        ../../../openpi_integration/ur5e_pick_place_policy.py)."""
        try:
            result = self.control.moveJ(
                list(np.asarray(joint_positions, dtype=float)), speed or self.speed, acceleration or self.acceleration)
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
        self.control.stopL()

    def close(self):
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.io.disconnect()
