"""Isaac Sim scene-building helpers for the UR5e pick-and-place VLA data
pipeline -- shared by pick_place_scene.py / scripted_pick_place.py /
collect_demos.py.

Adapted from potato_drill_ws/src/potato_scan/isaac/isaac_sim_common.py's
robot-loading / RMPflow-setup / prim-pose-readout pattern (same UR5e asset,
same RMPflow config loader), but for a **gripper** end-effector instead of
the drill tip that project used -- this project needs to actually grasp and
move an object, not just contact one.

Written and reasoned about against Isaac Sim 5.1, WITHOUT the ability to
actually run Isaac Sim in the environment this was authored in -- same
"solid first draft, not verified" posture as the potato_scan project's own
isaac_scene.py / isaac_sim_common.py. The gripper section below is the least
certain part (exact asset path and joint-control API vary more across Isaac
Sim versions than the arm-loading path does) -- ADJUST markers flag it.

Only import this from a script already running inside Isaac Sim's own Kit
runtime (i.e. after `from isaacsim import SimulationApp; SimulationApp(...)`
has run).
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# ADJUST: relative path under the Isaac asset root for the UR5e USD. Same
# asset potato_drill_ws/isaac/isaac_sim_common.py uses -- if that project's
# copy needed adjusting on your install, apply the same fix here.
UR5E_ASSET_RELATIVE_PATH = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

# ADJUST: exact prim names inside the loaded UR5e USD. Check the Stage
# window after the robot loads if TF/camera/gripper placement looks wrong.
ROBOT_PRIM_PATH = "/World/ur5e"
TOOL_LINK_PRIM_PATH = "/World/ur5e/tool0"

# ADJUST: this is the least-verified path in this file. A Robotiq 2F-85 is
# the common eye-in-hand-friendly parallel gripper bundled with recent Isaac
# Sim asset packs; open the Asset Browser (Window -> Browsers -> Assets) and
# search "robotiq" or "2f-85" to confirm the exact path/USD name on your
# install, and update GRIPPER_ASSET_RELATIVE_PATH / GRIPPER_DRIVE_JOINT_NAME
# to match what you find (a mimic-jointed gripper may drive one joint and
# mirror the rest, or need all finger joints set together).
GRIPPER_ASSET_RELATIVE_PATH = "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
GRIPPER_DRIVE_JOINT_NAME = "finger_joint"
GRIPPER_OPEN_POS = 0.0     # radians -- ADJUST against your gripper's joint limits
GRIPPER_CLOSED_POS = 0.68  # radians -- ADJUST; should be a firm-but-not-overdriven close


def add_gripper(stage, parent_path, assets_root, prim_path="gripper"):
    """References the gripper USD as a child of the UR5e tool flange.
    Returns the gripper's prim path."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    full_path = f"{parent_path}/{prim_path}"
    add_reference_to_stage(assets_root + GRIPPER_ASSET_RELATIVE_PATH, full_path)
    return full_path


class GripperController:
    """Drives the gripper's finger joint to an open/closed position target.

    ADJUST: assumes a single drive joint (GRIPPER_DRIVE_JOINT_NAME) that the
    rest of the gripper's fingers mimic mechanically/via the USD's own mimic
    joint setup -- true for a stock Robotiq 2F-85 asset, but verify against
    whatever gripper USD you actually load. If your gripper instead needs
    each finger joint driven independently, extend set_target() to loop over
    a list of joint names instead of the single GRIPPER_DRIVE_JOINT_NAME.
    """

    def __init__(self, gripper_prim_path):
        from isaacsim.core.prims import Articulation

        self.articulation = Articulation(gripper_prim_path)
        self._drive_joint_index = None  # resolved lazily, once the articulation is initialized

    def _resolve_joint_index(self):
        if self._drive_joint_index is None:
            joint_names = list(self.articulation.dof_names)
            self._drive_joint_index = joint_names.index(GRIPPER_DRIVE_JOINT_NAME)
        return self._drive_joint_index

    def set_target(self, position):
        """position: 0.0 (fully open) .. 1.0 (fully closed), linearly
        mapped to [GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS]."""
        idx = self._resolve_joint_index()
        target = GRIPPER_OPEN_POS + float(np.clip(position, 0.0, 1.0)) * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        targets = self.articulation.get_joint_positions()
        targets[0, idx] = target
        self.articulation.set_joint_position_targets(targets)

    def open(self):
        self.set_target(0.0)

    def close(self):
        self.set_target(1.0)

    def get_normalized_position(self):
        """Current drive-joint position, mapped back to 0.0 (open) .. 1.0
        (closed) -- the inverse of set_target(), used when logging the
        gripper's current state as part of an observation."""
        idx = self._resolve_joint_index()
        current = float(self.articulation.get_joint_positions()[0, idx])
        span = GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS
        return float(np.clip((current - GRIPPER_OPEN_POS) / span, 0.0, 1.0))


def setup_rmpflow(robot_articulation):
    """ADJUST: the exact robot-name string load_supported_motion_policy_config
    expects for the UR5e may differ from "UR5e" on your install -- if this
    raises/returns None, call get_supported_robot_policy_pairs() (same
    module) to list the exact registered names and swap it in below. Same
    helper as potato_drill_ws's isaac_sim_common.py."""
    from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_motion_policy_config)

    rmp_config = load_supported_motion_policy_config("UR5e", "RMPflow")
    rmpflow = RmpFlow(**rmp_config)
    physics_dt = 1.0 / 60.0
    return rmpflow, ArticulationMotionPolicy(robot_articulation, rmpflow, physics_dt)


def prim_world_pose(prim):
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = mat.ExtractTranslation()
    quat = mat.ExtractRotationQuat()
    quat_xyzw = [quat.imaginary[0], quat.imaginary[1], quat.imaginary[2], quat.real]
    return np.array([translation[0], translation[1], translation[2]]), np.array(quat_xyzw)


def add_cube(stage, prim_path, position, size=0.04, color=(0.8, 0.1, 0.1)):
    """A small dynamic (rigid-body) cube to pick up."""
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(size)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(0.05)  # 50g -- light enough for a small parallel gripper
    return cube


def add_place_target_marker(stage, prim_path, position, size=0.10):
    """A flat, non-colliding marker showing where the cube should be placed
    -- visual/logging aid only, not a physical obstacle."""
    marker = UsdGeom.Cylinder.Define(stage, prim_path)
    marker.CreateRadiusAttr(size / 2.0)
    marker.CreateHeightAttr(0.002)
    marker.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.001))
    marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.1)])
    return marker
