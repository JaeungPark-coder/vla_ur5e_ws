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

# Verified against the actual Isaac Sim 5.1 ur5e.usd asset's prim tree: there
# is no "tool0" prim (that's a ROS URDF-ism, not what this USD uses) -- the
# real tool flange child Xform is wrist_3_link/flange. The asset also ships
# an empty /World/ur5e/Gripper placeholder Xform plus a dangling
# "robot_gripper_joint" fixed joint (body1 unset), presumably meant for
# exactly this kind of attachment, but unused here -- see GRIPPER_PRIM_PATH
# below for why.
ROBOT_PRIM_PATH = "/World/ur5e"
TOOL_LINK_PRIM_PATH = "/World/ur5e/wrist_3_link/flange"

# ADJUST: this is the least-verified path in this file. A Robotiq 2F-85 is
# the common eye-in-hand-friendly parallel gripper bundled with recent Isaac
# Sim asset packs; open the Asset Browser (Window -> Browsers -> Assets) and
# search "robotiq" or "2f-85" to confirm the exact path/USD name on your
# install, and update GRIPPER_ASSET_RELATIVE_PATH / GRIPPER_DRIVE_JOINT_NAME
# to match what you find (a mimic-jointed gripper may drive one joint and
# mirror the rest, or need all finger joints set together).
GRIPPER_ASSET_RELATIVE_PATH = "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
# A TOP-LEVEL prim, deliberately NOT nested under ROBOT_PRIM_PATH/
# TOOL_LINK_PRIM_PATH. Confirmed by direct testing: referencing the gripper
# as a descendant of the arm's own already-rooted articulation subtree
# (e.g. TOOL_LINK_PRIM_PATH/gripper) leaves its PhysX articulation view
# broken (`_physics_view`'s backend is None -- an AttributeError on
# `.is_homogeneous`/`.count` the moment you try to read/drive its joints),
# even though the arm's own SingleArticulation initializes fine and the
# *same* gripper asset referenced at a top-level sibling path (this one)
# initializes fine too. Kept kinematically attached to the flange instead by
# teleporting it to the flange's current world pose every tick (see
# GripperController.sync_pose_to_flange, called from step_towards()) rather
# than via USD Xform parenting or the dangling robot_gripper_joint.
GRIPPER_PRIM_PATH = "/World/gripper"
GRIPPER_DRIVE_JOINT_NAME = "finger_joint"
GRIPPER_OPEN_POS = 0.0     # radians -- ADJUST against your gripper's joint limits
GRIPPER_CLOSED_POS = 0.68  # radians -- ADJUST; should be a firm-but-not-overdriven close


GRIPPER_BASE_LINK_PRIM_PATH = f"{GRIPPER_PRIM_PATH}/Robotiq_2F_85/base_link"
# The actual colliding geometry on the arm's last link lives on
# wrist_3_link/collisions, not on flange itself -- flange (TOOL_LINK_PRIM_PATH)
# is just a coordinate-frame Xform with no collider of its own, so filtering
# against it would filter nothing.
WRIST_3_LINK_PRIM_PATH = f"{ROBOT_PRIM_PATH}/wrist_3_link"


def add_gripper(stage, assets_root, arm_link_prim_path=WRIST_3_LINK_PRIM_PATH):
    """References the gripper USD at the top-level GRIPPER_PRIM_PATH (see
    its comment for why this isn't nested under the arm), then filters out
    collisions between the gripper's base and the arm's own last link --
    sync_pose_to_flange's every-tick teleport makes the two interpenetrate
    there by design.

    KNOWN GAP, not resolved -- ADJUST: teleporting a fully dynamic rigid
    body's pose every tick (sync_pose_to_flange) is not fully physically
    stable. Direct testing found RMPflow tracking a fixed Cartesian target
    perfectly well while far away (error shrinking steadily tick over tick),
    then suddenly diverging once the gripper got close enough to the arm's
    wrist to start interpenetrating it, even with the collision filter
    below in place. Marking base_link kinematic (the usual fix for "moving
    an attached prop by teleporting its pose every frame") was tried and
    made it worse -- isaacsim.core.prims.Articulation's tensor-API view
    creation (_on_physics_ready's `assert self._physics_view.is_homogeneous`)
    appears not to support a kinematic root with dynamic, joint-driven
    children. Left as the batched-Articulation-plus-collision-filter version
    (this function) since it's the closer-to-working state reached so far --
    next things worth trying: a real PhysicsFixedJoint welding the gripper
    to wrist_3_link (reusing the asset's own dangling "robot_gripper_joint",
    body1 currently unset -- see isaac_sim_common.py's ROBOT_PRIM_PATH
    comment) instead of a per-tick teleport, or checking Isaac Sim's own
    "attach prop to end effector" sample for the currently-recommended
    pattern.

    Returns the gripper's prim path."""
    from isaacsim.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(assets_root + GRIPPER_ASSET_RELATIVE_PATH, GRIPPER_PRIM_PATH)

    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(GRIPPER_BASE_LINK_PRIM_PATH))
    filtered_pairs.CreateFilteredPairsRel().AddTarget(arm_link_prim_path)

    return GRIPPER_PRIM_PATH


class GripperController:
    """Drives the gripper's finger joint to an open/closed position target.

    ADJUST: assumes a single drive joint (GRIPPER_DRIVE_JOINT_NAME) that the
    rest of the gripper's fingers mimic mechanically/via the USD's own mimic
    joint setup -- true for a stock Robotiq 2F-85 asset, but verify against
    whatever gripper USD you actually load. If your gripper instead needs
    each finger joint driven independently, extend set_target() to loop over
    a list of joint names instead of the single GRIPPER_DRIVE_JOINT_NAME.

    Uses the vectorized `Articulation` class (batched, shape (1, num_dof) --
    a view over exactly one prim here), NOT `SingleArticulation` (which the
    arm uses, since RMPflow's ArticulationMotionPolicy requires its
    get_articulation_controller()). Either class works fine for the gripper
    *as long as GRIPPER_PRIM_PATH stays a top-level prim* -- this isn't about
    the class choice.
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
        self.articulation.set_joint_position_targets(
            positions=np.array([[target]]), joint_indices=np.array([idx])
        )

    def sync_pose_to_flange(self, flange_pos, flange_quat_xyzw):
        """Teleports the gripper's base to the tool flange's current world
        pose. Since GRIPPER_PRIM_PATH is a top-level prim (not USD-parented
        under the arm, see that constant's comment), this is what keeps it
        visually/functionally attached to the flange -- call once per tick,
        after the arm has moved (see PickPlaceScene.step_towards).

        CONFIRMED (2026-09-07, actual Isaac Sim run) root cause of a crash
        that killed collect_demos.py's smoke test ~2 episodes in ("Invalid
        PhysX transform detected" spam on every gripper/arm link followed by
        "PhysX error: Illegal BroadPhaseUpdateData", then a silent native
        exit): set_world_poses() only calls physics_view.set_root_transforms
        -- it does NOT touch velocities (checked its actual source). Every
        tick this snaps the body to a new position without zeroing the
        linear/angular velocity left over from the PREVIOUS tick's collision
        response (e.g. while interpenetrating wrist_3_link, see add_gripper's
        KNOWN GAP comment), so PhysX's broadphase does velocity-swept AABB
        prediction off an increasingly bogus velocity that never gets reset
        -- eventually producing an invalid/NaN-ish bound it rejects outright.
        Zeroing the root velocity right after every teleport keeps each
        step's kinematic snap self-contained instead of compounding."""
        quat_wxyz = np.array([[flange_quat_xyzw[3], flange_quat_xyzw[0], flange_quat_xyzw[1], flange_quat_xyzw[2]]])
        self.articulation.set_world_poses(positions=np.asarray([flange_pos]), orientations=quat_wxyz)
        self.articulation.set_velocities(np.zeros((1, 6)))

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


def add_shape(stage, shape, prim_path, position, size=0.04, color=(0.8, 0.1, 0.1)):
    """A small dynamic (rigid-body) pickable object -- cube/sphere/cylinder,
    used by pick_place_scene.spawn_random_objects for the multi-object
    hybrid-pipeline scene (isaac/object_configs.py's vocabulary). `size` is
    the cube edge length / sphere-and-cylinder radius, in meters."""
    if shape == "cube":
        geom = UsdGeom.Cube.Define(stage, prim_path)
        geom.CreateSizeAttr(size)
    elif shape == "sphere":
        geom = UsdGeom.Sphere.Define(stage, prim_path)
        geom.CreateRadiusAttr(size / 2.0)
    elif shape == "cylinder":
        geom = UsdGeom.Cylinder.Define(stage, prim_path)
        geom.CreateRadiusAttr(size / 2.0)
        geom.CreateHeightAttr(size)
    else:
        raise ValueError(f"unknown shape {shape!r} -- expected cube/sphere/cylinder")

    geom.AddTranslateOp().Set(Gf.Vec3d(*position))
    geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(0.05)  # 50g -- light enough for a small parallel gripper
    return geom


def add_cube(stage, prim_path, position, size=0.04, color=(0.8, 0.1, 0.1)):
    """Phase 1-5's single-object case -- thin wrapper so existing callers
    (scripted_pick_place.py, collect_demos.py, residual_rl_train_env.py via
    pick_place_scene.reset()) don't need to change."""
    return add_shape(stage, "cube", prim_path, position, size=size, color=color)


def add_place_target_marker(stage, prim_path, position, size=0.10):
    """A flat, non-colliding marker showing where the cube should be placed
    -- visual/logging aid only, not a physical obstacle."""
    marker = UsdGeom.Cylinder.Define(stage, prim_path)
    marker.CreateRadiusAttr(size / 2.0)
    marker.CreateHeightAttr(0.002)
    marker.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.001))
    marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.1)])
    return marker
