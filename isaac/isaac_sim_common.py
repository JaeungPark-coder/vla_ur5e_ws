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
import os

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
# "robot_gripper_joint" fixed joint (body1 unset), presumably meant for a
# separately-attached gripper -- unused here because select_gripper_variant
# below takes a different approach (a USD variant selection that welds the
# gripper into the arm's own articulation) instead. (This comment used to
# point at a GRIPPER_PRIM_PATH constant for "why" -- that belonged to an
# earlier teleport-based attachment scheme and no longer exists; see
# select_gripper_variant's own docstring for the actual reasoning.)
ROBOT_PRIM_PATH = "/World/ur5e"
TOOL_LINK_PRIM_PATH = "/World/ur5e/wrist_3_link/flange"

# The gripper comes from the UR asset's own "Gripper" USD variant set, not
# from a separately-referenced gripper USD. Selecting a variant makes the
# gripper part of the ARM'S OWN ARTICULATION, which is what Isaac's supported
# manipulator API expects (isaacsim.robot.manipulators.ParallelGripper drives
# joints by index *within the robot's articulation*, and SingleManipulator
# wraps exactly that arrangement -- see Isaac's own
# isaacsim/robot/manipulators/tests/test_single_manipulators.py, which does
#     robot = add_reference_to_stage(".../ur10e/ur10e.usd", "/World/ur10e")
#     robot.GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_140")
# and then addresses "finger_joint" straight off the robot).
#
# This replaces two approaches that were both tried and both confirmed
# broken, because both tried to keep the gripper as a SEPARATE articulation
# and join it to the arm:
#   * per-tick teleport of the gripper root to the flange -- its finger links
#     accumulated the implied velocity of every snap and diverged to world
#     positions around 1e15 m, with RTX reporting bounding boxes past its
#     extents limit for every finger prim;
#   * a UsdPhysics.FixedJoint weld -- PhysX absorbed the gripper into the
#     arm's articulation, so it stopped being an articulation of its own and
#     the app died during World.reset() with "Failed to find articulation at
#     '/World/gripper/Robotiq_2F_85'".
# Neither is kept as a fallback: both are proven wrong, and leaving them
# selectable only invites running into them again.
#
# ADJUST: the variant names below are candidates tried in order. The ur10e
# asset ships "Robotiq_2f_140"; the ur5e asset's set has not been inspected
# from here, so select_gripper_variant lists what the asset actually offers
# if none of these match.
GRIPPER_VARIANT_SET_NAME = "Gripper"
GRIPPER_VARIANT_CANDIDATES = ["Robotiq_2F_85", "Robotiq_2f_85", "Robotiq_2F_140", "Robotiq_2f_140"]
GRIPPER_DRIVE_JOINT_NAME = "finger_joint"
GRIPPER_OPEN_POS = 0.0     # radians -- ADJUST against your gripper's joint limits
# MEASURED (2026-09-11, check_finger_limits.py): commanding finger_joint past
# 0.68 rad still tracks -- 0.80 is reached cleanly, and the drive stalls at
# 0.8203 rad regardless of how far past that it is commanded (0.68, 1.0 and
# 1.2 rad were all tried; 1.0 and 1.2 both stall at the identical 0.8203),
# which is this asset's real hard mechanical limit. The previous
# GRIPPER_CLOSED_POS=0.68 was only ~83% of true full closure -- for a
# Robotiq 2F-85's ~85mm stroke that leaves roughly 15mm of gap still open
# between the pads at "closed", easily enough to miss a 40mm cube depending
# on standoff distance. 0.80 leaves a small margin below the 0.8203 hard
# stop rather than commanding into it continuously (matches the original
# comment's "firm but not overdriven" intent, now against a measured limit
# instead of a guess).
GRIPPER_CLOSED_POS = 0.80  # radians -- see MEASURED note above; hard limit is 0.8203

WRIST_3_LINK_PRIM_PATH = f"{ROBOT_PRIM_PATH}/wrist_3_link"

# pivot_dwell_check.py's own docstring names this gap directly: "there is no
# stiffness, damping, mass or inertia set anywhere in this codebase." A
# stiffness value raised by hand (in the Isaac Sim GUI, against the live
# asset) is exactly the kind of change that gap makes invisible -- it isn't
# in git, so nothing here can log it, compare it, or revert it. The helpers
# below close that: rescale_gripper_mass_to_spec makes the gripper's mass an
# explicit, code-authored value instead of an undocumented PhysX default,
# and get/set_joint_drive_gains make the arm's drive gains readable and
# settable in code (see pivot_dwell_check.py's --stiffness-scale/--damping-
# scale).

# Robotiq's own 2F-85 datasheet lists 0.925 kg total mass.
# rescale_gripper_mass_to_spec uses this only as a target for the TOTAL --
# see its docstring for why the per-link split is read from PhysX's own
# auto-computed (collider-volume-based) proportions rather than guessed.
GRIPPER_TOTAL_MASS_KG = 0.925

# Wrist joints nearest the payload, per pivot_dwell_check.py's DYNAMICS
# hypothesis (gravity + the gripper's mass sagging against gains too soft to
# hold a loaded pose). UNVERIFIED against this asset's exact joint prim names
# beyond what self.robot.dof_names already confirms elsewhere in this
# codebase -- ADJUST if PrimRange traversal below finds none of these.
ARM_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


# The 6 arm link names, needed only to disambiguate the ONE gripper link
# name that collides with one of them -- see
# _resolve_gripper_body_indices. Not ARM_JOINT_NAMES (joints, not links).
ARM_LINK_NAMES = (
    "base_link", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
)


def _resolve_gripper_body_indices(body_names, expected_names, arm_link_names=ARM_LINK_NAMES):
    """Map each of the gripper's own link names to its index in the
    articulation's flat body list (Articulation.get_body_masses()'s column
    order), handling the one gripper link name that collides with an arm
    link's -- CONFIRMED (2026-09-21, direct inspection): both the arm and
    the gripper have a link literally named "base_link" (the gripper's own
    housing), and PhysX/the articulation view disambiguates by suffixing
    the second occurrence it registers ("base_link_0"), not by full USD
    path. Non-colliding names (every other gripper link -- knuckles,
    fingers -- none of which share a name with an arm link) match exactly.
    Raises rather than guessing if a name can't be resolved unambiguously --
    silently assuming a name/order here is exactly how the even-split
    mistake this replaces went unnoticed until an actual A/B run caught it.
    """
    import re
    body_names = list(body_names)
    indices = {}
    used = set()
    for name in expected_names:
        if name not in arm_link_names:
            if name not in body_names:
                raise RuntimeError(f"gripper link {name!r} not found in articulation "
                                    f"body_names {body_names} -- asset's gripper link naming "
                                    f"may have changed.")
            idx = body_names.index(name)
            indices[name] = idx
            used.add(idx)
            continue
        pattern = re.compile(rf"^{re.escape(name)}_\d+$")
        candidates = [i for i, n in enumerate(body_names) if pattern.match(n) and i not in used]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected exactly one suffixed duplicate of arm-colliding link name {name!r} "
                f"for the gripper's own link (found {len(candidates)}) -- inspect body_names "
                f"manually: {body_names}")
        indices[name] = candidates[0]
        used.add(candidates[0])
    return indices


def rescale_gripper_mass_to_spec(robot_prim, robot, total_mass_kg=GRIPPER_TOTAL_MASS_KG):
    """Give the gripper links an explicit, code-authored mass that PRESERVES
    PhysX's own auto-computed per-link proportions and only rescales the
    total to match Robotiq's published 0.925 kg 2F-85 spec, instead of
    guessing a distribution by hand.

    CONFIRMED 2026-09-21 (direct inspection, this exact asset): with no
    MassAPI authored at all (this codebase's actual state before this
    function), PhysX's own density-from-collider-volume computation already
    gives a realistic distribution -- base_link_0 (the gripper's housing)
    at ~610g, each finger/knuckle at 13-39g, summing to ~847g total, only
    ~8% under the real spec, NOT "arbitrarily wrong" the way an unverified
    guess would be. This replaces an earlier attempt (set_gripper_mass_
    properties, since removed) that split the total EVENLY across all 9
    links instead of preserving this -- an actual A/B sweep run confirmed
    that caused severe PhysX divergence (0 "Invalid PhysX transform" events
    in baseline vs. 5513 with the even split enabled, reach-check distances
    up to 5x10^11 mm), almost certainly because an even split hands small
    links (knuckles, fingers) far more mass relative to the housing than
    their real proportions, and a reduced-coordinate articulation solver is
    sensitive to exactly that kind of adjacent-link mass-ratio mismatch.
    This function only ever multiplies each link's already-stable
    auto-computed mass by the same scalar, so the proportions that mattered
    are untouched -- it corrects the total, nothing else.

    Must run AFTER world.reset() + robot.initialize() (needs a live physics
    view to read PhysX's own computed masses from -- unlike the collider
    setup, this cannot happen before first Play). The result is authored
    back into USD (UsdPhysics.MassAPI), not just written to the live
    physics view, so it survives the Stop+Play cycle every
    PickPlaceScene.reset() does; USD physics schema is only re-parsed at
    Play time, so it takes effect starting with the NEXT reset(), which is
    how every caller already uses this class.

    Returns the number of links rescaled -- 0 means the same "wrong Gripper
    prim tree" failure add_gripper_colliders already guards against.
    """
    stage = robot_prim.GetStage()
    gripper_root = stage.GetPrimAtPath(f"{robot_prim.GetPath()}/Gripper")
    if not gripper_root.IsValid():
        raise RuntimeError(f"no Gripper prim under {robot_prim.GetPath()} -- call "
                            "select_gripper_variant first")

    link_prims = {prim.GetName(): prim for prim in Usd.PrimRange(gripper_root)
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    if not link_prims:
        return 0

    body_names = robot._articulation_view.body_names
    body_masses = np.asarray(robot._articulation_view.get_body_masses()[0])
    name_to_index = _resolve_gripper_body_indices(body_names, tuple(link_prims))

    auto_masses = {name: float(body_masses[idx]) for name, idx in name_to_index.items()}
    auto_total = sum(auto_masses.values())
    scale = total_mass_kg / auto_total

    for name, prim in link_prims.items():
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(auto_masses[name] * scale)

    print(f"rescale_gripper_mass_to_spec: auto-computed total was {auto_total * 1000:.1f}g, "
          f"rescaled x{scale:.3f} to hit {total_mass_kg * 1000:.1f}g, proportions preserved "
          f"({', '.join(f'{n}={auto_masses[n] * scale * 1000:.1f}g' for n in sorted(link_prims))})",
          flush=True)
    return len(link_prims)


def _find_joint_prim(robot_prim, joint_name):
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() == joint_name:
            return prim
    return None


def get_joint_drive_gains(robot_prim, joint_names=ARM_JOINT_NAMES):
    """Current (stiffness, damping) of each named joint's angular drive, read
    straight off the USD prim -- so a gain changed by hand outside git (e.g.
    directly in the Isaac Sim GUI) is visible here instead of staying
    invisible to this codebase the way it was before this function existed.
    Returns {joint_name: (stiffness, damping) or None (joint/drive not found)}.
    """
    gains = {}
    for name in joint_names:
        joint_prim = _find_joint_prim(robot_prim, name)
        if joint_prim is None:
            gains[name] = None
            continue
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        stiffness_attr = drive.GetStiffnessAttr()
        damping_attr = drive.GetDampingAttr()
        if not stiffness_attr and not damping_attr:
            gains[name] = None
            continue
        gains[name] = (
            stiffness_attr.Get() if stiffness_attr else None,
            damping_attr.Get() if damping_attr else None,
        )
    return gains


def set_joint_drive_gains(robot_prim, stiffness=None, damping=None, joint_names=ARM_JOINT_NAMES):
    """Explicitly set (stiffness, damping) on each named joint's angular
    drive, in code, so a comparison run is reproducible and logged instead of
    depending on whatever the GUI happens to currently hold. Either argument
    left None leaves that gain alone. Returns the joint names actually found
    and changed."""
    changed = []
    for name in joint_names:
        joint_prim = _find_joint_prim(robot_prim, name)
        if joint_prim is None:
            continue
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        if stiffness is not None:
            drive.CreateStiffnessAttr(float(stiffness))
        if damping is not None:
            drive.CreateDampingAttr(float(damping))
        changed.append(name)
    return changed


def select_gripper_variant(robot_prim, candidates=None):
    """Turn on the arm asset's built-in gripper by selecting a variant.

    Returns the selected variant name. Raises with the list of variants the
    asset actually offers if none of `candidates` is among them -- guessing
    an asset path has cost this project enough runs already.
    """
    candidates = list(candidates or GRIPPER_VARIANT_CANDIDATES)
    variant_sets = robot_prim.GetVariantSets()
    if not variant_sets.HasVariantSet(GRIPPER_VARIANT_SET_NAME):
        raise RuntimeError(
            f"{robot_prim.GetPath()} has no {GRIPPER_VARIANT_SET_NAME!r} variant set "
            f"(it offers: {list(variant_sets.GetNames())}). This asset cannot supply a gripper "
            f"this way; pick a UR asset that ships one, e.g. ur10e.usd.")

    variant_set = variant_sets.GetVariantSet(GRIPPER_VARIANT_SET_NAME)
    available = list(variant_set.GetVariantNames())
    for name in candidates:
        if name in available:
            variant_set.SetVariantSelection(name)
            return name
    raise RuntimeError(
        f"none of {candidates} is in {robot_prim.GetPath()}'s {GRIPPER_VARIANT_SET_NAME!r} "
        f"variant set. Available: {available}. Set GRIPPER_VARIANT_CANDIDATES to one of those.")


def add_gripper_colliders(robot_prim):
    """The Robotiq_2F_85 variant this asset ships has NO collision geometry
    at all on any of its 9 links -- CONFIRMED 2026-09-11 (check_gripper_
    collision.py): every link carries a RigidBodyAPI but zero CollisionAPIs.
    Fingers with no collider pass straight through anything they close on,
    which is why the scripted expert's grasp -- with the flange->tool0 frame
    math independently verified correct to 2mm tracking error -- still never
    moved the cube by so much as a millimetre. Call this once, right after
    select_gripper_variant, before world.reset().

    The reason there is no collider to just "turn on": each link's visible
    geometry lives under a `<link>/visuals` Xform authored with
    `instanceable=True` (a handful of shared USD prototypes referenced by
    all the links, e.g. both fingers point at the same pad mesh) -- and
    PhysX cannot author a per-instance collider on a shared prototype's
    geometry; every instance would fight over one collider. This un-instances
    each such Xform (giving that link its own private copy of the mesh, no
    longer shared -- fine for one robot in one scene, would need
    reconsidering for many parallel envs) and applies a convex-hull collider
    to every Mesh prim underneath. convexHull rather than exact triangleMesh
    because PhysX requires convex collision geometry for non-static rigid
    bodies in general, and these pads are simple enough that the hull is a
    close approximation of the real shape.

    Returns the number of Mesh prims a collider was added to -- 0 means
    nothing was found (wrong path, or with_gripper=False), which the caller
    should treat as a hard failure rather than silently grasping nothing.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

    stage = robot_prim.GetStage()
    gripper_root = stage.GetPrimAtPath(f"{robot_prim.GetPath()}/Gripper")
    if not gripper_root.IsValid():
        raise RuntimeError(f"no Gripper prim under {robot_prim.GetPath()} -- call "
                            "select_gripper_variant first")

    # Same cap as add_shape's cube, applied at the LINK level (not the mesh)
    # since PhysxRigidBodyAPI is a rigid-body property -- these are the two
    # links whose pads actually contact a grasped object. Found by prim NAME
    # rather than a hardcoded "Robotiq_2F_85" path segment: the variant
    # folder's actual casing didn't match the variant-set string that
    # selected it (selecting "Robotiq_2f_85" produced a "Robotiq_2F_85"
    # child), so a path built from `variant` would be wrong, and a different
    # candidate (e.g. Robotiq_2F_140) would use a different folder entirely.
    PAD_LINK_NAMES = ("left_inner_finger", "right_inner_finger")
    n_pad_links = 0
    for prim in Usd.PrimRange(gripper_root):
        if prim.GetName() in PAD_LINK_NAMES:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(0.5)
            n_pad_links += 1

    # Pass 1: un-instance. Only the prim that actually AUTHORS
    # instanceable=True (the per-link "visuals" Xform) can have that flag
    # cleared -- its descendants only exist as read-only "instance proxies"
    # while it stays instanced, and a plain (non-instance-proxy) PrimRange
    # does not descend into them at all, which is exactly why the very first
    # collision check found nothing under /visuals: there was nothing there
    # to find without either un-instancing first or passing
    # Usd.TraverseInstanceProxies() to look inside.
    un_instanced = 0
    for prim in Usd.PrimRange(gripper_root):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            un_instanced += 1

    # Pass 2: now that nothing is instanced, the meshes are ordinary prims
    # reachable by a plain PrimRange -- apply a collider to each.
    #
    # The inner-finger pads (left/right_inner_finger -- the meshes that
    # actually touch a grasped object, confirmed by name: "finger4step" is
    # the pad body, "fingertipsstep" the rubber insert on top of it) get
    # convexDecomposition instead of a single convexHull. reach_probe
    # (2026-09-14) measured the cube launched >2m from a standing start
    # during an already-gradual gripper close, after tracking had converged
    # to a normal ~20-40mm grasp offset -- consistent with a single hull
    # puffing out these pads' grip-face concavity and starting the close
    # already deeply interpenetrating, so the solver's one-step correction
    # is enormous. Everything else (knuckles, outer fingers, base_link) is
    # rigid housing that never touches the object, where a coarse hull is
    # fine and decomposition would only cost more.
    PAD_LINK_MARKERS = ("left_inner_finger", "right_inner_finger")
    n_colliders = 0
    n_decomposed = 0
    for prim in Usd.PrimRange(gripper_root):
        if prim.IsA(UsdGeom.Mesh):
            path = str(prim.GetPath())
            is_pad = any(f"/{marker}/" in path for marker in PAD_LINK_MARKERS)
            approximation = "convexDecomposition" if is_pad else "convexHull"
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr(approximation)
            n_colliders += 1
            n_decomposed += is_pad

    print(f"add_gripper_colliders: un-instanced {un_instanced} prim(s), "
          f"added colliders to {n_colliders} mesh(es) "
          f"({n_decomposed} convexDecomposition on the grip pads, "
          f"{n_colliders - n_decomposed} convexHull elsewhere), "
          f"capped max depenetration velocity on {n_pad_links} pad link(s)", flush=True)
    return n_colliders


class GripperController:
    """Opens and closes the gripper by driving one joint of the ARM's
    articulation.

    The gripper is part of that articulation (see select_gripper_variant), so
    there is no separate articulation view, no pose to keep in sync, and
    nothing that can drift away from the flange. `sync_pose_to_flange` is
    kept only so callers that used the old attachment scheme still work; it
    does nothing.

    ADJUST: assumes a single drive joint (GRIPPER_DRIVE_JOINT_NAME) whose
    fingers mimic it -- true for a stock Robotiq 2F-85/2F-140. If your
    gripper needs each finger driven independently, extend set_target to
    write several indices.
    """

    def __init__(self, robot_articulation, drive_joint_name=GRIPPER_DRIVE_JOINT_NAME):
        self.robot = robot_articulation
        self.drive_joint_name = drive_joint_name
        self._drive_joint_index = None

    def _resolve_joint_index(self):
        if self._drive_joint_index is None:
            dof_names = list(self.robot.dof_names)
            if self.drive_joint_name not in dof_names:
                raise RuntimeError(
                    f"{self.drive_joint_name!r} is not a joint of the robot articulation. "
                    f"Its joints are: {dof_names}. Either the gripper variant did not take "
                    f"effect, or this gripper names its drive joint differently -- set "
                    f"GRIPPER_DRIVE_JOINT_NAME accordingly.")
            self._drive_joint_index = dof_names.index(self.drive_joint_name)
        return self._drive_joint_index

    def set_target(self, position):
        """position: 0.0 (fully open) .. 1.0 (fully closed), linearly mapped
        to [GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS]."""
        from isaacsim.core.utils.types import ArticulationAction

        idx = self._resolve_joint_index()
        target = GRIPPER_OPEN_POS + float(np.clip(position, 0.0, 1.0)) * (
            GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        self.robot.apply_action(ArticulationAction(
            joint_positions=np.array([target]), joint_indices=np.array([idx])))

    def sync_pose_to_flange(self, flange_pos, flange_quat_xyzw):
        """No-op: the gripper is part of the arm's articulation, so physics
        holds it on the flange. Kept for call-site compatibility."""

    def open(self):
        self.set_target(0.0)

    def close(self):
        self.set_target(1.0)

    def get_normalized_position(self):
        """Drive-joint position mapped back to 0.0 (open) .. 1.0 (closed) --
        the inverse of set_target, for logging gripper state in an
        observation."""
        idx = self._resolve_joint_index()
        current = float(np.asarray(self.robot.get_joint_positions())[idx])
        span = GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS
        return float(np.clip((current - GRIPPER_OPEN_POS) / span, 0.0, 1.0))


class NullGripper:
    """Stand-in with GripperController's interface that does nothing, for
    measurements that involve no grasping (check_cameras.py)."""

    def set_target(self, position):
        pass

    def sync_pose_to_flange(self, flange_pos, flange_quat_xyzw):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def get_normalized_position(self):
        return 0.0


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
    """CONFIRMED 2026-09-16 (check_pose_readout_multireset.py): reliable for
    an ARTICULATED prim (recomputed from live joint state every step -- this
    is how the robot/gripper links are read, and that tracking is fine), but
    for a prim that gets `stage.RemovePrim` + re-`Define`-d at the SAME path
    across multiple episodes (a free rigid body like the cube in
    pick_place_scene.reset(), previously), ComputeLocalToWorldTransform
    reads correctly on the FIRST episode and then FREEZES at that first
    episode's transform forever, regardless of how many further resets or
    physics steps happen -- across 5 fresh episodes, error against the true
    spawn position went 0mm, 156mm, 260mm, 184mm, 267mm, never correcting
    itself. Most likely Fabric's own stage/sim-history cache, keyed by prim
    path, not being invalidated by a raw USD remove+recreate at that same
    path within one still-running World/Kit session (the same session logs
    "gFabricState->gUsdStageToSimStageWithHistoryMap had 1 outstanding
    SimStageWithHistory(s)" on shutdown, consistent with this).
    Everything downstream that scored against a stale cube read this way
    (grasp_succeeded, place_error_m, check_cameras' reach check, and this
    session's own reach-rate/cube-drift diagnostics) was measuring against
    the WRONG position for every episode after the first.
    Do not call this on a free rigid body prim that gets removed and
    recreated across episodes -- keep the SAME persistent prim and
    reposition it via SingleRigidPrim.set_world_pose (see
    pick_place_scene.reset()) instead. Also do not call this immediately
    after such a set_world_pose and before the next world.step(): a raw
    USD/Fabric read like this one only reflects a physics-view write once
    physics has stepped and flushed state back -- CONFIRMED 2026-09-16
    (diag_single_rigid_prim_fix.py), it reads the PREVIOUS pose for exactly
    one call in between."""
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
    # Caps how fast PhysX is allowed to push two interpenetrating bodies
    # apart. reach_probe (2026-09-14) measured this object launched >2m from
    # a standing start during an already-gradual gripper close -- a solver
    # resolving deep penetration (from the grip pads' convex approximation,
    # or just an off-center close) in one step rather than a real contact
    # force. 0.5 m/s still separates a real overlap within a few ticks but
    # can no longer produce that -- see the matching cap on the gripper pad
    # links in add_gripper_colliders.
    from pxr import PhysxSchema
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateMaxDepenetrationVelocityAttr(0.5)
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
