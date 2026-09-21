"""Builds and resets the pick-and-place Isaac Sim scene: a UR5e + gripper, a
randomly-placed cube, a fixed place-target zone, and two cameras (base_rgb
overview + wrist_rgb eye-in-hand) -- matching openpi's UR5 example contract
(examples/ur5/README.md): state = joints+gripper, two RGB cameras, no right
wrist.

Import this only AFTER `from isaacsim import SimulationApp; SimulationApp(...)`
has already run in the importing script (see collect_demos.py) -- omni/pxr/
isaacsim.* aren't importable otherwise.

Written and reasoned about WITHOUT the ability to run Isaac Sim in the
environment this was authored in -- treat as a solid first draft, not
verified to run. The multi-camera replicator setup in _setup_cameras is the
least-verified part (single-camera annotator attach is confirmed working
from potato_drill_ws's isaac_scene.py; attaching two separate "rgb"
annotator instances to two different render products, one each, is the
expected pattern but wasn't testable here) -- check Isaac Sim's own
multi-camera replicator example if the two camera feeds come back
cross-wired.
"""
import math

import numpy as np
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
from scipy.spatial.transform import Rotation as Rot

from isaacsim.core.api import World
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim

import camera_framing
from isaac_sim_common import (
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    select_gripper_variant, add_gripper_colliders, rescale_gripper_mass_to_spec,
    GripperController, setup_rmpflow, prim_world_pose,
    GRIPPER_VARIANT_SET_NAME,
    add_cube, add_shape, add_place_target_marker, NullGripper,
)
import object_configs

CUBE_PRIM_PATH = "/World/cube"
TARGET_MARKER_PRIM_PATH = "/World/place_target"
BASE_CAMERA_PRIM_PATH = "/World/base_camera"
WRIST_CAMERA_PRIM_PATH = f"{TOOL_LINK_PRIM_PATH}/wrist_camera"

# Workspace bounds the cube is randomized within (robot base frame, meters) --
# ADJUST to whatever's actually reachable/visible on your table setup.
#
# All four edges pulled in by 0.07 on 2026-09-21 (X: 0.35/0.55 -> 0.42/0.48;
# Y: -0.20/0.20 -> -0.13/0.13). check_rmpflow_stability.py (30 fresh
# episodes, 15 each on the GPU and CPU physics pipelines -- same on both, so
# this is a real RMPflow/geometry effect, not a GPU-pipeline artifact) found
# the tool-tracking residual spiking to 200mm-1.4m (finite, never NaN/Inf)
# concentrated near the spawn range's edges -- but NOT one specific edge:
# X>=0.49 was the first one caught, then a separate run turned up a
# X=0.456/Y=0.19 spawn (near the Y edge instead) also spiking to 1.09m, and
# a third run (after narrowing X alone) turned up a Y=-0.175 spawn (the
# OTHER Y edge) spiking to 887mm. Three of four edges had independently
# shown this before any of them were deliberately targeted, which is why
# all four are pulled in together here instead of chasing them one at a
# time -- treat this as a real, reproducible edge-proximity risk, not a
# bounded region on one axis.
#
# NOT a validated safe boundary, and NOT free: this leaves only a 0.06 x
# 0.26m spawn area for a 0.04m cube, which may be too little spatial
# randomization for what reset()'s own comment calls out (forcing the
# eventual policy to look at the image, not memorize one pose) -- re-run
# check_rmpflow_stability.py --with-gripper --episodes 20 (both --device cpu
# and the GPU default) after any further change here, and reconsider the
# margin if training data ends up too repetitive.
CUBE_X_RANGE = (0.42, 0.48)
CUBE_Y_RANGE = (-0.13, 0.13)
CUBE_SIZE_M = 0.04  # edge length of the spawned cube
CUBE_Z = 0.02  # resting height for a 4cm cube on the table surface
PLACE_TARGET_POSITION = np.array([0.45, 0.30, 0.0])

# Episode-outcome thresholds (see grasp_succeeded / place_error_m below).
# The cube rests with its centre at CUBE_Z=0.02 and the scripted expert
# lifts it ~0.15 m, so 0.08 cleanly separates "picked up" from "still on the
# table" without demanding the full lift height.
LIFT_Z_THRESHOLD = 0.08
# Absolute floor for the peak red-cube pixel count the base camera must reach
# at some point during an episode (see cube_pixels_visible). Measured on a
# real collected episode with the cameras working: the peak is ~130 px. The
# first broken run peaked at 0. 30 sits well clear of both.
#
# It is only a floor now. The threshold actually used,
# MIN_CUBE_PIXELS_IN_BASE_VIEW, is derived from what this camera geometry can
# achieve (see BASE_FRAMING below), because a flat 30 turned out to sit at
# 15% of the best case here -- it catches a camera aimed at nothing, not a
# camera framing the task badly, and the latter is what the lost runs were.
MIN_CUBE_PIXELS_FLOOR = 30
# Fraction of a frame allowed to be near-black before the camera counts as
# aimed past the scene rather than at it. Measured across real episodes: a
# working base camera runs 0.03-0.10, while the mis-aimed wrist camera and
# the clipped first run run 0.34-1.00.
MAX_DARK_FRACTION = 0.30

# Eye-in-hand mount geometry, in the flange frame.
#
# The camera sits BESIDE the tool axis, not on it, and is aimed at a point a
# little way along that axis -- roughly where the tool tip and the object it
# is reaching for both are. That is how a real wrist camera is bracketed, and
# it avoids the two ways the earlier mounts failed: a fixed 5 cm offset put
# the camera inside the UR5e's own wrist_3_link (solid black at every
# orientation), and displacing it ALONG its viewing direction drove it
# through the workpiece and out the far side (at 0.18 m it ended up below the
# table, z = -0.008).
#
# WRIST_CAMERA_LATERAL_M must clear the wrist link's radius. Aiming at
# WRIST_CAMERA_FOCUS_M along the tool axis, rather than pointing straight
# down that axis, keeps the target centred despite the lateral offset.
#
# VALIDATED 2026-09-21 (check_wrist_mount_raycast.py's two-stage search: a
# cheap FOV-cone + PhysX-raycast screen over a 294-candidate grid across 5
# fresh grasp poses, narrowed to 10 survivors, THEN rendered for real):
# 0.16 was the largest lateral tested and the winner -- worst-case 351 cube
# px, mean 555, across 3 fresh render samples, 0% near-black. Re-run that
# script (wider LATERAL_CANDIDATES) if a still-larger offset might do
# better; this was only tested up to 0.16.
WRIST_CAMERA_LATERAL_M = 0.16
WRIST_CAMERA_FOCUS_M = 0.12
# How far BACK along the approach axis the camera sits, i.e. behind the
# gripper looking forward over it, the way a real eye-in-hand bracket is
# built. Without this the camera sat in the flange's own plane, and at the
# grasp -- where the flange has descended to the cube -- it ended up level
# with the table (camera_check/report.txt records it at z=0.024 with the cube
# at z=0.02), aiming its view straight past the cube into the floor. Every
# direction/lateral combination swept there saw exactly zero cube pixels.
WRIST_CAMERA_BACK_M = 0.12
# Wrist cameras are wide-angle; the USD default (~23 deg with a 50 mm lens on
# the standard aperture) is far too narrow to hold an object that sits off to
# one side of a laterally-offset camera.
WRIST_CAMERA_HORIZONTAL_FOV_DEG = 70.0
WRIST_CAMERA_FOCAL_LENGTH_MM = 24.0

# VALIDATED 2026-09-21 (same check_wrist_mount_raycast.py run as
# WRIST_CAMERA_LATERAL_M's winner -- see that constant's comment): 15
# degrees down. History worth keeping, because it shows why this needed a
# real search rather than a plausible-sounding external number: an earlier
# attempt set this to 30 (citing an external comparison of frontal vs.
# ~30-degree-down eye-in-hand mounts), and the one direct measurement taken
# of THAT value made things WORSE, not better (39% near-black at tilt=0 vs.
# 60% at tilt=30, both against the OLD rot=(0,0,0)/lateral=0.12 mount, and
# without the fill light below -- see WRIST_CAMERA_FLANGE_ROT_EULER and
# _setup_cameras' fillDomeLight comment). It later turned out most of that
# near-black reading was a missing scene light, not the mount at all; 15
# degrees down only won once BOTH the light was fixed and the search moved
# from a single hand-picked sample to a real geometry-then-render sweep.
WRIST_CAMERA_DOWN_TILT_DEG = 15.0

# Which way the camera looks, as XYZ Euler degrees taking the FLANGE frame to
# the tool's approach direction.
#
# VALIDATED 2026-09-21: check_wrist_mount_raycast.py's two-stage search
# (294 (direction, lateral, tilt) candidates screened by FOV-cone membership
# + an unoccluded PhysX raycast to the cube's centre across 5 fresh grasp
# poses -- no rendering, so cheap enough to cover that many -- then the top
# 10 survivors actually rendered, 3 fresh grasp samples each) found
# (180, 0, 0) @ lateral=0.16, tilt=15 as the winner: worst-case 351 cube px,
# mean 555, 0% near-black across all 3 render samples. Two things had to be
# fixed FIRST for this search to mean anything, both confirmed the same
# day: the reach/tracking instability that made every earlier attempt's
# grasp poses unreliable (see CUBE_X_RANGE/CUBE_Y_RANGE's comment and
# rescale_gripper_mass_to_spec), and the scene having no fill light at all
# (see _setup_cameras' fillDomeLight comment) -- every candidate this same
# search geometrically verified as "cube in FOV, unoccluded" still rendered
# 57-84% near-black before that light existed, which is why every previous
# hand-picked or partially-swept value here was untrustworthy. Re-run
# check_wrist_mount_raycast.py after any further change to gripper mass,
# CUBE_X_RANGE/CUBE_Y_RANGE, or scene lighting -- any of those can shift
# which grasp poses this was validated against.
WRIST_CAMERA_FLANGE_ROT_EULER = (180.0, 0.0, 0.0)
BASE_CAMERA_POSITION = (0.9, 0.0, 0.5)
# What the base camera looks at: between the cube spawn area (CUBE_X_RANGE x
# CUBE_Y_RANGE at CUBE_Z) and the place target, so both are in frame.
BASE_CAMERA_AIM_POINT = (0.45, 0.10, 0.02)
# 256x256 matches openpi's LIBERO example image shape -- keep the
# training-side transform (UR5eInputs) consistent with whatever's set here.
CAMERA_RESOLUTION = (256, 256)
# How many pixels across the cube ought to be for the policy to have real
# evidence of where it is during the approach. Viewpoint quality is one of
# the larger levers on behaviour-cloning success, and a target that is a
# couple of pixels wide at reset gives essentially no signal in exactly the
# phase that needs it. Used as the yardstick in BASE_FRAMING below; whether
# it is reachable at all is part of what that reports.
TARGET_CUBE_SPAN_PX = 30.0

# Named so camera_framing.py's own near-clip check (it already reads
# scene.get("CAMERA_NEAR_CLIP_M"), added before this constant existed) has
# something to find instead of silently skipping. Value unchanged from the
# literal 0.01/10000.0 _setup_cameras used to hardcode -- see that function's
# own comment for why 0.01 (1cm) replaced USD's 1m default near clip.
CAMERA_NEAR_CLIP_M = 0.01
CAMERA_FAR_CLIP_M = 10000.0

# Base camera's horizontal FOV, realized via focal length / aperture below
# so camera_projection.py's pinhole math (which takes this same value) is
# actually accurate -- ADJUST alongside BASE_CAMERA_FOCAL_LENGTH_MM if you
# change the camera's framing; the two must stay consistent.
BASE_CAMERA_HORIZONTAL_FOV_DEG = 60.0
BASE_CAMERA_FOCAL_LENGTH_MM = 24.0

# What this camera can actually show, computed rather than assumed. Distance
# and field of view only ever act together, as the width of the swath the
# image covers, so span_px = resolution * cube_size / visible_width -- and
# everything that must stay in frame (the cube's spawn spread plus the place
# target) puts a floor under that width and therefore a ceiling on the cube's
# pixel size that no camera placement beats. See camera_framing.py, which
# also runs standalone to print the full report and the ways out.
BASE_FRAMING = camera_framing.framing_analysis(
    object_size_m=CUBE_SIZE_M,
    camera_position=BASE_CAMERA_POSITION,
    sample_points=[(x, y, CUBE_Z) for x in CUBE_X_RANGE for y in CUBE_Y_RANGE],
    hfov_deg=BASE_CAMERA_HORIZONTAL_FOV_DEG,
    resolution_px=CAMERA_RESOLUTION[0],
    must_cover_m=camera_framing.workspace_span_m(
        CUBE_Y_RANGE, PLACE_TARGET_POSITION[1]),
    target_span_px=TARGET_CUBE_SPAN_PX,
    reference_point=BASE_CAMERA_AIM_POINT,
)

# The episode-level guard collect_demos.py enforces: the cube has to get about
# as visible as this camera is capable of making it, not merely visible.
MIN_CUBE_PIXELS_IN_BASE_VIEW = max(
    MIN_CUBE_PIXELS_FLOOR, int(round(BASE_FRAMING.usable_peak_area_px)))

# Multi-object scene (isaac/object_configs.py's vocabulary) used by the LLM
# + open-vocabulary hybrid pipeline (hybrid_pick_place_demo.py) -- kept
# separate from the single-cube CUBE_PRIM_PATH/reset() Phase 1-5 already use.
OBJECTS_PARENT_PRIM_PATH = "/World/objects"
MIN_OBJECT_SEPARATION_M = 0.12  # reject a random layout where two objects would overlap/collide

# MEASURED (2026-09-10) on this asset, from the gripper's own finger geometry:
# the fingertips sit 120mm out from the flange along the FLANGE'S +Z, while
# the frame RMPflow drives ("tool0") has its +Z along the flange's +X -- the
# two are exactly 90 degrees apart. Nothing accounted for either fact, which
# is why the scripted expert never grasped anything: its waypoints, meant as
# "put the grasp point here", went to RMPflow unchanged, so the arm placed
# the FLANGE there, rotated 90 degrees off, with the fingers 120mm past it
# ploughing into the table. That contact is what pushed the arm upward the
# longer it held the pose (the residual grew 24mm -> 43mm between a 60- and
# a 120-tick hold, while the same run without a gripper stayed at 5mm).
GRIPPER_TCP_OFFSET_M = 0.12


def _look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)):
    """Rotation placing a camera at `eye` so it images `target`. USD cameras
    look along their own local -Z, so that axis -- not +Z -- is what gets
    aimed. Works in whatever frame `eye`/`target` are expressed in."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)

    z_axis = -forward
    up = np.asarray(up, dtype=float)
    if abs(float(np.dot(up, z_axis))) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return Rot.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))


def _look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Orientation (Gf.Quatf, for a USD OrientOp) placing a camera at `eye`
    so that it images `target`. USD cameras look along their own local -Z,
    so that axis -- not +Z -- is what gets aimed."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)

    z_axis = -forward  # camera local +Z points away from what it looks at
    up = np.asarray(up, dtype=float)
    if abs(float(np.dot(up, z_axis))) > 0.999:  # degenerate: looking straight up/down
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    q = Rot.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()  # xyzw
    return Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


class PickPlaceScene:
    def __init__(self, with_gripper=True):
        """with_gripper=False builds the scene with no gripper (NullGripper).
        Only for measurements that do not involve grasping -- see
        check_cameras.py."""
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root -- check Nucleus connection.")

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()

        # CONFIRMED 2026-09-21: add_default_ground_plane's own SphereLight
        # (overhead, intensity 100000) is the ONLY light this scene ever had
        # -- there was no other light anywhere in this file. Measured at
        # reset: base_rgb mean=72 (tolerable, high up and unobstructed) but
        # wrist_rgb mean=27 (the wrist camera sits low, close to the
        # gripper/table, partly shadowed from a single overhead point
        # source). This alone explained a near_black reading that a wrist
        # mount / lateral / down-tilt search (check_wrist_mount_raycast.py)
        # could NOT get below ~57-84% no matter the angle -- every candidate
        # it geometrically verified (in-FOV, unoccluded raycast to the
        # cube) still rendered mostly dark. Adding this fill dome light
        # alone, with no mount/angle change at all, dropped wrist_rgb's
        # near_black from >40% to 0.2% (mean 27 -> 79) in a direct A/B
        # check. Intensity chosen to roughly match the ambient light level
        # a real tabletop scene would have without blowing out the
        # existing overhead SphereLight's highlights -- ADJUST if either
        # camera looks over/under-exposed once this can be eyeballed.
        UsdLux.DomeLight.Define(self.stage, "/World/fillDomeLight").CreateIntensityAttr(1500.0)

        robot_prim = add_reference_to_stage(assets_root + UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH)
        # Turn the asset's own gripper on BEFORE any articulation view is
        # created: selecting the variant changes the prim tree, so it has to
        # happen while the stage is still being composed.
        if with_gripper:
            variant = select_gripper_variant(robot_prim)
            print(f"gripper: selected {GRIPPER_VARIANT_SET_NAME} variant {variant!r}", flush=True)
            # CONFIRMED (2026-09-11, check_gripper_collision.py): this variant
            # ships zero collision geometry on any of its 9 links -- the
            # fingers pass through anything they close on. Without this, no
            # amount of correct frame math will ever grasp anything.
            n_colliders = add_gripper_colliders(robot_prim)
            if n_colliders == 0:
                raise RuntimeError(
                    "add_gripper_colliders found no meshes to add colliders to -- the gripper "
                    "prim tree has probably changed shape; inspect it before trusting any grasp.")
        # SingleArticulation (unbatched), not the vectorized multi-env
        # `Articulation` class -- ArticulationMotionPolicy (RMPflow, below)
        # requires get_articulation_controller(), which only the single-robot
        # API provides.
        #
        # Deliberately NOT registered via world.scene.add(): that path (Scene
        # ._finalize -> SingleArticulation.initialize() during World.reset())
        # produced a physics view with `_physics_view is None` for this asset
        # ("is_homogeneous" AttributeError) on this Isaac Sim version. Calling
        # world.reset() then .initialize() directly, ourselves, works
        # reliably instead -- but per SingleArticulation.initialize()'s own
        # docstring this needs to be redone after every *hard* reset (Stop+
        # Play, which a non-soft world.reset() -- the default -- triggers),
        # so reset() below re-initializes it on every episode too. The
        # gripper (GripperController, isaac_sim_common.py) uses the batched
        # `Articulation` class instead and does not need this dance.
        self.robot = SingleArticulation(ROBOT_PRIM_PATH, name="ur5e_arm")
        self.world.reset()
        self.robot.initialize()

        if with_gripper:
            # Needs a LIVE physics view to read PhysX's own auto-computed
            # per-link masses from (see rescale_gripper_mass_to_spec's own
            # docstring for why those proportions are trusted rather than
            # guessed) -- cannot happen before this first robot.initialize().
            # Authors the result into USD, so the very next world.reset()
            # (PickPlaceScene.reset(), called by every caller before an
            # episode actually runs) is what picks it up.
            n_mass_links = rescale_gripper_mass_to_spec(robot_prim, self.robot)
            if n_mass_links == 0:
                raise RuntimeError(
                    "rescale_gripper_mass_to_spec found no RigidBodyAPI links under Gripper -- "
                    "the gripper prim tree has probably changed shape; inspect it before "
                    "trusting any grasp.")

        # The gripper's joints are part of self.robot's articulation now (the
        # variant selected above), so the controller just drives one of its
        # joints -- nothing to attach, nothing to keep in sync.
        if with_gripper:
            self.gripper = GripperController(self.robot)
            print(f"robot articulation joints: {list(self.robot.dof_names)}", flush=True)
        else:
            # No gripper at all -- see NullGripper. Only for measurements that
            # involve no grasping (check_cameras.py).
            self.gripper = NullGripper()

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)
        self._sync_gripper_to_flange()
        self.physics_dt = 1.0 / 60.0

        self._setup_cameras()
        self._rng = np.random.default_rng()
        self.cube_position = None
        self._object_spawn_generation = 0
        # SingleRigidPrim for the cube, constructed once reset() has created
        # the cube prim and played physics at least once -- see reset()'s
        # own comment for why this replaced set_rigid_body_translation.
        self.cube_rigid_prim = None

    def _setup_cameras(self):
        base_cam = UsdGeom.Camera.Define(self.stage, BASE_CAMERA_PRIM_PATH)
        base_cam.AddTranslateOp().Set(Gf.Vec3d(*BASE_CAMERA_POSITION))
        # CONFIRMED (2026-09-08) that the previous placeholder tilt
        # (RotateXYZ 0/55/180) pointed this camera at the sky: a collected
        # frame showed nothing but the background grid, no table, robot or
        # cube. Aim it at the workspace instead of guessing Euler angles.
        # A USD camera images along its local -Z, so -Z is what has to land
        # on the target point.
        base_cam.AddOrientOp().Set(_look_at_quat(BASE_CAMERA_POSITION, BASE_CAMERA_AIM_POINT))
        # Realize BASE_CAMERA_HORIZONTAL_FOV_DEG via focal length/aperture so
        # camera_projection.py's pinhole math (fed that same constant) is
        # actually accurate -- ADJUST if this doesn't match the rendered FOV
        # on your Isaac Sim version's camera attribute conventions.
        import math
        horizontal_aperture_mm = 2.0 * BASE_CAMERA_FOCAL_LENGTH_MM * math.tan(
            math.radians(BASE_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
        base_cam.CreateFocalLengthAttr(BASE_CAMERA_FOCAL_LENGTH_MM)
        base_cam.CreateHorizontalApertureAttr(horizontal_aperture_mm)
        # CONFIRMED 2026-09-15: verticalAperture was never set here, so it
        # sat at USD's schema default (15.2908mm) regardless of
        # horizontalAperture -- for this camera's 24mm focal length that is
        # a ~35.3deg vertical FOV against the intended 60deg horizontal, on
        # a SQUARE 256x256 render. Every frame this scene has ever rendered
        # was vertically compressed relative to horizontal by that ratio;
        # found chasing why check_cameras.py's wrist sweep showed the cube
        # vanishing on 3 of 4 in-plane ROLLS of its own winning direction --
        # a real roll of a genuinely square FOV can only move content
        # between edges, never make it disappear, so a non-square FOV that
        # trades horizontal reach for vertical reach as it rotates was the
        # only remaining explanation. Matching it to horizontalAperture
        # gives square pixels for this square CAMERA_RESOLUTION -- for a
        # non-square resolution this would need scaling by
        # (vertical_px / horizontal_px) instead.
        base_cam.CreateVerticalApertureAttr(horizontal_aperture_mm)
        # CONFIRMED (2026-09-07, by inspecting an actual collected dataset): a
        # USD camera's default clippingRange is (1.0, 1000000) -- a 1 METRE
        # near plane. The base camera sits ~0.66m from the cube and the wrist
        # camera a few centimetres from it, so BOTH were clipping the entire
        # task away: the first 100-episode collection came out with
        # all-black wrist_image frames (mean=0.0, std=0.0) and base frames
        # containing zero red pixels, i.e. the cube was never once visible in
        # the training data. Verified separately in the sibling potato_scan
        # project by a camera-distance sweep: with the default range, zero
        # points land on a target 0.15m away; with near=0.01 the same pose
        # returns tens of thousands.
        base_cam.CreateClippingRangeAttr().Set(Gf.Vec2f(CAMERA_NEAR_CLIP_M, CAMERA_FAR_CLIP_M))

        # CONFIRMED (2026-09-08, by inspecting collected frames): mounting the
        # wrist camera on the flange with translation only left it imaging the
        # sky. Two reasons, the same pair the sibling potato_scan project hit.
        # A USD camera images along its local -Z, not the +Z this project's
        # eye-in-hand convention assumes; and poses are commanded to RMPflow,
        # which drives the frame its config calls "tool0" -- a frame that
        # shares the flange's position but sits ~(-90, -90, 0) degrees away in
        # orientation on this asset. Measure that offset from RMPflow's own
        # forward kinematics rather than hard-coding it, then mount the camera
        # so it images along the commanded tool's +Z (the approach direction).
        q0 = np.asarray(self.robot.get_joint_positions())[:6]
        _, tool0_rot = self.rmpflow.get_end_effector_pose(q0)
        tool0_rot = np.asarray(tool0_rot)
        r_tool0 = (Rot.from_matrix(tool0_rot) if tool0_rot.shape == (3, 3)
                   else Rot.from_quat(tool0_rot[[1, 2, 3, 0]]))
        _, flange_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        self.r_flange_to_tool0 = Rot.from_quat(flange_quat).inv() * r_tool0

        wrist_cam = UsdGeom.Camera.Define(self.stage, WRIST_CAMERA_PRIM_PATH)
        # Both ops are set by set_wrist_camera_flange_rotation, which places
        # the camera beside the tool axis -- see WRIST_CAMERA_LATERAL_M.
        self._wrist_cam_translate_op = wrist_cam.AddTranslateOp()
        self._wrist_cam_orient_op = wrist_cam.AddOrientOp()
        wrist_cam.CreateClippingRangeAttr().Set(Gf.Vec2f(CAMERA_NEAR_CLIP_M, CAMERA_FAR_CLIP_M))
        wrist_aperture_mm = 2.0 * WRIST_CAMERA_FOCAL_LENGTH_MM * math.tan(
            math.radians(WRIST_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
        wrist_cam.CreateFocalLengthAttr(WRIST_CAMERA_FOCAL_LENGTH_MM)
        wrist_cam.CreateHorizontalApertureAttr(wrist_aperture_mm)
        # Same fix, same reason, as the base camera above: unset
        # verticalAperture defaults to USD's 15.2908mm regardless of this
        # camera's own 70deg-horizontal aperture, a ~2x FOV mismatch on this
        # camera's shorter focal length -- worse than the base camera's, and
        # what actually produced the vanishing-on-roll symptom this was
        # chased down from.
        wrist_cam.CreateVerticalApertureAttr(wrist_aperture_mm)
        self.set_wrist_camera_flange_rotation(WRIST_CAMERA_FLANGE_ROT_EULER)

        # Render products + annotators last, once both cameras exist. These
        # belong to _setup_cameras, not to any of the setters below: without
        # them get_observation() raises AttributeError on base_rgb_annotator.
        self.base_rp = rep.create.render_product(BASE_CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
        self.wrist_rp = rep.create.render_product(WRIST_CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
        self.base_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.wrist_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.base_rgb_annotator.attach([self.base_rp])
        self.wrist_rgb_annotator.attach([self.wrist_rp])

    def derived_wrist_rotation(self):
        """The flange->camera rotation implied by the measured flange->tool0
        offset -- what WRIST_CAMERA_FLANGE_ROT_EULER = None resolves to.
        Returned as a scipy Rotation so check_cameras.py can report the
        Euler triple it corresponds to alongside the swept candidates."""
        # The approach direction is the GRIPPER's axis -- the flange's own +Z,
        # measured from the finger geometry -- not tool0's +Z, which sits 90
        # degrees away (see GRIPPER_TCP_OFFSET_M). Aiming the camera down
        # tool0's axis is why it kept framing the arm and the target marker
        # instead of the cube, and why sweeping 18 mount candidates found
        # nothing: every one of them was measured off the wrong axis.
        # In the flange frame the grip frame is the identity, so all that is
        # left is USD's own -Z imaging convention.
        return Rot.from_euler("x", 180.0, degrees=True)

    def set_wrist_camera_flange_rotation(self, euler_xyz_deg, lateral_m=None, focus_m=None,
                                          back_m=None, down_tilt_deg=None):
        """Mount the wrist camera for a given tool-approach direction.

        `euler_xyz_deg` (XYZ degrees, flange -> approach frame; None uses
        derived_wrist_rotation()) fixes which way the tool reaches. The camera
        is then placed `lateral_m` to the side of that axis and aimed at a
        point `focus_m` along it -- see WRIST_CAMERA_LATERAL_M for why beside
        rather than on, and why aimed rather than parallel.

        `down_tilt_deg` (see WRIST_CAMERA_DOWN_TILT_DEG) additionally rotates
        the approach axis, about the same lateral bracket axis used to offset
        the camera, before the lateral/focus/back placement above is applied
        -- an attempt at "look down past the fingers" rather than "look
        straight along the tool axis", per the finger-occlusion symptom at
        close standoff. UNVERIFIED sign/magnitude against this asset -- see
        WRIST_CAMERA_DOWN_TILT_DEG's comment.

        Settable at runtime so check_cameras.py can sweep mounts within one
        session instead of needing a restart per guess."""
        if euler_xyz_deg is None:
            r_approach = self.derived_wrist_rotation()
            self.wrist_camera_flange_rot_euler = None
        else:
            r_approach = Rot.from_euler(
                "xyz", np.asarray(euler_xyz_deg, dtype=float), degrees=True)
            self.wrist_camera_flange_rot_euler = tuple(float(v) for v in euler_xyz_deg)

        lateral = WRIST_CAMERA_LATERAL_M if lateral_m is None else float(lateral_m)
        focus = WRIST_CAMERA_FOCUS_M if focus_m is None else float(focus_m)
        back = WRIST_CAMERA_BACK_M if back_m is None else float(back_m)
        down_tilt = WRIST_CAMERA_DOWN_TILT_DEG if down_tilt_deg is None else float(down_tilt_deg)
        self.wrist_camera_lateral_m = lateral
        self.wrist_camera_focus_m = focus
        self.wrist_camera_back_m = back
        self.wrist_camera_down_tilt_deg = down_tilt

        # A USD camera images along its own local -Z, so that is the tool's
        # approach direction expressed in the flange frame.
        view_dir = r_approach.apply(np.array([0.0, 0.0, -1.0]))
        # Any direction perpendicular to the tool axis will do for the
        # bracket; pick one deterministically.
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(reference, view_dir))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        side = np.cross(view_dir, reference)
        side /= np.linalg.norm(side)

        if down_tilt != 0.0:
            # Rotate the approach axis about the same `side` axis the lateral
            # offset already uses, so the tilt stays in the plane the bracket
            # geometry below is built in.
            view_dir = Rot.from_rotvec(np.radians(down_tilt) * side).apply(view_dir)
            view_dir /= np.linalg.norm(view_dir)

        # Behind the gripper and off to one side, aimed at a point out along
        # the approach axis -- so at the grasp the camera looks down over the
        # gripper at the cube instead of sitting level with it.
        eye = side * lateral - view_dir * back
        target = view_dir * focus
        r_cam = _look_at_rotation(eye, target)

        self._wrist_cam_translate_op.Set(Gf.Vec3d(*eye))
        q = r_cam.as_quat()  # xyzw
        self._wrist_cam_orient_op.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))

    def wrist_camera_world_position(self):
        """Where the wrist camera actually ended up, for diagnosing a mount
        that renders black because it is buried inside the arm."""
        pos, _ = prim_world_pose(self.stage.GetPrimAtPath(WRIST_CAMERA_PRIM_PATH))
        return pos

    def reset(self):
        """Randomizes the cube's spawn position (domain randomization so
        the eventual policy has to look at the image, not memorize one
        pose), resets the robot to its default pose, and opens the
        gripper. Returns the initial observation (see get_observation)."""
        cube_x = self._rng.uniform(*CUBE_X_RANGE)
        cube_y = self._rng.uniform(*CUBE_Y_RANGE)
        self.cube_position = np.array([cube_x, cube_y, CUBE_Z])

        # CONFIRMED 2026-09-16 (check_pose_readout_multireset.py): removing
        # and recreating this prim at the same path every episode -- what
        # this used to do -- left prim_world_pose/get_cube_position's
        # ComputeLocalToWorldTransform read frozen at the FIRST episode's
        # position forever, across every episode after it (spawn-vs-read
        # error: 0, 156, 184, 260, 267mm over 5 fresh episodes in that
        # repro). Fixed by creating the prim ONCE and repositioning the
        # SAME prim on every reset instead of tearing it down -- but the
        # first attempt at that reposition (a raw USD translate op, set
        # BEFORE world.reset()) turned out to have its own, separate bug:
        # CONFIRMED 2026-09-16 (diag_reset_stages.py) that world.reset()'s
        # Stop+Play cycle discards whatever the USD attribute holds at Stop
        # time and snaps the rigid body back to the pose PhysX cached from
        # its very FIRST Play -- every episode after the first landed at
        # the exact same fixed point (e.g. [0.396, -0.177, 0.02]) regardless
        # of what had just been authored, which is what check_cameras.py's
        # --sweep_wrist --wrist_samples 5 was actually measuring as "the
        # tool didn't reach the cube" in 32/34 samples: RMPflow tracking
        # was fine (2-23mm), the cube itself wasn't where cube_position said.
        # Fix: reposition through SingleRigidPrim.set_world_pose AFTER
        # world.reset(), not through a raw USD op before it -- it writes
        # straight into the live PhysX rigid-body view (see
        # isaacsim.core.prims RigidPrim.set_world_poses:
        # physics_view.set_transforms(...) when the handle is valid),
        # bypassing the Stop/Play cache entirely. CONFIRMED zero drift
        # across 10 fresh episodes (diag_single_rigid_prim_fix.py) once
        # done in this order. Same "construct once, initialize() every
        # episode" pattern as self.robot below -- constructing a NEW
        # SingleRigidPrim after every reset crashed with "Simulation view
        # object is invalidated".
        cube_prim = self.stage.GetPrimAtPath(CUBE_PRIM_PATH)
        cube_prim_is_new = not cube_prim.IsValid()
        if cube_prim_is_new:
            add_cube(self.stage, CUBE_PRIM_PATH, self.cube_position, size=CUBE_SIZE_M)

        if not self.stage.GetPrimAtPath(TARGET_MARKER_PRIM_PATH).IsValid():
            add_place_target_marker(self.stage, TARGET_MARKER_PRIM_PATH, PLACE_TARGET_POSITION)

        self.world.reset()
        # A non-soft world.reset() is a hard reset (Stop+Play) under the
        # hood, which invalidates the arm's physics handles since it's not
        # registered via world.scene (see __init__'s comment) -- redo it
        # every episode, not just once at construction.
        self.robot.initialize()

        if self.cube_rigid_prim is None:
            # Only constructible once physics has Played at least once
            # (mirrors self.robot needing world.reset() before initialize()).
            # cube_prim_is_new is True here, so the cube is already sitting
            # at self.cube_position from add_cube -- no reposition needed
            # this episode.
            self.cube_rigid_prim = SingleRigidPrim(CUBE_PRIM_PATH, name="cube_rigid")
            self.cube_rigid_prim.initialize()
        else:
            self.cube_rigid_prim.initialize()
            self.cube_rigid_prim.set_world_pose(position=self.cube_position)
            self.cube_rigid_prim.set_linear_velocity(np.zeros(3))
            self.cube_rigid_prim.set_angular_velocity(np.zeros(3))

        self._sync_gripper_to_flange()
        self.gripper.open()
        # A fresh world.reset() hasn't rendered a frame yet -- the replicator
        # annotators return an empty array until at least one render pass
        # happens, so get_observation() right after reset would report
        # base_rgb/wrist_rgb with shape (0,) instead of (256, 256, 4).
        self.world.step(render=True)
        return self.get_observation()

    def _sync_gripper_to_flange(self):
        """STALE NAME/DOCSTRING, kept 2026-09-21 only to avoid touching every
        call site: this used to teleport a separately-attached gripper prim
        onto the flange every tick, back when the gripper was NOT part of
        the arm's own articulation (the "teleport" attachment mode --
        GRIPPER_PRIM_PATH, referenced by the old version of this docstring,
        no longer exists anywhere in isaac_sim_common.py). Since
        select_gripper_variant welds the gripper into the arm's OWN
        articulation via a USD variant selection, GripperController.
        sync_pose_to_flange is a documented no-op and this call does
        nothing for the with_gripper=True path -- there is no separate prim
        left to sync. Confirmed misleading enough on its own to produce a
        very reasonable but incorrect hypothesis for an unrelated PhysX
        divergence found by a 2026-09-21 gripper-mass A/B test."""
        flange_pos, flange_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        self.gripper.sync_pose_to_flange(flange_pos, flange_quat)

    def _grip_pose_to_tool0(self, target_pos, target_rotvec):
        """Convert a GRIP pose -- where the fingertips should be, and which
        way the gripper should point (+Z of the given rotation) -- into the
        tool0 pose RMPflow is driven with.

        Two corrections, both measured rather than assumed (see
        GRIPPER_TCP_OFFSET_M): step back GRIPPER_TCP_OFFSET_M along the
        approach so the FINGERS land on the target rather than the flange,
        and compose the fixed flange->tool0 rotation so "point the gripper
        this way" is not off by the 90 degrees between those frames."""
        r_grip = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float))
        approach = r_grip.apply(np.array([0.0, 0.0, 1.0]))
        flange_pos = np.asarray(target_pos, dtype=float) - GRIPPER_TCP_OFFSET_M * approach
        # tool0 and the flange share a position, so only the rotation composes
        quat_xyzw = (r_grip * self.r_flange_to_tool0).as_quat()
        return flange_pos, quat_xyzw[[3, 0, 1, 2]]

    def grip_point_world(self):
        """Where the fingertips currently are -- the point the scripted
        waypoints and any reach measurement are about. The flange prim alone
        is GRIPPER_TCP_OFFSET_M short of it."""
        pos, quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        return pos + GRIPPER_TCP_OFFSET_M * Rot.from_quat(quat).apply(np.array([0.0, 0.0, 1.0]))

    def step_towards(self, target_pos, target_rotvec, target_gripper):
        """One control tick: sets the RMPflow Cartesian target, applies one
        articulation action, drives the gripper toward target_gripper
        (0=open .. 1=closed), steps physics/render once, then re-attaches the
        (top-level, not USD-parented) gripper to wherever the flange actually
        ended up. Deliberately NOT a blocking settle loop (contrast with the
        potato_scan RL train envs' _move_and_settle) -- the caller
        (scripted_pick_place.py / collect_demos.py) controls the logging
        cadence by calling this once per logged frame."""
        flange_pos, tool0_quat_wxyz = self._grip_pose_to_tool0(target_pos, target_rotvec)
        self.rmpflow.set_end_effector_target(flange_pos, tool0_quat_wxyz)
        self.rmpflow.update_world()
        action = self.articulation_policy.get_next_articulation_action(self.physics_dt)
        self.robot.apply_action(action)
        self.gripper.set_target(target_gripper)
        self.world.step(render=True)
        self._sync_gripper_to_flange()

    def apply_joint_targets(self, joint_positions, gripper_target):
        """Low-level control primitive for policies that already output
        joint-space actions (openpi's UR5e contract, and anything built on
        top of it like the residual RL policy in residual_rl_train_env.py)
        -- sets the articulation's joint position targets DIRECTLY, no
        RMPflow/Cartesian IK involved. Contrast with step_towards, which is
        Cartesian-space and RMPflow-driven, used only by the scripted
        demonstrator. Same apply_action(ArticulationAction(...)) approach
        pick_place_scene_bridge.py uses inline for real-time policy
        inference (self.robot is a SingleArticulation, unbatched -- see
        __init__'s comment -- not the vectorized Articulation class, so
        set_joint_position_targets isn't available/correct here); factored
        out here so residual_rl_train_env.py doesn't have to duplicate it.
        Also re-syncs the gripper to the flange afterward, same as
        step_towards -- it's a top-level, not USD-parented prim that only
        stays attached via this per-tick teleport."""
        from isaacsim.core.utils.types import ArticulationAction
        self.robot.apply_action(
            ArticulationAction(joint_positions=np.asarray(joint_positions, dtype=float)[:6]))
        self.gripper.set_target(gripper_target)
        self.world.step(render=True)
        self._sync_gripper_to_flange()

    def get_observation(self):
        # The fingertips, not the bare flange: waypoints are expressed as grip
        # points (see _grip_pose_to_tool0), so the pose reported back has to
        # be the same quantity or the scripted expert interpolates from a
        # start GRIPPER_TCP_OFFSET_M away from where it thinks it is.
        _, tool_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        tool_pos = self.grip_point_world()
        # Confirmed against the actual asset: self.robot.dof_names is exactly
        # the 6 arm joints (shoulder_pan..wrist_3), no extra joints ahead of
        # them -- SingleArticulation.get_joint_positions() is unbatched
        # (shape (6,)), unlike the vectorized Articulation API.
        joint_pos = np.asarray(self.robot.get_joint_positions())[:6]
        gripper_pos = self.gripper.get_normalized_position()
        return {
            "joints": joint_pos.astype(np.float32),
            "gripper": np.array([gripper_pos], dtype=np.float32),
            "base_rgb": self.base_rgb_annotator.get_data(),
            "wrist_rgb": self.wrist_rgb_annotator.get_data(),
            "tool_pos": tool_pos,
            "tool_quat": tool_quat,
        }

    def get_cube_position(self):
        pos, _ = prim_world_pose(self.stage.GetPrimAtPath(CUBE_PRIM_PATH))
        return pos

    # --- episode outcome checks -------------------------------------------
    # collect_demos.py logs a demonstration only if the scripted expert
    # actually did the task. Without this every episode was logged
    # unconditionally, so a failed grasp still went into the training set as
    # if it were a demonstration, teaching the policy to mime the motion
    # whether or not it picked anything up. Ground truth is free here.

    def grasp_succeeded(self, max_cube_z, lift_z_threshold=LIFT_Z_THRESHOLD):
        """Was the cube ever actually lifted clear of the table? The cube
        rests with its centre at CUBE_Z (0.02 m); the scripted expert lifts
        it to roughly STANDOFF_HEIGHT above that, so anything that never got
        past `lift_z_threshold` was never really grasped."""
        return bool(max_cube_z >= lift_z_threshold)

    def place_error_m(self):
        """Planar distance from the cube's final resting position to the
        place target -- the same xy-error measure hybrid_pick_place_demo.py
        and vla_policy_client.py's eval_mode report, so demonstration
        acceptance and policy evaluation are scored the same way."""
        return float(np.linalg.norm(self.get_cube_position()[:2] - PLACE_TARGET_POSITION[:2]))

    @staticmethod
    def cube_pixels_visible(base_rgb):
        """How many pixels of the (red) cube the base camera can see in this
        frame. add_shape colours the cube (0.8, 0.1, 0.1), so count pixels
        whose red channel clearly dominates both others -- robust to the
        scene's blue-ish ambient light in a way a fixed per-channel
        threshold is not.

        collect_demos.py accumulates the per-episode maximum of this and
        refuses to collect a whole dataset in which the cube is never
        properly visible. That is the check the first 100-episode run
        needed: it produced 21,000 perfectly well-formed frames in which
        the cube appeared exactly zero times."""
        img = np.asarray(base_rgb)
        if img.size == 0 or img.ndim != 3:
            return 0
        i = img[..., :3].astype(np.int16)
        return int(((i[..., 0] - np.maximum(i[..., 1], i[..., 2])) > 40).sum())

    def preflight_check(self, verbose=True):
        """Cheap sanity check that the scene is actually renderable and the
        task is visible BEFORE committing to a long collection run.

        This exists because two full collection runs were already lost to
        silent camera faults that a single glance at a frame would have
        caught: a default 1.0 m near-clip plane that removed the entire
        task from both cameras (all-black wrist frames, mean 0.0), and a
        base camera whose guessed Euler tilt aimed it at the sky. Both are
        fixed in _setup_cameras now, and both are exactly what this checks
        for -- non-degenerate images, and the (red) cube visible to the base
        camera.

        Returns (ok, list_of_problem_strings)."""
        obs = self.get_observation()
        problems = []

        for name in ("base_rgb", "wrist_rgb"):
            img = np.asarray(obs[name])
            if img.size == 0:
                problems.append(f"{name}: annotator returned an empty array (no render pass has run yet)")
                continue
            if img.ndim != 3 or img.shape[:2] != tuple(CAMERA_RESOLUTION):
                problems.append(f"{name}: expected {CAMERA_RESOLUTION} HxW, got shape {img.shape}")
                continue
            rgb = img[..., :3].astype(np.float32)
            if float(rgb.std()) < 1.0:
                problems.append(
                    f"{name}: image is flat (std={rgb.std():.3f}, mean={rgb.mean():.1f}) -- "
                    "camera is clipping the scene away or pointing at nothing")
            elif float((rgb.max(axis=2) < 12).mean()) > MAX_DARK_FRACTION:
                # Not flat, but mostly void: this is what the wrist camera
                # looked like while aimed past the workspace -- a sliver of
                # background grid against black sky, which passes a plain
                # std check.
                problems.append(
                    f"{name}: {100 * float((rgb.max(axis=2) < 12).mean()):.0f}% of the frame is "
                    f"near-black -- the camera is aimed past the scene, not at it")

        # Deliberately NOT checking cube visibility here. Measured on an
        # actual collected episode: at reset the base camera sees the cube
        # in ~1 pixel (the arm's default pose sits between camera and cube),
        # and it only becomes clearly visible around frame 90 once it has
        # been lifted. A reset-time cube check would therefore reject a
        # perfectly good setup. Visibility is tracked across the whole first
        # episode instead -- see cube_pixels_visible, called from
        # collect_demos.py.

        # Framing is reported, not failed on: a camera that cannot make the
        # cube big enough is a design limit to decide about, not a fault to
        # abort on, and the numbers are the same every run. The episode-level
        # guard (MIN_CUBE_PIXELS_IN_BASE_VIEW, derived from these numbers) is
        # what actually stops a bad collection.
        if verbose:
            print("preflight framing (base camera):")
            for line in camera_framing.describe(BASE_FRAMING):
                print(f"  {line}")
            print(f"  episodes must peak at >= {MIN_CUBE_PIXELS_IN_BASE_VIEW} px "
                  f"({100 * MIN_CUBE_PIXELS_IN_BASE_VIEW / BASE_FRAMING.expected_peak_area_px:.0f}%"
                  f" of the achievable {BASE_FRAMING.expected_peak_area_px:.0f} px)")
            if not BASE_FRAMING.target_reachable:
                print(f"  NOTE: {BASE_FRAMING.target_span_px:.0f} px across is out of reach "
                      f"here, so the wrist camera carries the approach -- run "
                      f"camera_framing.py for the alternatives")

            for p in problems:
                print(f"preflight PROBLEM -- {p}")
            if not problems:
                print("preflight: cameras OK, cube visible")
        return (len(problems) == 0), problems

    def get_object_position(self, prim_path):
        """Ground-truth world position of any prim spawned by
        spawn_random_objects -- used by hybrid_pick_place_demo.py to check
        whether the object actually named/moved matches the one the LLM +
        detector picked, and whether it ended up in the target zone."""
        pos, _ = prim_world_pose(self.stage.GetPrimAtPath(prim_path))
        return pos

    def get_base_camera_pose(self):
        """(position, rotation_matrix) of the base camera, read from its
        actual USD transform -- used by hybrid_pick_place_demo.py with
        camera_projection.py rather than re-deriving the rotation from the
        Euler constants in _setup_cameras, so any convention mismatch
        between USD's RotateXYZ and camera_projection's expected matrix is
        sidestepped entirely (the ground-truth transform is read back
        directly, the same prim_world_pose helper used everywhere else in
        this file)."""
        from scipy.spatial.transform import Rotation as Rot
        pos, quat_xyzw = prim_world_pose(self.stage.GetPrimAtPath(BASE_CAMERA_PRIM_PATH))
        return pos, Rot.from_quat(quat_xyzw).as_matrix()

    def spawn_random_objects(self, n=3):
        """Multi-object scene for the LLM + open-vocabulary hybrid pipeline
        (hybrid_pick_place_demo.py) -- separate from the single-cube
        CUBE_PRIM_PATH/reset() Phase 1-5 already use, so this is purely
        additive. Clears any previously-spawned objects, then spawns `n`
        distinct (color, shape) pairs from object_configs.OBJECT_VOCABULARY
        at random non-overlapping positions. Returns
        {description: {"prim_path", "position"}}.

        CONFIRMED (2026-09-07, actual Isaac Sim run) root cause of a native
        RTX/Hydra crash ("invalid points buffer, expected 8, got N" on
        objN, then a hard crash inside librtx.scenedb/librtx.hydra a couple
        episodes later, in BOTH headless and GUI runs): reusing the same
        prim path (f"{...}/obj_{i}") across spawns while its USD TYPE
        changes (obj_1 is a Cube one episode, a Sphere the next -- shapes
        are assigned by index, not fixed per slot) leaves Hydra's
        per-path render-index entry holding the PREVIOUS type's cached
        topology (a Cube's implicit 8-point proxy) against the NEW prim's
        real geometry (a tessellated sphere/cylinder mesh has far more
        points) -- exactly matching the "expected 8, got N" pattern.
        Giving every spawn generation its own fresh prim paths (instead of
        RemovePrim + redefine at the SAME path) means Hydra only ever sees
        brand-new paths, never a type change at an existing one."""
        if self.stage.GetPrimAtPath(OBJECTS_PARENT_PRIM_PATH).IsValid():
            self.stage.RemovePrim(OBJECTS_PARENT_PRIM_PATH)
        self._object_spawn_generation += 1

        chosen = object_configs.sample_objects(n, self._rng)
        positions = []
        objects = {}
        for i, (color, shape) in enumerate(chosen):
            for _ in range(20):  # a handful of rejection-sampling attempts per object
                candidate = np.array([
                    self._rng.uniform(*CUBE_X_RANGE),
                    self._rng.uniform(*CUBE_Y_RANGE),
                    CUBE_Z,
                ])
                if all(np.linalg.norm(candidate[:2] - p[:2]) >= MIN_OBJECT_SEPARATION_M for p in positions):
                    break
            positions.append(candidate)

            prim_path = f"{OBJECTS_PARENT_PRIM_PATH}/gen{self._object_spawn_generation}_obj_{i}"
            add_shape(self.stage, shape, prim_path, candidate, color=object_configs.COLOR_RGB[color])
            objects[object_configs.description(color, shape)] = {
                "prim_path": prim_path, "position": candidate,
            }

        # Same hard-reset dance reset() does above: a non-soft world.reset()
        # invalidates self.robot's physics handles (see __init__'s comment),
        # and a fresh reset hasn't rendered a frame yet so the camera
        # annotators would return empty arrays to whatever calls
        # get_observation() right after this (e.g. hybrid_pick_place_demo.py).
        self.world.reset()
        self.robot.initialize()
        self._sync_gripper_to_flange()
        self.gripper.open()
        self.world.step(render=True)
        return objects
