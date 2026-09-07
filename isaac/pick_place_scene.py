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
import numpy as np
import omni.replicator.core as rep
from pxr import UsdGeom, Gf
from scipy.spatial.transform import Rotation as Rot

from isaacsim.core.api import World
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import SingleArticulation

from isaac_sim_common import (
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    add_gripper, GripperController, setup_rmpflow, prim_world_pose,
    add_cube, add_shape, add_place_target_marker,
)
import object_configs

CUBE_PRIM_PATH = "/World/cube"
TARGET_MARKER_PRIM_PATH = "/World/place_target"
BASE_CAMERA_PRIM_PATH = "/World/base_camera"
WRIST_CAMERA_PRIM_PATH = f"{TOOL_LINK_PRIM_PATH}/wrist_camera"

# Workspace bounds the cube is randomized within (robot base frame, meters) --
# ADJUST to whatever's actually reachable/visible on your table setup.
CUBE_X_RANGE = (0.35, 0.55)
CUBE_Y_RANGE = (-0.20, 0.20)
CUBE_Z = 0.02  # resting height for a 4cm cube on the table surface
PLACE_TARGET_POSITION = np.array([0.45, 0.30, 0.0])

# Same eye-in-hand offset convention potato_scan's camera uses (translation
# only, no extra rotation -- camera +Z follows tool0's own orientation).
WRIST_CAMERA_OFFSET = (0.0, -0.05, 0.05)
BASE_CAMERA_POSITION = (0.9, 0.0, 0.5)
# 256x256 matches openpi's LIBERO example image shape -- keep the
# training-side transform (UR5eInputs) consistent with whatever's set here.
CAMERA_RESOLUTION = (256, 256)

# Base camera's horizontal FOV, realized via focal length / aperture below
# so camera_projection.py's pinhole math (which takes this same value) is
# actually accurate -- ADJUST alongside BASE_CAMERA_FOCAL_LENGTH_MM if you
# change the camera's framing; the two must stay consistent.
BASE_CAMERA_HORIZONTAL_FOV_DEG = 60.0
BASE_CAMERA_FOCAL_LENGTH_MM = 24.0

# Multi-object scene (isaac/object_configs.py's vocabulary) used by the LLM
# + open-vocabulary hybrid pipeline (hybrid_pick_place_demo.py) -- kept
# separate from the single-cube CUBE_PRIM_PATH/reset() Phase 1-5 already use.
OBJECTS_PARENT_PRIM_PATH = "/World/objects"
MIN_OBJECT_SEPARATION_M = 0.12  # reject a random layout where two objects would overlap/collide


class PickPlaceScene:
    def __init__(self):
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root -- check Nucleus connection.")

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()

        add_reference_to_stage(assets_root + UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH)
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

        gripper_path = add_gripper(self.stage, assets_root)
        self.gripper = GripperController(gripper_path)
        self.world.reset()
        self.robot.initialize()

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)
        self._sync_gripper_to_flange()
        self.physics_dt = 1.0 / 60.0

        self._setup_cameras()
        self._rng = np.random.default_rng()
        self.cube_position = None

    def _setup_cameras(self):
        base_cam = UsdGeom.Camera.Define(self.stage, BASE_CAMERA_PRIM_PATH)
        base_cam.AddTranslateOp().Set(Gf.Vec3d(*BASE_CAMERA_POSITION))
        # ADJUST: placeholder look-down/inward tilt so the workspace is in
        # frame -- replace with the exact orientation for your table
        # geometry (or compute via pose_utils.look_at_rotation, as
        # potato_scan's scan_controller does for its camera poses) if the
        # render looks off.
        base_cam.AddRotateXYZOp().Set(Gf.Vec3d(0.0, 55.0, 180.0))
        # Realize BASE_CAMERA_HORIZONTAL_FOV_DEG via focal length/aperture so
        # camera_projection.py's pinhole math (fed that same constant) is
        # actually accurate -- ADJUST if this doesn't match the rendered FOV
        # on your Isaac Sim version's camera attribute conventions.
        import math
        horizontal_aperture_mm = 2.0 * BASE_CAMERA_FOCAL_LENGTH_MM * math.tan(
            math.radians(BASE_CAMERA_HORIZONTAL_FOV_DEG) / 2.0)
        base_cam.CreateFocalLengthAttr(BASE_CAMERA_FOCAL_LENGTH_MM)
        base_cam.CreateHorizontalApertureAttr(horizontal_aperture_mm)

        wrist_cam = UsdGeom.Camera.Define(self.stage, WRIST_CAMERA_PRIM_PATH)
        wrist_cam.AddTranslateOp().Set(Gf.Vec3d(*WRIST_CAMERA_OFFSET))

        self.base_rp = rep.create.render_product(BASE_CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
        self.wrist_rp = rep.create.render_product(WRIST_CAMERA_PRIM_PATH, CAMERA_RESOLUTION)
        self.base_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.wrist_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.base_rgb_annotator.attach([self.base_rp])
        self.wrist_rgb_annotator.attach([self.wrist_rp])

    def reset(self):
        """Randomizes the cube's spawn position (domain randomization so
        the eventual policy has to look at the image, not memorize one
        pose), resets the robot to its default pose, and opens the
        gripper. Returns the initial observation (see get_observation)."""
        cube_x = self._rng.uniform(*CUBE_X_RANGE)
        cube_y = self._rng.uniform(*CUBE_Y_RANGE)
        self.cube_position = np.array([cube_x, cube_y, CUBE_Z])

        if self.stage.GetPrimAtPath(CUBE_PRIM_PATH).IsValid():
            self.stage.RemovePrim(CUBE_PRIM_PATH)
        add_cube(self.stage, CUBE_PRIM_PATH, self.cube_position)

        if not self.stage.GetPrimAtPath(TARGET_MARKER_PRIM_PATH).IsValid():
            add_place_target_marker(self.stage, TARGET_MARKER_PRIM_PATH, PLACE_TARGET_POSITION)

        self.world.reset()
        # A non-soft world.reset() is a hard reset (Stop+Play) under the
        # hood, which invalidates the arm's physics handles since it's not
        # registered via world.scene (see __init__'s comment) -- redo it
        # every episode, not just once at construction.
        self.robot.initialize()
        self._sync_gripper_to_flange()
        self.gripper.open()
        # A fresh world.reset() hasn't rendered a frame yet -- the replicator
        # annotators return an empty array until at least one render pass
        # happens, so get_observation() right after reset would report
        # base_rgb/wrist_rgb with shape (0,) instead of (256, 256, 4).
        self.world.step(render=True)
        return self.get_observation()

    def _sync_gripper_to_flange(self):
        """Teleports the (top-level, not USD-parented -- see
        isaac_sim_common.GRIPPER_PRIM_PATH) gripper prim to the tool
        flange's current world pose, keeping it visually/functionally
        attached to the arm."""
        flange_pos, flange_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        self.gripper.sync_pose_to_flange(flange_pos, flange_quat)

    def step_towards(self, target_pos, target_rotvec, target_gripper):
        """One control tick: sets the RMPflow Cartesian target, applies one
        articulation action, drives the gripper toward target_gripper
        (0=open .. 1=closed), steps physics/render once, then re-attaches the
        (top-level, not USD-parented) gripper to wherever the flange actually
        ended up. Deliberately NOT a blocking settle loop (contrast with the
        potato_scan RL train envs' _move_and_settle) -- the caller
        (scripted_pick_place.py / collect_demos.py) controls the logging
        cadence by calling this once per logged frame."""
        target_quat_wxyz = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_quat()[[3, 0, 1, 2]]
        self.rmpflow.set_end_effector_target(np.asarray(target_pos, dtype=float), target_quat_wxyz)
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
        tool_pos, tool_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
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
        {description: {"prim_path", "position"}}."""
        if self.stage.GetPrimAtPath(OBJECTS_PARENT_PRIM_PATH).IsValid():
            self.stage.RemovePrim(OBJECTS_PARENT_PRIM_PATH)

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

            prim_path = f"{OBJECTS_PARENT_PRIM_PATH}/obj_{i}"
            add_shape(self.stage, shape, prim_path, candidate, color=object_configs.COLOR_RGB[color])
            objects[object_configs.description(color, shape)] = {
                "prim_path": prim_path, "position": candidate,
            }

        self.world.reset()
        self.gripper.open()
        return objects
