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
from isaacsim.core.prims import Articulation

from isaac_sim_common import (
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    add_gripper, GripperController, setup_rmpflow, prim_world_pose,
    add_cube, add_place_target_marker,
)

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


class PickPlaceScene:
    def __init__(self):
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root -- check Nucleus connection.")

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()

        add_reference_to_stage(assets_root + UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH)
        self.robot = Articulation(ROBOT_PRIM_PATH)
        self.world.reset()  # initializes physics handles for the articulation

        gripper_path = add_gripper(self.stage, TOOL_LINK_PRIM_PATH, assets_root)
        self.gripper = GripperController(gripper_path)

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)
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
        self.gripper.open()
        return self.get_observation()

    def step_towards(self, target_pos, target_rotvec, target_gripper):
        """One control tick: sets the RMPflow Cartesian target, applies one
        articulation action, drives the gripper toward target_gripper
        (0=open .. 1=closed), and steps physics/render once. Deliberately
        NOT a blocking settle loop (contrast with the potato_scan RL train
        envs' _move_and_settle) -- the caller (scripted_pick_place.py /
        collect_demos.py) controls the logging cadence by calling this once
        per logged frame."""
        target_quat_wxyz = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float)).as_quat()[[3, 0, 1, 2]]
        self.rmpflow.set_end_effector_target(np.asarray(target_pos, dtype=float), target_quat_wxyz)
        self.rmpflow.update_world()
        action = self.articulation_policy.get_next_articulation_action(self.physics_dt)
        self.robot.apply_action(action)
        self.gripper.set_target(target_gripper)
        self.world.step(render=True)

    def get_observation(self):
        tool_pos, tool_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        # ADJUST: assumes the arm's 6 DoF are the first 6 entries in the
        # articulation's DOF list (true for the stock UR5e asset with no
        # extra joints ahead of the arm) -- verify against
        # self.robot.dof_names on your install if this looks wrong.
        joint_pos = np.asarray(self.robot.get_joint_positions())[0, :6]
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
