"""Which simulated setup the scene builds -- VLA_SIM_PROFILE=v4 (default) or v5.

v4: everything as it was for the v1-v4 data and every evaluation in the README.  Nothing changes unless v5 is asked for, so
    old results stay reproducible.
v5: the simulator matched to the real robot (README "v5 decision", 2026-11-16).  One exception to "one change at a time",
    with a new baseline:
      * gripper mounted COAXIAL with wrist_3 / tool0 (v1-v4 had it 90 deg off -- README BLOCKER, confirmed on URSim),
        fingertips at the real TCP length from the flange face;
      * expert joint speeds held to a fraction of the UR5e limit (v4 peaked at 4.9 rad/s on wrist_3 against 3.14);
      * cameras, table height and cube size as in the lab;
      * v4's visual randomization and converged first frame kept (those are collect_demos.py options, not set here).

Values marked LAB are placeholders until measured in the lab (the user brings them).  `placeholders()` lists the ones still
unset, and collect_demos.py refuses to write a v5 dataset while any remain, unless --allow_placeholders (expert checks only).
Frames: UR base frame for anything measured on the real robot (what the teach pendant shows); Isaac world = UR base rotated
180 deg about z, converted here with ur_base_to_world.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

PROFILE = os.environ.get("VLA_SIM_PROFILE", "v4").strip().lower()
if PROFILE not in ("v4", "v5"):
    raise ValueError(f"VLA_SIM_PROFILE={PROFILE!r}: expected v4 or v5")

# The sim Robotiq 2F-85's own length, flange prim origin to the fingertip grip point (pick_place_scene.GRIPPER_TCP_OFFSET_M,
# measured 2026-09-10 on the asset).  A real gripper plus its coupling is usually longer; the difference becomes a spacer.
SIM_GRIPPER_LENGTH_M = 0.12


def ur_base_to_world(p):
    """UR base frame -> Isaac world (180 deg about z; fitted on the bridge's traces, residual 0.000 mm)."""
    p = np.asarray(p, dtype=float)
    return np.array([-p[0], -p[1], p[2]])


_LAB = {}   # name -> placeholder value, for the names still waiting on a lab measurement


def _lab(name, placeholder):
    _LAB[name] = placeholder
    return placeholder


if PROFILE == "v4":
    GRIPPER_MOUNT = "legacy"            # gripper along the flange prim's +Z, 90 deg off tool0
    GRIPPER_ROLL_DEG = 0.0
    GRIPPER_TCP_M = SIM_GRIPPER_LENGTH_M
    EXPERT_MAX_JOINT_SPEED_RAD_S = None  # no limit (as collected)
    TABLE_Z_M = 0.0
    CUBE_SIZE_M = 0.04
    BASE_CAMERA = None                  # None = pick_place_scene's v4 constants
    WRIST_CAMERA_TOOL0 = None
    CAMERA_RESOLUTION = (256, 256)
else:
    GRIPPER_MOUNT = "coaxial"
    # Rotation of the fingers about the tool axis.  0 keeps the finger closing direction where v4 had it relative to the grip
    # frame (the expert grasps with the same yaw).  LAB: which way the real fingers close relative to the tool flange's
    # x/y (teach pendant tool frame) -- matters for the wrist-camera view, not for the 90-deg-symmetric cube.
    GRIPPER_ROLL_DEG = _lab("GRIPPER_ROLL_DEG", 0.0)
    # LAB: flange face to the point midway between the fingertip pads, Robotiq 2F-85 + coupling (Robotiq's datasheet value
    # for the coupling used, or measured).  Must be >= SIM_GRIPPER_LENGTH_M; the excess is inserted as a spacer.
    GRIPPER_TCP_M = _lab("GRIPPER_TCP_M", SIM_GRIPPER_LENGTH_M)
    # Half the UR5e's 180 deg/s on every joint (the reviewer: "aim for about half the limit").
    EXPERT_MAX_JOINT_SPEED_RAD_S = 0.5 * np.pi
    # LAB: table surface height in the UR base frame (0 = the robot stands on the table the cube lies on).
    TABLE_Z_M = _lab("TABLE_Z_M", 0.0)
    # LAB: cube edge length.
    CUBE_SIZE_M = _lab("CUBE_SIZE_M", 0.04)
    # LAB: base camera in the UR base frame -- position (m), the point it looks at (m), horizontal FOV (deg) of the real
    # camera at the resolution used.  Placeholder = v4's camera expressed in the UR base frame.
    BASE_CAMERA = _lab("BASE_CAMERA", dict(position_ur=(-0.9, 0.0, 0.5), aim_ur=(-0.45, -0.10, 0.02), hfov_deg=60.0))
    # LAB: wrist camera in the TOOL0 frame (z along the tool axis, x/y as the teach pendant shows the tool flange): position
    # (m) and a point it looks at (m), plus its horizontal FOV.  None = keep v4's camera pose relative to the gripper.
    WRIST_CAMERA_TOOL0 = _lab("WRIST_CAMERA_TOOL0", None)
    # LAB: the real cameras' aspect ratio, as rendered here (openpi resizes with padding to 224x224 on both sides, so
    # matching the aspect matters, not the pixel count).  Placeholder = v4's square images.
    CAMERA_RESOLUTION = _lab("CAMERA_RESOLUTION", (256, 256))

if GRIPPER_TCP_M < SIM_GRIPPER_LENGTH_M - 1e-9:
    raise ValueError(f"GRIPPER_TCP_M {GRIPPER_TCP_M} is shorter than the sim gripper itself ({SIM_GRIPPER_LENGTH_M}); "
                     "a spacer cannot be negative -- shorten the sim gripper instead")
GRIPPER_SPACER_M = GRIPPER_TCP_M - SIM_GRIPPER_LENGTH_M


# The grip frame (+Z = the way the fingers point; every expert waypoint is written in it) expressed in the flange prim's frame.
# The asset's variant mounts the gripper along the flange prim's +Z (identity, v4); the real tool axis -- tool0 z, the
# wrist_3 axis -- is the flange prim's +X, so coaxial = 90 deg about the flange's y (taking +Z to +X), after the roll.
if GRIPPER_MOUNT == "coaxial":
    R_GRIP_IN_FLANGE = Rot.from_euler("y", 90.0, degrees=True) * Rot.from_euler("z", float(GRIPPER_ROLL_DEG), degrees=True)
else:
    R_GRIP_IN_FLANGE = Rot.identity()
# where realign_gripper_mount moves the gripper: R_GRIP_IN_FLANGE about the flange origin, then the spacer along the new axis
GRIPPER_SHIFT_IN_FLANGE = GRIPPER_SPACER_M * R_GRIP_IN_FLANGE.apply(np.array([0.0, 0.0, 1.0]))


def grip_point(flange_pos, flange_rot):
    """Fingertip position from the flange prim's world pose (flange_rot: scipy Rotation)."""
    return np.asarray(flange_pos, dtype=float) + GRIPPER_TCP_M * (flange_rot * R_GRIP_IN_FLANGE).apply([0.0, 0.0, 1.0])


def flange_target(grip_pos, r_grip, r_flange_to_tool0):
    """(flange position, tool0 rotation) that put the fingertips at grip_pos with the grip frame at r_grip -- what RMPflow
    and the diff-IK servo are commanded with.  world<-flange = world<-grip * (flange<-grip)^-1; tool0 shares the flange's
    position, and world<-tool0 = world<-flange * flange<-tool0."""
    approach = r_grip.apply([0.0, 0.0, 1.0])
    return (np.asarray(grip_pos, dtype=float) - GRIPPER_TCP_M * approach,
            r_grip * R_GRIP_IN_FLANGE.inv() * r_flange_to_tool0)


def placeholders():
    """Names of the LAB values still at their placeholder (empty for v4)."""
    return sorted(_LAB)


def describe():
    return (f"sim profile {PROFILE}: gripper {GRIPPER_MOUNT} (roll {GRIPPER_ROLL_DEG} deg, TCP {GRIPPER_TCP_M:.3f} m, spacer "
            f"{GRIPPER_SPACER_M:.3f} m), expert joint speed cap {EXPERT_MAX_JOINT_SPEED_RAD_S}, table z {TABLE_Z_M}, "
            f"cube {CUBE_SIZE_M} m, resolution {CAMERA_RESOLUTION}"
            + (f"; LAB placeholders: {', '.join(placeholders())}" if placeholders() else ""))
