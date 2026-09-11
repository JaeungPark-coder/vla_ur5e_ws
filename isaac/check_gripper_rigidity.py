"""Is the gripper actually bolted to the wrist?

Every grasp failure in this project so far has been blamed on a coordinate
convention, and each convention fix left the reach worse than before. This
script tests the assumption underneath all of them: that the gripper is a
rigid body on the end of the arm, so that "where the fingertips are" is a
FIXED offset from the frame RMPflow drives.

If it is rigid, the fingertip pose expressed in the tool0 frame is identical
at every arm configuration, and the TCP printed at the end is a constant this
project can finally rely on. If it drifts, no TCP constant can be correct and
no waypoint transform can fix the grasp -- the attachment has to be fixed
first.

The arm is driven the way the real pipeline drives it (RMPflow, smoothly,
one control tick at a time), NOT by teleporting joint positions. That
distinction matters: a first version of this measurement teleported with
set_joint_positions and read three ticks later, which makes even a
correctly-attached gripper look detached because the solver has not caught
up with the jump yet.

    python3 check_gripper_rigidity.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from isaac_sim_common import ROBOT_PRIM_PATH, prim_world_pose  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402

# Deliberately NOT imported from measure_frames: that module calls
# SimulationApp() at import time, so importing it from here starts Kit a
# second time inside an already-running Kit and takes the process down with a
# breakpad crash dump 118 ms in, before a single line of this script runs.
_REPORT = {"path": None}


def say(line=""):
    print(line, flush=True)
    if _REPORT["path"]:
        with open(_REPORT["path"], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def as_rotation(rot):
    """RMPflow returns its end-effector rotation as a 3x3 matrix on some
    versions and a wxyz quaternion on others -- accept both."""
    rot = np.asarray(rot)
    if rot.shape == (3, 3):
        return Rot.from_matrix(rot)
    if rot.shape == (4,):
        return Rot.from_quat(rot[[1, 2, 3, 0]])  # wxyz -> xyzw
    raise RuntimeError(f"unexpected end-effector rotation layout {rot.shape}: {rot}")

GRIPPER_ROOT = f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85"
# Links whose pose in the tool0 frame must be constant if the gripper is one
# rigid assembly bolted to the wrist. base_link is the coupling; the inner
# fingers are the pads that actually touch the cube.
TRACKED_LINKS = ["base_link", "left_inner_finger", "right_inner_finger"]

# Poses to drive through, as (tool0 position, tool0 +Z direction). Spread
# across the workspace so a gripper that merely looks attached at one
# configuration is caught.
PROBE_POSES = [
    ([0.45, 0.00, 0.35], [0.0, 0.0, -1.0]),
    ([0.35, 0.25, 0.45], [0.0, 0.0, -1.0]),
    ([0.55, -0.20, 0.30], [0.0, 0.0, -1.0]),
    ([0.40, 0.00, 0.50], [0.0, 0.0, -1.0]),
]
TICKS_PER_POSE = 150


def quat_wxyz_for_approach(direction):
    """An orientation whose +Z lies along `direction` -- the roll about it is
    arbitrary and irrelevant to a symmetric cube."""
    z = np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, z))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = Rot.from_matrix(np.column_stack((x, y, z))).as_quat()  # xyzw
    return q[[3, 0, 1, 2]]


_RIGID_READERS = {}


def physics_link_pose(path):
    """A link's world pose read from PHYSICS, not from the USD stage.

    The USD stage is the wrong place to ask. Reading these same gripper links
    with UsdGeom.ComputeLocalToWorldTransform returns the gripper root's pose
    for every one of them -- base_link and both fingers identical, at tool0's
    origin, at every arm configuration -- because physics writes link
    transforms to Fabric and those USD prims keep their authored values. The
    arm's own links happen to read correctly that way, which is what made the
    stale readings so easy to trust: the flange agrees with RMPflow's forward
    kinematics to 0.000 mm, so the same call on a finger looks equally sound
    and is silently meaningless.
    """
    from isaacsim.core.prims import SingleRigidPrim

    if path not in _RIGID_READERS:
        reader = SingleRigidPrim(path)
        try:
            reader.initialize()
        except Exception:
            pass  # already initialized by the World
        _RIGID_READERS[path] = reader
    pos, quat_wxyz = _RIGID_READERS[path].get_world_pose()
    pos = np.asarray(pos, dtype=float)
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return pos, Rot.from_quat(quat_wxyz[[1, 2, 3, 0]])


def link_poses_in_tool0(scene, compare_usd=False):
    """Every tracked gripper link's pose, expressed in the tool0 frame."""
    q = np.asarray(scene.robot.get_joint_positions())[:6]
    ee_pos, ee_rot = scene.rmpflow.get_end_effector_pose(q)
    r_ee = as_rotation(ee_rot)
    ee_pos = np.asarray(ee_pos, dtype=float)

    out = {}
    for name in TRACKED_LINKS:
        path = f"{GRIPPER_ROOT}/{name}"
        if not scene.stage.GetPrimAtPath(path).IsValid():
            continue
        try:
            pos, rot = physics_link_pose(path)
        except Exception as exc:
            say(f"    (physics read failed for {name}: {type(exc).__name__}: {exc})")
            continue
        out[name] = (r_ee.inv().apply(pos - ee_pos), r_ee.inv() * rot)
        if compare_usd:
            usd_pos, usd_quat = prim_world_pose(scene.stage.GetPrimAtPath(path))
            say(f"    {name:20s} physics {np.round(out[name][0] * 1000, 1)} mm   "
                f"usd {np.round(r_ee.inv().apply(usd_pos - ee_pos) * 1000, 1)} mm")
    return out


def drive(scene, position, approach, ticks):
    quat = quat_wxyz_for_approach(approach)
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(position, dtype=float), quat)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(0.0)  # held open throughout
        scene.world.step(render=False)


def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _REPORT["path"] = os.path.join(out_dir, "gripper_rigidity.txt")
    open(_REPORT["path"], "w").close()
    say(f"writing this report to {_REPORT['path']} as well as stdout")

    scene = PickPlaceScene(with_gripper=True)
    say(f"robot articulation joints: {list(scene.robot.dof_names)}")
    scene.reset()

    root = scene.stage.GetPrimAtPath(GRIPPER_ROOT)
    if not root.IsValid():
        say(f"FAILED: no gripper at {GRIPPER_ROOT}")
        return
    say(f"\ntracking {TRACKED_LINKS} under {GRIPPER_ROOT}")
    say("(pose in the tool0 frame -- a rigidly attached gripper reads the same at every pose)")

    samples = []
    for i, (position, approach) in enumerate(PROBE_POSES):
        drive(scene, position, approach, TICKS_PER_POSE)
        if i == 0:
            say("\n  physics vs USD readings of the same links (they should agree; "
                "where they do not, USD is stale):")
            link_poses_in_tool0(scene, compare_usd=True)
        poses = link_poses_in_tool0(scene)
        samples.append(poses)
        q = np.asarray(scene.robot.get_joint_positions())[:6]
        say(f"\n  [pose {i}] commanded tool0 {np.round(position, 3)}, arm q={np.round(q, 3)}")
        for name, (pos, rot) in poses.items():
            say(f"    {name:20s} pos {np.round(pos * 1000, 1)} mm   "
                f"rot {np.round(rot.as_euler('xyz', degrees=True), 1)} deg")

    say("\n--- verdict ---")
    rigid = True
    for name in TRACKED_LINKS:
        if not all(name in s for s in samples):
            continue
        positions = np.array([s[name][0] for s in samples])
        angles = [float(np.degrees((samples[0][name][1].inv() * s[name][1]).magnitude()))
                  for s in samples]
        pos_spread = float(np.abs(positions - positions[0]).max()) * 1000.0
        rot_spread = max(angles)
        say(f"  {name:20s} position varies {pos_spread:7.2f} mm, orientation {rot_spread:6.2f} deg")
        if pos_spread > 2.0 or rot_spread > 2.0:
            rigid = False

    if not rigid:
        say("\n  NOT RIGID. The gripper is not held on the wrist, so no TCP constant and no")
        say("  waypoint transform can make the scripted expert grasp anything. Fix the")
        say("  attachment before touching the waypoints or the cameras again.")
        return

    pads = [s for s in samples if "left_inner_finger" in s and "right_inner_finger" in s]
    tcp = np.mean([0.5 * (s["left_inner_finger"][0] + s["right_inner_finger"][0]) for s in pads],
                  axis=0)
    opening = np.mean([s["right_inner_finger"][0] - s["left_inner_finger"][0] for s in pads], axis=0)
    dist = float(np.linalg.norm(tcp))
    say(f"\n  RIGID. The grasp point sits at {np.round(tcp * 1000, 1)} mm in the tool0 frame "
        f"({dist * 1000:.1f} mm out).")
    say(f"  approach direction in tool0: {np.round(tcp / dist, 4)}")
    say(f"  opening axis in tool0:       {np.round(opening / np.linalg.norm(opening), 4)}")
    say(f"\n  set GRIPPER_TCP_IN_TOOL0 = {np.round(tcp, 4).tolist()}")
    say(f"      GRIPPER_OPENING_AXIS_IN_TOOL0 = "
        f"{np.round(opening / np.linalg.norm(opening), 4).tolist()}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="camera_check")
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
