"""Finds the gripper's grasp frame by grasping, not by algebra.

Four rounds of frame reasoning have now produced four different answers for
where the fingertips are relative to the frame RMPflow drives, and the
readouts they were based on cannot be trusted: the USD stage and the physics
view BOTH report base_link and both finger pads at exactly the wrist origin,
which is impossible for a real parallel gripper, so neither describes the
fingers. Rendered images show the gripper is attached and finger_joint does
drive (0.000 -> 0.680 on command), yet closing it where the fingertips are
believed to be leaves the cube completely unmoved.

So stop asking where the fingers are and ask what actually grasps. Each
candidate is a (approach axis in the tool0 frame, distance) pair. For each,
the arm is driven so that THAT candidate's idea of the fingertips lands on
the cube, the gripper closes, and the arm lifts. If the cube comes up with
it, the candidate is right -- there is no way to lift a cube with the wrong
grasp frame. The answer is whatever survives, and it is a physical fact
rather than an inference.

    python3 find_grasp_frame.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

# Approach axes to try, expressed in the tool0 frame. tool0 +Z is the tool
# axis the URDF puts out of the wrist; tool0 +Y is the flange's +Z, which an
# earlier measurement claimed the fingers ran along. The rest are here so a
# wrong assumption cannot survive by not being tested.
AXIS_CANDIDATES = {
    "+Z": [0.0, 0.0, 1.0],
    "+Y": [0.0, 1.0, 0.0],
    "-Y": [0.0, -1.0, 0.0],
    "+X": [1.0, 0.0, 0.0],
    "-X": [-1.0, 0.0, 0.0],
    "-Z": [0.0, 0.0, -1.0],
}
DISTANCE_CANDIDATES = [0.10, 0.13, 0.16]

SAFE_HEIGHT = 0.25   # above the cube, for the approach
LIFT_HEIGHT = 0.25   # how far above the table to lift after closing
CUBE_SEED = 7        # same cube spawn for every candidate


def tool0_quat_for(axis_in_tool0, world_direction=(0.0, 0.0, -1.0)):
    """Orientation for tool0 that points `axis_in_tool0` along
    `world_direction` (straight down by default). The rotation about that
    axis is unconstrained -- pick any, the cube is symmetric."""
    a = np.asarray(axis_in_tool0, dtype=float)
    a /= np.linalg.norm(a)
    d = np.asarray(world_direction, dtype=float)
    d /= np.linalg.norm(d)
    # Shortest rotation taking a to d.
    v = np.cross(a, d)
    c = float(np.dot(a, d))
    if np.linalg.norm(v) < 1e-8:
        r = Rot.identity() if c > 0 else Rot.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    else:
        r = Rot.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1.0, 1.0)))
    q = r.as_quat()  # xyzw
    return q[[3, 0, 1, 2]]


def drive(scene, tool0_pos, quat_wxyz, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), quat_wxyz)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def reached(scene, tool0_pos):
    q = np.asarray(scene.robot.get_joint_positions())[:6]
    ee_pos, _ = scene.rmpflow.get_end_effector_pose(q)
    return float(np.linalg.norm(np.asarray(ee_pos) - np.asarray(tool0_pos)))


# The cube's position readout was checked separately, by check_pose_readout.py:
# a cube dropped from 0.30 m reads 0.2973 -> 0.0583 -> 0.0200 through
# UsdGeom.ComputeLocalToWorldTransform, so isaac_sim_common.prim_world_pose
# does follow physics and "did the cube rise?" below is a real measurement.
#
# An earlier version of this file checked that inline and reported the
# readout frozen -- wrongly. It removed the cube, re-added it at 0.30 m and
# called world.reset(), which restores every registered object to the default
# state it was registered with, putting the cube back on the table before the
# read. The check was measuring its own reset, not the readout.


def try_candidate(scene, axis_name, axis, distance):
    scene._rng = np.random.default_rng(CUBE_SEED)
    scene.reset()
    cube = scene.cube_position.copy()
    quat = tool0_quat_for(axis)

    # tool0 sits `distance` back along the approach from where the fingertips
    # are meant to be; the approach points straight down, so that is simply
    # `distance` higher.
    above = np.array([cube[0], cube[1], SAFE_HEIGHT + distance])
    at = np.array([cube[0], cube[1], CUBE_Z + distance])
    lift = np.array([cube[0], cube[1], LIFT_HEIGHT + distance])

    drive(scene, above, quat, 200, 0.0)
    approach_err = reached(scene, above)
    drive(scene, at, quat, 200, 0.0)
    grasp_err = reached(scene, at)
    drive(scene, at, quat, 90, 1.0)     # close on the cube
    drive(scene, lift, quat, 220, 1.0)  # lift it

    cube_now = scene.get_cube_position()
    rise = float(cube_now[2] - CUBE_Z)
    return {
        "rise_mm": rise * 1000.0,
        "approach_err_mm": approach_err * 1000.0,
        "grasp_err_mm": grasp_err * 1000.0,
        "cube_xy_drift_mm": float(np.linalg.norm(cube_now[:2] - cube[:2])) * 1000.0,
    }


def run():
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()

    print(f"\nsweeping {len(AXIS_CANDIDATES)} approach axes x {len(DISTANCE_CANDIDATES)} "
          f"distances; a candidate is correct if the cube rises with the gripper", flush=True)
    results = []
    for axis_name, axis in AXIS_CANDIDATES.items():
        for distance in DISTANCE_CANDIDATES:
            r = try_candidate(scene, axis_name, axis, distance)
            results.append((r["rise_mm"], axis_name, distance, r))
            print(f"  tool0 {axis_name} @ {distance:.2f}m: cube rose {r['rise_mm']:7.1f} mm "
                  f"(drifted {r['cube_xy_drift_mm']:5.1f} mm), tool0 tracking err "
                  f"{r['grasp_err_mm']:5.1f} mm at the grasp", flush=True)

    results.sort(key=lambda x: -x[0])
    best_rise, best_axis, best_distance, best = results[0]
    print("\n--- verdict ---", flush=True)
    if best_rise < 50.0:
        print(f"  NOTHING GRASPS. The best candidate (tool0 {best_axis} @ {best_distance}m) "
              f"lifted the cube only {best_rise:.1f} mm.", flush=True)
        print("  The grasp frame is not the problem, or not the only one -- suspect the "
              "gripper's contact physics (open/closed joint limits, finger collision, cube "
              "mass/friction) rather than the waypoint transform.", flush=True)
        return
    print(f"  tool0 {best_axis} @ {best_distance:.2f} m lifted the cube {best_rise:.1f} mm.",
          flush=True)
    axis = np.array(AXIS_CANDIDATES[best_axis], dtype=float)
    print("\n  Set in pick_place_scene.py:", flush=True)
    print(f"    GRIPPER_APPROACH_AXIS_IN_TOOL0 = {axis.tolist()}", flush=True)
    print(f"    GRIPPER_TCP_OFFSET_M = {best_distance}", flush=True)


def main():
    try:
        run()
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
