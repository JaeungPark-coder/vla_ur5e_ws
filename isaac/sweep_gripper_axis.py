"""Tests whether the gripper's TRUE approach axis is tool0 +Z (assumed by
DOWNWARD_ROTVEC=[0,pi,0] all along) or actually tool0 +Y -- which is what it
would be if the gripper mesh is built relative to the FLANGE's own +Z
(CONFIRMED earlier: flange +Z == tool0 +Y, from the fixed [90,0,90] deg
xyz-Euler flange->tool0 relationship measured in measure_frames.py).

Motivation: pin_offset_safe_height.py's dense grid, taken at a VERIFIED
well-conditioned pose (feasibility_gate: condition_number=7.1), found the
closed fingers only 2-9mm below tool0's own height but 13-95mm off to the
side -- i.e. reaching almost HORIZONTALLY, not down. That is exactly what
you'd see if the gripper's real approach axis is tool0 Y, not tool0 Z.

For each candidate "which tool0 axis should point down" (accessed by
composing DOWNWARD_ROTVEC with an extra fixed rotation), drives to a SAFE
height above the cube (gated by feasibility_gate so nothing here can wind up
again), closes the gripper, and scores by whether the cube visibly rises --
same decisive test used throughout this project, just applied to the
orientation instead of the standoff distance this time.
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import CUBE_Z, ROBOT_PRIM_PATH, PickPlaceScene  # noqa: E402
from feasibility_gate import FeasibilityGate  # noqa: E402

SEED = 7
DISTANCE = 0.09  # closer -- 0.15 gave zero contact for EVERY candidate (all
# five: bit-identical 0.0mm drift), meaning 150mm is simply farther than
# this gripper's total reach in any tested direction (~90-100mm, per
# pin_offset_safe_height.py's dense grid), not informative about orientation
# at all. Use a distance close to that measured magnitude instead.
# Candidates: (label, extra rotation applied ON TOP of "tool0 Z points down",
# so as to point a DIFFERENT tool0 local axis down instead).
# Rot.from_rotvec([0,pi,0]) already points tool0 Z down; composing with a
# further 90deg rotation about tool0 X or Y re-aims which local axis ends up
# pointing down without changing "the tool is generally reaching downward".
BASE = Rot.from_rotvec([0.0, np.pi, 0.0])
CANDIDATES = {
    "Z-down (current DOWNWARD_ROTVEC, assumed all along)": Rot.identity(),
    "Y-down (flange Z convention)": Rot.from_euler("x", 90, degrees=True),
    "Y-down (opposite roll)": Rot.from_euler("x", -90, degrees=True),
    "X-down": Rot.from_euler("y", 90, degrees=True),
    "X-down (opposite roll)": Rot.from_euler("y", -90, degrees=True),
}


def drive(scene, tool0_pos, quat_wxyz, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), quat_wxyz)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def try_axis(scene, gate, label, extra_rot):
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()
    r_total = extra_rot * BASE
    quat_xyzw = r_total.as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

    above = np.array([cube0[0], cube0[1], 0.25 + DISTANCE])
    at = np.array([cube0[0], cube0[1], CUBE_Z + DISTANCE])

    gate_result = gate.check(at, quat_wxyz)
    if not gate_result["ok"]:
        return None, gate_result["reason"]

    drive(scene, above, quat_wxyz, 200, 0.0)
    drive(scene, at, quat_wxyz, 200, 0.0)
    drive(scene, at, quat_wxyz, 90, 1.0)
    lift = above.copy()
    drive(scene, lift, quat_wxyz, 200, 1.0)

    cube_now = scene.get_cube_position()
    rise = float(cube_now[2] - cube0[2])
    drift = float(np.linalg.norm(cube_now[:2] - cube0[:2]))
    return (rise, drift), gate_result["reason"]


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        # Deliberately NOT calling scene.reset() here before the loop -- the
        # same redundant-reset bug hit repeatedly elsewhere this session: an
        # extra unseeded reset() before each candidate's own seeded reset()
        # produces a constant, bogus drift unrelated to the candidate
        # (confirmed here: "Z-down" and "Y-down" both read exactly 68.7mm
        # before this fix).
        gate = FeasibilityGate(ROBOT_PRIM_PATH)
        print("feasibility gate ready\n", flush=True)

        results = []
        for label, extra_rot in CANDIDATES.items():
            outcome, gate_reason = try_axis(scene, gate, label, extra_rot)
            if outcome is None:
                print(f"  {label:45s} -- SKIPPED (gate rejected: {gate_reason})", flush=True)
                continue
            rise, drift = outcome
            print(f"  {label:45s} -> rise={rise*1000:7.1f}mm drift={drift*1000:6.1f}mm "
                  f"(gate: {gate_reason})", flush=True)
            results.append((rise, label, drift))

        if results:
            results.sort(key=lambda r: -r[0])
            best_rise, best_label, best_drift = results[0]
            print("\n--- verdict ---", flush=True)
            if best_rise * 1000 > 50:
                print(f"SUCCESS: {best_label} lifted {best_rise*1000:.1f}mm -- this is the "
                      "gripper's real approach axis.", flush=True)
            else:
                print(f"Best: {best_label}, rise={best_rise*1000:.1f}mm -- still no clean lift, "
                      "but compare drift/rise across candidates for the closest one to refine "
                      "next.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
