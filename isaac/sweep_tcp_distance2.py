"""TCP-distance sweep, built on the exact call pattern instrument_grasp.py
already showed causes zero spurious cube disturbance -- one scene, one seed,
one reset before the whole sweep starts (not one per candidate), each
candidate a fresh `scene.reset()` INSIDE the loop but always from the SAME
seeded RNG state each time by re-seeding right before it. Logs the cube
position after every phase for every candidate, so a spurious bump is
visible (which phase, how much) rather than folded into a single before/after
number.
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])  # matches scripted_pick_place.DOWNWARD_ROTVEC exactly
DISTANCES = [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
SEED = 7


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)
    return scene.get_cube_position()


def try_distance(scene, distance):
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()

    above = [cube0[0], cube0[1], 0.25 + distance]
    at = [cube0[0], cube0[1], CUBE_Z + distance]
    lift = [cube0[0], cube0[1], 0.25 + distance]

    c1 = drive(scene, above, 220, 0.0)
    c2 = drive(scene, at, 220, 0.0)
    c3 = drive(scene, at, 90, 1.0)
    c4 = drive(scene, lift, 220, 1.0)

    def d(c):
        return float(np.linalg.norm(np.asarray(c) - cube0))

    print(f"  distance={distance:.2f}m: after above={d(c1)*1000:6.1f}mm  at_open={d(c2)*1000:6.1f}mm  "
          f"closed={d(c3)*1000:6.1f}mm  lift={d(c4)*1000:6.1f}mm  "
          f"-> rise={float(c4[2]-cube0[2])*1000:7.1f}mm", flush=True)
    return float(c4[2] - cube0[2]), d(c4)


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        # Deliberately NOT calling scene.reset() here before the loop --
        # instrument_grasp.py's one-shot version (exactly one seed + one
        # reset before its single trial) showed ZERO spurious cube
        # disturbance; this sweep's first version called an extra unseeded
        # reset() here before each candidate's own seeded reset() and showed
        # an identical 78.0mm bump on every candidate regardless of distance
        # -- too suspiciously constant to be a real per-distance physical
        # effect, and disappeared entirely once this redundant reset was
        # removed. Matches the one-reset-per-trial pattern that is known to
        # work.
        print(f"sweeping TCP distance over {DISTANCES}, per-phase cube displacement from spawn "
              "shown at each column", flush=True)
        results = []
        for dist in DISTANCES:
            rise, total_disp = try_distance(scene, dist)
            results.append((rise, dist, total_disp))

        results.sort(key=lambda r: -r[0])
        best_rise, best_d, best_disp = results[0]
        print("\n--- verdict ---", flush=True)
        if best_rise * 1000 < 50.0:
            print(f"  BEST distance={best_d:.2f}m only lifted {best_rise*1000:.1f}mm.", flush=True)
        else:
            print(f"  distance={best_d:.2f}m lifted the cube {best_rise*1000:.1f}mm. "
                  f"Set GRIPPER_TCP_OFFSET_M = {best_d}", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
