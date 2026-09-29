"""2026-09-28's 50-episode batch had 2/50 attempts where the cube launched
far above its normal peak (max_cube_z 0.554m and 0.562m vs the usual
~0.16m), placed hundreds of mm off target. The cube already has
maxDepenetrationVelocity=0.5 capped (isaac_sim_common.add_shape, since
2026-09-14, specifically for this failure class) and the gripper pads have
the same cap -- so this is NOT the same mechanism that cap was built for,
or the cap is insufficient against whatever changed today (diffIK for
grasp/place approach, closed-loop correction). Reproduces the two known-bad
spawn positions exactly (bypassing RNG) with full per-tick logging (segment,
cube pos/vel, gripper closure, whether diffIK is driving) to catch the
actual moment/mechanism live instead of guessing.

    python3 check_contact_explosion.py --case 1   # cube spawned [0.428, -0.018] in the original run
    python3 check_contact_explosion.py --case 2   # cube spawned [0.479, -0.019]
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION, CUBE_Z  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

SEGMENT_NAMES = ["approach", "settle", "descend", "settle2", "close", "lift",
                  "transport", "descend2", "settle3", "release", "retract"]

CASES = {
    1: np.array([0.428, -0.018, CUBE_Z]),
    2: np.array([0.479, -0.019, CUBE_Z]),
}


def say(line=""):
    print(line, flush=True)


def run(cube_xy):
    scene = PickPlaceScene(with_gripper=True)

    class _FixedRNG:
        def __init__(self, x, y):
            self._vals = [x, y]
            self._i = 0

        def uniform(self, *_a, **_k):
            v = self._vals[self._i % 2]
            self._i += 1
            return v

    scene._rng = _FixedRNG(cube_xy[0], cube_xy[1])
    obs = scene.reset()
    say(f"cube forced to {np.round(scene.cube_position, 4).tolist()} (target {cube_xy[:2].tolist()})")

    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    diffik_start, diffik_end = policy.diffik_frame_range()
    grasp_diffik_start, grasp_diffik_end = policy.grasp_diffik_frame_range()
    segment_bounds = np.cumsum([n for _, _, n in policy.waypoints])
    assert len(segment_bounds) == len(SEGMENT_NAMES)

    seg_idx = 0
    max_cube_z = -np.inf
    prev_cube_z = float(scene.get_cube_position()[2])
    for tick, (target_pos, target_rotvec, target_gripper) in enumerate(policy.generate_frames(), start=1):
        use_diffik = (grasp_diffik_start <= tick <= grasp_diffik_end) or (diffik_start <= tick <= diffik_end)
        if use_diffik:
            scene.step_towards_diffik(target_pos, target_rotvec, target_gripper)
        else:
            scene.step_towards(target_pos, target_rotvec, target_gripper)
        while seg_idx < len(segment_bounds) - 1 and tick > segment_bounds[seg_idx]:
            seg_idx += 1
        seg_name = SEGMENT_NAMES[seg_idx]

        cube_pos = scene.get_cube_position()
        cube_vel = scene.cube_rigid_prim.get_linear_velocity()
        cube_z = float(cube_pos[2])
        dz_this_tick = cube_z - prev_cube_z
        max_cube_z = max(max_cube_z, cube_z)

        # Only print when something interesting is happening: a big
        # per-tick jump, or periodically for context.
        if abs(dz_this_tick) > 0.005 or tick % 60 == 0:
            say(f"{tick:>5} {seg_name:>10} diffik={use_diffik!s:>5} "
                f"cube_z={cube_z*1000:8.1f}mm dz_tick={dz_this_tick*1000:+7.2f}mm "
                f"cube_vz={float(cube_vel[2]):+8.3f}m/s grip_act={scene.gripper.get_normalized_position():.3f}")
        prev_cube_z = cube_z

    say(f"\nfinal: max_cube_z={max_cube_z:.4f}m place_error={scene.place_error_m()*1000:.1f}mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    try:
        run(CASES[args.case])
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
