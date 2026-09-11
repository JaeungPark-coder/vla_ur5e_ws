"""Small perturbations around the last measured near-miss offset
([34.3, 89.2, -6.1]mm, which produced a 15.9mm nudge but no lift -- real
contact, not yet a clean pinch). Tries several nearby candidates directly
with a grasp-lift test rather than another expensive full grid search."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
BASE = np.array([34.3, 89.2, -6.1]) / 1000.0
CANDIDATES = {
    "base": BASE,
    "+x10": BASE + [0.010, 0, 0],
    "-x10": BASE + [-0.010, 0, 0],
    "+y10": BASE + [0, 0.010, 0],
    "-y10": BASE + [0, -0.010, 0],
    "+z10 (deeper)": BASE + [0, 0, -0.010],
    "-z10 (shallower)": BASE + [0, 0, 0.010],
    "-x15-y15": BASE + [-0.015, -0.015, 0],
    "-x20": BASE + [-0.020, 0, 0],
}


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def try_offset(scene, offset):
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()
    tool0_command = cube0 - offset
    above = tool0_command.copy()
    above[2] += 0.15
    drive(scene, above, 200, 0.0)
    drive(scene, tool0_command, 200, 0.0)
    drive(scene, tool0_command, 90, 1.0)
    lift = tool0_command.copy()
    lift[2] += 0.15
    drive(scene, lift, 200, 1.0)
    cube_now = scene.get_cube_position()
    rise = float(cube_now[2] - cube0[2])
    drift = float(np.linalg.norm(cube_now[:2] - cube0[:2]))
    return rise, drift


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()
        results = []
        for name, offset in CANDIDATES.items():
            rise, drift = try_offset(scene, offset)
            results.append((rise, name, offset, drift))
            print(f"  {name:20s} offset={np.round(offset*1000,1).tolist()}mm -> "
                  f"rise={rise*1000:7.1f}mm drift={drift*1000:6.1f}mm", flush=True)
        results.sort(key=lambda r: -r[0])
        best_rise, best_name, best_offset, best_drift = results[0]
        print("\n--- verdict ---", flush=True)
        if best_rise * 1000 > 50:
            print(f"SUCCESS: {best_name} lifted {best_rise*1000:.1f}mm. "
                  f"GRIPPER_TCP_OFFSET_WORLD = {np.round(best_offset,4).tolist()}", flush=True)
        else:
            print(f"Best so far: {best_name}, rise={best_rise*1000:.1f}mm, drift={best_drift*1000:.1f}mm "
                  "-- still no clean lift.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
