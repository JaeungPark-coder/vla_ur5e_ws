"""Instead of position-then-snap-closed (shown highly sensitive to exact
XY alignment, and even non-deterministic on a near-miss -- three identical
trials of the same offset gave 15.9/134.6/262.3mm of drift), close GRADUALLY
while already at the approach height, the way a real compliant grasp
works: the V-shaped pads naturally funnel/center a nearby object as they
close, rather than needing to already be perfectly centered before closing.
Tries this at a few XY offsets around the last measured centroid, since it
should be much more tolerant of being slightly off."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
Z_OFFSET = -0.006  # from the measured centroid: fingers sit ~6mm below tool0 at this height
CANDIDATES = {
    "xy(34,89)": np.array([0.034, 0.089]),
    "xy(20,70)": np.array([0.020, 0.070]),
    "xy(45,100)": np.array([0.045, 0.100]),
    "xy(25,90)": np.array([0.025, 0.090]),
    "xy(40,70)": np.array([0.040, 0.070]),
}


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def try_compliant(scene, xy_offset):
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()
    tool0_xy = cube0[:2] - xy_offset
    approach_z = cube0[2] - Z_OFFSET + 0.10  # well above, same lateral offset
    at_z = cube0[2] - Z_OFFSET               # hover right at the measured finger height

    above = np.array([tool0_xy[0], tool0_xy[1], approach_z])
    at = np.array([tool0_xy[0], tool0_xy[1], at_z])
    drive(scene, above, 200, 0.0)
    drive(scene, at, 150, 0.0)

    # Gradually close over many ticks WHILE holding position -- gives the
    # pads time to make and adjust contact rather than snapping shut.
    for frac in np.linspace(0.0, 1.0, 60):
        scene.rmpflow.set_end_effector_target(at, DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(float(frac))
        scene.world.step(render=False)
    for _ in range(60):
        scene.rmpflow.set_end_effector_target(at, DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(1.0)
        scene.world.step(render=False)

    lift = above.copy()
    drive(scene, lift, 200, 1.0)
    cube_now = scene.get_cube_position()
    rise = float(cube_now[2] - cube0[2])
    drift = float(np.linalg.norm(cube_now[:2] - cube0[:2]))
    return rise, drift


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        results = []
        for name, xy in CANDIDATES.items():
            rise, drift = try_compliant(scene, xy)
            results.append((rise, name, drift))
            print(f"  {name:15s} -> rise={rise*1000:7.1f}mm drift={drift*1000:6.1f}mm", flush=True)
        results.sort(key=lambda r: -r[0])
        best_rise, best_name, best_drift = results[0]
        print("\n--- verdict ---", flush=True)
        if best_rise * 1000 > 50:
            print(f"SUCCESS with compliant close: {best_name}, rise={best_rise*1000:.1f}mm", flush=True)
        else:
            print(f"Best: {best_name}, rise={best_rise*1000:.1f}mm, drift={best_drift*1000:.1f}mm -- "
                  "still no clean lift even with gradual close.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
