"""Renders the 'closed' phase at a few TCP distances so the pad-vs-cube
relationship can be judged by eye instead of guessed at further -- the
0.08-0.16m numeric sweep found literally zero contact at every single
distance, which means the straight-down-tool0-Z distance model itself may be
wrong (not just mis-calibrated), and a picture will show that immediately."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCES = [0.10, 0.14, 0.20, 0.28]
SEED = 7


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def main():
    from PIL import Image
    try:
        scene = PickPlaceScene(with_gripper=True)
        out_dir = "camera_check"
        os.makedirs(out_dir, exist_ok=True)
        for d in DISTANCES:
            scene._rng = np.random.default_rng(SEED)
            scene.reset()
            cube0 = scene.cube_position.copy()
            above = [cube0[0], cube0[1], 0.25 + d]
            at = [cube0[0], cube0[1], CUBE_Z + d]
            drive(scene, above, 220, 0.0)
            drive(scene, at, 220, 0.0)
            drive(scene, at, 90, 1.0)
            for _ in range(2):
                scene.world.render()
            obs = scene.get_observation()
            arr = np.asarray(obs["base_rgb"])
            path = os.path.join(out_dir, f"render_d{d:.2f}.png")
            if arr.size:
                Image.fromarray(arr[..., :3].astype(np.uint8)).save(path)
            cube_now = scene.get_cube_position()
            print(f"d={d:.2f}m: cube {np.round(cube0,3)} -> {np.round(cube_now,3)}, wrote {path}",
                  flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
