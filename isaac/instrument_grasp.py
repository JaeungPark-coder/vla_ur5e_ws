"""Where exactly does the cube get bumped? Logs its position after every
phase of one grasp attempt (now that the gripper has real colliders), plus a
rendered frame at each phase, instead of just the before/after summary
find_grasp_frame.py's sweep gives."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])  # tool0 +Z straight down
DISTANCE = 0.13


def drive(scene, tool0_pos, ticks, gripper, label, out_dir):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)
    cube = scene.get_cube_position()
    q = np.asarray(scene.robot.get_joint_positions())[:6]
    ee_pos, _ = scene.rmpflow.get_end_effector_pose(q)
    print(f"[{label}] tool0 target {np.round(tool0_pos, 3)} -> reached {np.round(np.asarray(ee_pos), 3)}, "
          f"cube now at {np.round(cube, 4)}", flush=True)
    for _ in range(2):
        scene.world.render()
    obs = scene.get_observation()
    from PIL import Image
    arr = np.asarray(obs["base_rgb"])
    if arr.size:
        Image.fromarray(arr[..., :3].astype(np.uint8)).save(os.path.join(out_dir, f"instr_{label}.png"))
    return cube


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene._rng = np.random.default_rng(7)
        scene.reset()
        cube0 = scene.cube_position.copy()
        print(f"cube spawned at {np.round(cube0, 4)}", flush=True)
        out_dir = "camera_check"
        os.makedirs(out_dir, exist_ok=True)

        cube = cube0
        above = [cube0[0], cube0[1], 0.25 + DISTANCE]
        at = [cube0[0], cube0[1], CUBE_Z + DISTANCE]
        lift = [cube0[0], cube0[1], 0.25 + DISTANCE]

        cube = drive(scene, above, 220, 0.0, "1_above", out_dir)
        cube = drive(scene, at, 220, 0.0, "2_at_open", out_dir)
        cube = drive(scene, at, 90, 1.0, "3_closed", out_dir)
        cube = drive(scene, lift, 220, 1.0, "4_lift", out_dir)

        print(f"\nrise from spawn: {(cube[2] - cube0[2]) * 1000:.1f} mm, "
              f"xy drift: {np.linalg.norm(cube[:2] - cube0[:2]) * 1000:.1f} mm", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
