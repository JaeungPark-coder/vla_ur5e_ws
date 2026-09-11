"""Renders the arm at a known pose so the gripper can be judged by eye.

Three rounds of frame algebra have now disagreed with each other about where
the gripper is and which way it points, and the numeric readouts are the
reason: the USD stage and the physics view BOTH report every gripper link at
the wrist origin, which cannot be true of a real parallel gripper and means
neither readout describes the fingers. A rendered image is not ambiguous.

The arm is driven with the transform the tracking measurement actually
converged with (command the requested orientation as tool0's, back off along
tool0 +Z by the TCP), to a height where the whole gripper is in the base
camera's view, then down to the grasp.

    python3 look_at_gripper.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

# Straight down: an orientation whose +Z is world -Z.
DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
TCP_OFFSET_M = 0.12  # only shifts the commanded height; the picture is the point


def drive(scene, tool0_pos, ticks=200, gripper=0.0):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)
    for _ in range(3):
        scene.world.render()


def save(frame, path):
    from PIL import Image

    arr = np.asarray(frame)
    if arr.size == 0:
        print(f"  {path}: EMPTY frame", flush=True)
        return
    Image.fromarray(arr[..., :3].astype(np.uint8)).save(path)
    print(f"  wrote {path}", flush=True)


def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()
    print(f"cube at {np.round(scene.cube_position, 3)}", flush=True)

    cube = scene.cube_position
    shots = [
        ("high", [cube[0], cube[1], 0.40], 0.0),
        ("approach", [cube[0], cube[1], CUBE_Z + TCP_OFFSET_M + 0.10], 0.0),
        ("at_grasp_open", [cube[0], cube[1], CUBE_Z + TCP_OFFSET_M], 0.0),
        ("at_grasp_closed", [cube[0], cube[1], CUBE_Z + TCP_OFFSET_M], 1.0),
    ]
    for name, tool0_pos, grip in shots:
        drive(scene, tool0_pos, ticks=220, gripper=grip)
        obs = scene.get_observation()
        q = np.asarray(scene.robot.get_joint_positions())
        ee_pos, _ = scene.rmpflow.get_end_effector_pose(q[:6])
        print(f"[{name}] commanded tool0 {np.round(tool0_pos, 3)} -> reached "
              f"{np.round(np.asarray(ee_pos), 3)}, cube now at "
              f"{np.round(scene.get_cube_position(), 3)}, finger_joint={q[6]:.3f}", flush=True)
        save(obs["base_rgb"], os.path.join(out_dir, f"look_{name}_base.png"))
        save(obs["wrist_rgb"], os.path.join(out_dir, f"look_{name}_wrist.png"))


def main():
    try:
        run("camera_check")
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
