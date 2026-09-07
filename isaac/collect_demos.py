"""Runs N randomized episodes of the scripted pick-and-place policy inside
Isaac Sim and writes them straight into a LeRobot dataset -- the training
data for the openpi UR5e LoRA fine-tune (see ../openpi_integration/).

NOT a ROS2 node -- run only via `<isaac-sim-install-dir>/python.sh` (needs
Isaac Sim's own Kit runtime), same as potato_drill_ws's isaac_scene.py:

    <isaac-sim-install-dir>/python.sh collect_demos.py --num_episodes 50 --repo_id your_hf_username/ur5e_pick_place

Dataset schema (image/wrist_image/state/actions, add_frame + save_episode)
matches openpi's own examples/libero/convert_libero_data_to_lerobot.py, and
the state/action layout (joints[6] + gripper[1], both 7-dim) matches
openpi's examples/ur5/README.md UR5Inputs/UR5Outputs contract exactly --
this is what lets the ur5e_pick_place_policy.py transforms in
../openpi_integration/ consume this dataset with no reshaping.

Written and reasoned about WITHOUT the ability to run Isaac Sim (or
LeRobot's dataset writer) in the environment this was authored in -- treat
as a solid first draft, not verified to run. Smoke-test with a small
--num_episodes (e.g. 3-5) before committing to a full collection run.
"""
import argparse
import os
import shutil

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

PROMPT = "pick up the cube and place it in the target zone"
CONTROL_FPS = 30  # matches the LeRobot dataset's `fps` metadata -- keep in sync with steps_per_segment choices


def _to_rgb_uint8(rgba_or_rgb):
    """Replicator's "rgb" annotator returns HxWx4 (RGBA) uint8 -- LeRobot's
    "image" dtype expects HxWx3."""
    arr = np.asarray(rgba_or_rgb)
    return arr[..., :3] if arr.shape[-1] == 4 else arr


def collect(num_episodes: int, repo_id: str, push_to_hub: bool):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    image_shape = (256, 256, 3)  # keep in sync with pick_place_scene.CAMERA_RESOLUTION
    # Field names here are chosen to match openpi's examples/ur5/README.md
    # LeRobotUR5DataConfig repack_transform verbatim (image/wrist_image/
    # joints/gripper on the dataset side, remapped to base_rgb/wrist_rgb/
    # joints/gripper for UR5eInputs) -- see ../openpi_integration/
    # ur5e_pick_place_policy.py. Deliberately joints(6)+gripper(1) as
    # separate fields rather than one combined "state" vector, so no
    # reshaping is needed on the openpi side.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=CONTROL_FPS,
        features={
            "image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
            "joints": {"dtype": "float32", "shape": (6,), "names": ["joints"]},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},  # next joints[6] + gripper[1]
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    scene = PickPlaceScene()

    for ep in range(num_episodes):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)

        n_logged = 0
        for target_pos, target_rotvec, target_gripper in policy.generate_frames():
            obs = scene.get_observation()
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            next_obs = scene.get_observation()

            # Log the ACHIEVED next joint state as the action, not the
            # Cartesian waypoint -- matches openpi's UR5 contract (state and
            # action share the same joints+gripper space), and sidesteps
            # needing to introspect RMPflow's ArticulationAction internals.
            action = np.concatenate([next_obs["joints"], next_obs["gripper"]]).astype(np.float32)

            dataset.add_frame({
                "image": _to_rgb_uint8(obs["base_rgb"]),
                "wrist_image": _to_rgb_uint8(obs["wrist_rgb"]),
                "joints": obs["joints"].astype(np.float32),
                "gripper": obs["gripper"].astype(np.float32),
                "actions": action,
                "task": PROMPT,
            })
            n_logged += 1

        dataset.save_episode()
        print(f"episode {ep + 1}/{num_episodes}: logged {n_logged} frames "
              f"(cube spawned at {np.round(scene.cube_position, 3)})")

    if push_to_hub:
        dataset.push_to_hub(tags=["ur5e", "pick_place", "vla_ur5e_ws"], private=False, push_videos=True,
                             license="apache-2.0")

    print(f"done -- dataset written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--repo_id", type=str, default="your_hf_username/ur5e_pick_place")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    try:
        collect(args.num_episodes, args.repo_id, args.push_to_hub)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
