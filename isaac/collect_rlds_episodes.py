"""Collects demonstration episodes for OpenVLA fine-tuning -- same
multi-object scene as the hybrid pipeline (pick_place_scene.spawn_random_objects,
Phase 6), but logging in the convention OpenVLA/Open-X-Embodiment datasets
actually use: a SINGLE camera image (not base+wrist like the π0 side) and
**Cartesian end-effector pose deltas** for actions (not joint positions --
see README.md's Phase 7 section for why these two VLAs need genuinely
different action-space data, and why this can't just reuse
collect_demos.py's dataset).

Writes one `.npy` file per episode (a list of per-step dicts) into
raw_episodes/, in the flat format `openvla_integration/ur5e_pick_place_dataset_builder.py`'s
_generate_examples expects to read -- building the actual TFDS/RLDS dataset
is a separate step (`cd openvla_integration && tfds build`), kept out of
this script so the Isaac-Sim-side collection stays simple.

NOT a ROS2 node -- run via Isaac Sim's own python.sh:
    <isaac-sim-install-dir>/python.sh collect_rlds_episodes.py --num_episodes 5

Written and reasoned about WITHOUT the ability to run Isaac Sim in the
environment this was authored in -- treat as a solid first draft. This is
the LESS-verified half of Phase 7 (see README.md) -- the RLDS/TFDS side
(ur5e_pick_place_dataset_builder.py) is where problems are more likely to
show up than in this collection script itself, since this script mostly
reuses already-exercised pick_place_scene/scripted_pick_place machinery.
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_RLDS_COLLECT_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "openvla_integration", "raw_episodes")


def _pose_vec(position, quat_xyzw):
    """(3,) position + (3,) rotation-vector -- a compact 6-dof pose
    representation, consistent between state and the action-delta
    computation below."""
    rotvec = Rot.from_quat(quat_xyzw).as_rotvec()
    return np.concatenate([position, rotvec])


def collect_episode(scene, target_description, target_position):
    obs = scene.get_observation()
    policy = ScriptedPickPlace(obs["tool_pos"], target_position, PLACE_TARGET_POSITION)

    steps = []
    prev_pose = _pose_vec(obs["tool_pos"], obs["tool_quat"])
    prev_gripper = float(obs["gripper"][0])

    for target_pos, target_rotvec, target_gripper in policy.generate_frames():
        image = np.ascontiguousarray(np.asarray(scene.get_observation()["base_rgb"])[..., :3])
        state = np.concatenate([prev_pose, [prev_gripper]]).astype(np.float32)  # (7,): xyz+rotvec+gripper

        scene.step_towards(target_pos, target_rotvec, target_gripper)

        next_obs = scene.get_observation()
        next_pose = _pose_vec(next_obs["tool_pos"], next_obs["tool_quat"])
        next_gripper = float(next_obs["gripper"][0])

        # Cartesian EE pose DELTA + absolute gripper -- the OpenVLA/Open-X
        # action convention, distinct from the π0 side's joint-space logging.
        action = np.concatenate([next_pose - prev_pose, [next_gripper]]).astype(np.float32)

        steps.append({
            "image": image,
            "state": state,
            "action": action,
            "language_instruction": f"pick up the {target_description} and place it in the target zone",
        })

        prev_pose, prev_gripper = next_pose, next_gripper

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--n_objects", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene = PickPlaceScene()
    rng = np.random.default_rng()

    try:
        for ep in range(args.num_episodes):
            objects = scene.spawn_random_objects(args.n_objects)
            target_description = rng.choice(list(objects.keys()))
            target_position = objects[target_description]["position"]

            steps = collect_episode(scene, target_description, target_position)

            out_path = os.path.join(OUTPUT_DIR, f"episode_{ep:05d}.npy")
            np.save(out_path, steps, allow_pickle=True)
            print(f"episode {ep + 1}/{args.num_episodes}: {len(steps)} steps, "
                  f"target={target_description!r} -> {out_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
