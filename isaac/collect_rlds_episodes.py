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


# ADJUST: standard ROS/URDF extrinsic-XYZ RPY, matching scipy's as_euler("xyz")
# / from_euler("xyz") -- the best-supported guess for OXE's actual axis
# convention (OpenVLA's public transforms.py defers this dataset-specific
# rotation handling to an undefined `relabel_bridge_actions` helper, so it
# could not be pinned down from source). Verify against a real OpenVLA
# checkout's dataloader before trusting this verbatim, same posture as every
# other ADJUST marker in this project.
EULER_SEQ = "xyz"


def _state_vec(position, quat_xyzw, gripper):
    """(8,) state matching OpenVLA's StateEncoding.POS_EULER exactly: EEF
    XYZ(3) + Roll-Pitch-Yaw(3) + PAD(1) + Gripper(1) -- verified against
    OpenVLA's own prismatic/vla/datasets/rlds/oxe/configs.py source
    (2026-09-11), not guessed. A previous version of this file logged a
    7-dim (xyz + rotvec, no PAD slot) vector -- one dimension short of what
    POS_EULER's fixed-index unpacking expects, so every field from the
    rotation onward would land one slot to the left of where OpenVLA reads
    it. See verify_action_encoding.py, which round-trips this against the
    scripted expert's own trajectory."""
    rpy = Rot.from_quat(quat_xyzw).as_euler(EULER_SEQ)
    return np.concatenate([position, rpy, [0.0], [gripper]]).astype(np.float32)


def _delta_action(prev_pos, prev_quat_xyzw, next_pos, next_quat_xyzw, next_gripper):
    """(7,) action matching OpenVLA's ActionEncoding.EEF_POS: EEF Delta
    XYZ(3) + Delta Roll-Pitch-Yaw(3) + Gripper(1).

    Delta ORIENTATION is NOT `next_rpy - prev_rpy`, and a previous version of
    this file computed the (then-rotvec) rotational delta by exactly that
    kind of plain vector subtraction -- which is only valid in the
    infinitesimal limit, and produces wildly wrong deltas whenever the
    logged rotation sits near a representation singularity. This project's
    entire scripted trajectory holds the gripper at
    scripted_pick_place.DOWNWARD_ROTVEC = [0, pi, 0], exactly the rotvec
    representation's worst point (|rotvec| = pi, the antipodal wraparound) --
    verify_action_encoding.py's stress test showed plain-subtraction deltas
    inflated >10x their true physical size on 243/499 ticks under realistic
    (1-4 deg) RMPflow tracking noise. Composing the actual relative rotation
    (prev.inv() * next) before converting to Euler avoids that regardless of
    representation or noise level."""
    delta_pos = np.asarray(next_pos, dtype=float) - np.asarray(prev_pos, dtype=float)
    r_prev = Rot.from_quat(prev_quat_xyzw)
    r_next = Rot.from_quat(next_quat_xyzw)
    delta_rpy = (r_prev.inv() * r_next).as_euler(EULER_SEQ)
    return np.concatenate([delta_pos, delta_rpy, [next_gripper]]).astype(np.float32)


def collect_episode(scene, target_description, target_position):
    obs = scene.get_observation()
    policy = ScriptedPickPlace(obs["tool_pos"], target_position, PLACE_TARGET_POSITION)

    steps = []
    prev_pos, prev_quat = obs["tool_pos"], obs["tool_quat"]
    prev_gripper = float(obs["gripper"][0])

    for target_pos, target_rotvec, target_gripper in policy.generate_frames():
        image = np.ascontiguousarray(np.asarray(scene.get_observation()["base_rgb"])[..., :3])
        state = _state_vec(prev_pos, prev_quat, prev_gripper)  # (8,): xyz+rpy+pad+gripper

        scene.step_towards(target_pos, target_rotvec, target_gripper)

        next_obs = scene.get_observation()
        next_pos, next_quat = next_obs["tool_pos"], next_obs["tool_quat"]
        next_gripper = float(next_obs["gripper"][0])

        # Cartesian EE pose DELTA + absolute gripper -- the OpenVLA/Open-X
        # action convention, distinct from the π0 side's joint-space logging.
        action = _delta_action(prev_pos, prev_quat, next_pos, next_quat, next_gripper)  # (7,)

        steps.append({
            "image": image,
            "state": state,
            "action": action,
            "language_instruction": f"pick up the {target_description} and place it in the target zone",
        })

        prev_pos, prev_quat, prev_gripper = next_pos, next_quat, next_gripper

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
