"""OpenVLA inference/evaluation, one function call per trial: loads a
LoRA-fine-tuned OpenVLA checkpoint (produced by the `finetune.py` command in
../openvla_integration/openvla_transform_snippet.py) and runs it against
Isaac Sim's PickPlaceScene -- single camera + Cartesian end-effector-delta
actions, OpenVLA's own convention, distinct from the pi0 side's
joint-space/dual-camera one (see collect_rlds_episodes.py's docstring for
why these two VLAs need separate data/action spaces). The third arm of this
project's VLA vs. hybrid vs. VLA comparison (see README.md Phase 7).

NOT a ROS2 node -- run via Isaac Sim's own python.sh:
    <isaac-sim-install-dir>/python.sh openvla_pick_place_demo.py \
        --checkpoint <path-to-your-finetuned-openvla-checkpoint> --n_trials 20

Requires `transformers`, `torch`, `pillow` in the Isaac Sim python.sh
environment (`pip install transformers torch pillow`).

Logs one (trial, success, steps, xy_error_mm) row per trial to --results_csv
-- same shape hybrid_pick_place_demo.py and vla_policy_client.py's eval_mode
use, so all three pipelines' results can be compared directly (e.g. loaded
side by side in a notebook).

ADJUST -- more so than any other file in this project (see
openvla_transform_snippet.py's own warning): the exact rotation-delta
convention OpenVLA's action head was fine-tuned to output (a delta rotation
*vector*, matching what collect_rlds_episodes.py logs, vs. some other
rotation convention OpenVLA's own config expects) has never been confirmed
against a real fine-tuned checkpoint. If the arm's orientation drifts
strangely during a trial, check this first. Also note _apply_action's
linear addition of rotation vectors (current + delta) is an approximation,
not a rigorous rotation composition -- acceptable here only because this
task holds a fixed downward orientation and never needs a large
re-orientation; do not reuse this shortcut for a task that does.

Written and reasoned about WITHOUT the ability to run Isaac Sim, a real
fine-tuned OpenVLA checkpoint, or the two together in the environment this
was authored in -- treat as a solid first draft, not verified to run
end-to-end.
"""
import argparse
import csv
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_OPENVLA_DEMO_HEADLESS", "0") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION, LIFT_Z_THRESHOLD  # noqa: E402

# Same values hybrid_pick_place_demo.py / residual_rl_train_env.py use, kept
# consistent across all three pipelines so "success" means the same thing
# everywhere in the comparison.
SUCCESS_XY_TOLERANCE_M = 0.03
HOLDING_GRIPPER_THRESHOLD = 0.5
# 2026-09-23: this used to redeclare 0.03 locally -- a stray duplicate of
# what turned out to be the WRONG value (see residual_rl_train_env.py's own
# 2026-09-23 fix; the right one, isaac/pick_place_scene.LIFT_Z_THRESHOLD, is
# 0.08). Now imported directly instead of copied, so this file can't drift
# from it again.
LIFTED_Z_THRESHOLD = LIFT_Z_THRESHOLD


class OpenVLAPolicy:
    """Thin wrapper around a LoRA-fine-tuned OpenVLA checkpoint, following
    OpenVLA's own documented local-inference pattern (its README's
    `predict_action` usage) -- not this project's invention. Runs in-process
    (no server/client split like openpi's WebsocketClientPolicy) since
    that's how OpenVLA itself is normally invoked."""

    def __init__(self, checkpoint_path, unnorm_key="ur5e_pick_place"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.unnorm_key = unnorm_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(self.device)

    def predict_action(self, image, instruction):
        """image: HxWx3 uint8 RGB array (base_rgb only -- OpenVLA's side of
        this project uses a single camera, see module docstring). Returns a
        (7,) array: delta xyz(3) + delta rotvec(3) + absolute gripper(1),
        matching collect_rlds_episodes.py's exact action convention."""
        import torch
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(np.asarray(image)[..., :3])
        # OpenVLA's own documented prompt template for a fine-tuned checkpoint.
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        return np.asarray(action, dtype=np.float32)


def _apply_action(scene, action):
    """Converts OpenVLA's delta-pose action into the absolute Cartesian
    target scene.step_towards expects, and applies it for one control tick."""
    obs = scene.get_observation()
    current_rotvec = Rot.from_quat(obs["tool_quat"]).as_rotvec()

    target_pos = obs["tool_pos"] + action[:3]
    target_rotvec = current_rotvec + action[3:6]  # linear approx -- see module docstring
    target_gripper = float(np.clip(action[6], 0.0, 1.0))

    scene.step_towards(target_pos, target_rotvec, target_gripper)
    return target_gripper


def run_trial(scene, policy, n_objects, max_steps, rng):
    """Spawns n_objects random objects, picks one as the target (same
    random-choice-among-spawned-objects distribution collect_rlds_episodes.py
    trained on), and queries OpenVLA once per control tick until it either
    places the target within tolerance or max_steps runs out.

    Returns (success, steps_taken, final_xy_error_mm)."""
    objects = scene.spawn_random_objects(n_objects)
    target_description = rng.choice(list(objects.keys()))
    instruction = f"pick up the {target_description} and place it in the target zone"
    print(f"spawned objects: {list(objects.keys())} -- target: {target_description!r}")

    target_prim_path = objects[target_description]["prim_path"]
    was_holding = False
    xy_error_mm = -1.0

    for step in range(1, max_steps + 1):
        obs = scene.get_observation()
        action = policy.predict_action(obs["base_rgb"], instruction)
        target_gripper = _apply_action(scene, action)

        object_pos = scene.get_object_position(target_prim_path)
        is_holding = target_gripper >= HOLDING_GRIPPER_THRESHOLD and object_pos[2] > LIFTED_Z_THRESHOLD
        xy_error_mm = 1000.0 * float(np.linalg.norm(object_pos[:2] - PLACE_TARGET_POSITION[:2]))
        released_here = was_holding and not is_holding
        was_holding = is_holding

        if released_here and xy_error_mm <= SUCCESS_XY_TOLERANCE_M * 1000.0:
            return True, step, xy_error_mm

    return False, max_steps, xy_error_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a LoRA-fine-tuned OpenVLA checkpoint directory.")
    parser.add_argument("--unnorm_key", type=str, default="ur5e_pick_place")
    parser.add_argument("--n_objects", type=int, default=1,
                         help="Use 1 for a like-for-like comparison against pi0's single-cube training data.")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--results_csv", type=str, default="openvla_eval_results.csv")
    args = parser.parse_args()

    scene = PickPlaceScene()
    policy = OpenVLAPolicy(args.checkpoint, unnorm_key=args.unnorm_key)
    rng = np.random.default_rng()

    try:
        with open(args.results_csv, "w", newline="") as f:
            csv.writer(f).writerow(["trial", "success", "steps", "xy_error_mm"])

        successes = 0
        for trial in range(args.n_trials):
            print(f"\n=== trial {trial + 1}/{args.n_trials} ===")
            success, steps, xy_error_mm = run_trial(scene, policy, args.n_objects, args.max_steps, rng)
            successes += int(success)
            print(f"{'SUCCESS' if success else 'FAILURE'} -- steps={steps}, xy_error_mm={xy_error_mm:.1f}")
            with open(args.results_csv, "a", newline="") as f:
                csv.writer(f).writerow([trial + 1, success, steps, xy_error_mm])

        print(f"\n{successes}/{args.n_trials} trials succeeded (see {args.results_csv})")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
