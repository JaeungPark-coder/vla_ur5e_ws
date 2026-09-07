"""Ties together the LLM + open-vocabulary hybrid pipeline for one trial:
spawn a random multi-object scene, parse a free-form instruction with an
LLM, locate the named object among several with Grounding DINO, ray-cast
its pixel to a 3D position, then execute the pick-and-place with the
EXISTING scripted controller (scripted_pick_place.ScriptedPickPlace,
unmodified) -- modeled on Pusan National University's RoboCup@Home-winning
"타이디보이" team's LLM-based task-planning approach (their EGPSR mission),
kept and compared against (not replacing) this project's pure end-to-end
π0 VLA pipeline. See README.md's Phase 6 section for the full architecture
rationale and comparison protocol.

NOT a ROS2 node -- run via Isaac Sim's own python.sh:
    <isaac-sim-install-dir>/python.sh hybrid_pick_place_demo.py --instruction "pick up the red cube and put it in the target zone"

Requires ANTHROPIC_API_KEY set, and `anthropic`/`transformers`/`torch`
installed in the Isaac Sim python.sh environment (`pip install anthropic
transformers torch`).

Written and reasoned about WITHOUT the ability to run Isaac Sim, Grounding
DINO, and the Claude API together in the environment this was authored in
-- treat as a solid first draft, not verified to run end-to-end. Each stage
prints its intermediate result specifically so a failed trial's root cause
(LLM misparse vs detector miss vs bad localization vs execution failure) is
easy to tell apart.
"""
import argparse
import csv
import os
import sys

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_HYBRID_DEMO_HEADLESS", "0") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perception"))  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, BASE_CAMERA_HORIZONTAL_FOV_DEG, CAMERA_RESOLUTION, PLACE_TARGET_POSITION, CUBE_Z,
)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
import camera_projection  # noqa: E402
from llm_command_parser import parse_command, CommandParseError  # noqa: E402
from object_detector import ObjectDetector  # noqa: E402

SUCCESS_XY_TOLERANCE_M = 0.03  # matches residual_rl_train_env.py's success criterion, for consistency


def run_trial(scene, detector, instruction, n_objects=3):
    objects = scene.spawn_random_objects(n_objects)
    print(f"spawned objects: {list(objects.keys())}")

    try:
        parsed = parse_command(instruction, object_vocabulary=list(objects.keys()))
    except CommandParseError as exc:
        print(f"LLM command parsing failed: {exc}")
        return False, 0, -1.0, -1.0
    target_description = parsed["object"]
    print(f'LLM parsed "{instruction}" -> object={target_description!r} destination={parsed["destination"]!r}')

    obs = scene.get_observation()
    detection = detector.locate(obs["base_rgb"], target_description)
    if detection is None:
        print(f"detector did not find {target_description!r} in the scene -- aborting trial")
        return False, 0, -1.0, -1.0
    pixel_x, pixel_y, confidence = detection
    print(f"detected {target_description!r} at pixel ({pixel_x:.1f}, {pixel_y:.1f}), confidence={confidence:.2f}")

    camera_pos, camera_rot = scene.get_base_camera_pose()
    target_position = camera_projection.pixel_to_table_position(
        (pixel_x, pixel_y), camera_pos, camera_rot, CAMERA_RESOLUTION, BASE_CAMERA_HORIZONTAL_FOV_DEG, CUBE_Z)
    if target_position is None:
        print("camera ray never crosses the table plane -- aborting trial")
        return False, 0, -1.0, -1.0

    ground_truth_position = objects[target_description]["position"]
    localization_error_mm = 1000.0 * float(np.linalg.norm(target_position[:2] - ground_truth_position[:2]))
    print(f"localization error vs ground truth: {localization_error_mm:.1f}mm")

    policy = ScriptedPickPlace(obs["tool_pos"], target_position, PLACE_TARGET_POSITION)
    for target_pos, target_rotvec, target_gripper in policy.generate_frames():
        scene.step_towards(target_pos, target_rotvec, target_gripper)

    final_position = scene.get_object_position(objects[target_description]["prim_path"])
    placed_xy_error_mm = 1000.0 * float(np.linalg.norm(final_position[:2] - PLACE_TARGET_POSITION[:2]))
    success = placed_xy_error_mm <= SUCCESS_XY_TOLERANCE_M * 1000.0
    print(f"final placement error: {placed_xy_error_mm:.1f}mm -- {'SUCCESS' if success else 'FAILURE'}")
    return success, policy.total_frames(), placed_xy_error_mm, localization_error_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", type=str,
                         default="pick up the red cube and put it in the target zone")
    parser.add_argument("--n_objects", type=int, default=3)
    parser.add_argument("--n_trials", type=int, default=1)
    parser.add_argument("--results_csv", type=str, default="hybrid_eval_results.csv")
    args = parser.parse_args()

    scene = PickPlaceScene()
    detector = ObjectDetector()

    try:
        with open(args.results_csv, "w", newline="") as f:
            csv.writer(f).writerow(["trial", "success", "steps", "xy_error_mm", "localization_error_mm"])

        successes = 0
        for trial in range(args.n_trials):
            print(f"\n=== trial {trial + 1}/{args.n_trials} ===")
            success, steps, xy_error_mm, localization_error_mm = run_trial(
                scene, detector, args.instruction, args.n_objects)
            successes += int(success)
            with open(args.results_csv, "a", newline="") as f:
                csv.writer(f).writerow([trial + 1, success, steps, xy_error_mm, localization_error_mm])
        print(f"\n{successes}/{args.n_trials} trials succeeded (see {args.results_csv})")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
