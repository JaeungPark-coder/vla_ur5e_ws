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
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_RLDS_COLLECT_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, LIFT_Z_THRESHOLD, PLACE_TARGET_POSITION  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "openvla_integration", "raw_episodes")

# The quality gate every episode has to clear before it is written. Imported
# rather than reimplemented so the collector and the standalone pre-training
# check (openvla_integration/validate_dataset.py) can never drift apart --
# and so this file's encoding is checked by the same three identities that
# tool checks. validate_dataset imports only numpy/scipy, no Isaac Sim.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "openvla_integration"))
from validate_dataset import validate_episode, EULER_SEQ  # noqa: E402


# Euler axis order for the RPY fields. CONFIRMED extrinsic (fixed-axis)
# lowercase "xyz": scipy reads a lowercase axis string as extrinsic and an
# uppercase one as INTRINSIC, which is a different rotation for the same
# three numbers, and the OpenVLA / Bridge / DROID stack this dataset targets
# is extrinsic throughout (transforms3d's `sxyz` default across the Berkeley
# tooling, DROID's own scipy `from_euler("xyz")`, and the shared Octo/DROID
# `tensorflow_graphics.from_euler` Rz@Ry@Rx composition all agree).
#
# Imported from the validator rather than redeclared so the two cannot
# drift: validate_dataset refuses an uppercase value at import, and its
# metamorphic check re-derives extrinsic-XYZ from first principles and
# requires scipy's reading of THIS constant to match -- which is what
# catches an axis order that collector and validator would otherwise be
# consistently wrong about together.


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


def collect_episode(scene, target_description, target_position, target_prim_path=None):
    """Records one episode and returns (steps, max_target_z).

    max_target_z is how high the TARGET object ever got, read from ground
    truth. main() uses it to throw away attempts where the scripted expert
    never actually picked anything up -- see PickPlaceScene.grasp_succeeded,
    whose own comment describes what happens without it: "a failed grasp
    still went into the training set as if it were a demonstration,
    teaching the policy to mime the motion whether or not it picked
    anything up". collect_demos.py has done this from the start on the
    pi0/LeRobot side; this path had no equivalent, which matters all the
    more while the grasp itself is still being debugged.
    """
    obs = scene.get_observation()
    policy = ScriptedPickPlace(obs["tool_pos"], target_position, PLACE_TARGET_POSITION)

    steps = []
    max_target_z = -np.inf
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
        if target_prim_path is not None:
            max_target_z = max(max_target_z, float(
                scene.get_object_position(target_prim_path)[2]))

    return steps, max_target_z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--n_objects", type=int, default=3)
    parser.add_argument("--max_attempts", type=int, default=0,
                        help="cap on collection attempts including rejected ones "
                             "(default 0: 3x --num_episodes)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scene = PickPlaceScene()
    rng = np.random.default_rng()

    n_saved = 0
    n_rejected = 0
    try:
        for attempt in range(args.max_attempts or args.num_episodes * 3):
            if n_saved >= args.num_episodes:
                break

            objects = scene.spawn_random_objects(args.n_objects)
            target_description = rng.choice(list(objects.keys()))
            target_position = objects[target_description]["position"]

            steps, max_target_z = collect_episode(
                scene, target_description, target_position,
                target_prim_path=objects[target_description]["prim_path"])

            # Check BEFORE writing. Every expensive failure on this project so
            # far has been a dataset that was perfectly well-formed and
            # completely useless -- black wrist frames, a target that never
            # appeared in a single frame, rotation deltas inflated 10x by
            # subtracting representations instead of composing them. All of
            # those are invisible unless something looks at the numbers, and
            # all of them are cheap to catch here and expensive to discover
            # after a fine-tune.
            ok, problems = validate_episode(steps)

            # Two independent questions, and both have to be yes. The gate
            # above asks whether the RECORDING is sound (frames, encoding,
            # the state/action identities); this asks whether the episode
            # demonstrates the TASK. An episode can be flawlessly recorded
            # and still show the gripper closing on nothing.
            # NOTE: only the lift is checked, not the placement --
            # place_error_m() measures the single-cube scene that reset()
            # builds, not the multi-object one spawn_random_objects does.
            if not scene.grasp_succeeded(max_target_z):
                ok = False
                problems = problems + [
                    f"the target was never lifted (peaked at z={max_target_z:.3f}m, "
                    f"needs >= {LIFT_Z_THRESHOLD:.3f}m) -- the gripper closed on nothing, "
                    f"so this shows the motion without the grasp"]

            if not ok:
                n_rejected += 1
                print(f"attempt {attempt + 1}: REJECTED ({len(steps)} steps, "
                      f"target={target_description!r})")
                for problem in problems:
                    print(f"    - {problem}")

                # A first episode that fails on framing or encoding is not bad
                # luck, it is a broken setup: every later episode will fail the
                # same way. Stop now rather than after another 99.
                if n_saved == 0 and n_rejected >= 3:
                    raise RuntimeError(
                        "the first 3 attempts all failed validation, which means the scene or "
                        "the encoding is wrong rather than the run being unlucky. Fix what the "
                        "problems above report before collecting -- run isaac/check_cameras.py "
                        "if they are about visibility, and isaac/verify_action_encoding.py if "
                        "they are about the state/action identities.")
                continue

            out_path = os.path.join(OUTPUT_DIR, f"episode_{n_saved:05d}.npy")
            np.save(out_path, steps, allow_pickle=True)
            n_saved += 1
            print(f"episode {n_saved}/{args.num_episodes}: {len(steps)} steps, "
                  f"target={target_description!r} -> {out_path}")

        if n_saved < args.num_episodes:
            print(f"\nWARNING: only {n_saved}/{args.num_episodes} episodes passed validation "
                  f"in {n_saved + n_rejected} attempts. Collecting more will not help until "
                  f"the reported problems are fixed.")
        else:
            print(f"\ncollected {n_saved} episodes, rejected {n_rejected} "
                  f"({n_rejected / max(n_saved + n_rejected, 1):.0%} of attempts)")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
