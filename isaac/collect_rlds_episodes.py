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
from pick_place_scene import (  # noqa: E402
    PickPlaceScene, LIFT_Z_THRESHOLD, PLACE_TARGET_POSITION, ROBOT_PRIM_PATH)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, set_joint_max_force, zero_follower_joint_drives)

# 2026-09-28: see collect_demos.py's matching constants -- same first-ever
# successful grasp-lift config (IsaacLab's FRANKA_ROBOTIQ_GRIPPER_CFG
# reference gains), wired in here too since PickPlaceScene(finger_kp=None,
# finger_kd=None) resolves to this project's own broken 20000/500/no-limit
# default (GripperController._fix_drive_gains), which has never produced a
# successful lift.
GRIPPER_FINGER_KP = 17.0
GRIPPER_FINGER_KD = 0.02
GRIPPER_FINGER_EFFORT_LIMIT = 1650.0

# 2026-09-28: closed-loop place correction -- see collect_demos.py's
# matching constants for the full rationale (safe now specifically because
# descend2/settle3/release already use step_towards_diffik; an earlier
# fixed-trim attempt at this same problem, back when the place approach was
# still RMPflow-driven, made results worse by pushing into a different
# RMPflow equilibrium).
PLACE_CORRECTION_TOLERANCE_M = 0.003
PLACE_CORRECTION_MAX_TICKS = 30

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
    diffik_start, diffik_end = policy.diffik_frame_range()
    settle3_end = policy.settle3_end_tick()
    grasp_diffik_start, grasp_diffik_end = policy.grasp_diffik_frame_range()
    settle2_end = policy.settle2_end_tick()

    steps = []
    max_target_z = -np.inf
    state_holder = {"pos": obs["tool_pos"], "quat": obs["tool_quat"], "gripper": float(obs["gripper"][0])}

    def log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik):
        nonlocal max_target_z
        image = np.ascontiguousarray(np.asarray(scene.get_observation()["base_rgb"])[..., :3])
        prev_pos, prev_quat, prev_gripper = state_holder["pos"], state_holder["quat"], state_holder["gripper"]
        state = _state_vec(prev_pos, prev_quat, prev_gripper)  # (8,): xyz+rpy+pad+gripper

        # 2026-09-28: place approach uses direct Jacobian servoing, not
        # RMPflow -- see collect_demos.py's matching comment / README.
        if use_diffik:
            scene.step_towards_diffik(target_pos, target_rotvec, target_gripper)
        else:
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

        state_holder["pos"], state_holder["quat"], state_holder["gripper"] = next_pos, next_quat, next_gripper
        if target_prim_path is not None:
            max_target_z = max(max_target_z, float(
                scene.get_object_position(target_prim_path)[2]))

    for tick, (target_pos, target_rotvec, target_gripper) in enumerate(policy.generate_frames(), start=1):
        use_diffik = (grasp_diffik_start <= tick <= grasp_diffik_end) or (diffik_start <= tick <= diffik_end)
        log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik=use_diffik)

        if tick == settle2_end or tick == settle3_end:
            # Closed-loop correction at both the grasp-approach dwell
            # (settle2) and the place-approach dwell (settle3) -- see
            # collect_demos.py's matching comment / README's 2026-09-28
            # section. Logged, not hidden: each correction tick becomes a
            # real recorded step.
            for _ in range(PLACE_CORRECTION_MAX_TICKS):
                residual_m = float(np.linalg.norm(
                    np.asarray(target_pos, dtype=float) - scene.grip_point_world()))
                if residual_m <= PLACE_CORRECTION_TOLERANCE_M:
                    break
                log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik=True)

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
    scene = PickPlaceScene(finger_kp=GRIPPER_FINGER_KP, finger_kd=GRIPPER_FINGER_KD)

    # CONFIRMED 2026-09-28, load-bearing -- see collect_demos.py's matching
    # comment: must run before the first spawn_random_objects() call below,
    # not after, since set_joint_max_force on finger_joint's prim reverts its
    # LIVE controller gain as a side effect, and spawn_random_objects's own
    # world.reset()+reapply_drive_gains() (just added -- see pick_place_scene.py)
    # is what restores it, every attempt.
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    set_joint_max_force(robot_prim, GRIPPER_FINGER_EFFORT_LIMIT, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
    follower_names = zero_follower_joint_drives(robot_prim)
    if not follower_names:
        raise RuntimeError(
            "zero_follower_joint_drives found no follower/passive gripper joints to zero -- "
            "check resolve_gripper_follower_joint_names, the grasp fix depends on this.")

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
        # See collect_demos.py's matching comment: simulation_app.close()'s
        # fastShutdown can race ordinary buffered print() output, not just
        # tracebacks -- flush unconditionally rather than hunting down every
        # print() call above.
        import sys
        sys.stdout.flush()
        simulation_app.close()


if __name__ == "__main__":
    main()
