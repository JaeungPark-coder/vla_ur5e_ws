"""Runs N randomized episodes of the scripted pick-and-place policy inside
Isaac Sim and writes them straight into a LeRobot dataset -- the training
data for the openpi UR5e LoRA fine-tune (see ../openpi_integration/).

NOT a ROS2 node -- it needs Isaac Sim's own Kit runtime, so run it with the
env_isaaclab conda environment's interpreter (Isaac Sim 5.1 is pip-installed
there, so there is no python.sh wrapper; `python3` already resolves to it):

    python3 collect_demos.py --num_episodes 50 --repo_id your_hf_username/ur5e_pick_place

Dataset schema (image/wrist_image/state/actions, add_frame + save_episode)
matches openpi's own examples/libero/convert_libero_data_to_lerobot.py, and
the state/action layout (joints[6] + gripper[1], both 7-dim) matches
openpi's examples/ur5/README.md UR5Inputs/UR5Outputs contract exactly --
this is what lets the ur5e_pick_place_policy.py transforms in
../openpi_integration/ consume this dataset with no reshaping.

Every attempt is scored against ground truth before it is kept: the cube
must actually leave the table and end up within --place_tolerance_m of the
target, or the attempt is discarded (images and all) and retried. So
--num_episodes is the number of SUCCESSFUL episodes, and the closing success
rate is a direct measurement of whether the grasp is stable. Without this
every attempt was logged unconditionally, so a failed grasp went into the
training set as a demonstration of the task it had just failed to do.

Run check_cameras.py first. Two collection runs were lost to cameras aimed
away from the task, which no amount of downstream care recovers from.

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
from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, MIN_CUBE_PIXELS_IN_BASE_VIEW, ROBOT_PRIM_PATH)
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, set_joint_max_force, zero_follower_joint_drives)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

PROMPT = "pick up the cube and place it in the target zone"
CONTROL_FPS = 30  # matches the LeRobot dataset's `fps` metadata -- keep in sync with steps_per_segment choices

# 2026-09-28: the first-ever successful grasp-lift (5/5, ~2x LIFT_Z_THRESHOLD,
# see README's 2026-09-28 section) used IsaacLab's real FRANKA_ROBOTIQ_GRIPPER_CFG
# reference gains, not this project's own as-shipped-then-"fixed" 20000/500/
# no-limit default -- PickPlaceScene(finger_kp=None, finger_kd=None) resolves
# to that broken default (see GripperController._fix_drive_gains), so leaving
# these unset here would silently keep collecting against the config that has
# never once produced a successful lift. These three go together -- see
# check_free_space_closure.py's docstring for why (finger_effort_limit and
# zero_follower_pd applied together with this kp/kd, not separately tested).
GRIPPER_FINGER_KP = 17.0
GRIPPER_FINGER_KD = 0.02
GRIPPER_FINGER_EFFORT_LIMIT = 1650.0

# 2026-09-28: closed-loop place correction (see the settle3_end handling
# below) -- small on purpose. This is safe to add now specifically BECAUSE
# descend2/settle3/release already use step_towards_diffik: the earlier
# fixed-trim attempt at this same problem (moving the TARGET by a measured
# constant, back when the place approach was still RMPflow-driven) made 3/5
# episodes much worse by pushing into a different RMPflow equilibrium
# ("basin-hopping", see README). With diffIK already converging close
# (2-20mm observed), a few more small corrective diffIK ticks toward the
# SAME target carries none of that risk.
PLACE_CORRECTION_TOLERANCE_M = 0.003
PLACE_CORRECTION_MAX_TICKS = 30


def _to_rgb_uint8(rgba_or_rgb):
    """Replicator's "rgb" annotator returns HxWx4 (RGBA) uint8 -- LeRobot's
    "image" dtype expects HxWx3."""
    arr = np.asarray(rgba_or_rgb)
    return arr[..., :3] if arr.shape[-1] == 4 else arr


def max_attempts_for(num_episodes: int) -> int:
    """Budget for retrying rejected episodes. Generous enough to absorb the
    occasional physics hiccup, tight enough that a systematically broken
    grasp stops the run early instead of spinning forever."""
    return max(10, 3 * num_episodes)


def collect(num_episodes: int, repo_id: str, push_to_hub: bool, place_tolerance_m: float = 0.03):
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

    scene = PickPlaceScene(finger_kp=GRIPPER_FINGER_KP, finger_kd=GRIPPER_FINGER_KD)

    # CONFIRMED 2026-09-28, load-bearing: this must run BEFORE the first
    # scene.reset() below, not after. set_joint_max_force on finger_joint's
    # prim reverts its LIVE controller gain (the finger_kp/finger_kd just
    # requested above) back to the raw USD default as a side effect --
    # scene.reset() calling reapply_drive_gains() afterwards is what restores
    # the requested gain. Every subsequent scene.reset() (once per attempt,
    # below) reapplies it again, so this block only needs to run once here.
    # See set_joint_max_force's own docstring and check_grasp_alignment.py's
    # matching comment for the full story.
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    set_joint_max_force(robot_prim, GRIPPER_FINGER_EFFORT_LIMIT, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
    follower_names = zero_follower_joint_drives(robot_prim)
    if not follower_names:
        raise RuntimeError(
            "zero_follower_joint_drives found no follower/passive gripper joints to zero -- "
            "check resolve_gripper_follower_joint_names, the grasp fix depends on this.")

    # Look before leaping: two full collection runs were already lost to
    # camera faults that produced perfectly well-formed datasets full of
    # useless frames (see PickPlaceScene.preflight_check). Refuse to start
    # rather than discover it 100 episodes later.
    scene.reset()
    ok, problems = scene.preflight_check()
    if not ok:
        raise RuntimeError(
            "preflight check failed -- refusing to collect:\n  " + "\n  ".join(problems))

    attempted = 0
    n_success = 0
    place_errors = []

    while n_success < num_episodes:
        attempted += 1
        if attempted > max_attempts_for(num_episodes):
            raise RuntimeError(
                f"gave up after {attempted - 1} attempts with only {n_success}/{num_episodes} "
                f"successful episodes. The scripted expert is not reliably completing the task -- "
                f"this is a scene/gripper problem, not a data problem, and collecting more will "
                f"not fix it. The grasp is the usual cause: check that the gripper variant "
                f"actually took effect (the run prints the robot's joint names at startup -- "
                f"{GRIPPER_DRIVE_JOINT_NAME!r} must be among them) and that "
                f"GRIPPER_OPEN_POS/GRIPPER_CLOSED_POS match this gripper's real joint limits.")

        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        diffik_start, diffik_end = policy.diffik_frame_range()
        settle3_end = policy.settle3_end_tick()
        grasp_diffik_start, grasp_diffik_end = policy.grasp_diffik_frame_range()
        settle2_end = policy.settle2_end_tick()

        n_logged = 0
        max_cube_z = -np.inf
        peak_cube_px = 0

        def log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik):
            nonlocal n_logged, max_cube_z, peak_cube_px
            obs = scene.get_observation()
            # 2026-09-28: descend2/settle3/release (the place approach) use
            # direct Jacobian servoing instead of RMPflow -- see
            # step_towards_diffik's docstring / README's 2026-09-28 section
            # for why (a persistent 25-56mm RMPflow tracking bias at the
            # place target that a 400-tick dwell does not resolve).
            if use_diffik:
                scene.step_towards_diffik(target_pos, target_rotvec, target_gripper)
            else:
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
            max_cube_z = max(max_cube_z, float(scene.get_cube_position()[2]))
            peak_cube_px = max(peak_cube_px, scene.cube_pixels_visible(obs["base_rgb"]))

        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(policy.generate_frames(), start=1):
            use_diffik = (grasp_diffik_start <= tick <= grasp_diffik_end) or (diffik_start <= tick <= diffik_end)
            log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik=use_diffik)

            # 2026-09-28: closed-loop correction at BOTH the grasp-approach
            # dwell (settle2, before close) and the place-approach dwell
            # (settle3, before release) -- same mechanism, same reasoning
            # (see README's 2026-09-28 sections; PLACE_CORRECTION_* despite
            # the name applies to both, it's the tolerance/budget for "keep
            # re-measuring and correcting via diffIK until close enough").
            if tick == settle2_end or tick == settle3_end:
                for _ in range(PLACE_CORRECTION_MAX_TICKS):
                    residual_m = float(np.linalg.norm(
                        np.asarray(target_pos, dtype=float) - scene.grip_point_world()))
                    if residual_m <= PLACE_CORRECTION_TOLERANCE_M:
                        break
                    log_one_tick(target_pos, target_rotvec, target_gripper, use_diffik=True)

        # Only a demonstration that actually demonstrates the task is worth
        # keeping. clear_episode_buffer() also removes the frames' image
        # files and rewinds the episode index, so a rejected attempt leaves
        # no trace in the dataset.
        lifted = scene.grasp_succeeded(max_cube_z)
        place_error = scene.place_error_m()
        placed = place_error <= place_tolerance_m
        if lifted and placed:
            dataset.save_episode()
            n_success += 1
            place_errors.append(place_error)
            outcome = "OK"
        else:
            dataset.clear_episode_buffer()
            outcome = "REJECTED (" + ("never lifted" if not lifted
                                      else f"placed {place_error * 1000:.0f}mm off target") + ")"

        print(f"attempt {attempted} -> episode {n_success}/{num_episodes}: {outcome}, "
              f"{n_logged} frames, cube spawned at {np.round(scene.cube_position, 3)}, "
              f"max_cube_z={max_cube_z:.3f}m, peak_cube_px={peak_cube_px}")

        # Stop the moment a SAVED episode's cube was never actually visible.
        # A dataset the cube never appears in trains a policy that cannot
        # possibly find it, and that is exactly how the first 100-episode
        # run was lost -- 21,000 well-formed frames, zero of them showing
        # the cube. 2026-09-23: this used to only check when n_success==1
        # (the very first success), which catches a broken-from-the-start
        # setup but not a mid-run regression (camera mount drift, a lighting
        # change, anything that degrades framing after episode 1 already
        # passed) -- the scripted expert drives off GROUND-TRUTH cube_
        # position, not vision, so a later episode can still lift+place
        # correctly (get saved) with the cube barely visible in its
        # recorded frames, and nothing would catch it. Checked on every
        # SAVED episode now, not gated by n_success -- rejected episodes
        # aren't saved either way, so checking those would only cost time
        # for no dataset-quality benefit.
        if lifted and placed and peak_cube_px < MIN_CUBE_PIXELS_IN_BASE_VIEW:
            raise RuntimeError(
                f"episode {n_success} (attempt {attempted}) was saved but the cube peaked at "
                f"only {peak_cube_px} pixels in the base camera across the whole episode (need "
                f">= {MIN_CUBE_PIXELS_IN_BASE_VIEW}). The task is not visible in the "
                f"observations, so no amount of data will help. Run check_cameras.py and fix "
                f"the framing before collecting more.")

    success_rate = n_success / attempted
    print(f"\ncollected {n_success} successful episodes from {attempted} attempts "
          f"(success rate {success_rate:.0%})")
    if place_errors:
        print(f"placement error: mean {np.mean(place_errors) * 1000:.1f}mm, "
              f"max {np.max(place_errors) * 1000:.1f}mm")
    if success_rate < 0.9:
        print("WARNING: the scripted expert is failing often. These demonstrations are still "
              "valid (failures were discarded), but a flaky expert usually means an unstable "
              "grasp -- worth fixing before training, since the policy has to reproduce it.")

    # CONFIRMED (2026-09-07) by inspecting a finished collection: without
    # this, the parquet files are left without their footer metadata and the
    # dataset cannot be loaded at all (LeRobotDataset.finalize's own docstring
    # says as much). main()'s simulation_app.close() tears the process down
    # via Kit's fastShutdown before any interpreter-exit hook could do it, so
    # the last chunk of a run was silently truncated -- file-004.parquet of
    # the first 100-episode collection ended up unreadable, and a 2-episode
    # run lost everything. This also joins the image-writer subprocesses,
    # which otherwise linger holding several GB of RAM each.
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(tags=["ur5e", "pick_place", "vla_ur5e_ws"], private=False, push_videos=True,
                             license="apache-2.0")

    print(f"done -- dataset written to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="number of SUCCESSFUL episodes to collect -- attempts where the "
                             "scripted expert failed to pick and place the cube are discarded "
                             "and retried, so this is the dataset size you actually get")
    parser.add_argument("--repo_id", type=str, default="your_hf_username/ur5e_pick_place")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--place_tolerance_m", type=float, default=0.03,
                        help="max cube-to-target planar error for an episode to count as a "
                             "demonstration; matches the success threshold vla_policy_client.py's "
                             "eval_mode and hybrid_pick_place_demo.py already use")
    args = parser.parse_args()

    # Print any traceback BEFORE closing the app: simulation_app.close() runs
    # Kit's fastShutdown, which kills the process before Python can report an
    # escaping exception, making a crash look like a clean exit.
    try:
        collect(args.num_episodes, args.repo_id, args.push_to_hub, args.place_tolerance_m)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        # CONFIRMED 2026-09-28: not just the exception path -- collect()'s own
        # final summary prints (the last "attempt N -> ..." line, "collected
        # N successful episodes...", "done -- dataset written to...") were
        # observed missing from a real run's log despite the run completing
        # correctly (verified by loading the resulting dataset directly:
        # exactly the right episode/frame count, fully finalized). Those
        # print()s aren't flush=True and stdout is block-buffered once
        # redirected to a file/pipe -- the SAME fastShutdown race the
        # exception-path comment above already describes, just for ordinary
        # buffered output instead of a traceback. Flush unconditionally here
        # rather than hunting down every print() call in collect().
        import sys
        sys.stdout.flush()
        simulation_app.close()


if __name__ == "__main__":
    main()
