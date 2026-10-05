"""Does a recorded episode's `actions`, played back the way the policy serving
path plays them, reproduce the demonstration?

Why (2026-10-05, found by reading code): collect_demos.py records
`action = next_obs["joints"] + gripper` -- the MEASURED next-tick position, not
the commanded target. The serving path (pick_place_scene_bridge.py) sends
`target = state + delta` through a plain position drive, with no velocity
target. A position drive covers less than the gap it is given, so the arm may
advance more slowly than the demonstration, and the gripper -- whose recorded
"action" is ~0.7 while holding although 1.0 was commanded -- may not squeeze.
This script measures both, on the exact physics path the bridge uses
(scene.apply_joint_targets == apply_action + gripper.set_target + one
world.step), with no ROS and no policy server.

Procedure per episode seed:
  1. reseed the scene RNG, reset, run the scripted expert exactly as
     collect_demos.py does, recording (joints_t, action_t) every tick;
  2. for every replay variant: reseed (=> identical cube spawn), reset, play
     the recorded actions back open loop and score the outcome.

Arm variants:  abs      target = recorded next joints (absolute)
               delta    target = q_now + K * (recorded next - recorded now)
                        -- chunk[0] re-inferred every tick; K>1 is a lead to compensate lag
               chunk    blocks of K ticks: at the block start take q0 = q_now, then
                        target_t = q0 + (recorded next_t - recorded state at block start)
                        -- what serving with execute_horizon=K does: every chunk row is
                        relative to the state at INFERENCE time (AbsoluteActions)
               deltav   like delta, but ALSO sends a joint-velocity target of
                        K*(recorded next - recorded now)/dt -- what RMPflow's own
                        ArticulationAction carries (position AND velocity) and the
                        bridge does not
               cmdabs   target = the COMMANDED joint target the expert actually sent to the
                        drive that tick (scene.last_command_joints), absolute
               cmddelta target = q_now + (commanded target - recorded state), i.e. what a
                        policy trained on commanded-target labels would produce
Gripper variants: cmd   command = the gripper command the expert actually sent (0..1)
                  raw   command = clip(recorded gripper action)   (old client)
                  hyst  command = GripperHysteresis(recorded)     (new client)
                  scale command = clip(recorded / 0.70)  -- continuous: the ~0.7 stall
                        value maps to a full 1.0 command, with no snap-shut

Reports per variant: lifted / placed, max cube z, place error, arm path-length
ratio vs the recording (1.0 = same speed), final joint RMS error.

    ISAAC_ENV=/home/icrs/bigdisk/conda_envs/env_isaaclab
    LD_PRELOAD=$ISAAC_ENV/lib/libstdc++.so.6 $ISAAC_ENV/bin/python check_action_replay.py \
        --episodes 3 --out /tmp/replay.json

Needs a free GPU. Do not run next to a training job that holds the memory.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "vla_bridge"))

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, ROBOT_PRIM_PATH,
    GRIPPER_FINGER_KP, GRIPPER_FINGER_KD, GRIPPER_FINGER_EFFORT_LIMIT)
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, set_joint_max_force, zero_follower_joint_drives)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
from vla_bridge.gripper_command import GripperHysteresis  # noqa: E402
from vla_bridge.feedforward import chunk_velocities  # noqa: E402

# same values as collect_demos.py (kept in sync by hand -- it cannot be imported,
# it starts its own SimulationApp at import time)
PLACE_CORRECTION_TOLERANCE_M = 0.005
PLACE_CORRECTION_MAX_TICKS = 30
PLACE_TOLERANCE_M = 0.03


def build_scene():
    scene = PickPlaceScene(finger_kp=GRIPPER_FINGER_KP, finger_kd=GRIPPER_FINGER_KD)
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    set_joint_max_force(robot_prim, GRIPPER_FINGER_EFFORT_LIMIT, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
    if not zero_follower_joint_drives(robot_prim):
        raise RuntimeError("zero_follower_joint_drives found no follower joints")
    scene.reset()
    return scene


def record_expert(scene, seed):
    """One scripted-expert episode, logged like collect_demos.py. Returns
    (states (T,7), actions (T,7), cmds (T,7), lifted, placed, place_error) -- state/actions
    are joints[6]+gripper[1]; action_t is the measured next-tick state."""
    scene._rng = np.random.default_rng(seed)
    obs = scene.reset()
    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    diffik_start, diffik_end = policy.diffik_frame_range()
    settle3_end = policy.settle3_end_tick()
    grasp_diffik_start, grasp_diffik_end = policy.grasp_diffik_frame_range()
    settle2_end = policy.settle2_end_tick()

    states, actions, cmds, cvels = [], [], [], []
    max_cube_z = -np.inf

    def log_tick(target_pos, target_rotvec, target_gripper, use_diffik):
        nonlocal max_cube_z
        o = scene.get_observation()
        if use_diffik:
            scene.step_towards_diffik(target_pos, target_rotvec, target_gripper)
        else:
            scene.step_towards(target_pos, target_rotvec, target_gripper)
        n = scene.get_observation()
        states.append(np.concatenate([o["joints"], o["gripper"]]).astype(np.float32))
        actions.append(np.concatenate([n["joints"], n["gripper"]]).astype(np.float32))
        cmds.append(np.concatenate([scene.last_command_joints, [scene.last_command_gripper]]).astype(np.float32))
        v = scene.last_command_velocities
        cvels.append(np.zeros(6, dtype=np.float32) if v is None else np.nan_to_num(v).astype(np.float32))
        max_cube_z = max(max_cube_z, float(scene.get_cube_position()[2]))

    for tick, (pos, rot, grip) in enumerate(policy.generate_frames(), start=1):
        use_diffik = (grasp_diffik_start <= tick <= grasp_diffik_end) or (diffik_start <= tick <= diffik_end)
        log_tick(pos, rot, grip, use_diffik)
        if tick == settle2_end or tick == settle3_end:
            for _ in range(PLACE_CORRECTION_MAX_TICKS):
                residual = float(np.linalg.norm(np.asarray(pos, dtype=float) - scene.grip_point_world()))
                if residual <= PLACE_CORRECTION_TOLERANCE_M:
                    break
                log_tick(pos, rot, grip, use_diffik=True)

    place_error = scene.place_error_m()
    lifted = scene.grasp_succeeded(max_cube_z)
    record_expert.last_velocities = np.stack(cvels)  # side channel: keeps the return signature
    return (np.stack(states), np.stack(actions), np.stack(cmds), lifted,
            lifted and place_error <= PLACE_TOLERANCE_M, place_error)


def replay(scene, seed, states, actions, cmds, arm_mode, gain, grip_mode, thresholds, noise_sigma=0.0, cvels=None):
    scene._rng = np.random.default_rng(seed)   # identical cube spawn to the recording
    scene.reset()
    hyst = GripperHysteresis(*thresholds) if grip_mode == "hyst" else None
    max_cube_z = -np.inf
    achieved = []
    # Stand-in for a policy's imperfect outputs: temporally correlated (AR(1), rho
    # 0.95) noise added to every arm target. White noise would be smoothed away by the
    # drive and prove nothing.
    noise_rng = np.random.default_rng(seed + 7)
    noise = np.zeros(6)
    rho = 0.95
    for t in range(len(actions)):
        o = scene.get_observation()
        q = np.asarray(o["joints"], dtype=float)
        achieved.append(q)
        if arm_mode == "abs":
            target = actions[t, :6].astype(float)
        elif arm_mode.startswith("chunkvn"):
            # The serving recipe with a noisy policy: at each block start build the next 50 rows
            # (relative to the block-start state) + AR(1) noise, take the velocities from THOSE
            # rows with the real vla_bridge.feedforward.chunk_velocities, then execute the first K.
            block = int(gain)
            window = int(arm_mode.split("w")[1]) if "w" in arm_mode else 1
            if t % block == 0:
                q0, s0 = q.copy(), states[t, :6].astype(float)
                rows = q0 + (actions[t:t + 50, :6].astype(float) - s0)
                n_rows = np.zeros_like(rows)
                nz = noise.copy()
                for r in range(len(rows)):
                    nz = rho * nz + np.sqrt(1 - rho ** 2) * noise_sigma * noise_rng.standard_normal(6)
                    n_rows[r] = nz
                rows = rows + n_rows
                chunk_vels = chunk_velocities(rows, scene.physics_dt, window=window, max_speed=3.0)
            target = rows[t % block]
            vel = chunk_vels[t % block]
        elif arm_mode == "chunkv":
            # execute_horizon=K with velocity feed-forward: rows are relative to the state at
            # block start (what delta training + AbsoluteActions gives), velocity from the
            # difference of consecutive rows
            block = int(gain)
            if t % block == 0:
                q0, s0 = q.copy(), states[t, :6].astype(float)
            target = q0 + (actions[t, :6].astype(float) - s0)
        elif arm_mode == "cmdposvel":
            target = cmds[t, :6].astype(float)
        elif arm_mode == "absvel":
            target = actions[t, :6].astype(float)
        elif arm_mode == "cmdabs":
            target = cmds[t, :6].astype(float)
        elif arm_mode == "cmddelta":
            target = q + gain * (cmds[t, :6].astype(float) - states[t, :6].astype(float))
        elif arm_mode == "chunk":
            block = int(gain)
            if t % block == 0:
                q0, s0 = q.copy(), states[t, :6].astype(float)
            target = q0 + (actions[t, :6].astype(float) - s0)
        else:  # delta / deltav, as the policy produces it
            target = q + gain * (actions[t, :6].astype(float) - states[t, :6].astype(float))
        if noise_sigma > 0 and not arm_mode.startswith("chunkvn"):
            noise = rho * noise + np.sqrt(1 - rho ** 2) * noise_sigma * noise_rng.standard_normal(6)
            target = target + noise
        g_rec = float(actions[t, 6])
        if hyst is not None:
            cmd = hyst.update(g_rec)
        elif grip_mode == "cmd":
            cmd = float(np.clip(cmds[t, 6], 0.0, 1.0))
        elif grip_mode == "scale":
            cmd = float(np.clip(g_rec / 0.70, 0.0, 1.0))
        else:
            cmd = float(np.clip(g_rec, 0.0, 1.0))
        if arm_mode.startswith("chunkvn"):
            from isaacsim.core.utils.types import ArticulationAction
            scene.robot.apply_action(ArticulationAction(
                joint_positions=target, joint_velocities=vel, joint_indices=np.arange(6)))
            scene.gripper.set_target(cmd)
            scene.world.step(render=True)
        elif arm_mode in ("cmdposvel", "absvel", "chunkv"):
            from isaacsim.core.utils.types import ArticulationAction
            if arm_mode == "cmdposvel":
                vel = cvels[t].astype(float)
            else:  # absvel / chunkv
                vel = (actions[t, :6].astype(float) - states[t, :6].astype(float)) / scene.physics_dt
            scene.robot.apply_action(ArticulationAction(
                joint_positions=target, joint_velocities=vel, joint_indices=np.arange(6)))
            scene.gripper.set_target(cmd)
            scene.world.step(render=True)
        elif arm_mode == "deltav":
            from isaacsim.core.utils.types import ArticulationAction
            scene.robot.apply_action(ArticulationAction(
                joint_positions=target, joint_velocities=(target - q) / scene.physics_dt,
                joint_indices=np.arange(6)))
            scene.gripper.set_target(cmd)
            scene.world.step(render=True)
        else:
            scene.apply_joint_targets(target, cmd)
        max_cube_z = max(max_cube_z, float(scene.get_cube_position()[2]))
    achieved = np.stack(achieved)
    rec = states[:, :6].astype(float)
    rec_path = float(np.abs(np.diff(rec, axis=0)).sum())
    got_path = float(np.abs(np.diff(achieved, axis=0)).sum())
    place_error = scene.place_error_m()
    lifted = scene.grasp_succeeded(max_cube_z)
    return {
        "lifted": bool(lifted),
        "placed": bool(lifted and place_error <= PLACE_TOLERANCE_M),
        "max_cube_z": round(max_cube_z, 4),
        "place_error_mm": round(1000 * place_error, 1),
        "path_ratio": round(got_path / rec_path, 3) if rec_path > 0 else None,
        "final_joint_rms_rad": round(float(np.sqrt(np.mean((achieved[-1] - rec[-1]) ** 2))), 4),
        "mean_joint_rms_rad": round(float(np.sqrt(np.mean((achieved - rec) ** 2))), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gains", type=float, nargs="*", default=[],
                    help="delta-mode gains K to try (target = q + K*delta)")
    ap.add_argument("--vel", action="store_true",
                    help="also run cmdposvel (the expert's own position+velocity commands, a control that must "
                         "reproduce the demo) and absvel (recorded next position + velocity from consecutive "
                         "positions, what serving could send)")
    ap.add_argument("--chunkv", type=int, nargs="*", default=[],
                    help="chunk-block lengths with velocity feed-forward")
    ap.add_argument("--noisechunkv", type=float, nargs="*", default=[],
                    help="chunkv variants (the final serving recipe) with AR(1) noise of this sigma on the position targets")
    ap.add_argument("--noisechunkv_k", type=int, default=25)
    ap.add_argument("--noisy_vel_windows", type=int, nargs="*", default=[],
                    help="with --noisechunkv: replace the plain chunkv noise runs by chunkvn runs (noisy rows, velocities "
                         "from the real serving function) for each smoothing window")
    ap.add_argument("--noisevel", type=float, nargs="*", default=[],
                    help="absvel variants with AR(1) noise of this sigma on the position targets")
    ap.add_argument("--cmd", action="store_true",
                    help="also run the commanded-target variants (cmdabs, cmddelta) with the commanded gripper")
    ap.add_argument("--noise", type=float, nargs="*", default=[],
                    help="extra abs-target variants with AR(1) noise of this sigma (rad) per joint")
    ap.add_argument("--vel_gains", type=float, nargs="*", default=[],
                    help="deltav gains K (position AND velocity target from the delta)")
    ap.add_argument("--chunks", type=int, nargs="*", default=[5, 10, 25, 50],
                    help="chunk-block lengths to try (execute_horizon equivalents)")
    ap.add_argument("--grip_modes", nargs="*", default=["raw", "hyst"])
    ap.add_argument("--no_abs", action="store_true", help="skip the absolute-target variant")
    ap.add_argument("--close_above", type=float, default=0.45)
    ap.add_argument("--open_below", type=float, default=0.25)
    ap.add_argument("--max_expert_attempts", type=int, default=4)
    ap.add_argument("--dataset_ticks", type=int, default=0,
                    help="instead of the normal run: replay the first N ticks of the DATASET's first episode "
                         "(absolute targets, no ROS) and print the tracking error -- separates 'the dataset does "
                         "not match this simulator' from 'the ROS bridge path is wrong'")
    ap.add_argument("--dataset_episodes", type=int, nargs="*", default=[0],
                    help="episode indices to use with --dataset_ticks")
    ap.add_argument("--out", default="action_replay_results.json")
    args = ap.parse_args()

    scene = build_scene()
    if args.dataset_ticks:
        import glob
        import pandas as pd
        frames = []
        for path in sorted(glob.glob(os.path.expanduser(
                "~/.cache/huggingface/lerobot/vla_ur5e_ws/ur5e_pick_place_v2/data/chunk-*/*.parquet"))):
            frames.append(pd.read_parquet(path, columns=["episode_index", "joints", "gripper", "actions"]))
        df = pd.concat(frames)
        for ep_idx in args.dataset_episodes:
            ep = df[df["episode_index"] == ep_idx].iloc[:args.dataset_ticks]
            st = np.concatenate([np.stack(ep["joints"].values), np.stack(ep["gripper"].values).reshape(len(ep), -1)], 1)
            ac = np.stack(ep["actions"].values)
            r = replay(scene, args.seed, st, ac, ac, "abs", 1.0, "raw", (0.45, 0.25))
            print(f"DATASET episode {ep_idx}, first {len(ep)} ticks, absolute targets in-process: "
                  f"mean_rms={r['mean_joint_rms_rad']} final_rms={r['final_joint_rms_rad']} "
                  f"path_ratio={r['path_ratio']}", flush=True)
        simulation_app.close()
        return
    variants = (([] if args.no_abs else [("abs", 1.0, 0.0)]) + [("delta", k, 0.0) for k in args.gains]
                + [("deltav", k, 0.0) for k in args.vel_gains] + [("chunk", k, 0.0) for k in args.chunks]
                + [("abs", 1.0, sg) for sg in args.noise])
    results = []
    t_start = time.time()

    for ep in range(args.episodes):
        recorded = None
        for attempt in range(args.max_expert_attempts):
            seed = args.seed + 1000 * ep + attempt
            states, actions, cmds, lifted, placed, place_err = record_expert(scene, seed)
            print(f"[ep {ep}] expert seed={seed}: T={len(actions)} lifted={lifted} placed={placed} "
                  f"place_err={1000 * place_err:.1f}mm  (elapsed {time.time() - t_start:.0f}s)", flush=True)
            if placed:
                recorded = (seed, states, actions, cmds)
                break
        if recorded is None:
            print(f"[ep {ep}] no successful expert recording in {args.max_expert_attempts} attempts -- skipped", flush=True)
            continue
        seed, states, actions, cmds = recorded
        plan = [(a, g, sg, gm) for a, g, sg in variants for gm in args.grip_modes]
        plan += [("chunkv", float(k), 0.0, "raw") for k in args.chunkv]
        plan += [("absvel", 1.0, sg, "raw") for sg in args.noisevel]
        if args.noisy_vel_windows:
            plan += [(f"chunkvn:w{w}", float(args.noisechunkv_k), sg, "raw")
                     for sg in args.noisechunkv for w in args.noisy_vel_windows]
        else:
            plan += [("chunkv", float(args.noisechunkv_k), sg, "raw") for sg in args.noisechunkv]
        if args.vel:
            plan += [("cmdposvel", 1.0, 0.0, "cmd"), ("absvel", 1.0, 0.0, "raw")]
        if args.cmd:
            plan += [("cmdabs", 1.0, 0.0, "cmd"), ("cmddelta", 1.0, 0.0, "cmd")]
        for arm_mode, gain, sigma, grip_mode in plan:
            if True:
                r = replay(scene, seed, states, actions, cmds, arm_mode, gain, grip_mode,
                           (args.close_above, args.open_below), noise_sigma=sigma,
                           cvels=record_expert.last_velocities if recorded else None)
                r.update(episode=ep, seed=seed, arm=arm_mode, gain=gain, gripper=grip_mode, noise=sigma)
                results.append(r)
                print(f"[ep {ep}] {arm_mode:5s} K={gain:<5} sig={sigma:<5} grip={grip_mode:5s} -> lifted={r['lifted']!s:5} "
                      f"placed={r['placed']!s:5} zmax={r['max_cube_z']:.3f} err={r['place_error_mm']:6.1f}mm "
                      f"path_ratio={r['path_ratio']} final_rms={r['final_joint_rms_rad']}  "
                      f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)

    # summary: one line per variant
    print("\n=== summary (per variant, over episodes) ===", flush=True)
    keys = sorted({(r["arm"], r["gain"], r["noise"], r["gripper"]) for r in results})
    for arm, gain, sigma, grip in keys:
        rs = [r for r in results if (r["arm"], r["gain"], r["noise"], r["gripper"]) == (arm, gain, sigma, grip)]
        n = len(rs)
        print(f"{arm:5s} K={gain:<4} sig={sigma:<5} grip={grip:5s}: placed {sum(r['placed'] for r in rs)}/{n}, "
              f"lifted {sum(r['lifted'] for r in rs)}/{n}, "
              f"mean path_ratio {np.mean([r['path_ratio'] for r in rs]):.3f}, "
              f"mean final_rms {np.mean([r['final_joint_rms_rad'] for r in rs]):.4f} rad", flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    simulation_app.close()


if __name__ == "__main__":
    main()
