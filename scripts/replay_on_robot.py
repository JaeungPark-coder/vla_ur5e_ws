#!/usr/bin/env python3
"""Replay one demonstration's action rows on a UR arm (URSim or the real robot), no policy: real-robot preparation step 4.

The rows go through exactly the path the policy client uses for robot_backend 'rtde' -- safety check, then
UR5eInterface-style streaming (vla_bridge/joint_streamer.py: row_dt / speed_scale per row, 500 Hz servoJ, adaptive slowdown
above max_joint_speed) in blocks of K rows -- so a clean replay checks the joint convention, the timing and the limits
together, before any policy output reaches the arm.  The actual joints are recorded at 500 Hz and compared with the command.

  --lift_m 0.10   replays the same motion 10 cm higher ("in the air"): every row is re-solved by IK so the flange moves up
                  by lift_m with its orientation unchanged.  Use it for the first runs on hardware.
  --dry_run       checks and prints the plan, sends nothing.

    <python with ur_rtde, numpy, pyarrow> scripts/replay_on_robot.py --host 127.0.0.1 \
        --dataset ~/.cache/huggingface/lerobot/Jaeung12/ur5e_pick_place_v4 --episode 0 --limits limits.json --speed_scale 0.25
Real robot: e-stop in reach, speed slider low, --lift_m 0.10 first, --gripper robotiq only once the arm part is clean.
"""
import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'vla_bridge'))
from vla_bridge.gripper_mapping import GripperMap  # noqa: E402
from vla_bridge.joint_streamer import JointRowStreamer, segment_durations  # noqa: E402
from vla_bridge.safety_limits import SafetyLimits  # noqa: E402
from vla_bridge.ur5e_kinematics import flange  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='127.0.0.1')
ap.add_argument('--dataset', required=True)
ap.add_argument('--episode', type=int, default=0)
ap.add_argument('--limits', default=None, help='safety_limits JSON (scripts/derive_safety_limits.py)')
ap.add_argument('--no_limits', action='store_true', help='URSim only')
ap.add_argument('--speed_scale', type=float, default=0.25)
ap.add_argument('--max_joint_speed', type=float, default=1.0)
ap.add_argument('--row_dt', type=float, default=1.0 / 60.0)
ap.add_argument('--k', type=int, default=25, help='rows per block, as the policy executes them')
ap.add_argument('--pause_s', type=float, default=0.0, help='pause between blocks (emulates inference time)')
ap.add_argument('--lift_m', type=float, default=0.0)
ap.add_argument('--approach_speed', type=float, default=0.3, help='moveJ to the first row, rad/s')
ap.add_argument('--lookahead', type=float, default=0.1)
ap.add_argument('--gain', type=float, default=300)
ap.add_argument('--gripper', choices=['none', 'robotiq'], default='none')
ap.add_argument('--gripper_calibration', type=float, nargs='+', default=[0.0, 0.0, 1.0, 255.0])
ap.add_argument('--dry_run', action='store_true')
ap.add_argument('--yes', action='store_true', help='do not ask before moving')
ap.add_argument('--json', default=None)
a = ap.parse_args()


def load_episode(root, ep):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(root, 'data', 'chunk-*', '*.parquet'))):
        t = pq.read_table(f, columns=['joints', 'actions', 'episode_index']).to_pandas()
        t = t[t.episode_index == ep]
        if len(t):
            return np.stack(t.joints.to_numpy()).astype(float), np.stack(t.actions.to_numpy()).astype(float)
    sys.exit(f'episode {ep} not found under {root}')


def lift_rows(q_rows, dz, iters=200, damping=1e-4):
    """Same flange orientation, flange position + dz (UR base z), by damped least squares from the previous solution.
    Near the fully stretched start pose (q = 0) a lifted flange is out of reach; there the damped solution is the closest
    reachable pose (less lift), so the achieved lift per row is returned and reported rather than required."""
    out, q, lift, rot_res = [], q_rows[0].copy(), [], []
    for row in q_rows:
        T = flange(row)[0]
        z0 = T[2, 3]
        T[2, 3] += dz
        for _ in range(iters):
            F = flange(q)[0]
            e = np.concatenate([T[:3, 3] - F[:3, 3],
                                0.5 * (np.cross(F[:3, 0], T[:3, 0]) + np.cross(F[:3, 1], T[:3, 1]) + np.cross(F[:3, 2], T[:3, 2]))])
            if np.linalg.norm(e) < 1e-7:
                break
            Jn = np.zeros((6, 6))
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = 1e-6
                G = flange(q + dq)[0]
                Jn[:3, j] = (G[:3, 3] - F[:3, 3]) / 1e-6
                Rd = G[:3, :3] @ F[:3, :3].T
                Jn[3:, j] = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / 2e-6
            q = q + Jn.T @ np.linalg.solve(Jn @ Jn.T + damping * np.eye(6), e)
        F = flange(q)[0]
        lift.append(F[2, 3] - z0)
        rot_res.append(np.linalg.norm(0.5 * sum(np.cross(F[:3, i], T[:3, i]) for i in range(3))))
        out.append(q.copy())
    return np.array(out), np.array(lift), np.array(rot_res)


joints, actions = load_episode(os.path.expanduser(a.dataset), a.episode)
rows, grip = actions[:, :6], actions[:, 6]
if a.lift_m:
    rows, lift, rot_res = lift_rows(rows, a.lift_m)
    part = np.nonzero(lift < 0.99 * a.lift_m)[0]
    print(f'lifted by {a.lift_m} m: achieved lift min {lift.min():.3f} / median {np.median(lift):.3f} m; '
          f'{len(part)} rows partly lifted (first {part[:1].tolist()}, last {part[-1:].tolist()}); '
          f'orientation residual max {rot_res.max():.1e} rad; joint change max {np.abs(rows - actions[:, :6]).max():.3f} rad')
    if rot_res.max() > 0.01:
        sys.exit('lift IK changed the tool orientation by > 0.01 rad on some row; try a smaller --lift_m')
q0 = rows[0]
limits = SafetyLimits.from_json(a.limits) if a.limits else None
if limits is None and not a.no_limits:
    sys.exit('give --limits, or --no_limits (URSim only)')
if limits is not None:
    if a.lift_m and limits.tool_box_max is not None:
        limits.tool_box_max = limits.tool_box_max + np.array([0, 0, a.lift_m])
    bad = limits.check_chunk(q0, rows)
    if bad:
        sys.exit('refused by safety limits:\n  ' + '\n  '.join(bad))
durs = segment_durations(q0, rows, a.row_dt, a.max_joint_speed, a.speed_scale)
print(f'episode {a.episode}: {len(rows)} rows, nominal {len(rows) * a.row_dt:.1f} s at 60 Hz -> planned {durs.sum():.1f} s '
      f'(speed_scale {a.speed_scale}, {int(np.sum(durs > a.row_dt / a.speed_scale + 1e-9))} rows slowed by the '
      f'{a.max_joint_speed} rad/s cap); blocks of {a.k}; gripper {a.gripper}')
if a.dry_run:
    sys.exit(0)

import rtde_control  # noqa: E402
import rtde_receive  # noqa: E402

ctrl = rtde_control.RTDEControlInterface(a.host, 500.0)
recv = rtde_receive.RTDEReceiveInterface(a.host, 500.0)
gripper = gmap = None
if a.gripper == 'robotiq':
    from vla_bridge.robotiq_socket_gripper import RobotiqSocketGripper
    gripper, gmap = RobotiqSocketGripper(a.host), GripperMap.from_flat(a.gripper_calibration)
    gripper.activate()
q_now = np.array(recv.getActualQ())
print(f'now {np.round(q_now, 3).tolist()}\nfirst row {np.round(q0, 3).tolist()}  (max diff {np.abs(q_now - q0).max():.3f} rad)')
if not a.yes and input('moveJ to the first row and replay? [y/N] ').strip().lower() != 'y':
    sys.exit('aborted')
ctrl.moveJ(list(q0), a.approach_speed, 0.5)

log, stop = [], threading.Event()


def record():
    while not stop.is_set():
        log.append((time.monotonic(), recv.getActualQ(), recv.getTargetQ()))
        time.sleep(0.002)


rec = threading.Thread(target=record, daemon=True)
streamer = JointRowStreamer(ctrl, lookahead_time=a.lookahead, gain=a.gain, v_max=a.max_joint_speed, speed_scale=a.speed_scale)
last_counts = [None]


def on_row_for(offset):
    def on_row(i):
        if gripper is None:
            return
        c = gmap.to_counts(grip[offset + i])
        if last_counts[0] is None or abs(c - last_counts[0]) >= 2:
            gripper.set_position(c)
            last_counts[0] = c
    return on_row


cmd_t, stats = [], []
rec.start()
t_start = time.monotonic()
try:
    q_prev = q0
    for s in range(0, len(rows), a.k):
        block = rows[s:s + a.k]
        cmd_t.append((time.monotonic(), s))
        stats.append(streamer.execute(q_prev, block, a.row_dt, on_row=on_row_for(s)))
        q_prev = block[-1]
        if a.pause_s:
            t0 = time.monotonic()
            while time.monotonic() - t0 < a.pause_s:   # keep servoing the hold point, as the client does between chunks
                p = ctrl.initPeriod()
                ctrl.servoJ(list(q_prev), 0, 0, 0.002, a.lookahead, a.gain)
                ctrl.waitPeriod(p)
    t_total = time.monotonic() - t_start
    time.sleep(0.5)
finally:
    stop.set()
    ctrl.servoStop()
    ctrl.stopScript()
rec.join()

T = np.array([t for t, _, _ in log]) - t_start
Q = np.array([q for _, q, _ in log])
# commanded path as a function of time (what the streamer sent), rebuilt from the planned durations and the block starts
cmd_times, cmd_rows = [], []
for (tb, s), st in zip(cmd_t, stats):
    d = segment_durations(rows[s - 1] if s else q0, rows[s:s + a.k], a.row_dt, a.max_joint_speed, a.speed_scale)
    cmd_times.extend((tb - t_start) + np.cumsum(d))
    cmd_rows.extend(rows[s:s + a.k])
cmd_times, cmd_rows = np.array(cmd_times), np.array(cmd_rows)
# for each row: where the arm actually was when the streamer reached it, and lookahead later
at_row = np.array([Q[np.searchsorted(T, t).clip(0, len(T) - 1)] for t in cmd_times])
at_row_lag = np.array([Q[np.searchsorted(T, t + a.lookahead).clip(0, len(T) - 1)] for t in cmd_times])
err, err_lag = np.abs(at_row - cmd_rows).max(1), np.abs(at_row_lag - cmd_rows).max(1)
vel = np.abs(np.diff(Q, axis=0)) / np.maximum(np.diff(T)[:, None], 1e-4)
res = dict(episode=a.episode, rows=len(rows), speed_scale=a.speed_scale, max_joint_speed=a.max_joint_speed, lift_m=a.lift_m,
           k=a.k, pause_s=a.pause_s, nominal_s=float(len(rows) * a.row_dt / a.speed_scale), planned_s=float(durs.sum()),
           wall_s=float(t_total), slowed_rows=int(sum(s['slowed_rows'] for s in stats)),
           track_err_at_row_rad=dict(median=float(np.median(err)), p95=float(np.percentile(err, 95)), max=float(err.max())),
           track_err_lookahead_later_rad=dict(median=float(np.median(err_lag)), p95=float(np.percentile(err_lag, 95)),
                                              max=float(err_lag.max())),
           final_err_rad=float(np.abs(Q[-1] - rows[-1]).max()),
           peak_joint_speed_rad_s=np.percentile(vel, 99.5, axis=0).round(3).tolist(), samples=len(T))
print(json.dumps(res, indent=1))
if a.json:
    with open(a.json, 'w') as f:
        json.dump(res, f, indent=1)
