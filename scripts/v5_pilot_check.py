#!/usr/bin/env python3
"""v5 PILOT check (README 2026-11-16 night) -- a pipeline test, NOT a result: does the v5 data and the closed loop start where
v5 says (sim_profile.READY_JOINTS) and does the gripper behave as intended.

Dataset (each episode):  first recorded joints within --tol of READY_JOINTS; gripper open at the start (<= 0.05); exactly one
    close (action crossing 0.5 upward) followed by one open; at the first close the tool axis (tool0 z, UR DH FK) <= 5 deg
    from straight down and the fingertip (FK + --tcp_m along it) within 3 cm of the cube-centre height.
Evaluation traces (each trial):  first joints within --tol of READY_JOINTS; gripper open at the start; and -- reported, not
    required, a 1k-step policy may well miss -- whether it closed, and the fingertip-to-cube distance at the first close.
    python3 scripts/v5_pilot_check.py --datasets <dir> ... --eval_dir eval_runs/v5pilot_1k_centre --out pilot_check.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'vla_bridge'))
sys.path.insert(0, os.path.join(HERE, '..', 'isaac'))
os.environ['VLA_SIM_PROFILE'] = 'v5'
import sim_profile  # noqa: E402
from vla_bridge.ur5e_kinematics import flange  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--datasets', nargs='*', default=[])
ap.add_argument('--eval_dir', default=None)
ap.add_argument('--tcp_m', type=float, default=float(sim_profile.GRIPPER_TCP_M))
ap.add_argument('--tol', type=float, default=0.02, help='rad, every joint')
ap.add_argument('--out', required=True)
a = ap.parse_args()

READY = np.asarray(sim_profile.READY_JOINTS, float)
W = np.diag([-1.0, -1.0, 1.0])                       # UR base -> Isaac world
CUBE_Z = float(sim_profile.TABLE_Z_M + sim_profile.CUBE_SIZE_M / 2)


def edges(g, hi=0.5, lo=0.3):
    """Close / open events with hysteresis: the measured value sits at ~0.50 while the fingers release (v4 data: 33 of 302
    episodes cross 0.5 several times there), so closed = above hi (0.5; a held cube reads ~0.58 in the closed loop), open again = below lo."""
    up, down, closed = [], [], False
    for i, v in enumerate(np.asarray(g, float)):
        if not closed and v > hi:
            closed = True
            up.append(i)
        elif closed and v < lo:
            closed = False
            down.append(i)
    return np.array(up, int), np.array(down, int)


ds_rows = []
for d in a.datasets:
    for f in sorted(glob.glob(os.path.join(d, 'data', 'chunk-*', '*.parquet'))):
        t = pq.read_table(f, columns=['joints', 'gripper', 'actions', 'episode_index']).to_pandas()
        for e, ep in t.groupby('episode_index'):
            j = np.stack(ep['joints'].to_numpy()).astype(float)
            g0 = float(np.ravel(ep['gripper'].iloc[0])[0])
            act = np.stack(ep['actions'].to_numpy()).astype(float)
            up, down = edges(act[:, 6])
            r = dict(dataset=os.path.basename(os.path.normpath(d)), episode=int(e), frames=len(j),
                     start_err_rad=float(np.abs(j[0] - READY).max()), gripper_start=g0,
                     closes=len(up), opens=len(down))
            if len(up):
                T = flange(j[up[0]])[0]
                r['axis_from_down_deg'] = float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1))))
                r['fingertip_z_minus_cube_z_m'] = float(T[2, 3] + a.tcp_m * T[2, 2] - CUBE_Z)
            r['ok'] = bool(r['start_err_rad'] <= a.tol and g0 <= 0.05 and r['closes'] == 1 and r['opens'] == 1
                           and len(down) and down[0] > up[0] and r.get('axis_from_down_deg', 99) <= 5.0
                           and abs(r.get('fingertip_z_minus_cube_z_m', 9)) <= 0.03)
            ds_rows.append(r)

ev_rows = []
if a.eval_dir:
    for f in sorted(glob.glob(os.path.join(a.eval_dir, 'trace', 'trial_*.npz'))):
        if f.endswith('_images.npz'):
            continue
        z = np.load(f)
        j, g, cube, grip = z['joints'], z['gripper'], z['cube'], z['grip_point']
        up, down = edges(np.asarray(g, float))
        r = dict(trial=os.path.basename(f)[:-4], ticks=len(j), start_err_rad=float(np.abs(j[0] - READY).max()),
                 gripper_start=float(g[0]), closed=bool(len(up)), opens_after_close=int(np.sum(down > up[0])) if len(up) else 0)
        if len(up):
            k = up[0]
            r['close_tick'] = int(k)
            r['fingertip_cube_xy_mm'] = float(np.linalg.norm(grip[k, :2] - cube[k, :2]) * 1000)
            r['fingertip_minus_cube_z_mm'] = float((grip[k, 2] - cube[k, 2]) * 1000)
        r['start_ok'] = bool(r['start_err_rad'] <= a.tol and r['gripper_start'] <= 0.05)
        ev_rows.append(r)
    res_csv = os.path.join(a.eval_dir, 'results.csv')
    successes = None
    if os.path.exists(res_csv):
        lines = open(res_csv).read().strip().splitlines()[1:]
        successes = sum(1 for ln in lines if ln.split(',')[1] == 'True'), len(lines)

summary = dict(
    PILOT='v5 pipeline test run -- NOT for conclusions (README 2026-11-16 night)',
    ready_joints=READY.tolist(), tol_rad=a.tol, tcp_m=a.tcp_m,
    dataset=dict(episodes=len(ds_rows), ok=sum(r['ok'] for r in ds_rows),
                 start_err_max_rad=max((r['start_err_rad'] for r in ds_rows), default=None),
                 gripper_start_max=max((r['gripper_start'] for r in ds_rows), default=None),
                 close_count_hist={str(k): sum(r['closes'] == k for r in ds_rows) for k in sorted({r['closes'] for r in ds_rows})},
                 axis_at_close_max_deg=max((r.get('axis_from_down_deg', np.nan) for r in ds_rows), default=None),
                 fingertip_vs_cube_z_at_close_m=dict(
                     median=float(np.nanmedian([r.get('fingertip_z_minus_cube_z_m', np.nan) for r in ds_rows])) if ds_rows else None)),
    evaluation=None if not a.eval_dir else dict(
        trials=len(ev_rows), start_ok=sum(r['start_ok'] for r in ev_rows),
        start_err_max_rad=max((r['start_err_rad'] for r in ev_rows), default=None),
        closed=sum(r['closed'] for r in ev_rows),
        fingertip_cube_xy_mm_at_close_median=float(np.median([r['fingertip_cube_xy_mm'] for r in ev_rows if r['closed']]))
        if any(r['closed'] for r in ev_rows) else None,
        successes_of_trials=successes),
)
summary['start_pose_as_intended'] = bool(ds_rows and all(r['start_err_rad'] <= a.tol for r in ds_rows)
                                         and all(r['start_ok'] for r in ev_rows))
summary['gripper_as_intended_in_data'] = bool(ds_rows and all(r['ok'] for r in ds_rows))
print(json.dumps(summary, indent=1))
with open(a.out, 'w') as f:
    json.dump(dict(summary=summary, dataset_episodes=ds_rows, eval_trials=ev_rows), f, indent=1)
