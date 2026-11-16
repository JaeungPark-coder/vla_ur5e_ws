#!/usr/bin/env python3
"""Checks a v5 expert dataset against what v5 is for (README "v5 decision"), from the recorded joints alone (no Isaac):

  1. joint speed: per-row change of the recorded next-joint actions, as rad/s at 60 rows/s, against the expert cap
     (sim_profile v5: pi/2) -- v4 peaked at 4.9 rad/s on wrist_3;
  2. coaxial grasp: the tool axis (tool0 z = wrist_3 axis, UR DH FK) at the first close command must point down
     (v1-v4: 89.6 deg from down, median over 89 grasps -- the BLOCKER);
  3. the fingertip at that moment (FK + --tcp_m along the tool axis), so the grasp heights can be compared with the cube.

    python3 scripts/v5_expert_check.py <lerobot dataset dir> [<more dirs> ...] [--tcp_m 0.12] [--cap 1.5708] [--json out.json]
Pass: every episode's peak speed <= cap x 1.05 on every joint, and every grasp axis within --axis_tol_deg of straight down.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'vla_bridge'))
from vla_bridge.ur5e_kinematics import flange  # noqa: E402

WORLD_FROM_BASE = np.diag([-1.0, -1.0, 1.0])
ap = argparse.ArgumentParser()
ap.add_argument('dataset', nargs='+')
ap.add_argument('--tcp_m', type=float, default=0.12)
ap.add_argument('--cap', type=float, default=np.pi / 2)
ap.add_argument('--row_hz', type=float, default=60.0)
ap.add_argument('--axis_tol_deg', type=float, default=5.0)
ap.add_argument('--json', default=None)
a = ap.parse_args()

eps = []
for ds, f in [(d, f) for d in a.dataset for f in sorted(glob.glob(os.path.join(d, 'data', 'chunk-*', '*.parquet')))]:
    t = pq.read_table(f, columns=['joints', 'actions', 'episode_index']).to_pandas()
    for e, ep in t.groupby('episode_index'):
        j = np.stack(ep['joints'].to_numpy()).astype(float)
        act = np.stack(ep['actions'].to_numpy()).astype(float)
        speed = np.abs(np.diff(np.vstack([j[:1], act[:, :6]]), axis=0)) * a.row_hz
        c = np.nonzero(act[:, 6] > 0.5)[0]
        rec = dict(dataset=os.path.basename(os.path.normpath(ds)), episode=int(e), frames=len(j), peak_speed=speed.max(0).round(3).tolist(),
                   p99_speed=np.percentile(speed, 99, axis=0).round(3).tolist())
        if len(c):
            T = flange(j[c[0]])[0]
            z_w = WORLD_FROM_BASE @ T[:3, 2]
            rec.update(close_frame=int(c[0]), axis_from_down_deg=float(np.degrees(np.arccos(np.clip(-z_w[2], -1, 1)))),
                       fingertip_world=(WORLD_FROM_BASE @ (T[:3, 3] + a.tcp_m * T[:3, 2])).round(4).tolist())
        eps.append(rec)
if not eps:
    sys.exit(f'no episodes under {a.dataset}')
peak = np.array([r['peak_speed'] for r in eps])
axis = np.array([r.get('axis_from_down_deg', np.nan) for r in eps])
speed_ok = bool(np.all(peak <= a.cap * 1.05))
axis_ok = bool(np.all(np.isfinite(axis)) and np.all(axis <= a.axis_tol_deg))
summary = dict(datasets=[os.path.abspath(d) for d in a.dataset], episodes=len(eps), cap_rad_s=a.cap,
               peak_speed_max=peak.max(0).round(3).tolist(), peak_speed_median=np.median(peak, 0).round(3).tolist(),
               episodes_over_cap=int(np.sum(np.any(peak > a.cap * 1.05, axis=1))),
               grasp_axis_from_down_deg=dict(median=float(np.nanmedian(axis)), max=float(np.nanmax(axis)),
                                             missing=int(np.sum(~np.isfinite(axis)))),
               fingertip_z_at_close=dict(median=float(np.median([r['fingertip_world'][2] for r in eps if 'fingertip_world' in r])))
               if np.any(np.isfinite(axis)) else None,
               speed_ok=speed_ok, coaxial_grasp_ok=axis_ok, PASS=speed_ok and axis_ok)
print(json.dumps(summary, indent=1))
if a.json:
    with open(a.json, 'w') as f:
        json.dump(dict(summary=summary, episodes=eps), f, indent=1)
