#!/usr/bin/env python3
"""Safety limits for the real arm, derived from the demonstrations a policy was trained on (vla_bridge/safety_limits.py).

    python3 scripts/derive_safety_limits.py <lerobot dataset dir> --tool_offset 0 0 <TCP m> [--table_z <m>] --out limits.json

  joint box     = min/max over every recorded joint vector and action row, widened by --joint_margin (rad)
  max_row_step  = per-joint largest change between consecutive action rows, x --step_factor (floor --step_floor)
  tool box      = min/max of the tool point (FK, UR base frame, --tool_offset in the flange frame) over every action row,
                  widened by --tool_margin; with --table_z the floor is table_z + --clearance instead of the data minimum
The tool offset has to be the REAL gripper's (coaxial: 0 0 <TCP length>) once the data come from a coaxially mounted sim
gripper (v5).  v1-v4 data were recorded with the sim gripper 90 deg off (0 0.12 0 in the DH flange frame; README BLOCKER).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'vla_bridge'))
from vla_bridge.ur5e_kinematics import tool_point  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('dataset')
ap.add_argument('--tool_offset', type=float, nargs=3, required=True)
ap.add_argument('--joint_margin', type=float, default=0.15)
ap.add_argument('--step_factor', type=float, default=1.5)
ap.add_argument('--step_floor', type=float, default=0.01)
ap.add_argument('--tool_margin', type=float, default=0.05)
ap.add_argument('--table_z', type=float, default=None, help='table surface z in the UR base frame (m)')
ap.add_argument('--clearance', type=float, default=0.005)
ap.add_argument('--out', required=True)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dataset, 'data', 'chunk-*', '*.parquet')))
if not files:
    sys.exit(f'no parquet files under {a.dataset}/data')
J, ACT, steps = [], [], []
for f in files:
    t = pq.read_table(f, columns=['joints', 'actions', 'episode_index']).to_pandas()
    for _, ep in t.groupby('episode_index'):
        j = np.stack(ep['joints'].to_numpy()).astype(float)
        act = np.stack(ep['actions'].to_numpy()).astype(float)[:, :6]
        J.append(j)
        ACT.append(act)
        steps.append(np.abs(np.diff(act, axis=0)).max(axis=0))
J, ACT, steps = np.vstack(J), np.vstack(ACT), np.vstack(steps)
allq = np.vstack([J, ACT])
tp = tool_point(ACT, a.tool_offset)
box_min, box_max = tp.min(0) - a.tool_margin, tp.max(0) + a.tool_margin
if a.table_z is not None:
    box_min[2] = a.table_z + a.clearance
out = dict(
    joint_min=(allq.min(0) - a.joint_margin).round(4).tolist(),
    joint_max=(allq.max(0) + a.joint_margin).round(4).tolist(),
    max_row_step=np.maximum(steps.max(0) * a.step_factor, a.step_floor).round(4).tolist(),
    tool_offset_in_flange=list(a.tool_offset),
    tool_box_min=box_min.round(4).tolist(), tool_box_max=box_max.round(4).tolist(),
    provenance=dict(dataset=os.path.abspath(a.dataset), episodes=len(steps), frames=len(J),
                    joint_margin=a.joint_margin, step_factor=a.step_factor, step_floor=a.step_floor,
                    tool_margin=a.tool_margin, table_z=a.table_z, clearance=a.clearance,
                    data_max_row_step=steps.max(0).round(4).tolist(),
                    data_row_step_p99=np.percentile(steps, 99, axis=0).round(4).tolist(),
                    data_tool_min=tp.min(0).round(4).tolist(), data_tool_max=tp.max(0).round(4).tolist()))
with open(a.out, 'w') as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
