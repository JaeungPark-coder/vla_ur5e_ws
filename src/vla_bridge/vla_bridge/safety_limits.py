"""Pre-execution checks for a chunk on the real arm: joint box, per-row step, tool-point workspace box.

The limits come from the demonstrations the policy was trained on (scripts/derive_safety_limits.py writes the JSON), not from
the robot's own limits: a policy that leaves the region its demonstrations covered is extrapolating, and the first runs should
stop rather than find out.  A chunk that fails any check is refused as a whole, before any row of it is sent.

JSON keys: joint_min, joint_max, max_row_step (rad, 6 each); tool_offset_in_flange (m, 3); tool_box_min, tool_box_max
(m, 3, UR base frame, or null to skip).  Anything else in the file (provenance) is ignored here.
"""
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vla_bridge.ur5e_kinematics import tool_point


@dataclass
class SafetyLimits:
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_row_step: np.ndarray
    tool_offset_in_flange: np.ndarray
    tool_box_min: Optional[np.ndarray] = None
    tool_box_max: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, d):
        arr = lambda k, n: None if d.get(k) is None else np.asarray(d[k], dtype=float).reshape(n)  # noqa: E731
        lim = cls(arr('joint_min', 6), arr('joint_max', 6), arr('max_row_step', 6),
                  arr('tool_offset_in_flange', 3) if d.get('tool_offset_in_flange') is not None else np.zeros(3),
                  arr('tool_box_min', 3), arr('tool_box_max', 3))
        if lim.joint_min is None or lim.joint_max is None or lim.max_row_step is None:
            raise ValueError('safety limits need joint_min, joint_max and max_row_step')
        if np.any(lim.joint_min >= lim.joint_max) or np.any(lim.max_row_step <= 0):
            raise ValueError('safety limits are inconsistent (joint_min >= joint_max or max_row_step <= 0)')
        return lim

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def check_chunk(self, q_start, rows):
        """List of human-readable violations, empty if the chunk may run.  q_start is where the first row is reached from."""
        rows = np.atleast_2d(np.asarray(rows, dtype=float))[:, :6]
        out = []
        if not np.all(np.isfinite(rows)):
            return ['non-finite joint target']
        lo, hi = rows < self.joint_min, rows > self.joint_max
        for r, j in zip(*np.nonzero(lo | hi)):
            out.append(f'row {r} joint {j} = {rows[r, j]:+.3f} outside [{self.joint_min[j]:+.3f}, {self.joint_max[j]:+.3f}]')
            if len(out) >= 5:
                break
        prev = np.vstack([np.asarray(q_start, dtype=float)[None, :6], rows[:-1]])
        step = np.abs(rows - prev)
        for r, j in zip(*np.nonzero(step > self.max_row_step)):
            out.append(f'row {r} joint {j} step {step[r, j]:.4f} rad > {self.max_row_step[j]:.4f}')
            if len(out) >= 10:
                break
        if self.tool_box_min is not None and self.tool_box_max is not None:
            p = tool_point(rows, self.tool_offset_in_flange)
            bad = np.nonzero(np.any((p < self.tool_box_min) | (p > self.tool_box_max), axis=1))[0]
            for r in bad[:3]:
                out.append(f'row {r} tool point {np.round(p[r], 3).tolist()} outside box '
                           f'{np.round(self.tool_box_min, 3).tolist()}..{np.round(self.tool_box_max, 3).tolist()}')
        return out
