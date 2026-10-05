"""Joint-velocity feed-forward for serving, derived from the policy's own action chunk.

Why (measured 2026-10-05, isaac/check_action_replay.py): the scripted expert drove the
arm with RMPflow ArticulationActions that carry a position AND a velocity target every
tick (up to 2.2 rad/s). The serving path sends positions only. Replaying recorded
absolute targets that way lags the recording by ~0.30 rad RMS over an episode; adding
a velocity target v_t = (x_{t+1} - x_t) / dt brought it to 0.0002 rad (path length
ratio 1.00). A chunk's rows are consecutive target positions one tick apart, so the
velocity is just their difference.
"""
import numpy as np


def chunk_velocities(rows, dt):
    """rows: (n, 6) consecutive absolute joint targets, one control tick apart.
    Returns (n, 6) velocities in rad/s, v_i = (rows[i+1] - rows[i]) / dt; the last row
    repeats the previous velocity. None when there is only one row (nothing to
    difference)."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or len(rows) < 2:
        return None
    v = np.empty_like(rows)
    v[:-1] = (rows[1:] - rows[:-1]) / float(dt)
    v[-1] = v[-2]
    return v
