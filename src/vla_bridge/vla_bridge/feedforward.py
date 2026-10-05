"""Joint-velocity feed-forward for serving, derived from the policy's own action chunk.

Why (measured 2026-10-05, isaac/check_action_replay.py): the scripted expert drove the
arm with RMPflow ArticulationActions that carry a position AND a velocity target every
tick (up to 2.2 rad/s). The serving path sends positions only. Replaying recorded
absolute targets that way lags the recording by ~0.30 rad RMS over an episode; adding
a velocity target v_t = (x_{t+1} - x_t) / dt brought it to 0.0002 rad (path length
ratio 1.00). A chunk's rows are consecutive target positions one tick apart, so the
velocity is just their difference.

The catch (measured the same day): a difference of consecutive rows amplifies row noise
-- noise of 0.005 rad on position targets made the replay with this feed-forward diverge.
`window` averages the per-tick differences over that many neighbouring ticks and
`max_speed` clips the result, which is what keeps a jittery chunk from commanding wild
velocities. `window=1, max_speed=None` is the plain difference.
"""
import numpy as np


def chunk_velocities(rows, dt, window=1, max_speed=None):
    """rows: (n, 6) consecutive absolute joint targets, one control tick apart.
    Returns (n, 6) velocities in rad/s. The per-tick difference
    d_j = (rows[j+1] - rows[j]) / dt is averaged over `window` ticks centred on i (clamped at
    the ends of the chunk), and clipped to +-max_speed when given. The last row has no
    successor, so it uses the last difference. None when there is only one row."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or len(rows) < 2:
        return None
    d = np.empty_like(rows)
    d[:-1] = (rows[1:] - rows[:-1]) / float(dt)
    d[-1] = d[-2]
    window = max(1, int(window))
    if window > 1:
        h = window // 2
        n = len(rows)
        v = np.empty_like(d)
        for i in range(n):
            lo, hi = max(0, i - h), min(n, i + h + 1)
            v[i] = d[lo:hi].mean(axis=0)
    else:
        v = d
    if max_speed is not None:
        v = np.clip(v, -float(max_speed), float(max_speed))
    return v
