"""Offline comparison of a policy's action chunk with the recorded one at the SAME time step.

Both inputs are (n, 7) arrays of ABSOLUTE joint targets + gripper, i.e. exactly what the server returns after
AbsoluteActions and what the dataset stores in `actions`. Definitions (written down after a wrong comparison on
2026-11-10, where a model chunk was set against a recorded window from a near-stationary stretch of the episode):

  err_row0 / err_row24   max over joints of |model - recorded| at chunk row 0 / 24, in rad
  err_rms                RMS over all rows and joints, in rad
  step_mean_*            mean |row[i+1]-row[i]| per joint, in rad per tick   (how fast the chunk moves)
  jerk_ratio_*           mean |second difference| / mean |first difference|  (jaggedness relative to its own speed;
                         scale-free, so a slow window and a fast window are comparable)

Judge jitter with jerk_ratio over many windows (model vs recorded at the same t), not with raw step sizes.
"""
import numpy as np


def _step_and_jerk(rows):
    d = np.abs(np.diff(rows, axis=0))
    dd = np.abs(np.diff(rows, n=2, axis=0))
    step = float(d.mean())
    jerk = float(dd.mean())
    return step, (jerk / step if step > 1e-9 else float("nan"))


def chunk_comparison(model, recorded):
    model = np.asarray(model, dtype=float)
    recorded = np.asarray(recorded, dtype=float)
    n = min(len(model), len(recorded))
    if n < 4:
        raise ValueError(f"need at least 4 rows to compare, got {n}")
    m, r = model[:n, :6], recorded[:n, :6]
    err = np.abs(m - r)
    step_m, jr_m = _step_and_jerk(m)
    step_r, jr_r = _step_and_jerk(r)
    return {
        "rows": n,
        "err_row0": float(err[0].max()),
        "err_row24": float(err[24].max()) if n > 24 else float("nan"),
        "err_rms": float(np.sqrt((err ** 2).mean())),
        "gripper_mean_abs_err": float(np.abs(model[:n, 6] - recorded[:n, 6]).mean()),
        "step_mean_model": step_m, "step_mean_rec": step_r,
        "jerk_ratio_model": jr_m, "jerk_ratio_rec": jr_r,
    }
