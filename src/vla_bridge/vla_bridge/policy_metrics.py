"""Offline comparison of a policy's action chunk with the recorded one at the SAME time step.

Both inputs are (n, 7) arrays of ABSOLUTE joint targets + gripper, i.e. exactly what the server returns after
AbsoluteActions and what the dataset stores in `actions`. Definitions (written down after a wrong comparison on
2026-11-10, where a model chunk was set against a recorded window from a near-stationary stretch of the episode):

  err_row0 / err_row24   max over joints of |model - recorded| at chunk row 0 / 24, in rad
  err_rms                RMS over all rows and joints, in rad
  step_mean_*            mean |row[i+1]-row[i]| per joint, in rad per tick   (how fast the chunk moves)
  jerk_ratio_*           mean |second difference| / mean |first difference|  (jaggedness relative to its own speed;
                         scale-free, so a slow window and a fast window are comparable)

Judge jitter with jerk_ratio over many windows (model vs recorded at the same t), not with raw step sizes. The ratio explodes
when the recorded window is nearly stationary, so summarize_windows() only uses windows with enough recorded motion.
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


def summarize_windows(windows, min_motion=5e-4):
    """windows: list of chunk_comparison() dicts (one per probed (episode, t)). Errors are summarised over all windows;
    jitter only over windows whose RECORDED step_mean is >= min_motion rad/tick (a stationary window makes the ratio
    meaningless). `jerk_ratio_pair` is the median over windows of jerk_ratio_model / jerk_ratio_rec, i.e. a same-t pairing."""
    if not windows:
        raise ValueError("no windows")
    med = lambda xs: float(np.nanmedian(xs)) if len(xs) else float("nan")  # noqa: E731
    moving = [w for w in windows if w["step_mean_rec"] >= min_motion]
    pair = [w["jerk_ratio_model"] / w["jerk_ratio_rec"] for w in moving
            if w["jerk_ratio_rec"] and np.isfinite(w["jerk_ratio_rec"]) and w["jerk_ratio_rec"] > 1e-9]
    return {
        "windows": len(windows), "windows_moving": len(moving), "min_motion": min_motion,
        "err_row0": med([w["err_row0"] for w in windows]), "err_row24": med([w["err_row24"] for w in windows]),
        "err_rms": med([w["err_rms"] for w in windows]),
        "gripper_mean_abs_err": med([w["gripper_mean_abs_err"] for w in windows]),
        "step_ratio_model_over_rec": med([w["step_mean_model"] / w["step_mean_rec"] for w in moving]),
        "jerk_ratio_model": med([w["jerk_ratio_model"] for w in moving]),
        "jerk_ratio_rec": med([w["jerk_ratio_rec"] for w in moving]),
        "jerk_ratio_pair": med(pair),
    }
