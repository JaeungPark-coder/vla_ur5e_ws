"""policy_metrics.chunk_comparison -- the definitions the 3k-run analysis will rely on."""
import numpy as np
import pytest

from vla_bridge.policy_metrics import chunk_comparison, summarize_windows


def _ramp(n=50, speed=0.002):
    return np.stack([np.arange(n) * speed * (j + 1) for j in range(6)] + [np.zeros(n)], axis=1)


def test_identical_chunks_have_zero_error():
    a = _ramp()
    r = chunk_comparison(a, a)
    assert r["err_rms"] == 0 and r["err_row0"] == 0 and r["err_row24"] == 0


def test_error_is_in_radians_and_max_over_joints_for_the_rows():
    rec = _ramp()
    model = rec.copy()
    model[0, 2] += 0.05
    model[24, 4] -= 0.2
    r = chunk_comparison(model, rec)
    assert r["err_row0"] == pytest.approx(0.05) and r["err_row24"] == pytest.approx(0.2)


def test_a_smooth_ramp_has_zero_jerk_ratio_and_a_zigzag_has_a_large_one():
    smooth = _ramp()
    zigzag = smooth.copy()
    zigzag[1::2, :6] += 0.004          # alternate rows displaced
    assert chunk_comparison(smooth, smooth)["jerk_ratio_model"] == pytest.approx(0.0, abs=1e-9)
    assert chunk_comparison(zigzag, smooth)["jerk_ratio_model"] > 1.0


def test_jerk_ratio_is_scale_free_so_a_slow_and_a_fast_window_compare():
    # the 2026-11-10 mistake: comparing raw step sizes of a moving chunk with a nearly stationary one
    slow, fast = _ramp(speed=0.0001), _ramp(speed=0.01)
    noise = np.random.default_rng(0).normal(0, 1, (50, 7))
    rs = chunk_comparison(slow + 0.00001 * noise, slow)
    rf = chunk_comparison(fast + 0.001 * noise, fast)
    assert rs["step_mean_rec"] < rf["step_mean_rec"] / 50          # raw sizes differ by ~100x ...
    assert rs["jerk_ratio_model"] == pytest.approx(rf["jerk_ratio_model"], rel=0.5)  # ... the ratio does not


def test_a_stationary_recording_gives_nan_not_a_division_error():
    still = np.zeros((50, 7))
    assert np.isnan(chunk_comparison(still, still)["jerk_ratio_rec"])


def test_too_short_a_chunk_is_rejected():
    with pytest.raises(ValueError):
        chunk_comparison(np.zeros((3, 7)), np.zeros((3, 7)))


def _win(step_rec, jr_model, jr_rec, err=0.01):
    return {"step_mean_rec": step_rec, "step_mean_model": step_rec * 1.1, "jerk_ratio_model": jr_model,
            "jerk_ratio_rec": jr_rec, "err_row0": err, "err_row24": err, "err_rms": err, "gripper_mean_abs_err": 0.0}


def test_stationary_windows_are_left_out_of_the_jitter_summary_but_not_the_error_summary():
    ws = [_win(0.002, 0.4, 0.2), _win(0.003, 0.6, 0.2), _win(0.00008, 500.0, float("nan"), err=0.5)]
    s = summarize_windows(ws)
    assert s["windows"] == 3 and s["windows_moving"] == 2
    assert s["jerk_ratio_model"] == pytest.approx(0.5)          # the exploded stationary value is not in it
    assert s["jerk_ratio_pair"] == pytest.approx(2.5)           # median of 2.0 and 3.0, paired per window
    assert s["err_rms"] == pytest.approx(0.01)                  # errors still use every window


def test_no_moving_windows_gives_nan_not_an_exception():
    s = summarize_windows([_win(0.0001, 1.0, 1.0)])
    assert s["windows_moving"] == 0 and np.isnan(s["jerk_ratio_pair"])
