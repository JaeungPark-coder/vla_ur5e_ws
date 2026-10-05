"""chunk_velocities: velocity = difference of consecutive chunk rows / dt."""
import numpy as np
import pytest

from vla_bridge.feedforward import chunk_velocities


def test_constant_speed_motion_gives_a_constant_velocity():
    dt = 1.0 / 60.0
    rows = np.stack([np.full(6, 0.01 * i) for i in range(5)])        # 0.01 rad per tick
    v = chunk_velocities(rows, dt)
    assert v.shape == (5, 6)
    assert np.allclose(v, 0.01 / dt)                                  # 0.6 rad/s everywhere


def test_the_last_row_repeats_the_previous_velocity():
    rows = np.array([[0.0] * 6, [0.1] * 6, [0.3] * 6])
    v = chunk_velocities(rows, 1.0)
    assert np.allclose(v[0], 0.1) and np.allclose(v[1], 0.2) and np.allclose(v[2], 0.2)


def test_each_joint_is_differenced_independently_and_sign_is_kept():
    rows = np.array([[0, 0, 0, 0, 0, 0], [0.01, -0.02, 0, 0.03, 0, -0.005]], dtype=float)
    v = chunk_velocities(rows, 0.5)
    assert np.allclose(v[0], [0.02, -0.04, 0, 0.06, 0, -0.01])


@pytest.mark.parametrize('rows', [np.zeros((1, 6)), np.zeros(6), np.zeros((0, 6))])
def test_nothing_to_difference_returns_none(rows):
    assert chunk_velocities(rows, 1 / 60) is None


def test_a_window_averages_neighbouring_differences_and_leaves_constant_speed_alone():
    dt = 1.0
    rows = np.stack([np.full(6, 0.1 * i) for i in range(9)])
    assert np.allclose(chunk_velocities(rows, dt, window=5), 0.1)


def test_a_window_smooths_a_noisy_chunk():
    rng = np.random.default_rng(0)
    clean = np.stack([np.full(6, 0.01 * i) for i in range(60)])
    noisy = clean + rng.normal(0, 0.005, clean.shape)
    err_raw = np.abs(chunk_velocities(noisy, 1 / 60) - 0.6).mean()
    err_smooth = np.abs(chunk_velocities(noisy, 1 / 60, window=11) - 0.6).mean()
    assert err_smooth < err_raw / 3


def test_the_ends_use_the_differences_that_exist():
    rows = np.array([[0.0] * 6, [1.0] * 6, [3.0] * 6])
    v = chunk_velocities(rows, 1.0, window=3)
    assert np.allclose(v[0], 1.5) and np.allclose(v[1], 5.0 / 3) and np.allclose(v[2], 2.0)


def test_speeds_are_clipped_symmetrically():
    rows = np.array([[0.0] * 6, [10.0, -10.0, 0, 0, 0, 0], [20.0, -20.0, 0, 0, 0, 0]])
    v = chunk_velocities(rows, 1.0, max_speed=3.0)
    assert np.allclose(v[:, 0], 3.0) and np.allclose(v[:, 1], -3.0)
