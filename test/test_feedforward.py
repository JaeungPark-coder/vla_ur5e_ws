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
