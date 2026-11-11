"""The fake policy's position lookup must not get stuck on a recorded standstill.

Found 2026-11-11: the expert recordings contain stretches where the arm waits for 90+ ticks (the gripper closes). Returning
the FIRST of those near-identical frames made the next 50 rows a standstill again, so the fake policy replayed the same
frame forever and the evaluation-harness check silently measured nothing.
"""
import importlib.util
import pathlib

import numpy as np

_PATH = pathlib.Path(__file__).parent / "fakes" / "openpi_client" / "websocket_client_policy.py"
_spec = importlib.util.spec_from_file_location("fake_ws_policy", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
match = _mod.WebsocketClientPolicy._match


def _episode(dwell=100):
    moving = np.stack([np.linspace(0, 1, 50)] * 6 + [np.zeros(50)], axis=1)
    still = np.repeat(moving[-1:], dwell, axis=0)
    after = np.stack([np.linspace(1, 2, 50)] * 6 + [np.full(50, 0.7)], axis=1)
    return np.concatenate([moving, still, after])


def test_the_last_frame_of_a_standstill_is_returned_so_the_next_rows_move_on():
    states = _episode(dwell=100)
    q = states[60]                          # somewhere inside the dwell (frames 49..149 are identical)
    idx, dist = match(states, q, 0, len(states))
    assert dist < 1e-9 and idx >= 140       # end of the dwell, not frame 49
    assert np.abs(states[idx + 5, :6] - q[:6]).max() > 0.01   # the rows that follow actually move


def test_a_moving_stretch_is_matched_to_the_nearest_frame():
    states = _episode()
    idx, _ = match(states, states[20], 0, len(states))
    assert idx == 20


def test_the_search_window_starts_where_asked():
    states = _episode()
    idx, _ = match(states, states[170], 150, 100)
    assert idx == 170
