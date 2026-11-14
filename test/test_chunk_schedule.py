"""chunk_schedule.plan_iteration: tick-level latency emulation (sync / async)."""
import pytest

from vla_bridge.chunk_schedule import HOLD, NEW, PREV, plan_iteration


@pytest.mark.parametrize('mode', ['sync', 'async'])
def test_zero_delay_is_the_original_behaviour(mode):
    steps, nxt = plan_iteration(mode, 25, 0, 50, prev_pos=None)
    assert steps == [(NEW, i) for i in range(25)] and nxt == 25
    steps, _ = plan_iteration(mode, 25, 0, 50, prev_pos=25)
    assert steps == [(NEW, i) for i in range(25)]


def test_sync_holds_d_ticks_then_runs_k_rows_from_row_zero():
    steps, _ = plan_iteration('sync', 25, 10, 50, prev_pos=25)
    assert steps[:10] == [(HOLD, None)] * 10
    assert steps[10:] == [(NEW, i) for i in range(25)]


def test_async_first_iteration_stands_then_starts_at_row_zero():
    steps, nxt = plan_iteration('async', 25, 10, 50, prev_pos=None)
    assert steps == [(HOLD, None)] * 10 + [(NEW, i) for i in range(15)]
    assert len(steps) == 25 and nxt == 15


def test_async_runs_the_previous_tail_then_skips_the_first_d_rows():
    steps, nxt = plan_iteration('async', 25, 10, 50, prev_pos=25)
    assert steps == [(PREV, 25 + i) for i in range(10)] + [(NEW, 10 + i) for i in range(15)]
    assert nxt == 25


def test_async_rows_line_up_with_ticks_over_several_iterations():
    """Every chunk row j must be applied exactly at tick (request tick + j) after the first chunk."""
    K, d = 25, 10
    tick, prev_pos, req_ticks, applied = 0, None, [], []
    for it in range(4):
        req_ticks.append(tick)
        steps, prev_pos = plan_iteration('async', K, d, 50, prev_pos)
        for src, row in steps:
            if src == NEW:
                applied.append((it, row, tick))
            elif src == PREV:
                applied.append((it - 1, row, tick))
            tick += 1
    for it, row, t in applied:
        if it >= 1:
            assert t == req_ticks[it] + row
        else:                       # the first chunk started d ticks late from rest
            assert t == req_ticks[0] + d + row
    assert tick == 4 * K


def test_async_rejects_impossible_combinations():
    with pytest.raises(ValueError):
        plan_iteration('async', 40, 20, 50, prev_pos=None)   # K + d > chunk
    with pytest.raises(ValueError):
        plan_iteration('async', 10, 10, 50, prev_pos=None)   # d >= K
    with pytest.raises(ValueError):
        plan_iteration('later', 25, 10, 50)
