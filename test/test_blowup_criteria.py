"""One blow-up definition for the trace and the client CSV (2026-11-17: a cube flung sideways was 0 in the CSV)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "vla_bridge"))
sys.path.insert(0, os.path.join(HERE, "..", "isaac"))

from vla_bridge.blowup_criteria import judge_tick, mask_to_reasons, reasons_to_mask  # noqa: E402
from vla_bridge.eval_tracking import MaxHeightTracker  # noqa: E402
import trial_trace  # noqa: E402


def test_sideways_fling_is_a_blowup_without_any_height():
    # v5 pilot seed 9214: cube ends at (0.715, 1.632, 0.02) -- never above 0.25 m
    assert judge_tick([0.715, 1.632, 0.02]) == ["far"]
    assert judge_tick([0.45, 0.0, 0.02]) == []


def test_jump_needs_the_previous_tick():
    assert judge_tick([0.45, 0.03, 0.02]) == []
    assert judge_tick([0.45, 0.03, 0.02], [0.45, 0.0, 0.02]) == ["jump"]


def test_mask_roundtrip_and_dict_input():
    m = reasons_to_mask({"far": 3, "jump": 1})
    assert sorted(mask_to_reasons(m)) == ["far", "jump"]
    assert reasons_to_mask([]) == 0


def test_trace_mask_reaches_the_tracker_and_survives_a_later_quiet_message():
    t = trial_trace.TrialTrace()
    t.start(1, cube=[0.45, 0.0, 0.02])
    t.add(1, [0.45, 0.0, 0.02], [0.4, 0, 0.1], np.zeros(6), 0.0)
    t.add(2, [0.80, 0.70, 0.02], [0.4, 0, 0.1], np.zeros(6), 0.0)   # flung: jump + far
    tr = MaxHeightTracker()
    tr.start_trial(1)
    assert tr.update(1, 0.02, t.reasons_mask)
    assert tr.update(1, 0.02)                    # an old two-field message does not clear it
    assert sorted(mask_to_reasons(tr.blowup_mask)) == ["far", "jump"]
    tr.start_trial(2)
    assert tr.blowup_mask == 0
    assert not tr.update(1, 0.5, 1)              # previous trial's epoch is ignored
    assert tr.blowup_mask == 0


def test_trace_judges_heights_above_the_table():
    t = trial_trace.TrialTrace(table_z=0.75)
    t.start(0, cube=[0.45, 0.0, 0.77])
    t.add(1, [0.45, 0.0, 0.77], [0.4, 0, 0.9], np.zeros(6), 0.0)   # resting on a 0.75 m table: fine
    assert not t.summary()["blowup"]
    t.add(2, [0.45, 0.0, 0.745], [0.4, 0, 0.9], np.zeros(6), 0.0)  # below the table top: fell (and a 25 mm jump)
    assert "fell" in t.summary()["reasons"]
