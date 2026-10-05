"""The policy's gripper output becomes a close/open decision, with hysteresis.

Context: the dataset's gripper action is the MEASURED position (about 0.7
while holding the cube, 0.75 at most), while the demonstrator commanded 1.0.
Echoing ~0.7 back as the command gives the PD drive ~0 squeeze force. See
vla_bridge/gripper_command.py. These tests pin the decision logic only -- they
cannot say whether 0.45/0.25 are the right thresholds; that needs the simulator.
"""
import pytest

from vla_bridge.gripper_command import GripperHysteresis


def test_the_value_recorded_while_holding_becomes_a_full_close_command():
    # 0.63-0.73 is what the 152 recorded episodes show while the cube is held.
    g = GripperHysteresis()
    for held in (0.63, 0.70, 0.73, 0.75):
        assert g.update(held) == 1.0


def test_fully_open_stays_open():
    g = GripperHysteresis()
    assert [g.update(0.0) for _ in range(5)] == [0.0] * 5


def test_hysteresis_band_holds_the_last_command():
    g = GripperHysteresis(close_above=0.45, open_below=0.25)
    # starts open; values inside the band do not close it
    assert g.update(0.30) == 0.0
    assert g.update(0.44) == 0.0
    assert g.update(0.50) == 1.0     # crosses close_above
    assert g.update(0.40) == 1.0     # back inside the band: stays closed
    assert g.update(0.26) == 1.0
    assert g.update(0.20) == 0.0     # crosses open_below
    assert g.update(0.40) == 0.0     # inside the band again: stays open


def test_a_value_hovering_near_one_threshold_cannot_make_it_chatter():
    g = GripperHysteresis(close_above=0.45, open_below=0.25)
    g.update(0.60)  # closed
    outputs = {g.update(v) for v in (0.34, 0.29, 0.31, 0.27, 0.33, 0.30)}
    assert outputs == {1.0}


def test_the_thresholds_are_exclusive_so_equality_does_not_flip():
    g = GripperHysteresis(close_above=0.45, open_below=0.25)
    assert g.update(0.45) == 0.0
    g.update(0.9)
    assert g.update(0.25) == 1.0


def test_reset_returns_to_open_so_the_next_episode_does_not_start_closed():
    g = GripperHysteresis()
    g.update(0.7)
    assert g.update(0.35) == 1.0
    g.reset()
    assert g.update(0.35) == 0.0


def test_initially_closed_is_respected_and_restored_by_reset():
    g = GripperHysteresis(initially_closed=True)
    assert g.update(0.35) == 1.0
    g.update(0.0)
    g.reset()
    assert g.update(0.35) == 1.0


@pytest.mark.parametrize('close_above,open_below', [(0.3, 0.3), (0.2, 0.4)])
def test_thresholds_without_a_band_are_rejected(close_above, open_below):
    with pytest.raises(ValueError):
        GripperHysteresis(close_above=close_above, open_below=open_below)
