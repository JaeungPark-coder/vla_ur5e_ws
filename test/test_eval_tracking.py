"""MaxHeightTracker: a blow-up inside one executed block must not be missed, and the
previous trial's maximum must never leak into the next one."""
from vla_bridge.eval_tracking import MaxHeightTracker


def test_the_running_maximum_is_kept():
    t = MaxHeightTracker()
    for z in (0.02, 0.17, 0.09):
        assert t.update(0, z)
    assert t.value == 0.17


def test_a_spike_reported_between_two_ordinary_samples_is_still_seen():
    # the bridge reports the max over all ticks since its last message, so one message
    # can carry a spike the client never sampled directly
    t = MaxHeightTracker()
    t.update(0, 0.16)
    t.update(0, 3.4)      # cube flung up and back down within one block
    t.update(0, 0.02)
    assert t.value == 3.4


def test_messages_from_before_the_reset_do_not_leak_into_the_next_trial():
    t = MaxHeightTracker()
    t.update(0, 0.31)                  # trial 1 blew up
    t.start_trial(min_epoch=1)         # reset requested; the bridge is still on epoch 0
    assert t.value == 0.0
    assert t.update(0, 0.31) is False  # the old re-published value arrives late
    assert t.value == 0.0
    assert t.update(1, 0.02)           # first message of the new scene
    assert t.value == 0.02


def test_epochs_may_skip_ahead():
    t = MaxHeightTracker()
    t.start_trial(min_epoch=3)
    assert t.update(5, 0.1) and t.value == 0.1


def test_without_resets_everything_counts():
    t = MaxHeightTracker()
    assert t.update(0, 0.05) and t.update(0, 0.2)
    assert t.value == 0.2
