"""joint_streamer / safety_limits / gripper_mapping: the real-robot execution path, without a robot."""
import numpy as np
import pytest

from vla_bridge.gripper_mapping import GripperMap
from vla_bridge.joint_streamer import JointRowStreamer, interpolate, segment_durations
from vla_bridge.safety_limits import SafetyLimits
from vla_bridge.ur5e_kinematics import flange, tool_point


class FakeControl:
    """RTDEControlInterface's servo-loop surface; time is counted, not slept."""

    def __init__(self):
        self.cmds, self.t = [], 0.0

    def initPeriod(self):
        return self.t

    def servoJ(self, q, v, a, t, lookahead, gain):
        self.cmds.append((self.t, np.array(q), t))

    def waitPeriod(self, t0):
        self.t = t0 + 0.002


def test_every_row_gets_row_dt_and_is_hit_exactly():
    q0 = np.zeros(6)
    rows = np.cumsum(np.full((25, 6), 0.005), axis=0)      # 0.3 rad/s: well under the cap
    c = FakeControl()
    seen = []
    st = JointRowStreamer(c, v_max=1.0).execute(q0, rows, 1 / 60, on_row=seen.append)
    assert seen == list(range(25))
    assert st['slowed_rows'] == 0 and st['planned_s'] == pytest.approx(25 / 60)
    assert len(c.cmds) == 25 * 9                              # ceil((1/60) / 0.002) = 9 per row
    assert c.t == pytest.approx(25 * 9 * 0.002)
    hits = np.array([q for _, q, _ in c.cmds])[8::9]
    np.testing.assert_allclose(hits, rows)                    # last sample of each segment is the row itself
    assert all(t == 0.002 for _, _, t in c.cmds)


def test_old_bug_is_gone_rows_are_not_collapsed_into_the_last_one():
    rows = np.linspace(0, 0.1, 25)[:, None] * np.ones(6)
    c = FakeControl()
    JointRowStreamer(c, v_max=1.0).execute(np.zeros(6), rows, 1 / 60)
    first = c.cmds[0][1]
    assert np.max(np.abs(first)) < 0.1 / 24                   # the first command is a small step, not the final target


def test_fast_segment_is_stretched_not_clipped():
    rows = np.array([[0.0, 0, 0, 0, 0, 0.05], [0.0, 0, 0, 0, 0, 0.06]])   # 3 rad/s then 0.6 rad/s on wrist_3
    d = segment_durations(np.zeros(6), rows, 1 / 60, v_max=1.0)
    assert d[0] == pytest.approx(0.05) and d[1] == pytest.approx(1 / 60)
    c = FakeControl()
    st = JointRowStreamer(c, v_max=1.0).execute(np.zeros(6), rows, 1 / 60)
    assert st['slowed_rows'] == 1
    q = np.array([q for _, q, _ in c.cmds])
    assert np.max(np.abs(np.diff(np.vstack([np.zeros(6), q]), axis=0))) <= 1.0 * 0.002 + 1e-12
    np.testing.assert_allclose(q[-1], rows[-1])


def test_speed_scale_stretches_uniformly():
    rows = np.cumsum(np.full((10, 6), 0.002), axis=0)
    d = segment_durations(np.zeros(6), rows, 1 / 60, v_max=1.0, speed_scale=0.25)
    assert np.allclose(d, 4 / 60)


def test_interpolate_zero_motion_row_still_takes_its_time():
    out = list(interpolate(np.zeros(6), np.zeros((2, 6)), [1 / 60, 1 / 60], 0.002))
    assert len(out) == 18 and [s for _, _, s in out].count(True) == 2


# ---- safety limits ---------------------------------------------------------------------------------------------------
LIM = dict(joint_min=[-1] * 6, joint_max=[1] * 6, max_row_step=[0.05] * 6, tool_offset_in_flange=[0, 0, 0.16],
           tool_box_min=[-1, -1, -0.02], tool_box_max=[1, 1, 1])


def test_safety_accepts_a_gentle_chunk_and_names_each_violation():
    lim = SafetyLimits.from_dict(LIM)
    q0 = np.array([0.0, -0.8, 0.8, -0.5, -0.5, 0.0])
    rows = q0 + np.cumsum(np.full((5, 6), 0.01), axis=0)
    assert lim.check_chunk(q0, rows) == []
    bad = rows.copy()
    bad[2, 1] = -1.2
    msgs = lim.check_chunk(q0, bad)
    assert any('joint 1' in m and 'outside' in m for m in msgs) and any('step' in m for m in msgs)
    assert lim.check_chunk(q0, np.full((2, 6), np.nan)) == ['non-finite joint target']


def test_safety_tool_box_uses_the_coaxial_tool_point():
    lim = SafetyLimits.from_dict(LIM)
    q = np.array([[0.0, -1.0, 1.4, -1.97, -1.57, 0.0]])
    z = tool_point(q, [0, 0, 0.16])[0, 2]
    lim.tool_box_min = np.array([-1, -1, z + 0.01])
    assert any('tool point' in m for m in lim.check_chunk(q[0], q))


def test_safety_rejects_inconsistent_files():
    with pytest.raises(ValueError):
        SafetyLimits.from_dict({**LIM, 'joint_min': [2] * 6})
    with pytest.raises(ValueError):
        SafetyLimits.from_dict({k: v for k, v in LIM.items() if k != 'max_row_step'})


# ---- kinematics (pinned to the URSim 5.26 readings, README "blocker confirmed on URSim") -----------------------------
def test_flange_at_home_matches_ursim():
    np.testing.assert_allclose(flange(np.zeros(6))[0, :3, 3], [-0.8172, -0.2329, 0.0628], atol=1e-4)


# ---- gripper mapping -------------------------------------------------------------------------------------------------
def test_gripper_map_default_is_the_old_linear_one_and_passes_through_calibration():
    m = GripperMap()
    assert m.to_counts(0.0) == 0 and m.to_counts(1.0) == 255 and m.to_counts(0.5) == 128
    m = GripperMap.from_flat([0.0, 3, 0.70, 140, 1.0, 230])
    assert m.to_counts(0.70) == 140 and m.to_fraction(140) == pytest.approx(0.70)
    assert m.to_counts(1.5) == 230 and m.to_fraction(0) == 0.0
    with pytest.raises(ValueError):
        GripperMap.from_flat([0.0, 10, 1.0, 5])
