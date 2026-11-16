"""Executes a chunk of policy rows on a UR arm at the rate the rows were recorded, with 500 Hz servoJ interpolation.

Why this exists (README "real-robot preparation", 2026-11-16): the rtde path of vla_policy_client used to send the K rows of a
chunk with servoJ back to back, with nothing pacing them.  servoJ returns immediately and the controller tracks only the latest
target, so the arm skipped straight to the last row -- a K/60 s motion squeezed into however long K socket writes take.

Each row is the joint target for one sim tick (action_dt_s = 1/60 s).  Here every row gets that much time, the segment between
consecutive rows is interpolated linearly and sent every servo_dt (2 ms = the e-series 500 Hz control rate) inside
initPeriod/waitPeriod, so the controller sees a smooth stream instead of steps.  A segment that would need more than v_max on
any joint is stretched (adaptive slowdown) instead of being clipped, so the path is kept and only the timing changes.
speed_scale < 1 slows everything down uniformly (first runs on hardware).

No ROS and no ur_rtde import: `control` is duck-typed (initPeriod, servoJ, waitPeriod), so the same code runs against
RTDEControlInterface, URSim, or the fake in test/test_joint_streamer.py.
"""
import time

import numpy as np

from vla_bridge.ur5e_kinematics import MAX_JOINT_SPEED_RAD_S


def segment_durations(q_start, rows, row_dt, v_max, speed_scale=1.0):
    """Seconds for each segment q_start->rows[0], rows[0]->rows[1], ...: row_dt / speed_scale, or longer if a joint would
    exceed v_max (rad/s, scalar or (6,))."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    prev = np.vstack([np.asarray(q_start, dtype=float)[None, :6], rows[:-1, :6]])
    need = np.max(np.abs(rows[:, :6] - prev) / np.broadcast_to(np.asarray(v_max, dtype=float), (6,)), axis=1)
    return np.maximum(row_dt / float(speed_scale), need)


def interpolate(q_start, rows, durations, servo_dt):
    """Yields (q (6,), row index of the segment being traversed, segment_started) every servo_dt.  The last sample of each
    segment lands exactly on its row, so no row is ever skipped."""
    prev = np.asarray(q_start, dtype=float)[:6]
    for i, (row, dur) in enumerate(zip(np.atleast_2d(rows), durations)):
        row = np.asarray(row, dtype=float)[:6]
        n = max(1, int(np.ceil(dur / servo_dt - 1e-9)))
        for k in range(1, n + 1):
            yield prev + (row - prev) * (k / n), i, k == 1
        prev = row


class JointRowStreamer:
    def __init__(self, control, servo_dt=0.002, lookahead_time=0.1, gain=300,
                 v_max=MAX_JOINT_SPEED_RAD_S * 0.5, speed_scale=1.0):
        """v_max defaults to half the UR5e joint limit, the margin the v5 expert is also held to."""
        self.control = control
        self.servo_dt = float(servo_dt)
        self.lookahead_time = float(lookahead_time)
        self.gain = float(gain)
        self.v_max = np.broadcast_to(np.asarray(v_max, dtype=float), (6,)).copy()
        self.speed_scale = float(speed_scale)

    def execute(self, q_start, rows, row_dt, on_row=None):
        """Stream rows (N,>=6) starting from q_start (the last commanded target if there is one, else the measured joints).
        on_row(i) is called when segment i starts -- the tick on which row i's gripper command applies in the sim.
        Returns timing statistics."""
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        durs = segment_durations(q_start, rows, row_dt, self.v_max, self.speed_scale)
        t_wall = time.monotonic()
        n_cmd = 0
        for q, i, started in interpolate(q_start, rows, durs, self.servo_dt):
            t0 = self.control.initPeriod()
            if started and on_row is not None:
                on_row(i)
            self.control.servoJ(list(q), 0.0, 0.0, self.servo_dt, self.lookahead_time, self.gain)
            self.control.waitPeriod(t0)
            n_cmd += 1
        nominal = len(rows) * row_dt / self.speed_scale
        return dict(rows=len(rows), commands=n_cmd, nominal_s=nominal, planned_s=float(durs.sum()),
                    wall_s=time.monotonic() - t_wall, slowed_rows=int(np.sum(durs > row_dt / self.speed_scale + 1e-9)))
