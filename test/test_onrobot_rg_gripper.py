"""onrobot_rg_gripper against a fake Compute Box: the Modbus framing, the closure-count convention, and the RG2's
habit of ignoring commands while it is still moving (the driver must send the latest target once it is idle)."""
import os
import socket
import struct
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vla_bridge'))
from vla_bridge.gripper_mapping import GripperMap  # noqa: E402
from vla_bridge.onrobot_rg_gripper import OnRobotRGGripper  # noqa: E402


class FakeComputeBox:
    """An RG2 behind Modbus TCP: fingers move at speed_mm_s, stop on an object of object_mm, busy while moving."""

    def __init__(self, speed_mm_s=100.0, object_mm=None, status_extra=0):
        self.speed = speed_mm_s * 10.0          # 0.1 mm / s
        self.object = None if object_mm is None else object_mm * 10.0
        self.status_extra = status_extra
        self.width0, self.goal, self.t0 = 1100.0, 1100.0, time.monotonic()
        self.writes, self.ignored = [], 0
        self.regs = {0: 0, 1: 0, 2: 0}
        self.srv = socket.socket()
        self.srv.bind(('127.0.0.1', 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _state(self):
        stop_at = self.goal
        if self.object is not None and self.goal < self.object <= self.width0:
            stop_at = self.object
        dist = abs(stop_at - self.width0)
        moved = min(dist, (time.monotonic() - self.t0) * self.speed)
        width = self.width0 + (moved if stop_at > self.width0 else -moved)
        busy = moved < dist
        grip = (not busy) and stop_at != self.goal
        return width, busy, grip

    def _serve(self):
        conn, _ = self.srv.accept()
        while True:
            head = conn.recv(7)
            if len(head) < 7:
                return
            tid, _, length, unit = struct.unpack('>HHHB', head)
            pdu = conn.recv(length - 1)
            fc = pdu[0]
            if fc == 3:
                addr, count = struct.unpack('>HH', pdu[1:5])
                width, busy, grip = self._state()
                val = {268: int(busy) | (int(grip) << 1) | self.status_extra, 275: int(round(width)), 267: int(round(width)),
                       258: 0}.get(addr, 0)
                body = struct.pack('>BBH', 3, 2, val)
            elif fc == 16:
                addr, count, nbytes = struct.unpack('>HHB', pdu[1:6])
                vals = struct.unpack(f'>{count}H', pdu[6:6 + nbytes])
                for k, v in enumerate(vals):
                    self.regs[addr + k] = v
                width, busy, _ = self._state()
                if self.regs.get(2) in (1, 16):
                    if busy:
                        self.ignored += 1
                    else:
                        self.writes.append(tuple(vals))
                        self.width0, self.goal, self.t0 = width, float(self.regs[1]), time.monotonic()
                body = struct.pack('>BHH', 16, addr, count)
            else:
                body = struct.pack('>BB', fc | 0x80, 1)
            conn.sendall(struct.pack('>HHHB', tid, 0, len(body) + 1, unit) + body)


def make(box, **kw):
    return OnRobotRGGripper('127.0.0.1', port=box.port, poll_hz=200.0, **kw)


def test_closes_on_a_cube_and_reads_closure_counts():
    box = FakeComputeBox(speed_mm_s=200.0, object_mm=40.0)
    g = make(box, force_n=20.0)
    try:
        assert g.get_current_position() == 0 and not g.is_object_detected()     # 110 mm open = 0 counts
        g.set_position(1100)                                                     # close fully ...
        assert g.wait_until_idle(3.0)
        assert box.writes == [(200, 0, 16)]                                      # 20 N, width 0, grip with offset
        assert g.get_current_position() == 700 and g.is_object_detected()        # ... stops on the 40 mm cube
        assert g.width_mm() == pytest.approx(40.0)
    finally:
        g.close()


def test_latest_target_is_sent_once_the_gripper_is_idle():
    box = FakeComputeBox(speed_mm_s=100.0)
    g = make(box)
    try:
        g.set_position(500)                      # 60 mm: 0.5 s of motion
        time.sleep(0.05)
        for c in (550, 600, 650, 300):           # the policy keeps changing its mind while the fingers move
            g.set_position(c)
            time.sleep(0.02)
        assert g.wait_until_idle(5.0)
        assert [w[1] for w in box.writes] == [600, 800]   # the first target, then only the latest (1100 - 300)
        assert box.ignored == 0                            # nothing was sent into a busy gripper
        assert g.get_current_position() == 300
    finally:
        g.close()


def test_deadband_and_calibrated_mapping():
    box = FakeComputeBox(speed_mm_s=500.0)
    g = make(box, deadband_counts=20)
    gm = GripperMap.from_flat([0.0, 0, 0.62, 690, 1.0, 1100])    # holding point: sim 0.62 <-> 40 mm cube minus 1 mm squeeze
    try:
        g.set_position(gm.to_counts(0.62))
        assert g.wait_until_idle(3.0)
        g.set_position(gm.to_counts(0.62) + 10)                   # 1 mm of jitter: not sent
        time.sleep(0.05)
        assert len(box.writes) == 1 and box.writes[0][1] == 1100 - 690
        assert gm.to_fraction(g.get_current_position()) == pytest.approx(0.62, abs=1e-3)
    finally:
        g.close()


def test_safety_fault_and_bad_force_are_refused():
    with pytest.raises(ValueError):
        make(FakeComputeBox(), force_n=45.0)                       # RG2 max 40 N
    g = make(FakeComputeBox(status_extra=1 << 3))
    try:
        with pytest.raises(RuntimeError, match='safety circuit 1'):
            g.activate()
        with pytest.raises(RuntimeError):
            g.set_position(100)
    finally:
        g.close()
