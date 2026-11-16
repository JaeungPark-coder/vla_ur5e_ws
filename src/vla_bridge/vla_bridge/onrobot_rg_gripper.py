"""OnRobot RG2 / RG6 driver over the OnRobot Compute Box's Modbus TCP server -- the real gripper is an RG2 (README
2026-11-16, lab photo), not the Robotiq 2F-85 the sim and robotiq_socket_gripper.py were written for.

Duck-typed like RobotiqSocketGripper, so robot_interface.UR5eInterface uses it unchanged:
    get_current_position() -> int "closure counts", is_object_detected() -> bool, set_position(counts).
Closure counts = (max width - width between the fingertips) in 0.1 mm: 0 = fully open, 1100 = closed (RG2), so they
increase with the sim's 0 (open) .. 1 (closed) fraction the way gripper_mapping.GripperMap requires.  Calibrate with the
holding point (sim fraction holding the cube <-> closure counts holding the real cube), e.g. for a 40 mm cube on an RG2:
1100 - 400 = 700 counts, minus a few for the squeeze.

Register map (OnRobot Compute Box, unit/slave 65; the one community drivers use, e.g. takuya-ki/onrobot-rg):
    write 0  target force, 0.1 N            (RG2 0..400, RG6 0..1200)
    write 1  target width, 0.1 mm           (RG2 0..1100, RG6 0..1600; with the fingertip offset under command 16)
    write 2  control: 1 grip, 8 stop, 16 grip with fingertip offset
    read 258 fingertip offset, 0.1 mm (signed)
    read 267 width without the offset, 0.1 mm;  read 275 width with the offset, 0.1 mm
    read 268 status bits: 0 busy, 1 grip detected, 2/4 safety switch 1/2 pushed, 3/5 safety circuit 1/2 tripped,
             6 safety error (switch pushed at power-on)
Two properties shape the design:
  * the RG2 IGNORES commands while busy (bit 0), and moves at 38..127 mm/s -- one opening-to-cube close takes a good
    fraction of a second, i.e. tens of policy rows;
  * a Modbus round trip must not sit inside the 500 Hz servoJ loop (joint_streamer.py) that calls set_position.
So set_position only records the target; a worker thread polls the status (poll_hz) and sends the latest target once the
gripper is idle and the target moved by >= deadband_counts from the last one sent.  Readings are the worker's cache.

Requires the Compute Box (Modbus TCP is served only there).  An RG2 driven only through the OnRobot URCap on the
teach pendant (no Compute Box on the network) cannot be reached this way -- README lists that as a lab question.
NOT verified on hardware; the protocol and the busy behaviour are tested against a fake server (test_onrobot_rg_gripper.py).
"""
import socket
import struct
import threading
import time

MODELS = {'rg2': dict(max_width=1100, max_force=400), 'rg6': dict(max_width=1600, max_force=1200)}
REG_FORCE, REG_WIDTH, REG_CONTROL = 0, 1, 2
REG_OFFSET, REG_WIDTH_RAW, REG_STATUS, REG_WIDTH_WITH_OFFSET = 258, 267, 268, 275
CMD_GRIP, CMD_STOP, CMD_GRIP_WITH_OFFSET = 1, 8, 16
SAFETY_BITS = {3: 'safety circuit 1 tripped', 5: 'safety circuit 2 tripped', 6: 'safety error at power-on'}


class ModbusTCP:
    """Just the two functions the Compute Box needs (03 read holding registers, 16 write multiple registers)."""

    def __init__(self, host, port=502, unit=65, timeout=1.0):
        self.unit = unit
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._tid = 0
        self._lock = threading.Lock()

    def _exchange(self, pdu):
        with self._lock:
            self._tid = (self._tid + 1) & 0xFFFF
            self._sock.sendall(struct.pack('>HHHB', self._tid, 0, len(pdu) + 1, self.unit) + pdu)
            head = self._recv(7)
            tid, proto, length, unit = struct.unpack('>HHHB', head)
            body = self._recv(length - 1)
        if tid != self._tid or proto != 0:
            raise ConnectionError(f'Modbus reply out of step (transaction {tid}, expected {self._tid})')
        if body[0] & 0x80:
            raise RuntimeError(f'Modbus exception {body[1]} for function {body[0] & 0x7F}')
        if body[0] != pdu[0]:
            raise ConnectionError(f'Modbus reply to function {body[0]}, expected {pdu[0]}')
        return body

    def _recv(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Modbus server closed the connection')
            buf += chunk
        return buf

    def read(self, address, count=1):
        body = self._exchange(struct.pack('>BHH', 3, address, count))
        if body[1] != 2 * count:
            raise ConnectionError(f'Modbus read of {count} registers returned {body[1]} bytes')
        return list(struct.unpack(f'>{count}H', body[2:2 + 2 * count]))

    def write(self, address, values):
        values = [int(v) & 0xFFFF for v in values]
        self._exchange(struct.pack(f'>BHHB{len(values)}H', 16, address, len(values), 2 * len(values), *values))

    def close(self):
        self._sock.close()


class OnRobotRGGripper:
    def __init__(self, host, port=502, unit=65, model='rg2', force_n=20.0, poll_hz=25.0, deadband_counts=20,
                 timeout=1.0):
        if model not in MODELS:
            raise ValueError(f'model {model!r}: expected one of {sorted(MODELS)}')
        self.max_width = MODELS[model]['max_width']
        max_force = MODELS[model]['max_force']
        self.force = int(round(force_n * 10))
        if not 0 < self.force <= max_force:
            raise ValueError(f'force_n {force_n}: the {model.upper()} takes 0..{max_force / 10:g} N')
        self.deadband = int(deadband_counts)
        self.poll_s = 1.0 / poll_hz
        self.bus = ModbusTCP(host, port, unit, timeout)
        self._target = None          # closure counts asked for
        self._sent = None            # closure counts last written
        self._status = None
        self._width = None
        self._error = None
        self.commands_sent = 0
        self._cv = threading.Condition()
        self._running = True
        self._poll()                 # one synchronous read: fail at construction, not on the first row
        self._thread = threading.Thread(target=self._run, name='onrobot-rg', daemon=True)
        self._thread.start()

    # ---- the worker ----------------------------------------------------------------------------------------------
    def _poll(self):
        status = self.bus.read(REG_STATUS)[0]
        width = self.bus.read(REG_WIDTH_WITH_OFFSET)[0]
        with self._cv:
            self._status, self._width = status, width
            self._cv.notify_all()

    def _run(self):
        while self._running:
            try:
                self._poll()
                with self._cv:
                    target, sent, busy = self._target, self._sent, bool(self._status & 1)
                if target is not None and not busy and (sent is None or abs(target - sent) >= self.deadband):
                    width = self.max_width - target
                    self.bus.write(REG_FORCE, [self.force, width, CMD_GRIP_WITH_OFFSET])
                    with self._cv:
                        self._sent = target
                        self.commands_sent += 1
                    self._poll()     # pick up busy at once, so the next decision does not see the stale idle state
            except Exception as exc:  # noqa: BLE001 -- reported through the readers, which UR5eInterface catches
                with self._cv:
                    self._error = exc
                    self._cv.notify_all()
                return
            time.sleep(self.poll_s)

    def _check(self):
        if self._error is not None:
            raise ConnectionError(f'OnRobot worker stopped: {self._error}')
        tripped = [why for bit, why in SAFETY_BITS.items() if self._status & (1 << bit)]
        if tripped:
            raise RuntimeError(f'OnRobot gripper: {", ".join(tripped)} -- power-cycle the gripper')

    # ---- the duck-typed interface --------------------------------------------------------------------------------
    def activate(self):
        """Nothing to activate on an RG; just refuses to start with a safety fault latched."""
        with self._cv:
            self._check()

    def set_position(self, counts):
        counts = int(max(0, min(self.max_width, counts)))
        with self._cv:
            self._check()
            self._target = counts

    def get_current_position(self):
        with self._cv:
            self._check()
            return int(self.max_width - min(self._width, self.max_width))

    def is_object_detected(self):
        with self._cv:
            self._check()
            return bool(self._status & 2)

    def is_busy(self):
        with self._cv:
            self._check()
            return bool(self._status & 1)

    def width_mm(self):
        with self._cv:
            self._check()
            return self._width / 10.0

    def wait_until_idle(self, timeout_s=3.0):
        """Blocks until the last target was sent and the gripper stopped moving (for scripts, not the control loop)."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                self._check()
                settled = self._target is None or (self._sent is not None and abs(self._target - self._sent) < self.deadband)
                if settled and not self._status & 1:
                    return True
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)

    def stop(self):
        self.bus.write(REG_CONTROL, [CMD_STOP])

    def close(self):
        self._running = False
        self._thread.join(timeout=2 * self.poll_s + 1.0)
        self.bus.close()
