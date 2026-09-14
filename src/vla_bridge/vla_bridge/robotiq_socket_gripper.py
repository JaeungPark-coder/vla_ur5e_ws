"""Robotiq gripper driver over the UR controller's "Socket" ADI interface --
the integration path robot_interface.py's own docstring already points at
("a Robotiq over its socket interface"). This is the plain ASCII command
server the Robotiq URCap exposes on port 63352 of the ROBOT CONTROLLER's own
PC (reachable at the same IP as `robot_ip` -- not the gripper directly, and
not a separate serial dongle). It is the same protocol UR's own `rq_*`
URScript functions and community drivers such as `pyRobotiqGripper` use:
one line in, one line back, e.g. "SET POS 128\n" / "GET POS\n" -> "POS 128".

NOT verified against real hardware -- same posture as the rest of this
project (see README's "written without the ability to run this" notes).
Grounded in Robotiq's published socket command reference, not guessed:
ADJUST if your gripper's URCap/firmware version answers differently --
`get_current_position()`/`is_object_detected()` are the two calls
`robot_interface.UR5eInterface` actually depends on, so verify those first
against a real gripper before trusting anything else here.

Usage (nothing else in this project constructs this for you -- see
vla_policy_client.py's `gripper_driver` parameter, which does):

    driver = RobotiqSocketGripper(robot_ip)
    driver.activate()  # once per gripper power-cycle
    interface = UR5eInterface(robot_ip, gripper_driver=driver)
"""
import socket
import time


class RobotiqSocketGripper:
    DEFAULT_PORT = 63352

    def __init__(self, robot_ip, port=DEFAULT_PORT, timeout=2.0):
        self.robot_ip = robot_ip
        self.port = port
        self.timeout = timeout
        self._sock = socket.create_connection((robot_ip, port), timeout=timeout)
        self._buf = b""

    def _readline(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1024)
            if not chunk:
                raise ConnectionError(
                    f"Robotiq socket at {self.robot_ip}:{self.port} closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii", errors="replace").strip()

    def _command(self, cmd):
        self._sock.sendall((cmd.strip() + "\n").encode("ascii"))
        return self._readline()

    def activate(self, wait=True, timeout_s=5.0):
        """SET ACT 1 (activation request) then SET GTO 1 (go-to mode, so a
        subsequent SET POS actually moves the gripper instead of just
        arming it) -- both required once per power cycle, per Robotiq's own
        activation sequence. wait=True polls GET STA until the gripper
        reports activated (STA 3), since a SET POS sent mid-activation is
        silently ignored on real hardware rather than queued."""
        self._command("SET ACT 1")
        if wait:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if self._command("GET STA") == "STA 3":
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError(
                    f"Robotiq gripper did not report activated (STA 3) within {timeout_s}s")
        self._command("SET GTO 1")

    def set_position(self, counts):
        """counts: 0 (open) .. 255 (closed) -- matches UR5eInterface's
        gripper_open_counts/gripper_closed_counts default mapping."""
        counts = int(max(0, min(255, counts)))
        reply = self._command(f"SET POS {counts}")
        if reply != "ack":
            raise RuntimeError(f"unexpected reply to SET POS {counts}: {reply!r}")

    def get_current_position(self):
        """-> int, 0..255. This is the call robot_interface.py's
        get_gripper_state() depends on for a measured (not echoed) reading."""
        reply = self._command("GET POS")
        return int(reply.split()[-1])

    def is_object_detected(self):
        """gOBJ status via GET OBJ: 0=moving, 1=stopped opening on contact,
        2=stopped closing on contact, 3=at requested position, no contact.
        1 or 2 means the fingers stopped early -- i.e. touched something
        before reaching the commanded position, which is the hardware
        grasp-success signal robot_interface.py surfaces to the policy."""
        reply = self._command("GET OBJ")
        status = int(reply.split()[-1])
        return status in (1, 2)

    def set_speed(self, speed):
        """speed: 0 (slowest) .. 255 (fastest). Optional tuning, not part
        of UR5eInterface's required contract."""
        self._command(f"SET SPE {int(max(0, min(255, speed)))}")

    def set_force(self, force):
        """force: 0 (lowest) .. 255 (highest). Optional tuning, not part
        of UR5eInterface's required contract."""
        self._command(f"SET FOR {int(max(0, min(255, force)))}")

    def close(self):
        self._sock.close()
