"""Drives the Robotiq socket driver against a fake UR controller.

The protocol is one line in, one line back over a long-lived socket, which
makes the interesting failures the stateful ones: a reply arriving split
across two packets, two replies arriving in one, and -- the one that costs
something -- a command that dies partway through and leaves its half-read
reply in the buffer for the NEXT command to pick up.

Nothing here opens a real socket. The driver is the only thing in this repo
that will talk to the gripper on day one, and it has never been run.
"""
import socket

import pytest

from vla_bridge.robotiq_socket_gripper import RobotiqSocketGripper


class FakeSocket:
    """A UR controller's socket server, scripted.

    `script` maps a command line to what the server sends back. A value that
    is a list is delivered one chunk per recv, so a reply can be split across
    packets; an Exception instance is raised from recv instead.
    """

    def __init__(self, script):
        self.script = script
        self.sent = []
        self.pending = []
        self.closed = False

    def sendall(self, payload):
        command = payload.decode('ascii').strip()
        self.sent.append(command)
        reply = self.script.get(command, 'ack')
        self.pending.extend(reply if isinstance(reply, list) else [reply])

    def recv(self, _bufsize):
        if not self.pending:
            return b''                      # server hung up
        chunk = self.pending.pop(0)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk if isinstance(chunk, bytes) else (chunk + '\n').encode('ascii')

    def close(self):
        self.closed = True


@pytest.fixture
def gripper(monkeypatch):
    """Builds a driver whose socket is whatever the test scripts."""
    def build(script):
        fake = FakeSocket(script)
        monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: fake)
        driver = RobotiqSocketGripper('192.168.1.100')
        return driver, fake
    return build


# --- framing -------------------------------------------------------------

def test_a_command_is_newline_terminated(gripper):
    driver, fake = gripper({'GET POS': 'POS 128'})
    driver.get_current_position()
    assert fake.sent == ['GET POS']


def test_a_reply_split_across_packets_is_reassembled(gripper):
    """The socket gives no guarantee a line arrives whole."""
    driver, _ = gripper({'GET POS': [b'PO', b'S 1', b'28\n']})
    assert driver.get_current_position() == 128


def test_two_replies_in_one_packet_are_not_merged(gripper):
    driver, _ = gripper({'GET POS': [b'POS 128\nOBJ 2\n'], 'GET OBJ': []})
    assert driver.get_current_position() == 128
    assert driver.is_object_detected() is True     # served from the buffer


def test_a_closed_connection_is_reported_as_one(gripper):
    driver, _ = gripper({'GET POS': []})
    with pytest.raises(ConnectionError):
        driver.get_current_position()


# --- the two calls UR5eInterface actually depends on ---------------------

def test_the_position_comes_back_as_counts(gripper):
    driver, _ = gripper({'GET POS': 'POS 0'})
    assert driver.get_current_position() == 0


@pytest.mark.parametrize('gobj,detected', [
    (0, False),     # still moving
    (1, True),      # stopped opening on contact
    (2, True),      # stopped closing on contact -- the grasp signal
    (3, False),     # reached the commanded position, touching nothing
])
def test_object_detection_follows_the_gobj_table(gripper, gobj, detected):
    driver, _ = gripper({'GET OBJ': f'OBJ {gobj}'})
    assert driver.is_object_detected() is detected


# --- set_position --------------------------------------------------------

@pytest.mark.parametrize('asked,sent', [
    (0, 0), (128, 128), (255, 255),
    (-5, 0), (300, 255),            # clamped, not rejected
    (127.6, 127),                   # a float from a 0..1 mapping
])
def test_position_commands_are_clamped_to_the_counts_range(gripper, asked, sent):
    driver, fake = gripper({})
    driver.set_position(asked)
    assert fake.sent == [f'SET POS {sent}']


def test_a_refused_position_command_raises(gripper):
    """Silence here would mean the gripper never moved and nothing said so."""
    driver, _ = gripper({'SET POS 128': 'not-ack'})
    with pytest.raises(RuntimeError):
        driver.set_position(128)


# --- activation ----------------------------------------------------------

def test_activation_waits_for_the_gripper_to_report_ready(gripper):
    """A SET POS sent mid-activation is silently ignored on real hardware,
    so the wait is what makes the first commanded move actually happen."""
    driver, fake = gripper({'GET STA': ['STA 0', 'STA 0', 'STA 3']})
    driver.activate()
    assert fake.sent == ['SET ACT 1', 'GET STA', 'GET STA', 'GET STA', 'SET GTO 1']


def test_go_to_mode_is_set_after_activation_not_before(gripper):
    """Without GTO the gripper arms but never moves on SET POS."""
    driver, fake = gripper({'GET STA': 'STA 3'})
    driver.activate()
    assert fake.sent.index('SET GTO 1') > fake.sent.index('SET ACT 1')


def test_activation_that_never_completes_times_out(gripper):
    driver, _ = gripper({'GET STA': ['STA 0'] * 200})
    with pytest.raises(TimeoutError):
        driver.activate(timeout_s=0.3)


# --- the stateful failure: a half-read reply left in the buffer ----------

def test_a_failed_read_does_not_desynchronise_the_next_one(gripper):
    """The expensive one.

    A command whose reply arrives partially and then stalls leaves those
    bytes in the buffer. UR5eInterface.get_gripper_state catches the error
    and falls back -- correctly -- but if the leftovers survive, the NEXT
    command reads the TAIL of the previous reply. Every later call is then
    answered one reply late: GET POS returns the OBJ status, reported as a
    measured position, with nothing anywhere saying it is wrong.
    """
    driver, fake = gripper({
        'GET POS': [b'POS 1', socket.timeout('timed out')],
        'GET OBJ': [b'OBJ 2\n'],
    })

    with pytest.raises(socket.timeout):
        driver.get_current_position()

    # the next command must be answered by its OWN reply
    assert driver.is_object_detected() is True


def test_a_reply_to_the_wrong_command_is_refused_rather_than_parsed(gripper):
    """Defence in depth for the same failure: even if the stream does slip,
    a position answered by an OBJ line must not be read as 3 counts."""
    driver, _ = gripper({'GET POS': 'OBJ 3'})
    with pytest.raises(ValueError):
        driver.get_current_position()


# --- the contract UR5eInterface duck-types -------------------------------

def test_it_satisfies_the_interface_robot_interface_expects(gripper):
    driver, _ = gripper({})
    for method in ('get_current_position', 'set_position', 'is_object_detected'):
        assert callable(getattr(driver, method, None)), method


def test_close_releases_the_socket(gripper):
    driver, fake = gripper({})
    driver.close()
    assert fake.closed
