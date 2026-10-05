"""Lockstep: one request <-> exactly one simulator tick.

Without it the bridge free-runs while the client blocks on policy.infer(), so
one action spans an unknown number of sim ticks, but the demonstrations are one
action per tick. These tests run the REAL protocol class (isaac/
lockstep_protocol.py) and the REAL client-side interface (vla_bridge/
isaac_robot_interface.py, with the ROS message modules stubbed) against a mock
bridge that counts ticks. They cannot say anything about Isaac Sim itself --
the actual bridge loop in pick_place_scene_bridge.py needs a live run.
"""
import sys
import time
import types

import pytest

from lockstep_protocol import LockstepGate


# --- the protocol class ---------------------------------------------------

def test_a_new_request_is_taken_once():
    gate = LockstepGate()
    assert gate.submit(1, [0.0] * 6, 0.0) is True
    assert gate.take() == (1, [0.0] * 6, 0.0)
    assert gate.take() is None
    assert gate.last_executed_id == 1


def test_stale_and_repeated_ids_never_run_a_tick_twice():
    gate = LockstepGate()
    gate.submit(5, [0.0] * 6)
    gate.take()
    assert gate.submit(5, [1.0] * 6) is False   # redelivery of the one just run
    assert gate.submit(3, [1.0] * 6) is False   # older than what already ran
    assert gate.take() is None


def test_only_the_newest_pending_request_is_kept():
    gate = LockstepGate()
    gate.submit(1, [0.0] * 6)
    assert gate.submit(2, [2.0] * 6) is True
    assert gate.submit(1, [9.0] * 6) is False
    assert gate.take()[0] == 2


def test_reset_drops_the_queued_request_but_keeps_counting_from_the_same_id():
    gate = LockstepGate()
    gate.submit(1, [0.0] * 6)
    gate.take()
    gate.submit(2, [0.0] * 6)
    gate.reset()
    assert gate.take() is None
    assert gate.last_executed_id == 1
    assert gate.submit(3, [0.0] * 6) is True


# --- a mock bridge + the real client-side interface -----------------------

class _Stamp:
    sec = 0


class _Header:
    def __init__(self):
        self.stamp = _Stamp()


class _JointState:
    def __init__(self):
        self.header = _Header()
        self.position = []


class _Float32:
    def __init__(self, data=0.0):
        self.data = data


@pytest.fixture
def interface_module(monkeypatch):
    """vla_bridge.isaac_robot_interface imports sensor_msgs/std_msgs at module
    level; stub them so this runs without ROS 2."""
    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.JointState = _JointState
    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Float32 = _Float32
    for name, mod in (('sensor_msgs', sensor_msgs), ('sensor_msgs.msg', sensor_msgs_msg),
                      ('std_msgs', std_msgs), ('std_msgs.msg', std_msgs_msg)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, 'vla_bridge.isaac_robot_interface', raising=False)
    import vla_bridge.isaac_robot_interface as module
    yield module
    sys.modules.pop('vla_bridge.isaac_robot_interface', None)


class _MockBridge:
    """What pick_place_scene_bridge.py does in lockstep mode, minus Isaac: take
    one request, advance ONE tick, answer with the same id in the next
    joint_state. `responsive=False` models a bridge that never answers."""

    def __init__(self, responsive=True):
        self.gate = LockstepGate()
        self.ticks = 0
        self.responsive = responsive
        self.executed = []        # (id, joints, gripper) per tick
        self.interface = None

    def on_joint_target(self, msg):
        position = list(msg.position)
        gripper = position[6] if len(position) >= 7 else None
        self.gate.submit(msg.header.stamp.sec, position[:6], gripper)
        if not self.responsive:
            return
        request = self.gate.take()
        if request is None:
            return
        self.ticks += 1                       # exactly one tick
        self.executed.append(request)
        reply = _JointState()
        reply.header.stamp.sec = request[0]
        reply.position = list(request[1]) + [0.0]
        self.interface._on_joint_state(reply)


class _Publisher:
    def __init__(self, sink):
        self.sink = sink
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)
        if self.sink is not None:
            self.sink(msg)


class _Node:
    def __init__(self, bridge):
        self.bridge = bridge
        self.publishers = {}

    def create_publisher(self, msg_type, topic, qos):
        sink = self.bridge.on_joint_target if msg_type is _JointState else None
        self.publishers[topic] = _Publisher(sink)
        return self.publishers[topic]

    def create_subscription(self, *args, **kwargs):
        return None


def _make(interface_module, responsive=True, lockstep=True, timeout=0.05):
    bridge = _MockBridge(responsive=responsive)
    node = _Node(bridge)
    robot = interface_module.IsaacSimRobotInterface(
        node, lockstep=lockstep, lockstep_timeout_s=timeout)
    bridge.interface = robot
    return robot, bridge, node


def test_every_move_joints_is_exactly_one_tick_and_ids_count_up(interface_module):
    robot, bridge, _ = _make(interface_module)
    for k in range(1, 6):
        assert robot.move_joints([0.1 * k] * 6) is True
    assert bridge.ticks == 5
    assert [e[0] for e in bridge.executed] == [1, 2, 3, 4, 5]
    assert robot.last_request_id() == 5 == robot.acked_request_id()


def test_the_gripper_command_travels_in_the_same_message_as_the_joint_target(interface_module):
    robot, bridge, node = _make(interface_module)
    robot.set_gripper(1.0)
    robot.move_joints([0.0] * 6)
    sent = node.publishers['/vla/joint_target'].sent[-1]
    assert len(sent.position) == 7 and sent.position[6] == 1.0
    assert bridge.executed[-1][2] == 1.0          # applied on that same tick


def test_without_an_answer_the_call_times_out_instead_of_hanging(interface_module):
    robot, bridge, _ = _make(interface_module, responsive=False, timeout=0.05)
    started = time.time()
    assert robot.move_joints([0.0] * 6) is False
    assert time.time() - started < 1.0
    assert bridge.ticks == 0


def test_a_reply_for_an_older_request_does_not_count_as_the_ack(interface_module):
    robot, bridge, _ = _make(interface_module, responsive=False, timeout=0.05)
    late = _JointState()
    late.header.stamp.sec = 0
    robot._on_joint_state(late)
    assert robot.move_joints([0.0] * 6) is False


def test_with_lockstep_off_nothing_about_the_old_message_changes(interface_module):
    robot, _, node = _make(interface_module, lockstep=False)
    robot._latest_joint_state = None
    robot.set_gripper(1.0)
    # no ack will ever come in this mode; the old tolerance poll just times out
    robot.settle_timeout_s = 0.05
    robot.move_joints([0.0] * 6)
    sent = node.publishers['/vla/joint_target'].sent[-1]
    assert len(sent.position) == 6
    assert sent.header.stamp.sec == 0
    assert robot.last_request_id() == 0
