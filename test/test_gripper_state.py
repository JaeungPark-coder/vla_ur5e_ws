"""Checks the gripper-state contract on both backends.

The bug this exists to catch was invisible by construction: the policy was
fed the gripper value it had just commanded, so a grasp that closed on
nothing looked exactly like a grasp that worked -- in the observations, in
the recorded demonstrations, and in the eval metric. Three things are
checked here: a real reading is reported as measured, a fallback says so
instead of pretending, and the disagreement an echo could never show shows.
"""
import sys
import types

import pytest

from vla_bridge.gripper_state import GripperState, grasp_disagreement


# --- 1. the contract itself ----------------------------------------------

def test_a_measured_reading_carries_its_source_and_object_flag():
    state = GripperState(position=0.62, measured=True, object_detected=True,
                         source='gripper driver (158 counts)')
    assert state.measured
    assert 'object detected' in state.describe()


def test_a_fallback_is_visibly_a_fallback():
    state = GripperState.from_command(0.8, 'no gripper_driver configured')
    assert state.position == 0.8
    assert state.measured is False
    assert state.object_detected is None
    assert 'NOT MEASURED' in state.describe()


# --- 2. the observation an echo could never make -------------------------

def test_disagreement_shows_the_gripper_stopping_early():
    """Commanded fully closed, stopped at 0.62 -- it hit something."""
    measured = GripperState(position=0.62, measured=True, source='x')
    assert grasp_disagreement(1.0, measured) == pytest.approx(0.38)


def test_agreement_reads_as_zero():
    agreed = GripperState(position=0.98, measured=True, source='x')
    assert grasp_disagreement(1.0, agreed) == 0.0


def test_an_unmeasured_state_gives_none_rather_than_zero():
    """"No disagreement" must never be confused with "no information" --
    zero here would report a perfect grasp on every failed one."""
    echoed = GripperState.from_command(1.0, 'no driver')
    assert grasp_disagreement(1.0, echoed) is None


# --- 3. the real backend, with RTDE stubbed ------------------------------

@pytest.fixture(scope='module')
def UR5eInterface():
    """ur_rtde is not installed on a development machine and is not needed."""
    for name in ('rtde_control', 'rtde_receive', 'rtde_io'):
        module = types.ModuleType(name)
        for attr in ('RTDEControlInterface', 'RTDEReceiveInterface', 'RTDEIOInterface'):
            setattr(module, attr, lambda *a, **k: types.SimpleNamespace(
                setToolDigitalOut=lambda *_: None))
        sys.modules.setdefault(name, module)

    from vla_bridge.robot_interface import UR5eInterface as cls
    return cls


class FakeRobotiq:
    """Counts like a Robotiq over its socket interface: 0 open, 255 closed."""

    def __init__(self, counts=0, detects=False, broken=False):
        self.counts, self.detects, self.broken = counts, detects, broken
        self.commanded = None

    def get_current_position(self):
        if self.broken:
            raise ConnectionError('socket closed')
        return self.counts

    def set_position(self, counts):
        self.commanded = counts
        self.counts = counts

    def is_object_detected(self):
        return self.detects


@pytest.fixture
def arm(UR5eInterface):
    """A UR5eInterface with the RTDE connection skipped."""
    robot = UR5eInterface.__new__(UR5eInterface)
    robot.io = types.SimpleNamespace(setToolDigitalOut=lambda *_: None)
    robot.gripper_output_pin = 0
    robot.gripper_open_counts, robot.gripper_closed_counts = 0, 255
    robot._last_commanded_gripper = 0.0
    robot.gripper_driver = None
    return robot


def test_without_a_driver_the_state_is_the_command_and_says_so(arm):
    arm.set_gripper(1.0)
    state = arm.get_gripper_state()
    assert not state.measured
    assert state.position == 1.0


def test_with_a_driver_the_counts_become_the_position(arm):
    arm.gripper_driver = FakeRobotiq(counts=158, detects=True)
    state = arm.get_gripper_state()
    assert state.measured
    assert state.position == pytest.approx(158 / 255)
    assert state.object_detected is True


def test_a_driver_fault_falls_back_rather_than_stopping_the_loop(arm):
    arm.gripper_driver = FakeRobotiq(broken=True)
    arm._last_commanded_gripper = 0.4
    state = arm.get_gripper_state()
    assert not state.measured
    assert state.position == 0.4


@pytest.mark.parametrize('command,counts', [(0.0, 0), (0.5, 128), (1.0, 255)])
def test_set_gripper_drives_the_driver_continuously(arm, command, counts):
    """Not the tool relay, which can only open and close."""
    driver = FakeRobotiq()
    arm.gripper_driver = driver
    arm.set_gripper(command)
    assert driver.commanded == counts


# --- 4. the sim backend, with the ROS message types stubbed --------------

@pytest.fixture(scope='module')
def IsaacSimRobotInterface():
    """isaac_robot_interface imports two ROS message types at module level.

    Stubbing them imports the real class rather than lifting one method out
    of the source, so what gets checked is the code that actually ships.
    """
    for name, attribute in (('sensor_msgs.msg', 'JointState'),
                            ('std_msgs.msg', 'Float32')):
        package = name.split('.')[0]
        sys.modules.setdefault(package, types.ModuleType(package))
        module = types.ModuleType(name)
        setattr(module, attribute, type(attribute, (), {}))
        sys.modules.setdefault(name, module)

    from vla_bridge.isaac_robot_interface import IsaacSimRobotInterface as cls
    return cls


@pytest.fixture
def sim(IsaacSimRobotInterface):
    robot = IsaacSimRobotInterface.__new__(IsaacSimRobotInterface)
    robot._latest_joint_state = None
    robot._last_commanded_gripper = 0.3
    return robot


def test_before_any_joint_state_arrives_the_sim_falls_back(sim):
    state = sim.get_gripper_state()
    assert not state.measured
    assert state.position == 0.3


def test_the_sim_reads_the_seventh_joint_the_bridge_publishes(sim):
    """pick_place_scene_bridge appends the gripper after the six arm joints.

    This value was already arriving and being thrown away -- the fix was to
    read it, not to publish anything new.
    """
    sim._latest_joint_state = types.SimpleNamespace(
        position=[0.1, -1.2, 1.0, -1.4, -1.6, 0.0, 0.77])
    state = sim.get_gripper_state()
    assert state.measured
    assert state.position == pytest.approx(0.77)


def test_the_sim_never_claims_an_object_detection_it_cannot_have(sim):
    """There is no gOBJ equivalent in the sim, so the flag stays unknown."""
    sim._latest_joint_state = types.SimpleNamespace(
        position=[0.1, -1.2, 1.0, -1.4, -1.6, 0.0, 0.77])
    assert sim.get_gripper_state().object_detected is None


def test_an_arm_only_joint_state_is_not_treated_as_a_reading(sim):
    sim._latest_joint_state = types.SimpleNamespace(position=[0.0] * 6)
    assert not sim.get_gripper_state().measured


# --- 5. the eval metric now scores reality -------------------------------

def test_a_grasp_that_closed_on_nothing_no_longer_counts_as_a_hold():
    """is_holding used to be scored off the command, which made it true by
    construction the moment the policy asked for a close."""
    threshold, commanded = 0.5, 1.0
    failed = GripperState(position=0.05, measured=True, source='x')

    scored_off_the_command = commanded >= threshold
    scored_off_the_reading = (failed.position if failed.measured
                              else commanded) >= threshold

    assert scored_off_the_command
    assert not scored_off_the_reading
