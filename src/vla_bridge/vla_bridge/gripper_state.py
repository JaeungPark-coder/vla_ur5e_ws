"""What the gripper is actually doing, as opposed to what it was told.

Both backends used to answer "where is the gripper?" by echoing the last
value commanded, and vla_policy_client fed that echo to the policy as
proprioception. That makes one particular observation impossible to make:

    commanded close, gripper did not close

which is exactly the observation a grasp policy needs. An echo can never
disagree with the command, so a grasp that closed on nothing, stalled on
something too large, or slipped afterwards all look identical to a grasp
that worked. Training on it teaches the policy that closing always succeeds,
and at run time it removes the only signal that would say otherwise.

This module holds the contract that fixes it. Both backends return a
GripperState that says where the gripper is AND whether anyone actually
measured that, so a fallback to the old behaviour is visible rather than
silent.

The measurement exists on both sides; neither is a platform limitation:

  * Isaac Sim already publishes it. pick_place_scene_bridge's
    publish_observation appends the gripper to joint_state, and the value it
    appends is GripperController.get_normalized_position() -- a reading of
    the drive joint, already normalised to 0..1. The backend was slicing it
    off with [:6] and then echoing the command instead.

  * A real Robotiq 2F-85 reports both position and object detection over its
    socket interface: the gOBJ status byte distinguishes "fingers stopped
    early because they hit something" from "fingers reached the requested
    position", which is a grasp-success signal straight from the hardware.
    Note it needs the socket/Modbus path -- with the discrete I/O coupling,
    object detection survives but position feedback does not.
"""
import dataclasses


@dataclasses.dataclass(frozen=True)
class GripperState:
    """position is 0.0 fully open .. 1.0 fully closed.

    `measured` is the field that matters. False means this is the last value
    commanded, standing in for a reading nobody took -- usable, but it cannot
    disagree with the command and so cannot report a failed grasp.

    `object_detected` is tri-state on purpose: True and False are findings,
    None means the question was not answerable (the simulator has no
    equivalent of gOBJ unless a contact sensor is wired up, and the relay
    fallback has no sensing at all).
    """
    position: float
    measured: bool
    object_detected: bool = None
    source: str = ''

    @classmethod
    def from_command(cls, position, reason):
        """The fallback: no reading available, so report the command and say
        so. `reason` should name what is missing, since that is the thing to
        fix before trusting gripper-dependent behaviour."""
        return cls(position=float(position), measured=False,
                   object_detected=None, source=f'command echo ({reason})')

    def describe(self):
        detail = '' if self.object_detected is None else (
            f", object {'detected' if self.object_detected else 'not detected'}")
        return (f'{self.position:.3f} '
                f'[{"measured" if self.measured else "NOT MEASURED"}: {self.source}]'
                f'{detail}')


def grasp_disagreement(commanded, state, tolerance=0.15):
    """How far the gripper ended up from where it was told to go, or None
    when that cannot be known.

    This is the quantity the echo made unobservable. A large disagreement
    after the gripper has had time to settle means it stopped early, which
    on a closing command means it hit something -- either the object (good)
    or nothing at all while stalling (not good). Returns None rather than 0.0
    for unmeasured state, so "no disagreement" is never confused with "no
    information".
    """
    if not state.measured:
        return None
    difference = abs(float(commanded) - state.position)
    return difference if difference > tolerance else 0.0
