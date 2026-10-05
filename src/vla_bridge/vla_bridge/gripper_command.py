"""Turns the policy's gripper output into an open/close COMMAND.

Why this exists (found 2026-10-05 by reading collect_demos.py against the
serving path -- not yet confirmed in the simulator):

The demonstrations COMMANDED the gripper fully closed (1.0, i.e. the
GRIPPER_CLOSED_POS joint target) while holding the cube, but the dataset's
gripper ACTION is the MEASURED position one tick later. A closed finger stalls
on the cube, so that reads ~0.63-0.73 (max over all 152 episodes: 0.75), and
the policy learns to output ~0.7. Sending 0.7 straight back as the command
makes the joint target about where the finger already stalled, so the PD
drive's squeeze force is ~0 and the grasp can slip.

The fix is to treat the output as a decision, not a position: close when it
rises above `close_above`, open when it falls below `open_below`, and hold the
last command in between. The gap between the two thresholds is the hysteresis
-- a value hovering around one number cannot make the gripper chatter.

The defaults are a first guess, chosen from the recorded range (0.0 open,
~0.7 holding); tune them on the first evaluation. They are parameters of
vla_policy_client (`gripper_close_threshold` / `gripper_open_threshold`).
"""


class GripperHysteresis:
    def __init__(self, close_above=0.45, open_below=0.25, initially_closed=False):
        if not open_below < close_above:
            raise ValueError(
                f'open_below ({open_below}) must be strictly less than close_above '
                f'({close_above}); otherwise there is no hysteresis band')
        self.close_above = float(close_above)
        self.open_below = float(open_below)
        self._initially_closed = bool(initially_closed)
        self._closed = self._initially_closed

    def update(self, policy_output):
        """Returns the command to send: 1.0 (closed) or 0.0 (open)."""
        value = float(policy_output)
        if value > self.close_above:
            self._closed = True
        elif value < self.open_below:
            self._closed = False
        return 1.0 if self._closed else 0.0

    def reset(self):
        """Back to the starting state -- call at the start of every episode,
        otherwise a trial that ended holding the cube starts the next one
        commanding closed."""
        self._closed = self._initially_closed
