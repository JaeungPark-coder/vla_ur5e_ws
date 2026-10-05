"""Optional: turns the policy's gripper output into an open/close COMMAND.

STATUS: OFF by default (vla_policy_client `gripper_hysteresis: false`). Written on a
guess that turned out wrong, kept for experiments. History, so nobody re-derives it:

The demonstrations COMMANDED the gripper fully closed (1.0) while holding the cube,
but the dataset's gripper ACTION is the MEASURED position one tick later. A closed
finger stalls on the cube, so that reads ~0.63-0.73 (max over all 152 episodes:
0.75), and the policy learns to output ~0.7. The guess: sending ~0.7 straight back
makes the joint target about where the finger already stalled, so the PD squeeze
force is ~0 and the grasp slips.

Measured 2026-10-05 (isaac/check_action_replay.py: 6 + 4 recorded episodes replayed
through the bridge's physics path with absolute arm targets): recorded value sent
as-is ("raw") lifted and placed 10/10; a continuous clip(v/0.7) ("scale") 6/6; this
hysteresis 8/10, with larger place errors (8-21 mm vs ~6 mm). The hysteresis jumps
the command to 1.0 the moment the output crosses `close_above`, i.e. it snaps the
gripper shut instead of ramping it as the demonstrations did (README, 2026-09-14:
snapping shut makes contact non-deterministic). The guess was therefore wrong for
this simulator; the raw value is fine. Only if a trained policy is later seen to
hover or chatter near the stall value is it worth turning this on, and then
prefer a smooth mapping over a snap.

The thresholds are a first guess from the recorded range (0.0 open, ~0.7 holding):
close when the output rises above `close_above`, open when it falls below
`open_below`, hold the last command in between.
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
