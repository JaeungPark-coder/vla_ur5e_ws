"""Per-trial maximum cube height as reported by the lockstep bridge.

Why (2026-11-10): eval_mode used to look at the cube once per executed block (up to 25
ticks), so a contact blow-up that flings the cube up and lets it fall within a block was
invisible, and the trial was scored as an ordinary failure. The bridge now keeps a running
maximum of the cube's height over EVERY tick and publishes it with each observation on
/vla/eval/cube_max_z as [epoch, max_z] (std_msgs/Float32MultiArray). `epoch` is the bridge's
reset counter, so a message produced before the scene was reset for the next trial can never
count towards that trial.
"""


class MaxHeightTracker:
    def __init__(self):
        self.min_epoch = 0
        self.value = 0.0

    def start_trial(self, min_epoch):
        """Call when a reset has been requested. Messages from an earlier epoch (the previous
        trial, still being re-published while the reset runs) are ignored from now on."""
        self.min_epoch = int(min_epoch)
        self.value = 0.0

    def update(self, epoch, max_z):
        """Feed one [epoch, max_z] message. Returns True when it counted."""
        if int(epoch) < self.min_epoch:
            return False
        self.value = max(self.value, float(max_z))
        return True
