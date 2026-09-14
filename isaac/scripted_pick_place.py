"""Ground-truth-driven waypoint "expert" demonstrator for the pick-and-place
task: approach above the cube, descend, grasp, lift, transport to the
target, descend, release, retract.

Used as the demonstration source for the first validation pass because we
have exact object pose in sim -- no teleoperation hardware needed to get an
initial dataset. Trajectories are scripted-perfect rather than
human-natural, which is a real gap worth revisiting with actual
teleoperation once the collect -> fine-tune -> serve -> deploy pipeline is
proven end-to-end (scripted demos risk teaching the policy an unnaturally
crisp, possibly-brittle motion style).
"""
import numpy as np

# Gripper pointing straight down (+Z of the tool aligned with -Z world) --
# ADJUST if your gripper's zero-orientation convention differs from
# potato_scan's tool0 convention this was modeled after.
DOWNWARD_ROTVEC = np.array([0.0, np.pi, 0.0])

STANDOFF_HEIGHT = 0.15  # meters above the table for approach/retract waypoints
GRASP_HEIGHT = 0.02     # tool height when grasping/placing -- matches the cube's resting height


class ScriptedPickPlace:
    def __init__(self, start_tool_pos, cube_position, target_position, steps_per_segment=90):
        """steps_per_segment=90 (was 30): measured with reach_probe against
        this exact scene, 30 ticks (0.5s) per segment moves the Cartesian
        target faster than RMPflow's default UR5e gains track it -- the tool
        was still 259mm from the cube at the moment the gripper closed. 90
        ticks (1.5s/segment) converges to ~40mm before the close segment
        starts. This changes demonstration *timing*, not the task -- widen
        further if a future asset/gain change makes tracking lag again."""
        self.start_tool_pos = np.asarray(start_tool_pos, dtype=float)
        self.waypoints = self._build_waypoints(
            np.asarray(cube_position, dtype=float), np.asarray(target_position, dtype=float), steps_per_segment)

    @staticmethod
    def _build_waypoints(cube_position, target_position, steps_per_segment):
        above_cube = cube_position + np.array([0.0, 0.0, STANDOFF_HEIGHT])
        at_cube = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT])
        above_target = target_position + np.array([0.0, 0.0, STANDOFF_HEIGHT])
        at_target = target_position + np.array([0.0, 0.0, GRASP_HEIGHT])

        # (target_position, target_gripper, num_control_ticks_to_reach_it)
        return [
            (above_cube, 0.0, steps_per_segment),        # approach from above
            (at_cube, 0.0, steps_per_segment),            # descend to the cube
            (at_cube, 1.0, steps_per_segment // 2),        # close the gripper in place
            (above_cube, 1.0, steps_per_segment),          # lift
            (above_target, 1.0, steps_per_segment),        # transport
            (at_target, 1.0, steps_per_segment),            # descend to the target
            (at_target, 0.0, steps_per_segment // 2),      # open the gripper, release
            (above_target, 0.0, steps_per_segment),         # retract
        ]

    def generate_frames(self):
        """Yields (target_pos, target_rotvec, target_gripper) once per
        control tick across the whole episode, linearly interpolating
        Cartesian position AND the gripper command within each segment
        (orientation held fixed downward throughout -- this task never needs
        to reorient).

        The gripper is interpolated for the same reason try_compliant_close.py
        found empirically: commanding it fully closed from the first tick of
        the "close" segment snaps the fingers shut in one control step, and
        that contact impulse was measured (reach_probe) to knock the arm
        259mm->374mm off target -- worse than never having converged at all.
        Ramping it over the segment (already steps_per_segment//2 ticks,
        close to that script's 60-tick ramp) lets contact form gradually."""
        current_pos = self.start_tool_pos
        current_gripper = 0.0
        for target_pos, target_gripper, num_ticks in self.waypoints:
            for i in range(1, num_ticks + 1):
                alpha = i / num_ticks
                interpolated_pos = current_pos + alpha * (target_pos - current_pos)
                interpolated_gripper = current_gripper + alpha * (target_gripper - current_gripper)
                yield interpolated_pos, DOWNWARD_ROTVEC, interpolated_gripper
            current_pos = target_pos
            current_gripper = target_gripper

    def frames_until_grasp(self):
        """Frame index at which the gripper has just finished closing on the
        cube -- the approach/descend/close segments, before the lift.

        Exposed because callers that want to look at "the grasp" were
        guessing a fraction of total_frames() and getting it wrong:
        check_cameras.py used 45%, which is frame 94 of 210, i.e. 19 frames
        INTO the lift, with the tool already 14.5 cm away from the cube. The
        boundary is a property of the waypoint list, so read it from there."""
        return sum(num_ticks for _, _, num_ticks in self.waypoints[:3])

    def total_frames(self):
        return sum(num_ticks for _, _, num_ticks in self.waypoints)
