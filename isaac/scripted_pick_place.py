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
    def __init__(self, start_tool_pos, cube_position, target_position, steps_per_segment=30):
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
        Cartesian position within each segment (orientation held fixed
        downward throughout -- this task never needs to reorient)."""
        current_pos = self.start_tool_pos
        for target_pos, target_gripper, num_ticks in self.waypoints:
            for i in range(1, num_ticks + 1):
                alpha = i / num_ticks
                interpolated_pos = current_pos + alpha * (target_pos - current_pos)
                yield interpolated_pos, DOWNWARD_ROTVEC, target_gripper
            current_pos = target_pos

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
