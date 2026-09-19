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

# physics_dt in pick_place_scene.py -- 90 ticks/1.5s in the comment below is
# this, not a coincidence.
CONTROL_HZ = 60.0

# CONFIRMED 2026-09-16 this needed to exist: a flat steps_per_segment=90 for
# EVERY segment, regardless of how far that segment actually has to travel,
# left the tool short of the cube on 80/91 (88%) of random grasp attempts
# sampled across a wrist-camera sweep (median 130mm off, worst 290mm) --
# `check_cameras.py --sweep_wrist`'s own multi-sample fix (same investigation)
# is what surfaced this; no camera mount could be validated against a grasp
# that mostly did not happen. steps_per_segment=90 was tuned once, at
# whatever distance reach_probe happened to command -- not against
# CUBE_X_RANGE x CUBE_Y_RANGE's full 0.20 x 0.40m spawn spread, where the
# first segment's distance (the arm's fixed start pose to `above_cube`)
# varies with where the cube landed. A short reach converges fine in 90
# ticks; a long one does not, and every attempt got the same fixed tick
# count regardless.
#
# MIN_TRACKING_SPEED_MPS is a conservative, NOT YET independently validated
# estimate of how fast a Cartesian target can move while RMPflow's default
# UR5e gains still track it closely -- chosen so that a segment gets AT
# LEAST steps_per_segment ticks (preserving the original, validated timing
# for short reaches) and MORE for longer ones, rather than replacing one
# unvalidated constant with another equally unvalidated one. Confirm/retune
# by re-running the same multi-sample reach-rate check
# (`check_cameras.py --sweep_wrist --wrist_samples 5`, watching the
# `tool is Xmm from the cube` / `WARNING: the tool did not actually reach`
# lines) and adjusting this until the failure rate actually drops -- not
# by trusting this number on its own.
MIN_TRACKING_SPEED_MPS = 0.15


def _ticks_for_distance(distance_m, floor_ticks):
    """At least `floor_ticks` (the original, validated short-reach timing),
    more if the distance would need it at MIN_TRACKING_SPEED_MPS."""
    return max(floor_ticks, int(np.ceil(distance_m / MIN_TRACKING_SPEED_MPS * CONTROL_HZ)))


# CONFIRMED 2026-09-16, and this is the actual fix the 88% miss rate above
# needed -- MIN_TRACKING_SPEED_MPS was not it. Logging cube position
# alongside tool position through the SAME 91-sample sweep showed the tool
# converging to within 2-39mm of its INTENDED target in nearly every sample
# (tracking was never the problem) while the CUBE independently drifted
# 0-301mm from where it spawned, WITH NO GRIPPER EVEN ATTACHED. Re-run WITH
# the gripper (the real configuration) confirmed the same thing: cube drift
# 34-225mm across 12 fresh samples, again with the tool tracking its
# intended target closely in most of them. The reach failures were never a
# camera or a tracking problem -- they were measuring distance to a cube
# that had already been knocked away by the descent that was supposed to
# approach it.
#
# The likely mechanism: `at_cube`'s Z (cube_position.z + GRASP_HEIGHT) is
# cube_position.z + 0.02 -- for a 4cm cube resting with its centre at
# cube_position.z, that is exactly the cube's TOP FACE, zero clearance.
# Segment 1 ("descend to the cube") is a pure vertical drop only if segment
# 0 ("approach from above") has fully converged in X/Y first -- and Track
# B's own pivot_dwell_check measured a real 20-40mm free-space tracking
# residual that does not vanish the instant a segment's tick budget ends.
# 20-40mm off-centre against a 40mm cube is enough to clip an edge instead
# of centring on top of it, and this cube is light (50g) -- exactly the
# shape of this project's own already-documented contact-explosion history
# (see isaac_sim_common.py's convexDecomposition/maxDepenetrationVelocity
# fixes for the CLOSE segment), just occurring one segment earlier, and
# just as present with no gripper attached at all.
#
# Fix: let the approach segment's residual actually settle -- Track B's own
# dwell measurement showed free-space error dropping to under ~20mm by
# tick 90-180 -- INTO A HOLD AT THE SAME POINT, before the vertical descent
# that risks contact begins. This is a zero-distance "segment" at
# above_cube, so _ticks_for_distance's floor applies unchanged.
#
# CONFIRMED 2026-09-19: this fix was only applied at ONE of the two places
# it was needed. collect_demos.py --num_episodes 5 (with_gripper=True, the
# real configuration) failed 0/5 -- a tick-by-tick trace showed the tool
# 21mm from the cube at the exact moment the CLOSE segment starts (right
# after descend, with no settle in between, unlike above_cube's settle
# before descend), the cube then shoved ~30mm sideways as the fingers
# closed on it off-centre instead of around it, and the gripper closing
# fully (1.00) on empty air -- the cube never left the table for the rest
# of the episode. The SAME pattern this comment already describes for
# "approach residual before descend" applies just as much to "descend
# residual before close": descend is itself a 13cm vertical move (STANDOFF_
# HEIGHT to GRASP_HEIGHT) that generates its own tracking residual and,
# unlike the approach-then-descend case, nothing let it settle before the
# next segment (closing) started. This is also the likely mechanism behind
# a separate 30.7m contact-explosion event seen in the same run: descend
# ending exactly when close begins means the tool may still have nonzero
# residual VELOCITY, not just position error, at first contact -- a moving
# fingertip hitting the cube is a different (and worse) impulse than a
# stationary one nudging it, which the existing maxDepenetrationVelocity
# cap (isaac_sim_common.py) is not designed to absorb.
SETTLE_TICKS = 90


class ScriptedPickPlace:
    def __init__(self, start_tool_pos, cube_position, target_position, steps_per_segment=90):
        """steps_per_segment=90 (was 30): measured with reach_probe against
        this exact scene, 30 ticks (0.5s) per segment moves the Cartesian
        target faster than RMPflow's default UR5e gains track it -- the tool
        was still 259mm from the cube at the moment the gripper closed. 90
        ticks (1.5s/segment) converges to ~40mm before the close segment
        starts, AT THE DISTANCE THAT WAS TESTED -- see MIN_TRACKING_SPEED_MPS
        above for why that no longer means every segment gets exactly 90."""
        self.start_tool_pos = np.asarray(start_tool_pos, dtype=float)
        self.waypoints = self._build_waypoints(
            self.start_tool_pos,
            np.asarray(cube_position, dtype=float), np.asarray(target_position, dtype=float), steps_per_segment)

    @staticmethod
    def _build_waypoints(start_tool_pos, cube_position, target_position, steps_per_segment):
        above_cube = cube_position + np.array([0.0, 0.0, STANDOFF_HEIGHT])
        at_cube = cube_position + np.array([0.0, 0.0, GRASP_HEIGHT])
        above_target = target_position + np.array([0.0, 0.0, STANDOFF_HEIGHT])
        at_target = target_position + np.array([0.0, 0.0, GRASP_HEIGHT])

        # (from_position, to_position, target_gripper, floor_ticks) -- the
        # actual tick count is derived per segment from the real distance
        # from_position -> to_position, below.
        segments = [
            (start_tool_pos, above_cube, 0.0, steps_per_segment),      # approach from above
            (above_cube, above_cube, 0.0, SETTLE_TICKS),               # settle before descending -- see SETTLE_TICKS
            (above_cube, at_cube, 0.0, steps_per_segment),             # descend to the cube
            (at_cube, at_cube, 0.0, SETTLE_TICKS),                     # settle before closing -- see SETTLE_TICKS
            (at_cube, at_cube, 1.0, steps_per_segment // 2),           # close the gripper in place
            (at_cube, above_cube, 1.0, steps_per_segment),             # lift
            (above_cube, above_target, 1.0, steps_per_segment),       # transport
            (above_target, at_target, 1.0, steps_per_segment),         # descend to the target
            (at_target, at_target, 0.0, steps_per_segment // 2),      # open the gripper, release
            (at_target, above_target, 0.0, steps_per_segment),         # retract
        ]

        # (target_position, target_gripper, num_control_ticks_to_reach_it)
        return [
            (to_pos, gripper, _ticks_for_distance(float(np.linalg.norm(to_pos - from_pos)), floor_ticks))
            for from_pos, to_pos, gripper, floor_ticks in segments
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
        cube -- the approach/settle/descend/close segments, before the lift.

        Exposed because callers that want to look at "the grasp" were
        guessing a fraction of total_frames() and getting it wrong:
        check_cameras.py used 45%, which is frame 94 of 210, i.e. 19 frames
        INTO the lift, with the tool already 14.5 cm away from the cube. The
        boundary is a property of the waypoint list, so read it from there.

        [:5] = approach, settle, descend, settle, close -- see
        _build_waypoints' `segments` list (a settle segment before descend
        was added 2026-09-16; a second one after descend, before close, was
        added 2026-09-19 -- keep this slice in sync with that list's order
        if it changes again)."""
        return sum(num_ticks for _, _, num_ticks in self.waypoints[:5])

    def total_frames(self):
        return sum(num_ticks for _, _, num_ticks in self.waypoints)
