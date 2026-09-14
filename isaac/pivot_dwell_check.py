"""Is the pose residual a frame error, contact, or dynamics? Measure it.

Three numbers on record disagree about what was wrong with the arm's
tracking: the residual grew 24 mm -> 43 mm between a 60- and a 120-tick
hold, while the same run without a gripper stayed at 5 mm. Two explanations
fit that, and they call for opposite work:

  CONTACT      the fingers were ploughing into the table, because nothing
               accounted for the 120 mm from flange to fingertips or the
               90 degrees between the flange and tool0. Holding the pose
               pressed harder, so the arm was pushed further off.
               If this was it, _grip_pose_to_tool0 already fixed it.

  DYNAMICS     gravity and the gripper's mass sagging against drive gains
               that are too soft to hold a loaded pose, or the controller
               slowly integrating away from the target. If this is it,
               nothing has been done about it: there is no stiffness,
               damping, mass or inertia set anywhere in this codebase.

The distinction is testable, and the test is cheap: contact needs something
to touch. So this holds a pose in FREE SPACE, where nothing is in reach, and
holds the actual grasp pose, where the fingers are at the table.

    growth in free space         -> dynamics; tune drive gains, set the
                                    gripper's mass and inertia
    growth only at the grasp     -> contact; the frame fix addressed it
    growth in neither            -> the residual is gone, and what remains
                                    is a static offset, which the pivot
                                    test below measures

PIVOT TEST

The standard way to check a tool centre point: keep the tool tip on one
physical spot and swing the arm around it. If the flange-to-fingertip
transform is right, the tip does not move. If it is wrong, rotating about a
point that is not really the tip drags the tip along an arc, and the spread
of where it lands is the size of the error. This commands one grip position
at several orientations and measures that spread -- which is a different
quantity from the dwell drift above, and a static offset shows up here while
being invisible there.

    python3 pivot_dwell_check.py                  # with the gripper fitted
    python3 pivot_dwell_check.py --no-gripper     # the payload ablation

Run both and compare: a residual that is present with the gripper and absent
without it is about the gripper (its mass, or its fingers touching
something), not about the arm.
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import PickPlaceScene  # noqa: E402
from scripted_pick_place import DOWNWARD_ROTVEC, GRASP_HEIGHT  # noqa: E402

# Tick counts the residual is sampled at. 60 and 120 are the two the original
# observation used, so the numbers here are directly comparable to it.
DWELL_SAMPLES = (1, 10, 30, 60, 90, 120, 180)

# A residual that moves less than this between the 60th and 180th tick is
# treated as settled rather than drifting. Chosen well under the 19 mm the
# original observation grew by, and around the scale a grasp can absorb.
SETTLED_TOLERANCE_M = 0.002

# Rolls about the approach axis, which leave the drilling/grasping geometry
# unchanged, plus two tilts. A correct TCP keeps the grip point still through
# all of them.
PIVOT_ROLLS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0)
PIVOT_TILTS_DEG = (0.0, 10.0, -10.0)


def hold(scene, grip_pos, rotvec, ticks, gripper=0.0):
    """Command one pose for `ticks` control ticks, returning the grip-point
    error at each sample point. The target never changes, so anything that
    moves is the arm failing to hold it."""
    errors = {}
    for tick in range(1, min(ticks, max(DWELL_SAMPLES)) + 1):
        scene.step_towards(grip_pos, rotvec, gripper)
        if tick in DWELL_SAMPLES:
            errors[tick] = float(np.linalg.norm(scene.grip_point_world() - grip_pos))
    return errors


def report_dwell(label, errors):
    print(f"\n  {label}")
    print("    " + "  ".join(f"{t:>5}" for t in sorted(errors)))
    print("    " + "  ".join(f"{errors[t] * 1000:>5.1f}" for t in sorted(errors)))
    print("    (ticks above, grip-point error in mm below)")

    late = [t for t in errors if t >= 60]
    if len(late) < 2:
        return None
    growth = errors[max(late)] - errors[min(late)]
    print(f"    growth from tick {min(late)} to {max(late)}: {growth * 1000:+.1f} mm")
    return growth


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-gripper", action="store_true",
                        help="build the scene with no gripper at all (NullGripper), "
                             "which removes both its mass and its fingers")
    parser.add_argument("--free-height", type=float, default=0.35,
                        help="height above the table for the free-space hold, chosen "
                             "so nothing is within reach of the fingers")
    args = parser.parse_args()

    with_gripper = not args.no_gripper
    scene = PickPlaceScene(with_gripper=with_gripper)
    scene.reset()

    print("=" * 68)
    print(f"PIVOT / DWELL CHECK  ({'with' if with_gripper else 'WITHOUT'} gripper)")
    print("=" * 68)

    free_pos = np.array([scene.cube_position[0], scene.cube_position[1], args.free_height])
    grasp_pos = scene.cube_position + np.array([0.0, 0.0, GRASP_HEIGHT])

    try:
        gate = None
        try:
            from feasibility_gate import FeasibilityGate
            from isaac_sim_common import ROBOT_PRIM_PATH
            gate = FeasibilityGate(ROBOT_PRIM_PATH)
        except Exception as exc:  # noqa: BLE001 -- the check is a nicety, not the point
            print(f"  (feasibility gate unavailable: {exc})")

        def reachable(grip_pos, rotvec):
            """Reachability, so an unreachable pose is not read as bad tracking."""
            if gate is None:
                return True, "unchecked"
            tool_pos, tool_quat_wxyz = scene._grip_pose_to_tool0(grip_pos, rotvec)
            verdict = gate.check(tool_pos, tool_quat_wxyz)
            return verdict["ok"], verdict["reason"]

        # --- 1. does the residual grow, and where -------------------------
        print("\n" + "-" * 68)
        print("DWELL: hold one pose and watch the error")
        print("-" * 68)

        ok, reason = reachable(free_pos, DOWNWARD_ROTVEC)
        print(f"  free-space target {np.round(free_pos, 3)} reachable: {ok} ({reason})")
        free_growth = report_dwell(
            "free space -- nothing within reach of the fingers",
            hold(scene, free_pos, DOWNWARD_ROTVEC, max(DWELL_SAMPLES)))

        scene.reset()
        ok, reason = reachable(grasp_pos, DOWNWARD_ROTVEC)
        print(f"\n  grasp target {np.round(grasp_pos, 3)} reachable: {ok} ({reason})")
        grasp_growth = report_dwell(
            "at the grasp -- the fingers are at the table",
            hold(scene, grasp_pos, DOWNWARD_ROTVEC, max(DWELL_SAMPLES)))

        # --- 2. is the tool centre point right ----------------------------
        print("\n" + "-" * 68)
        print("PIVOT: swing the arm around a fixed grip point")
        print("-" * 68)
        print(f"  commanding {np.round(free_pos, 3)} at {len(PIVOT_ROLLS_DEG)} rolls "
              f"x {len(PIVOT_TILTS_DEG)} tilts")
        print(f"\n  {'roll':>6} {'tilt':>6}  {'reached':>8}  {'error':>8}")
        print("  " + "-" * 34)

        landed = []
        base = Rot.from_rotvec(DOWNWARD_ROTVEC)
        for roll in PIVOT_ROLLS_DEG:
            for tilt in PIVOT_TILTS_DEG:
                rotvec = (base
                          * Rot.from_euler("z", roll, degrees=True)
                          * Rot.from_euler("x", tilt, degrees=True)).as_rotvec()
                ok, _ = reachable(free_pos, rotvec)
                if not ok:
                    print(f"  {roll:>5.0f}d {tilt:>5.0f}d  unreachable, skipped")
                    continue
                scene.reset()
                hold(scene, free_pos, rotvec, 60)
                reached = scene.grip_point_world()
                landed.append(reached)
                print(f"  {roll:>5.0f}d {tilt:>5.0f}d  {'yes':>8}  "
                      f"{np.linalg.norm(reached - free_pos) * 1000:>6.1f}mm")

        # --- 3. what it all means -----------------------------------------
        print("\n" + "=" * 68)
        print("VERDICT")
        print("=" * 68)

        if free_growth is not None and grasp_growth is not None:
            free_drifts = abs(free_growth) > SETTLED_TOLERANCE_M
            grasp_drifts = abs(grasp_growth) > SETTLED_TOLERANCE_M
            if free_drifts:
                print(f"  The residual grows by {free_growth * 1000:+.1f} mm in FREE SPACE,")
                print("  where there is nothing to touch. That is dynamics, not contact:")
                print("  tune the wrist drive stiffness/damping and set the gripper's")
                print("  mass and inertia -- none of which this codebase sets today.")
            elif grasp_drifts:
                print(f"  Free space is settled ({free_growth * 1000:+.1f} mm) but the grasp")
                print(f"  pose grows by {grasp_growth * 1000:+.1f} mm. That is contact, and it")
                print("  means the frame correction in _grip_pose_to_tool0 is doing its job")
                print("  in free space while the fingers still reach something at the grasp.")
            else:
                print("  Settled in both -- the growing residual is gone. Whatever remains")
                print("  is a constant offset, which is what the pivot spread below measures.")

        if len(landed) >= 2:
            landed = np.asarray(landed)
            spread = float(np.max(np.linalg.norm(
                landed[:, None, :] - landed[None, :, :], axis=-1)))
            mean_offset = float(np.linalg.norm(landed.mean(axis=0) - free_pos))
            print(f"\n  pivot spread across orientations: {spread * 1000:.1f} mm")
            print(f"  mean offset from the commanded point: {mean_offset * 1000:.1f} mm")
            print("\n  The spread is the frame error: a correct flange-to-fingertip")
            print("  transform keeps the grip point still while the arm rotates around")
            print("  it, so whatever the point moves is how wrong that transform is.")
            print("  The mean offset is different -- a constant bias, which tracking")
            print("  error and a wrong GRIPPER_TCP_OFFSET_M both produce, and which")
            print("  rotating cannot separate.")
            print(f"  These rolls span 180 degrees, so the tip traces a half circle of")
            print(f"  radius equal to the error, making the spread about TWICE it --")
            print(f"  roughly {spread * 500:.1f} mm of transform error here. A pure")
            print("  transform error averages out of the mean offset entirely, so the")
            print("  two numbers measure different faults and do not overlap.")

        print("\n  Then run the other half:  python3 pivot_dwell_check.py"
              f"{' ' if args.no_gripper else ' --no-gripper'}")
        print("  A residual present with the gripper and absent without it is about")
        print("  the gripper -- its mass, or its fingers touching something.")
    except BaseException:
        # Print BEFORE the finally closes Kit. isaac_scene.py's shutdown bug
        # was exactly this shape: simulation_app.close() can take the process
        # down with a native SIGSEGV during Py_FinalizeEx while annotators are
        # still attached, and a traceback that has not been flushed by then is
        # simply lost. Same idiom test_feasibility_gate.py already uses.
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
