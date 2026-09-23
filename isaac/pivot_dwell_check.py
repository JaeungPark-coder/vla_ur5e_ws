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
from isaac_sim_common import (  # noqa: E402
    ROBOT_PRIM_PATH, get_joint_drive_gains, set_joint_drive_gains)

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


def say(line=""):
    """print(..., flush=True) -- CONFIRMED 2026-09-15 this file needed it:
    a real run (69s of real dwell/pivot work, no exception, no traceback)
    produced a completely empty log when its stdout was redirected to a
    file. Plain print() left every line sitting in a block-buffered stdout
    that Kit's fastShutdown (see check_cameras.py's own say()/_REPORT
    workaround for the same failure) tore down before the interpreter's
    normal exit ever flushed it. The 2026-09-15 01:17 fix upstream only
    covered the exception path (its own flush=True print before re-raising);
    every print() on the SUCCESS path -- the entire report this script
    exists to produce -- was still exposed."""
    print(line, flush=True)


def settle_to(scene, grip_pos, rotvec, ticks=180):
    """Pre-converge to grip_pos BEFORE the dwell measurement starts, via a
    graduated above->descend approach, and don't count any of these ticks.

    CONFIRMED 2026-09-19 (vla_ur5e_ws/isaac/diag_pixel_to_mm_mapping.py,
    scratchpad): jumping the tool directly from scene.reset()'s home
    configuration to a single low/far target in one shot does not reliably
    converge within a normal tick budget -- one measured case landed 174mm
    off in Y after 120 ticks of doing exactly that. The scripted policy
    (scripted_pick_place.py) never does this; it always goes through a
    graduated approach(above)->settle->descend->settle path. hold() below
    was commanding its target in one jump from reset(), which is almost
    certainly why the 2026-09-15 run's own numbers opened at 432-500mm of
    error at tick 1 and needed hundreds of ticks just to approach -- most
    of DWELL_SAMPLES' budget was spent finishing that initial jump, not
    holding a converged pose, which is what this test needs to actually
    measure CONTACT vs DYNAMICS drift rather than approach-convergence
    drift."""
    above = grip_pos + np.array([0.0, 0.0, 0.15])
    for _ in range(ticks):
        scene.step_towards(above, rotvec, 0.0)
    for _ in range(ticks):
        scene.step_towards(grip_pos, rotvec, 0.0)


def hold(scene, grip_pos, rotvec, ticks, gripper=0.0):
    """Command one pose for `ticks` control ticks, returning the grip-point
    error at each sample point. The target never changes, so anything that
    moves is the arm failing to hold it. Caller is responsible for having
    already converged near grip_pos (see settle_to) -- this measures dwell
    drift from tick 1, not approach convergence."""
    errors = {}
    for tick in range(1, min(ticks, max(DWELL_SAMPLES)) + 1):
        scene.step_towards(grip_pos, rotvec, gripper)
        if tick in DWELL_SAMPLES:
            errors[tick] = float(np.linalg.norm(scene.grip_point_world() - grip_pos))
    return errors


def report_dwell(label, errors):
    say(f"\n  {label}")
    say("    " + "  ".join(f"{t:>5}" for t in sorted(errors)))
    say("    " + "  ".join(f"{errors[t] * 1000:>5.1f}" for t in sorted(errors)))
    say("    (ticks above, grip-point error in mm below)")

    late = [t for t in errors if t >= 60]
    if len(late) < 2:
        return None
    growth = errors[max(late)] - errors[min(late)]
    say(f"    growth from tick {min(late)} to {max(late)}: {growth * 1000:+.1f} mm")
    return growth


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-gripper", action="store_true",
                        help="build the scene with no gripper at all (NullGripper), "
                             "which removes both its mass and its fingers")
    parser.add_argument("--free-height", type=float, default=0.35,
                        help="height above the table for the free-space hold, chosen "
                             "so nothing is within reach of the fingers")
    parser.add_argument("--stiffness-scale", type=float, default=1.0,
                        help="scale factor applied to every arm joint's drive stiffness before "
                             "running the check (1.0 = leave whatever the asset/GUI currently "
                             "has). Use this instead of hand-editing the asset in the GUI so a "
                             "comparison run is reproducible and the before/after gains get "
                             "logged -- e.g. --stiffness-scale 0.5 to test half of whatever was "
                             "raised by hand, per this file's own DYNAMICS verdict.")
    parser.add_argument("--damping-scale", type=float, default=1.0,
                        help="scale factor applied to every arm joint's drive damping, same as "
                             "--stiffness-scale -- raising stiffness alone without damping to "
                             "match is a classic source of the exact contact blow-up this "
                             "project has already hit (see b5e7fb0's 5.26m explosion)")
    args = parser.parse_args()

    with_gripper = not args.no_gripper
    scene = PickPlaceScene(with_gripper=with_gripper)

    say("=" * 68)
    say(f"PIVOT / DWELL CHECK  ({'with' if with_gripper else 'WITHOUT'} gripper)")
    say("=" * 68)

    if with_gripper and (args.stiffness_scale != 1.0 or args.damping_scale != 1.0):
        robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        before = get_joint_drive_gains(robot_prim)
        say(f"\n  joint drive gains BEFORE scaling: {before}")
        for name, gains in before.items():
            if gains is None:
                continue
            stiffness, damping = gains
            set_joint_drive_gains(
                robot_prim, joint_names=(name,),
                stiffness=(stiffness * args.stiffness_scale) if stiffness is not None else None,
                damping=(damping * args.damping_scale) if damping is not None else None)
        after = get_joint_drive_gains(robot_prim)
        say(f"  joint drive gains AFTER scaling:  {after}")

    scene.reset()

    free_pos = np.array([scene.cube_position[0], scene.cube_position[1], args.free_height])
    grasp_pos = scene.cube_position + np.array([0.0, 0.0, GRASP_HEIGHT])

    try:
        gate = None
        try:
            from feasibility_gate import FeasibilityGate
            gate = FeasibilityGate(ROBOT_PRIM_PATH)
        except Exception as exc:  # noqa: BLE001 -- the check is a nicety, not the point
            say(f"  (feasibility gate unavailable: {exc})")

        def reachable(grip_pos, rotvec):
            """Reachability, so an unreachable pose is not read as bad tracking."""
            if gate is None:
                return True, "unchecked"
            tool_pos, tool_quat_wxyz = scene._grip_pose_to_tool0(grip_pos, rotvec)
            verdict = gate.check(tool_pos, tool_quat_wxyz)
            return verdict["ok"], verdict["reason"]

        # --- 1. does the residual grow, and where -------------------------
        say("\n" + "-" * 68)
        say("DWELL: hold one pose and watch the error")
        say("-" * 68)

        ok, reason = reachable(free_pos, DOWNWARD_ROTVEC)
        say(f"  free-space target {np.round(free_pos, 3)} reachable: {ok} ({reason})")
        settle_to(scene, free_pos, DOWNWARD_ROTVEC)
        free_growth = report_dwell(
            "free space -- nothing within reach of the fingers",
            hold(scene, free_pos, DOWNWARD_ROTVEC, max(DWELL_SAMPLES)))

        scene.reset()
        ok, reason = reachable(grasp_pos, DOWNWARD_ROTVEC)
        say(f"\n  grasp target {np.round(grasp_pos, 3)} reachable: {ok} ({reason})")
        settle_to(scene, grasp_pos, DOWNWARD_ROTVEC)
        grasp_growth = report_dwell(
            "at the grasp -- the fingers are at the table",
            hold(scene, grasp_pos, DOWNWARD_ROTVEC, max(DWELL_SAMPLES)))

        # --- 2. is the tool centre point right ----------------------------
        say("\n" + "-" * 68)
        say("PIVOT: swing the arm around a fixed grip point")
        say("-" * 68)
        say(f"  commanding {np.round(free_pos, 3)} at {len(PIVOT_ROLLS_DEG)} rolls "
              f"x {len(PIVOT_TILTS_DEG)} tilts")
        say(f"\n  {'roll':>6} {'tilt':>6}  {'reached':>8}  {'error':>8}")
        say("  " + "-" * 34)

        landed = []
        base = Rot.from_rotvec(DOWNWARD_ROTVEC)
        for roll in PIVOT_ROLLS_DEG:
            for tilt in PIVOT_TILTS_DEG:
                rotvec = (base
                          * Rot.from_euler("z", roll, degrees=True)
                          * Rot.from_euler("x", tilt, degrees=True)).as_rotvec()
                ok, _ = reachable(free_pos, rotvec)
                if not ok:
                    say(f"  {roll:>5.0f}d {tilt:>5.0f}d  unreachable, skipped")
                    continue
                scene.reset()
                # Same convergence bug settle_to() fixed for the two dwell
                # measurements above, left behind here in the 2026-09-19 pass:
                # 60 ticks straight from reset()'s home pose is nowhere near
                # enough to converge (the dwell tests opened at 432-500mm
                # doing exactly this), so what got read was how far each
                # orientation happened to get in 60 ticks, not where its
                # converged grip point sits. That is what the 2026-09-19
                # "~90-100mm transform error, 314-321mm mean offset" numbers
                # measured -- both with and without the gripper, exactly as an
                # approach-convergence artifact would be. Settle first, then
                # hold a little longer so the reading is a converged pose.
                settle_to(scene, free_pos, rotvec)
                hold(scene, free_pos, rotvec, 60)
                reached = scene.grip_point_world()
                landed.append(reached)
                say(f"  {roll:>5.0f}d {tilt:>5.0f}d  {'yes':>8}  "
                      f"{np.linalg.norm(reached - free_pos) * 1000:>6.1f}mm")

        # --- 3. what it all means -----------------------------------------
        say("\n" + "=" * 68)
        say("VERDICT")
        say("=" * 68)

        if free_growth is not None and grasp_growth is not None:
            # 2026-09-23: this used to be abs(growth) > SETTLED_TOLERANCE_M,
            # which cannot tell "still GROWING" from "still SHRINKING (fast
            # convergence not yet finished)" -- both have a large magnitude,
            # only one is the dynamics/contact problem this verdict is
            # supposed to diagnose. CONFIRMED wrong on a real run: it printed
            # "the residual GROWS by -311.3mm... that is dynamics" for a
            # case where growth was actually -311.3mm, i.e. SHRINKING.
            # Split on sign instead: only a positive change past tolerance
            # is "growing" in the sense this verdict means.
            free_growing = free_growth > SETTLED_TOLERANCE_M
            free_shrinking = free_growth < -SETTLED_TOLERANCE_M
            grasp_growing = grasp_growth > SETTLED_TOLERANCE_M
            grasp_shrinking = grasp_growth < -SETTLED_TOLERANCE_M
            if free_growing:
                say(f"  The residual grows by {free_growth * 1000:+.1f} mm in FREE SPACE,")
                say("  where there is nothing to touch. That is dynamics, not contact:")
                say("  tune the wrist drive stiffness/damping and set the gripper's")
                say("  mass and inertia -- none of which this codebase sets today.")
            elif grasp_growing:
                say(f"  Free space is settled ({free_growth * 1000:+.1f} mm) but the grasp")
                say(f"  pose grows by {grasp_growth * 1000:+.1f} mm. That is contact, and it")
                say("  means the frame correction in _grip_pose_to_tool0 is doing its job")
                say("  in free space while the fingers still reach something at the grasp.")
            elif free_shrinking or grasp_shrinking:
                say(f"  Neither residual is GROWING (free {free_growth * 1000:+.1f} mm, "
                    f"grasp {grasp_growth * 1000:+.1f} mm) -- but at least one is still "
                    f"moving by more than {SETTLED_TOLERANCE_M * 1000:.1f} mm in this late "
                    f"window, converging rather than settled. This is not the dynamics/")
                say("  contact failure mode above -- it means the probe target was commanded")
                say("  from too far away (a cold start) for this many ticks to fully settle;")
                say("  the pivot-spread numbers below are measured before convergence and")
                say("  may not be comparable to a run that did settle first.")
            else:
                say("  Settled in both -- the growing residual is gone. Whatever remains")
                say("  is a constant offset, which is what the pivot spread below measures.")

        if len(landed) >= 2:
            landed = np.asarray(landed)
            spread = float(np.max(np.linalg.norm(
                landed[:, None, :] - landed[None, :, :], axis=-1)))
            mean_offset = float(np.linalg.norm(landed.mean(axis=0) - free_pos))
            say(f"\n  pivot spread across orientations: {spread * 1000:.1f} mm")
            say(f"  mean offset from the commanded point: {mean_offset * 1000:.1f} mm")
            say("\n  The spread is the frame error: a correct flange-to-fingertip")
            say("  transform keeps the grip point still while the arm rotates around")
            say("  it, so whatever the point moves is how wrong that transform is.")
            say("  The mean offset is different -- a constant bias, which tracking")
            say("  error and a wrong GRIPPER_TCP_OFFSET_M both produce, and which")
            say("  rotating cannot separate.")
            say("  These rolls span 180 degrees, so the tip traces a half circle of")
            say("  radius equal to the error, making the spread about TWICE it --")
            say(f"  roughly {spread * 500:.1f} mm of transform error here. A pure")
            say("  transform error averages out of the mean offset entirely, so the")
            say("  two numbers measure different faults and do not overlap.")

        say("\n  Then run the other half:  python3 pivot_dwell_check.py"
              f"{' ' if args.no_gripper else ' --no-gripper'}")
        say("  A residual present with the gripper and absent without it is about")
        say("  the gripper -- its mass, or its fingers touching something.")
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
