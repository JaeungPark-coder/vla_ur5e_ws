"""check_place_dynamics.py found a persistent ~30-56mm RMPflow tracking bias
(commanded grip-point vs actual) that appears exactly when the gripper is
CLOSED (from the "close" segment onward) and does NOT go away even with a
400-tick (6.7s) dwell (check_place_dynamics.py's SETTLE_TICKS=400 rerun) --
it's a steady-state bias, not a settling-time problem. Before the descend
segment (gripper open, no cube grasped yet), the same dwell mechanism
converges tcp_err to ~13-18mm cleanly.

Two competing explanations:
  (a) self-collision-avoidance geometry: RMPflow's own UR5e/gripper self-
      collision model reacts differently to a CLOSED gripper pose (finger
      links occupy different space) than an open one, independent of
      whether anything is actually grasped.
  (b) payload/gravity-compensation mismatch: RMPflow's motion policy config
      assumes the robot's own inertia, not the extra ~50g cube mass at the
      end effector once something is actually held.

setup_rmpflow (isaac_sim_common.py) calls load_supported_motion_policy_config
("UR5e", "RMPflow") with no obstacle registration for the cube anywhere in
this codebase (confirmed by grep) -- so "RMPflow avoids the cube as an
obstacle" is NOT a live hypothesis here, only (a) and (b) above are.

Decisive, cheap test: reproduce the EXACT same approach->settle->descend
structure scripted_pick_place.py already uses and validates (same target
positions, same tick counts, and -- v2, CONFIRMED load-bearing -- the SAME
linear interpolation generate_frames() uses, not a step command straight to
the final target), but with the cube moved far away first (no payload/
contact possible either way) and the gripper commanded to a FIXED value
(0.0 or 1.0) for the whole descend+dwell -- the only thing that differs
between the two runs of this script is that one value. If tcp_err still
jumps to ~30-50mm in the CLOSED case, that is (a). If it stays tight like
the open case, the bias needs an actual held payload, i.e (b).

CONFIRMED 2026-09-28, v1 of this script (step command, not ramped):
open=38.9mm, closed=41.1mm at a FIXED target -- looked like (a) was ruled
out, but the "open" baseline itself was already 2-3x worse than the real
episode's actual (ramped) descend->settle2 convergence (13-18mm) at
similar-class positions, meaning v1's own baseline was contaminated by a
step-command artifact (see check_gravity_droop_prediction.py's docstring --
it hit the same ~40mm plateau from a third, unrelated step-commanded
approach). This is Koren & Borenstein (1991)'s classic potential-field
local-minimum/path-dependence issue -- RMPflow combines multiple RMPs via a
Riemannian metric, the same family of methods. Fixed in v2: descend now
ramps exactly like generate_frames() does.

Each case is its own process (matches check_free_space_closure.py's fix for
the same issue): constructing PickPlaceScene() twice in one Kit session
double-defines the camera xformOps and throws.

    python3 check_gripper_closed_tracking_bias.py --case open
    python3 check_gripper_closed_tracking_bias.py --case closed
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, CUBE_Z  # noqa: E402
from scripted_pick_place import STANDOFF_HEIGHT, GRASP_HEIGHT, DOWNWARD_ROTVEC  # noqa: E402

DESCEND_TICKS = 90     # matches scripted_pick_place.py's steps_per_segment for descend
DWELL_TICKS = 400      # matches check_place_dynamics.py's long-settle diagnostic run
FAR_AWAY = np.array([1.5, 1.5, 0.02])
FIXED_TARGET_XY = np.array([0.45, 0.0])  # centre of CUBE_X_RANGE x CUBE_Y_RANGE -- a realistic, reachable spot


def say(line=""):
    print(line, flush=True)


def run(gripper_target):
    scene = PickPlaceScene(with_gripper=True)
    obs = scene.reset()
    start_pos = obs["tool_pos"].copy()

    # Move the cube far out of the way -- no payload, no contact possible in
    # EITHER case, isolating gripper-state (self-collision geometry) from
    # payload/gravity-compensation effects. Reuse scene.cube_rigid_prim
    # (already constructed by reset()); constructing a NEW SingleRigidPrim
    # after a reset crashes ("Simulation view object is invalidated" -- see
    # pick_place_scene.py's own comment on this exact pitfall).
    scene.cube_rigid_prim.set_world_pose(position=FAR_AWAY)
    scene.cube_rigid_prim.set_linear_velocity(np.zeros(3))
    scene.world.step(render=False)

    # CUBE_Z + GRASP_HEIGHT, exactly matching scripted_pick_place.py's real
    # at_cube = cube_position + [0,0,GRASP_HEIGHT] (cube_position's own z IS
    # CUBE_Z) -- using GRASP_HEIGHT alone here (an earlier version of this
    # script's bug) put the target 2cm lower, right at/near the table
    # surface, which is a different confound (table-clearance) from the one
    # under test.
    above = np.array([FIXED_TARGET_XY[0], FIXED_TARGET_XY[1], CUBE_Z + STANDOFF_HEIGHT])
    at = np.array([FIXED_TARGET_XY[0], FIXED_TARGET_XY[1], CUBE_Z + GRASP_HEIGHT])

    # approach: RAMPED (linearly interpolated), exactly matching
    # ScriptedPickPlace.generate_frames()'s own alpha = i/num_ticks formula
    # -- a step command straight to the final target converges to a WORSE
    # steady state than a ramp does (see this file's docstring, v1's bug).
    for i in range(1, DESCEND_TICKS + 1):
        alpha = i / DESCEND_TICKS
        interp = start_pos + alpha * (above - start_pos)
        scene.step_towards(interp, DOWNWARD_ROTVEC, 0.0)
    # settle: dwell AT `above`, matching the real settle segment (zero-
    # distance, so ramping vs holding are the same thing here).
    for _ in range(DESCEND_TICKS):
        scene.step_towards(above, DOWNWARD_ROTVEC, 0.0)
    approach_err_mm = 1000.0 * float(np.linalg.norm(above - scene.grip_point_world()))
    say(f"after approach(ramped)+settle (gripper open, unrelated to the test): tcp_err={approach_err_mm:.1f}mm")

    # descend (RAMPED, above -> at) + dwell, AT THE FIXED gripper_target
    # under test the whole time -- structurally identical to descend->
    # settle2 (real episode) / descend2->settle3 (place side), just with the
    # cube moved away.
    for i in range(1, DESCEND_TICKS + 1):
        alpha = i / DESCEND_TICKS
        interp = above + alpha * (at - above)
        scene.step_towards(interp, DOWNWARD_ROTVEC, gripper_target)

    tcp_errs = []
    for tick in range(1, DWELL_TICKS + 1):
        scene.step_towards(at, DOWNWARD_ROTVEC, gripper_target)
        actual = scene.grip_point_world()
        tcp_err_mm = 1000.0 * float(np.linalg.norm(at - actual))
        tcp_errs.append(tcp_err_mm)
        if tick % 40 == 0 or tick == 1:
            say(f"  dwell tick {tick:>3}: tcp_err={tcp_err_mm:6.1f}mm actual_gripper={scene.gripper.get_normalized_position():.3f}")

    say(f"steady-state (last 50 of {DWELL_TICKS} dwell ticks): "
        f"mean={np.mean(tcp_errs[-50:]):.1f}mm max={np.max(tcp_errs[-50:]):.1f}mm min={np.min(tcp_errs[-50:]):.1f}mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", choices=["open", "closed"], required=True)
    args = parser.parse_args()
    try:
        say(f"\n=== case: gripper {args.case.upper()}, cube moved to {FAR_AWAY.tolist()} (no payload/contact) ===")
        run(0.0 if args.case == "open" else 1.0)
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
