"""check_gripper_closed_tracking_bias.py v3 (ramped, not step -- ruling out
the step-command artifact this time) STILL found a persistent ~39mm
steady-state tracking error at a FIXED point [0.45, 0.0] regardless of
gripper open/closed, using the exact same ramped approach real episodes use.
That point is the geometric CENTRE of CUBE_X_RANGE x CUBE_Y_RANGE -- i.e.
directly on the robot's sagittal plane (Y=0), a classic wrist-singularity
configuration for a straight-down tool orientation (wrist_1/wrist_3 gimbal
alignment). A bad choice of test point, not a gripper/payload finding.

This raises a cleaner, purely geometric hypothesis for the REAL episode's
own descend2/settle3 plateau (25-56mm, all 5 episodes, check_place_dynamics.py):
PLACE_TARGET_POSITION=[0.45, 0.3, 0.0] sits in a worse manipulability/
condition-number zone than the cube-spawn region (CUBE_X_RANGE x
CUBE_Y_RANGE, always Y in [-0.10, 0.10]) does -- independent of gripper
state or payload entirely. Y=0.3 is 3x further from the robot's sagittal
plane than any cube spawn point ever is.

Cheap, decisive, and no live physics stepping needed: FeasibilityGate
(feasibility_gate.py) already computes IK + Yoshikawa manipulability +
condition number for exactly this purpose (it's the upstream gate this
project already built for near-singular targets). Compare manipulability/
condition number at the REAL at_target (place side) vs a representative
at_cube (grasp side, cube-spawn-range) position -- same height, same
downward orientation, only XY differs.

    python3 check_place_manipulability.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import PLACE_TARGET_POSITION, CUBE_Z, ROBOT_PRIM_PATH, CUBE_X_RANGE, CUBE_Y_RANGE  # noqa: E402
from scripted_pick_place import GRASP_HEIGHT, DOWNWARD_ROTVEC  # noqa: E402
from feasibility_gate import FeasibilityGate, CONDITION_SOFT, CONDITION_HARD, MANIPULABILITY_MIN  # noqa: E402


def say(line=""):
    print(line, flush=True)


def report(gate, label, pos):
    # Bypass gate.check()'s workspace_ok() pre-filter -- its Z_MIN_M=0.08m
    # guard (tuned from a DIFFERENT confirmed-bad case at z=22mm) rejects
    # EVERY position this task's own grasp/place waypoints use
    # (CUBE_Z+GRASP_HEIGHT=0.04m, below that floor by design), so it refuses
    # before ever computing manipulability. Call IK + the numerical Jacobian
    # directly instead, same as gate.check() does internally past that gate.
    quat_wxyz = Rot.from_rotvec(DOWNWARD_ROTVEC).as_quat()[[3, 0, 1, 2]]
    joint_positions, ik_success = gate.solver.compute_inverse_kinematics(
        gate.ee_frame_name, np.asarray(pos, dtype=float), quat_wxyz)
    say(f"{label}: pos={np.round(pos, 4).tolist()}")
    if not ik_success:
        say("  ik_success=False (no IK solution)")
        return None
    jac = gate._numerical_jacobian(joint_positions)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    manipulability = float(np.prod(singular_values))
    condition_number = float(singular_values.max() / max(singular_values.min(), 1e-9))
    say(f"  ik_success=True joint_positions={np.round(joint_positions, 4).tolist()}")
    say(f"  manipulability={manipulability:.5f} (floor {MANIPULABILITY_MIN}) "
        f"condition_number={condition_number:.2f} (soft {CONDITION_SOFT}, hard {CONDITION_HARD})")
    return {"manipulability": manipulability, "condition_number": condition_number}


def main():
    try:
        gate = FeasibilityGate(ROBOT_PRIM_PATH)

        at_target = PLACE_TARGET_POSITION + np.array([0.0, 0.0, CUBE_Z + GRASP_HEIGHT])
        report(gate, "PLACE side (at_target, real place waypoint)", at_target)

        say()
        cube_x_mid = sum(CUBE_X_RANGE) / 2
        cube_y_mid = sum(CUBE_Y_RANGE) / 2
        at_cube_mid = np.array([cube_x_mid, cube_y_mid, CUBE_Z + GRASP_HEIGHT])
        report(gate, "GRASP side (at_cube, centre of cube-spawn range -- known bad/singular per the A/B test)", at_cube_mid)

        say()
        for x, y in [(CUBE_X_RANGE[0], CUBE_Y_RANGE[0]), (CUBE_X_RANGE[0], CUBE_Y_RANGE[1]),
                     (CUBE_X_RANGE[1], CUBE_Y_RANGE[0]), (CUBE_X_RANGE[1], CUBE_Y_RANGE[1])]:
            report(gate, f"GRASP side corner ({x},{y})", np.array([x, y, CUBE_Z + GRASP_HEIGHT]))
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
