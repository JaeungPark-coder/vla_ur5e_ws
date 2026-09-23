"""E-4: feasibility_gate.py's own workspace_ok() rejects every scripted
grasp/place waypoint outright -- at_cube and at_target (scripted_pick_place.py)
both sit at z=cube_centre_height_above_table+GRASP_HEIGHT=0.04m, under
Z_MIN_M=0.08. That rejection is the cheap, pure-numpy box check; it happens
BEFORE any IK solve, so it says nothing about whether z=40mm is actually
near-singular for this UR5e+RMPflow pair or whether Z_MIN_M=0.08 was just an
untested, conservative guess (see README's "Still open" note).

This calls the SAME IK solve and Jacobian condition-number computation
FeasibilityGate.check() would run, deliberately skipping workspace_ok(), at
the real at_cube/at_target corners plus a z-sweep at the cube-spawn centre --
so the actual manipulability/condition-number numbers exist to decide between
lowering Z_MIN_M and adding a pedestal under the cube.
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import ROBOT_PRIM_PATH, CUBE_X_RANGE, CUBE_Y_RANGE, PLACE_TARGET_POSITION, PickPlaceScene  # noqa: E402
from feasibility_gate import FeasibilityGate, CONDITION_SOFT, CONDITION_HARD, MANIPULABILITY_MIN  # noqa: E402

DOWN_QUAT_WXYZ = [0.0, 0.0, 1.0, 0.0]
GRASP_Z = 0.04  # at_cube / at_target height, per scripted_pick_place.py

CORNERS = {
    "at_cube corner (xmin,ymin)": [CUBE_X_RANGE[0], CUBE_Y_RANGE[0], GRASP_Z],
    "at_cube corner (xmin,ymax)": [CUBE_X_RANGE[0], CUBE_Y_RANGE[1], GRASP_Z],
    "at_cube corner (xmax,ymin)": [CUBE_X_RANGE[1], CUBE_Y_RANGE[0], GRASP_Z],
    "at_cube corner (xmax,ymax)": [CUBE_X_RANGE[1], CUBE_Y_RANGE[1], GRASP_Z],
    "at_cube centre": [np.mean(CUBE_X_RANGE), np.mean(CUBE_Y_RANGE), GRASP_Z],
    "at_target": [PLACE_TARGET_POSITION[0], PLACE_TARGET_POSITION[1], GRASP_Z],
}

Z_SWEEP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]


def probe(gate, pos):
    """Runs check()'s IK+Jacobian math directly, bypassing workspace_ok()."""
    joint_positions, ik_success = gate.solver.compute_inverse_kinematics(
        gate.ee_frame_name, np.asarray(pos, dtype=float), np.asarray(DOWN_QUAT_WXYZ, dtype=float))
    if not ik_success:
        return None
    jac = gate._numerical_jacobian(joint_positions)
    singular_values = np.linalg.svd(jac, compute_uv=False)
    manipulability = float(np.prod(singular_values))
    condition_number = float(singular_values.max() / max(singular_values.min(), 1e-9))
    return manipulability, condition_number


def verdict(manipulability, condition_number):
    if manipulability < MANIPULABILITY_MIN or condition_number > CONDITION_HARD:
        return "WOULD FAIL check() on manipulability/condition (near-singular)"
    if condition_number > CONDITION_SOFT:
        return "would PASS check(), soft warning"
    return "would PASS check() cleanly"


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()
        gate = FeasibilityGate(ROBOT_PRIM_PATH)
        print("gate built OK\n", flush=True)

        print(f"=== corners/target at the scripted grasp height z={GRASP_Z*1000:.0f}mm "
              f"(workspace_ok() rejects all of these; IK/Jacobian run anyway) ===", flush=True)
        for label, pos in CORNERS.items():
            result = probe(gate, pos)
            if result is None:
                print(f"  {label:30s} pos={pos} -- NO IK SOLUTION (unreachable)", flush=True)
                continue
            manipulability, condition_number = result
            print(f"  {label:30s} pos={pos} manipulability={manipulability:.4f} "
                  f"condition_number={condition_number:6.1f}  -- {verdict(manipulability, condition_number)}",
                  flush=True)

        centre_xy = [float(np.mean(CUBE_X_RANGE)), float(np.mean(CUBE_Y_RANGE))]
        print(f"\n=== z-sweep at cube-spawn centre xy={centre_xy} "
              f"(Z_MIN_M=0.08 is the current cutoff) ===", flush=True)
        for z in Z_SWEEP:
            pos = centre_xy + [z]
            result = probe(gate, pos)
            gated = "REJECTED by workspace_ok" if z < 0.08 else "passes workspace_ok"
            if result is None:
                print(f"  z={z*1000:5.1f}mm ({gated:24s}) -- NO IK SOLUTION (unreachable)", flush=True)
                continue
            manipulability, condition_number = result
            print(f"  z={z*1000:5.1f}mm ({gated:24s}) manipulability={manipulability:.4f} "
                  f"condition_number={condition_number:6.1f}  -- {verdict(manipulability, condition_number)}",
                  flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
