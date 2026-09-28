"""Quantitative check of the gravity-droop / steady-state-PD-error
hypothesis for the ~36-56mm place-side tracking bias check_place_dynamics.py
found (persists even at 400 ticks / 6.7s dwell -- not a settling-time issue).

Classical result for a proportional (no integral) joint drive under a
constant disturbance torque: steady-state joint error = tau_disturbance / kp,
exactly stationary regardless of how long you wait -- which matches the
400-tick plateau observation. Predicted here from first principles rather
than assumed (avoids this project's own previously-documented P8 mistake:
fitting a curve to two points and calling it confirmed):

    tau_gravity[6] = J_v(q)^T @ [0, 0, -m_cube * g]      (arm-joint gravity torque from the held cube)
    dtheta[6]      = tau_gravity / kp_arm                 (steady-state joint droop, no integral term)
    dx             = J_v(q) @ dtheta                       (linearized Cartesian droop)

Reuses FeasibilityGate._numerical_jacobian (central-difference Jacobian
through the same Lula FK RMPflow is built on -- already trusted elsewhere in
this project) rather than deriving a new Jacobian implementation. Arm joint
kp is read live via get_joint_drive_gains, not assumed, since this project
does not set ARM_JOINT_NAMES stiffness itself (whatever the UR5e asset ships
with is what's live). Cube mass (0.05kg / 50g) is CUBE_MASS in
isaac_sim_common.add_shape's MassAPI call -- also read from source, not
retyped.

    python3 check_gravity_droop_prediction.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION, CUBE_Z, ROBOT_PRIM_PATH  # noqa: E402
from scripted_pick_place import GRASP_HEIGHT, DOWNWARD_ROTVEC  # noqa: E402
from isaac_sim_common import ARM_JOINT_NAMES, get_joint_drive_gains  # noqa: E402
from feasibility_gate import FeasibilityGate  # noqa: E402

G = 9.81
CUBE_MASS_KG = 0.05  # matches isaac_sim_common.add_shape's MassAPI.CreateMassAttr(0.05)


def say(line=""):
    print(line, flush=True)


def run():
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()

    at_target = PLACE_TARGET_POSITION + np.array([0.0, 0.0, CUBE_Z + GRASP_HEIGHT])
    say(f"converging to at_target={np.round(at_target, 4).tolist()} (matches scripted_pick_place.py's real place waypoint)")
    for _ in range(180):
        scene.step_towards(at_target, DOWNWARD_ROTVEC, 1.0)

    q = np.asarray(scene.robot.get_joint_positions())[:6]
    tcp_err_mm = 1000.0 * float(np.linalg.norm(at_target - scene.grip_point_world()))
    say(f"converged joint_positions (rad): {np.round(q, 4).tolist()}")
    say(f"tcp_err at this pose right now: {tcp_err_mm:.1f}mm (context only, gripper carries no real cube here)")

    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    gains = get_joint_drive_gains(robot_prim, joint_names=ARM_JOINT_NAMES)
    kp = np.array([gains[name][0] for name in ARM_JOINT_NAMES], dtype=float)
    say(f"live arm joint kp ({ARM_JOINT_NAMES}): {np.round(kp, 2).tolist()}")

    gate = FeasibilityGate(ROBOT_PRIM_PATH)
    jac = gate._numerical_jacobian(q)  # (6,6): rows 0-2 = dposition/dq, rows 3-5 = drotation/dq
    jac_v = jac[:3, :]  # (3,6) linear-velocity Jacobian only -- gravity force has no direct moment input here

    tau_gravity = jac_v.T @ np.array([0.0, 0.0, -CUBE_MASS_KG * G])  # (6,) N*m per arm joint
    say(f"predicted per-joint gravity torque from the cube (N*m): {np.round(tau_gravity, 4).tolist()}")

    dtheta = tau_gravity / kp  # (6,) rad, steady-state droop with NO integral term
    say(f"predicted per-joint steady-state droop (mrad): {np.round(dtheta * 1000, 3).tolist()}")

    dx = jac_v @ dtheta  # (3,) m, linearized Cartesian droop
    dx_mm = 1000.0 * dx
    say(f"predicted Cartesian droop vector (mm): {np.round(dx_mm, 2).tolist()}")
    say(f"predicted Cartesian droop magnitude: {float(np.linalg.norm(dx_mm)):.2f}mm")

    say("\nobserved (check_place_dynamics.py, 5 episodes, descend2/settle3 plateau): 36-56mm")
    say("observed marginal jump from unloaded baseline (~13-18mm) to loaded plateau (~36-56mm): ~20-40mm")


def main():
    try:
        run()
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
