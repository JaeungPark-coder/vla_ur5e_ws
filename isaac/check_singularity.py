"""Confirms/refutes the wrist-singularity hypothesis for the target pose
that made RMPflow spin continuously in the GUI just now
([0.4618, 0.0636, 0.0222], tool0 pointing straight down).

A UR-style arm has a wrist singularity when wrist_2_joint's angle is near 0
or +-pi (the wrist_1/wrist_3 axes become co-linear, so infinitely many
joint-velocity combinations produce the same end-effector motion -- exactly
the kind of situation where a reactive controller like RMPflow can spin
without converging). It also has a shoulder singularity when the wrist
center lies on the shoulder's rotation axis, and an elbow singularity when
the arm is fully outstretched. Checks all three, plus just logs the joint
trajectory over time to SEE the spin numerically (do joint angles keep
changing after the position has already converged?).
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
TARGET = np.array([0.4618, 0.0636, 0.0222])  # the pose that spun in the GUI


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene._rng = np.random.default_rng(SEED)
        scene.reset()

        print(f"driving to the pose that spun in the GUI: {TARGET.tolist()}", flush=True)
        joint_log = []
        for tick in range(400):
            scene.rmpflow.set_end_effector_target(TARGET, DOWN_QUAT_WXYZ)
            scene.rmpflow.update_world()
            action = scene.articulation_policy.get_next_articulation_action(scene.physics_dt)
            scene.robot.apply_action(action)
            scene.world.step(render=False)
            q = np.asarray(scene.robot.get_joint_positions())[:6]
            joint_log.append(q.copy())
            if tick % 50 == 0 or tick == 399:
                ee_pos, _ = scene.rmpflow.get_end_effector_pose(q)
                err = float(np.linalg.norm(np.asarray(ee_pos) - TARGET))
                print(f"  tick {tick:3d}: q={np.round(np.degrees(q), 1)} deg  "
                      f"pos_err={err*1000:.1f}mm", flush=True)

        joint_log = np.array(joint_log)
        # Position should have converged well before tick 400 (RMPflow
        # normally settles within ~100-150 ticks elsewhere in this project).
        # If joint angles are STILL changing meaningfully in the last 100
        # ticks despite that, it's spinning in place, not converging.
        late_window = joint_log[-100:]
        late_range_deg = np.degrees(late_window.max(axis=0) - late_window.min(axis=0))
        print(f"\njoint range over the LAST 100 ticks (should be ~0 if settled): "
              f"{np.round(late_range_deg, 2).tolist()} deg", flush=True)
        spinning_joints = [i for i, r in enumerate(late_range_deg) if r > 5.0]
        dof_names = list(scene.robot.dof_names)[:6]
        if spinning_joints:
            print(f"STILL MOVING (>5deg range in the last 100 ticks): "
                  f"{[dof_names[i] for i in spinning_joints]}", flush=True)
        else:
            print("Settled -- no spinning in this run (may be pose-dependent/intermittent).",
                  flush=True)

        # Singularity checks on the final joint config.
        q_final = joint_log[-1]
        print(f"\nfinal joint angles (deg): "
              f"{dict(zip(dof_names, np.round(np.degrees(q_final), 2).tolist()))}", flush=True)
        wrist_2_deg = float(np.degrees(q_final[4]))
        print(f"\nwrist_2_joint = {wrist_2_deg:.2f} deg -- WRIST SINGULARITY risk if this is "
              f"near 0 or +-180 deg (co-linear wrist axes)", flush=True)
        shoulder_lift_deg = float(np.degrees(q_final[1]))
        elbow_deg = float(np.degrees(q_final[2]))
        print(f"shoulder_lift={shoulder_lift_deg:.1f} deg, elbow={elbow_deg:.1f} deg -- "
              f"ELBOW SINGULARITY risk if elbow is near 0/180 deg (arm fully "
              f"extended/folded)", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
