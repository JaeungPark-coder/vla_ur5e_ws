"""Two checks in one run: did tool0 actually converge to the offset-corrected
command (test_offset_grasp.py showed zero contact -- is that a tracking
failure or a wrong offset?), and if it converged fine, where did the fingers
actually end up this time (local grid search around the cube)?"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
OFFSET_WORLD = np.array([0.089, 0.090, -0.033])
RADIUS_MM = 12
GRID_MM = list(range(-140, 141, 20))
FINGER_NAMES = {"left_inner_finger", "right_inner_finger", "left_outer_finger", "right_outer_finger",
                "left_outer_knuckle", "right_outer_knuckle", "left_inner_knuckle", "right_inner_knuckle"}


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def main():
    try:
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf

        scene = PickPlaceScene(with_gripper=True)
        scene._rng = np.random.default_rng(SEED)
        scene.reset()
        cube0 = scene.cube_position.copy()
        tool0_command = cube0 - OFFSET_WORLD
        print(f"cube at {np.round(cube0, 4)}, commanding tool0 to {np.round(tool0_command, 4)}",
              flush=True)

        above = tool0_command.copy()
        above[2] += 0.20
        drive(scene, above, 220, 0.0)
        drive(scene, tool0_command, 220, 0.0)

        q = np.asarray(scene.robot.get_joint_positions())[:6]
        ee_pos, ee_rot = scene.rmpflow.get_end_effector_pose(q)
        ee_pos = np.asarray(ee_pos)
        err = float(np.linalg.norm(ee_pos - tool0_command))
        print(f"tool0 reached {np.round(ee_pos, 4)} -- tracking error {err*1000:.1f}mm from command",
              flush=True)

        drive(scene, tool0_command, 90, 1.0)

        query = get_physx_scene_query_interface()
        for z in (CUBE_Z, CUBE_Z + 0.03, CUBE_Z + 0.06, CUBE_Z + 0.09, CUBE_Z + 0.12):
            print(f"-- z={z*1000:.0f}mm --", flush=True)
            found = []
            for dx_mm in GRID_MM:
                row = []
                for dy_mm in GRID_MM:
                    hits = []

                    def report_fn(hit):
                        short = str(hit.rigid_body).split("/")[-1]
                        if short not in hits:
                            hits.append(short)
                        return True

                    px = cube0[0] + dx_mm / 1000.0
                    py = cube0[1] + dy_mm / 1000.0
                    query.overlap_sphere(RADIUS_MM / 1000.0, Gf.Vec3f(px, py, z), report_fn, False)
                    fh = [h for h in hits if h in FINGER_NAMES]
                    if fh:
                        row.append("F")
                        found.append((dx_mm, dy_mm))
                    elif hits:
                        row.append("o")
                    else:
                        row.append(".")
                print("  " + "".join(row), flush=True)
            if found:
                centroid = np.mean(found, axis=0)
                print(f"  finger centroid at this height: dx={centroid[0]:.0f}mm dy={centroid[1]:.0f}mm",
                      flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
