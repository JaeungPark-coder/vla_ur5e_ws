"""Which gripper parts occupy which WORLD Z heights, at a fixed XY above the
cube? If the fingers point down at the table (as assumed throughout), they
should be the LOWEST (closest to the cube) parts found. If they point up
toward the wrist instead, base_link/knuckles will be lower than the fingers
-- exactly what probe_gripper_gap.py's single-point overlap hinted at
(base_link found at 100mm, no finger part found even out to 100mm)."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCE = 0.12
SEED = 7
RADIUS_MM = 25
# Sample world Z heights from below the table up through well above tool0's
# commanded height, at the cube's XY.
Z_SAMPLES_MM = list(range(-20, 260, 20))


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

        above = [cube0[0], cube0[1], 0.25 + DISTANCE]
        at = [cube0[0], cube0[1], CUBE_Z + DISTANCE]
        drive(scene, above, 220, 0.0)
        drive(scene, at, 220, 0.0)
        drive(scene, at, 90, 1.0)

        q = np.asarray(scene.robot.get_joint_positions())[:6]
        ee_pos, _ = scene.rmpflow.get_end_effector_pose(q)
        print(f"cube xy={np.round(cube0[:2],4)}, tool0 reached {np.round(np.asarray(ee_pos),4)}, "
              f"cube z={cube0[2]:.4f}", flush=True)

        query = get_physx_scene_query_interface()
        for z_mm in Z_SAMPLES_MM:
            z = z_mm / 1000.0
            hits = []

            def report_fn(hit):
                path = str(hit.rigid_body)
                short = path.split("/")[-1]
                if short not in hits:
                    hits.append(short)
                return True

            probe_point = (float(cube0[0]), float(cube0[1]), z)
            query.overlap_sphere(RADIUS_MM / 1000.0, Gf.Vec3f(*probe_point), report_fn, False)
            gripper_hits = [h for h in hits if h not in ("base_link", "wrist_3_link", "flange",
                                                          "wrist_2_link", "wrist_1_link")]
            tag = "  <== TOOL0 HEIGHT" if abs(z - float(np.asarray(ee_pos)[2])) < 0.012 else ""
            tag += "  <== CUBE HEIGHT" if abs(z - cube0[2]) < 0.012 else ""
            print(f"  world z={z_mm:4d}mm: {hits}{tag}", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
