"""How far are the closed fingers from the cube, really? Uses a PhysX
overlap query at the CUBE's own position instead of any gripper-link pose
reading (every one of those has been unreliable on this asset) -- grows a
probe sphere around the cube until it actually touches gripper geometry,
which pins down the true miss distance directly rather than guessing another
TCP standoff to try.

    python3 probe_gripper_gap.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCE = 0.12
SEED = 7
RADII_MM = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]


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

        cube_now = scene.get_cube_position()
        print(f"cube at {np.round(cube_now, 4)} after closing at distance={DISTANCE}", flush=True)

        query = get_physx_scene_query_interface()
        for r_mm in RADII_MM:
            hits = []

            def report_fn(hit):
                path = str(hit.rigid_body)
                if "ur5e" in path and path not in hits:
                    hits.append(path)
                return True

            query.overlap_sphere(r_mm / 1000.0, Gf.Vec3f(*cube_now.tolist()), report_fn, False)
            print(f"  r={r_mm:4d}mm around the cube: {len(hits)} robot part(s) found"
                  + (f" -- {hits[:6]}" if hits else ""), flush=True)
            if hits:
                break
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
