"""Brute-force 2D grid search for where the closed fingers actually are in
world space, since map_gripper_along_z.py found NONE of them anywhere along
the vertical line through the cube (only base_link/wrist links, out to
240mm) -- the offset from tool0 to the true grasp point has a lateral (X/Y)
component this project never accounted for, not just a Z distance."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCE = 0.12
SEED = 7
RADIUS_MM = 8
XY_RANGE_MM = list(range(40, 161, 10))
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

        above = [cube0[0], cube0[1], 0.25 + DISTANCE]
        at = [cube0[0], cube0[1], CUBE_Z + DISTANCE]
        drive(scene, above, 220, 0.0)
        drive(scene, at, 220, 0.0)
        drive(scene, at, 90, 1.0)

        q = np.asarray(scene.robot.get_joint_positions())[:6]
        ee_pos, _ = scene.rmpflow.get_end_effector_pose(q)
        ee_pos = np.asarray(ee_pos)
        print(f"cube at {np.round(cube0,4)}, tool0 reached {np.round(ee_pos,4)}", flush=True)

        query = get_physx_scene_query_interface()
        # Scan a horizontal XY grid at each of a few Z heights spanning the
        # standoff region between the cube and tool0.
        for z in (CUBE_Z + 0.07, CUBE_Z + 0.08, CUBE_Z + 0.09, CUBE_Z + 0.10, CUBE_Z + 0.11):
            print(f"\n-- z={z*1000:.0f}mm --", flush=True)
            found_any = False
            for dx_mm in XY_RANGE_MM:
                row = []
                for dy_mm in XY_RANGE_MM:
                    hits = []

                    def report_fn(hit):
                        short = str(hit.rigid_body).split("/")[-1]
                        if short not in hits:
                            hits.append(short)
                        return True

                    px = cube0[0] + dx_mm / 1000.0
                    py = cube0[1] + dy_mm / 1000.0
                    query.overlap_sphere(RADIUS_MM / 1000.0, Gf.Vec3f(px, py, z), report_fn, False)
                    finger_hits = [h for h in hits if h in FINGER_NAMES]
                    if finger_hits:
                        found_any = True
                        row.append("F")
                    elif hits:
                        row.append("o")
                    else:
                        row.append(".")
                print(f"  dx={0:>4} " + "".join(row) if False else
                      f"  " + "".join(row), flush=True)
            if not found_any:
                print("  (no finger part found anywhere in this +/-150mm grid at this height)",
                      flush=True)
        print(f"\ngrid columns are dy = {XY_RANGE_MM} mm (world Y offset from cube), rows are "
              f"dx = {XY_RANGE_MM} mm (world X offset from cube). F=finger/knuckle, o=other robot "
              "part, .=nothing", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
