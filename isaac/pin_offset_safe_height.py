"""Nail the TCP offset at the SAFE configuration only (tool0 ~140mm above
the cube, well clear of ground contact) with a proper dense 3D grid instead
of sparse z-slices -- verify_offset_and_relocate.py found a COMPLETELY
different offset direction at a low (53mm) tool0 height, which is
consistent with ground contact physically perturbing the gripper (it now
has real colliders) rather than the offset being unstable in general. This
stays at the height already shown safe (zero spurious cube displacement
across the whole distance sweep) and searches densely enough to pin the
centroid with confidence, then verifies with an actual grasp-lift."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCE = 0.12  # tool0 standoff used throughout the safe-height sweeps
SEED = 7
RADIUS_MM = 10
XY_MM = list(range(-20, 161, 15))
Z_MM_ABOVE_CUBE = list(range(0, 131, 10))
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
        print(f"cube at {np.round(cube0, 4)}, tool0 reached {np.round(ee_pos, 4)} "
              f"(height {(ee_pos[2] - CUBE_Z) * 1000:.0f}mm above cube -- safely clear of ground)",
              flush=True)

        query = get_physx_scene_query_interface()
        all_found = []
        for z_above_mm in Z_MM_ABOVE_CUBE:
            z = CUBE_Z + z_above_mm / 1000.0
            for dx_mm in XY_MM:
                for dy_mm in XY_MM:
                    hits = []

                    def report_fn(hit):
                        short = str(hit.rigid_body).split("/")[-1]
                        if short not in hits:
                            hits.append(short)
                        return True

                    px = cube0[0] + dx_mm / 1000.0
                    py = cube0[1] + dy_mm / 1000.0
                    query.overlap_sphere(RADIUS_MM / 1000.0, Gf.Vec3f(px, py, z), report_fn, False)
                    if any(h in FINGER_NAMES for h in hits):
                        all_found.append((dx_mm, dy_mm, z_above_mm))

        if not all_found:
            print("\nNO FINGER FOUND anywhere in this dense grid either -- something more basic is "
                  "wrong (finger colliders not where expected, or grid too small/offset). Widen "
                  "XY_MM/Z_MM_ABOVE_CUBE.", flush=True)
            return

        arr = np.array(all_found, dtype=float)
        centroid_mm = arr.mean(axis=0)
        print(f"\n{len(all_found)} grid points hit a finger part.", flush=True)
        print(f"centroid: dx={centroid_mm[0]:.0f}mm dy={centroid_mm[1]:.0f}mm "
              f"z_above_cube={centroid_mm[2]:.0f}mm", flush=True)
        print(f"range: dx=[{arr[:,0].min():.0f},{arr[:,0].max():.0f}] "
              f"dy=[{arr[:,1].min():.0f},{arr[:,1].max():.0f}] "
              f"z=[{arr[:,2].min():.0f},{arr[:,2].max():.0f}]", flush=True)

        # Now verify: command tool0 so the centroid offset lands exactly on
        # the cube, using the SAME safe approach height/config, and check
        # for an actual lift.
        offset_world = np.array([centroid_mm[0], centroid_mm[1], centroid_mm[2] - DISTANCE * 1000]) / 1000.0
        # offset_world is (fingers - tool0): fingers sit at cube + (dx,dy) and
        # at height CUBE_Z + z_above_cube, while tool0 sits at CUBE_Z + DISTANCE.
        print(f"\nimplied offset (fingers - tool0), world mm: {np.round(offset_world*1000,1).tolist()}",
              flush=True)

        scene._rng = np.random.default_rng(SEED)
        scene.reset()
        cube1 = scene.cube_position.copy()
        tool0_command = cube1 - offset_world
        print(f"verifying: commanding tool0 to {np.round(tool0_command,4)}", flush=True)
        above2 = tool0_command.copy()
        above2[2] += 0.15
        drive(scene, above2, 220, 0.0)
        drive(scene, tool0_command, 220, 0.0)
        drive(scene, tool0_command, 90, 1.0)
        lift = tool0_command.copy()
        lift[2] += 0.15
        drive(scene, lift, 220, 1.0)
        cube_now = scene.get_cube_position()
        rise = float(cube_now[2] - cube1[2])
        drift = float(np.linalg.norm(cube_now[:2] - cube1[:2]))
        print(f"cube now at {np.round(cube_now,4)} -- rise={rise*1000:.1f}mm drift={drift*1000:.1f}mm",
              flush=True)
        if rise * 1000 > 50:
            print(f"\nSUCCESS. GRIPPER_TCP_OFFSET_WORLD (at this safe standoff/orientation) = "
                  f"{np.round(offset_world, 4).tolist()}", flush=True)
        else:
            print("\nSTILL NO LIFT even with the dense-grid centroid -- see verdict notes.",
                  flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
