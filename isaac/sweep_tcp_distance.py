"""Empirical TCP-distance sweep, axis and orientation held fixed at the
project's own already-verified-correct values (tool0 +Z straight down,
DOWNWARD_ROTVEC's exact roll) -- only the standoff distance varies.

Deliberately does NOT try to read any gripper link's pose: every pose-readout
method tried so far (UsdGeom.ComputeLocalToWorldTransform, SingleRigidPrim
physics readout) returns the wrist origin for every gripper link regardless
of arm configuration, on this specific asset, even after real collision
geometry was added -- a Fabric/USD sync gap specific to these links (the arm's
OWN links read correctly by the same methods). So this scores candidates the
only way that has been reliable throughout this project: did the cube
actually rise.

    python3 sweep_tcp_distance.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])  # matches scripted_pick_place.DOWNWARD_ROTVEC exactly
DISTANCES = [0.06, 0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]
SEED = 7


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def try_distance(scene, distance):
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()

    above = [cube0[0], cube0[1], 0.30 + distance]
    at = [cube0[0], cube0[1], CUBE_Z + distance]
    lift = [cube0[0], cube0[1], 0.30 + distance]

    drive(scene, above, 200, 0.0)
    drive(scene, at, 200, 0.0)
    drive(scene, at, 90, 1.0)
    drive(scene, lift, 200, 1.0)

    cube_now = scene.get_cube_position()
    rise = float(cube_now[2] - cube0[2])
    xy_drift = float(np.linalg.norm(cube_now[:2] - cube0[:2]))
    return rise, xy_drift


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()
        print(f"sweeping TCP distance over {DISTANCES}, orientation fixed at DOWNWARD_ROTVEC "
              "(tool0 +Z straight down)", flush=True)
        results = []
        for d in DISTANCES:
            rise, drift = try_distance(scene, d)
            results.append((rise, d, drift))
            print(f"  distance={d:.2f}m: cube rose {rise * 1000:7.1f} mm, xy drift {drift * 1000:6.1f} mm",
                  flush=True)

        results.sort(key=lambda r: -r[0])
        best_rise, best_d, best_drift = results[0]
        print("\n--- verdict ---", flush=True)
        if best_rise * 1000 < 50.0:
            print(f"  BEST distance={best_d:.2f}m only lifted {best_rise*1000:.1f}mm -- still not "
                  "a real grasp. Suspect the finger CLOSED joint target (GRIPPER_CLOSED_POS=0.68 "
                  "rad in isaac_sim_common.py) doesn't close far enough/too far to pinch a 40mm "
                  "cube at this distance, or friction is too low -- not the standoff distance.",
                  flush=True)
        else:
            print(f"  distance={best_d:.2f}m lifted the cube {best_rise*1000:.1f}mm "
                  f"(xy drift {best_drift*1000:.1f}mm). Set GRIPPER_TCP_OFFSET_M = {best_d}",
                  flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
