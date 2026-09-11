"""Does the gripper have collision geometry at all?

A functional grasp sweep can only tell you that nothing grasps; this says
why. Listing the prims under the gripper earlier showed every link carrying a
single "visuals" child and nothing else, while the arm's own links have a
"collisions" child (Kit logs warnings about
/World/ur5e/wrist_3_link/collisions/wrist3). Fingers with no collider pass
straight through the cube no matter how perfectly the waypoints are aimed --
which would explain a cube that does not move by even a millimetre when the
gripper closes on it, and would mean none of the frame corrections attempted
so far could ever have worked.

    python3 check_gripper_collision.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaac_sim_common import ROBOT_PRIM_PATH  # noqa: E402
from pick_place_scene import CUBE_PRIM_PATH, PickPlaceScene  # noqa: E402

GRIPPER_ROOT = f"{ROBOT_PRIM_PATH}/Gripper"


def describe(stage, root_path, label):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        print(f"{label}: no prim at {root_path}", flush=True)
        return 0, 0
    n_collide = n_rigid = 0
    print(f"\n{label} ({root_path}):", flush=True)
    for prim in Usd.PrimRange(root):
        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        n_collide += int(has_collision)
        n_rigid += int(has_rigid)
        if has_collision or has_rigid or prim.GetName() in ("collisions", "visuals"):
            flags = []
            if has_rigid:
                flags.append("RigidBody")
            if has_collision:
                flags.append("Collision")
            print(f"    {str(prim.GetPath()).replace(root_path, ''):55s} "
                  f"{prim.GetTypeName():10s} {' '.join(flags)}", flush=True)
    print(f"  -> {n_rigid} rigid bodies, {n_collide} colliders", flush=True)
    return n_rigid, n_collide


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()
        stage = scene.stage

        arm_rigid, arm_collide = describe(stage, f"{ROBOT_PRIM_PATH}/wrist_3_link",
                                          "ARM, one link for comparison")
        grip_rigid, grip_collide = describe(stage, GRIPPER_ROOT, "GRIPPER")
        cube = stage.GetPrimAtPath(CUBE_PRIM_PATH)
        print(f"\nCUBE {CUBE_PRIM_PATH}: RigidBody={cube.HasAPI(UsdPhysics.RigidBodyAPI)} "
              f"Collision={cube.HasAPI(UsdPhysics.CollisionAPI)}", flush=True)

        print("\n--- verdict ---", flush=True)
        if grip_collide == 0:
            print("  The gripper has NO collision geometry. Its fingers cannot touch anything,", flush=True)
            print("  so no waypoint transform, TCP offset or camera mount could ever have made", flush=True)
            print("  it grasp. Give the finger links colliders (or select an asset variant that", flush=True)
            print("  ships them) before any of the frame work is worth re-testing.", flush=True)
        else:
            print(f"  The gripper has {grip_collide} colliders, so contact is possible and the", flush=True)
            print("  failure to grasp is about where/how it closes, not about missing geometry.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
