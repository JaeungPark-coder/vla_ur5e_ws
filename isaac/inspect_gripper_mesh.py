"""What geometry actually exists under the gripper's /visuals Xforms, and
does the arm's own /collisions sibling (present but empty per the collision
check) hold anything either -- both needed before deciding how to add
colliders."""
import os
from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pxr import Usd, UsdGeom  # noqa: E402
from isaac_sim_common import ROBOT_PRIM_PATH  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402

GRIPPER_ROOT = f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85"


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()
        stage = scene.stage
        root = stage.GetPrimAtPath(GRIPPER_ROOT)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts = mesh.GetPointsAttr().Get()
                n = len(pts) if pts else 0
                print(f"MESH {prim.GetPath()}  points={n}", flush=True)
            elif prim.IsA(UsdGeom.Gprim) and not prim.IsA(UsdGeom.Mesh):
                print(f"GPRIM({prim.GetTypeName()}) {prim.GetPath()}", flush=True)

        print("\n--- arm wrist_3_link/collisions, for comparison ---", flush=True)
        wc = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/wrist_3_link/collisions")
        if wc.IsValid():
            for prim in Usd.PrimRange(wc):
                print(f"  {prim.GetPath()}  type={prim.GetTypeName()}", flush=True)
        else:
            print("  (no /collisions prim on wrist_3_link either)", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
