import os
from isaacsim import SimulationApp
HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})
from pxr import Usd  # noqa: E402
from isaac_sim_common import ROBOT_PRIM_PATH  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402

try:
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()
    stage = scene.stage
    root = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85/left_inner_finger/visuals")
    print(f"root valid={root.IsValid()} path={root.GetPath()}", flush=True)
    for prim in Usd.PrimRange(root):
        print(f"  {prim.GetPath()}  type={prim.GetTypeName()!r}  specifier={prim.GetSpecifier()}", flush=True)
except BaseException:
    import traceback
    print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
finally:
    simulation_app.close()
