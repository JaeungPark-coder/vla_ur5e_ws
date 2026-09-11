import os
from isaacsim import SimulationApp
HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})
from pxr import Usd, UsdGeom  # noqa: E402
from isaac_sim_common import ROBOT_PRIM_PATH  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402

try:
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()
    stage = scene.stage
    finger = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85/left_inner_finger")
    print(f"finger IsInstance={finger.IsInstance()} IsInstanceProxy={finger.IsInstanceProxy()} "
          f"IsInstanceable={finger.IsInstanceable()}", flush=True)
    vis = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85/left_inner_finger/visuals")
    print(f"visuals IsInstance={vis.IsInstance()} IsInstanceProxy={vis.IsInstanceProxy()} "
          f"IsInstanceable={vis.IsInstanceable()}", flush=True)
    print(f"visuals children (no predicate filter): {[c.GetName() for c in vis.GetAllChildren()]}", flush=True)
    print(f"visuals children (default): {[c.GetName() for c in vis.GetChildren()]}", flush=True)

    print("\nfull traversal WITH instance proxies:", flush=True)
    for prim in Usd.PrimRange(finger, Usd.TraverseInstanceProxies()):
        print(f"  {prim.GetPath()}  type={prim.GetTypeName()!r}", flush=True)

    print("\nPrototypes on stage:", flush=True)
    for proto in stage.GetPrototypes():
        print(f"  {proto.GetPath()}", flush=True)
except BaseException:
    import traceback
    print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
finally:
    simulation_app.close()
