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
    for side in ("left_inner_finger", "right_inner_finger"):
        link_path = f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85/{side}"
        link = stage.GetPrimAtPath(link_path)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                   useExtentsHint=False)
        # local bound of the link itself, in the link's OWN frame -- traverse
        # instance proxies so the instanced pad mesh is actually included.
        box = cache.ComputeLocalBound(link)
        rng = box.ComputeAlignedRange()
        print(f"{side}: local bbox min={list(rng.GetMin())} max={list(rng.GetMax())} "
              f"empty={rng.IsEmpty()}", flush=True)
        # ComputeLocalBound might not descend into instance proxies depending on
        # Isaac Sim version -- cross-check with a direct per-mesh union.
        lo = [1e9, 1e9, 1e9]
        hi = [-1e9, -1e9, -1e9]
        xf = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                mbox = cache.ComputeWorldBound(prim)
                mrng = mbox.ComputeAlignedRange()
                if mrng.IsEmpty():
                    continue
                wmin, wmax = mrng.GetMin(), mrng.GetMax()
                corners = [(wmin[0], wmin[1], wmin[2]), (wmax[0], wmax[1], wmax[2])]
                for cx in (wmin[0], wmax[0]):
                    for cy in (wmin[1], wmax[1]):
                        for cz in (wmin[2], wmax[2]):
                            local = xf.GetInverse().Transform((cx, cy, cz))
                            for k in range(3):
                                lo[k] = min(lo[k], local[k])
                                hi[k] = max(hi[k], local[k])
        print(f"{side}: mesh-union local bbox min={lo} max={hi}", flush=True)
except BaseException:
    import traceback
    print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
finally:
    simulation_app.close()
