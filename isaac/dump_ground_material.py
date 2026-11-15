"""Prints the ground plane's grid material (OmniPBR) inputs with resolved texture paths, and saves the textures to --out.
    GPU=0 scripts/isaac_py.sh dump_ground_material.py --out DIR"""
import os
import sys
out = sys.argv[sys.argv.index("--out") + 1]
sys.argv = [sys.argv[0]]
import check_action_replay as c  # noqa: E402
from pxr import UsdShade  # noqa: E402
try:
    import omni.client
    scene = c.build_scene(); scene.reset(); st = scene.stage
    sh = UsdShade.Shader(st.GetPrimAtPath("/World/defaultGroundPlane/Looks/theGrid/Shader"))
    os.makedirs(out, exist_ok=True)
    for i in sh.GetInputs():
        v = i.Get()
        rp = getattr(v, "resolvedPath", None)
        print("in", i.GetBaseName(), v, "->", rp, flush=True)
        if rp:
            res, _, content = omni.client.read_file(rp)
            print("  read", res, len(memoryview(content)) if content else 0, flush=True)
            if content:
                open(os.path.join(out, os.path.basename(v.path)), "wb").write(memoryview(content).tobytes())
    print("layers", [l.identifier for l in st.GetPrimAtPath("/World/defaultGroundPlane/Looks/theGrid/Shader").GetPrimStack()], flush=True)
except BaseException:
    import traceback; print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
