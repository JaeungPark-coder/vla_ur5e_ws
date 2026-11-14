"""One-factor visual perturbations of the pick-place scene for the visual-variation evaluation (README "visual variation").

The scene has never changed its look: one dome light (1500) + the ground plane's sphere light, the default grid ground,
a (0.8, 0.1, 0.1) cube, a green target marker, fixed cameras. A spec string changes one (or more) of these, for a whole run:

    light=0.5            all lights' intensity x 0.5 (dome + ground-plane sphere light)
    tint=1.0:0.8:0.6     dome light colour (r:g:b)
    ground=0.2:0.2:0.2   ground plane colour (a UsdPreviewSurface bound over its own material)
    cube=0.5:0.05:0.05   cube display colour
    basecam=0:0.03:0     base camera moved by (dx:dy:dz) m, orientation unchanged (a re-mounted camera, not re-aimed)
    wristtilt=5          wrist camera tilted down by this many degrees (set_wrist_camera_flange_rotation's down_tilt_deg)

    e.g. "light=0.5" or "basecam=0:0.03:0,wristtilt=3". Empty / "none" = unchanged.
Applied after the scene's first reset (the cube prim exists then). All of these are USD attributes, which a world.reset()
(Stop+Play) does not touch, so they hold for every trial of the run. `restore(scene)` puts the defaults back (for the
image check that renders several variants in one process).
"""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import pick_place_scene as pps

GROUND_MATERIAL_PATH = "/World/Looks/visualVariantGround"
_defaults = {}


def parse(spec):
    out = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item or item == "none":
            continue
        k, v = item.split("=", 1)
        vals = [float(x) for x in v.split(":")]
        out[k.strip()] = vals[0] if len(vals) == 1 else vals
    unknown = set(out) - {"light", "tint", "ground", "cube", "basecam", "wristtilt"}
    if unknown:
        raise ValueError(f"unknown visual perturbation keys {sorted(unknown)}")
    return out


def _lights(stage):
    return [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight) or p.IsA(UsdLux.SphereLight) or p.IsA(UsdLux.DistantLight)]


def _ground_prim(stage):
    for path in ("/World/defaultGroundPlane", "/World/GroundPlane"):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            return p
    raise RuntimeError("no ground plane prim found")


def _capture(scene):
    if _defaults:
        return
    st = scene.stage
    _defaults["lights"] = {}
    for p in _lights(st):
        lt = UsdLux.LightAPI(p)
        col = lt.GetColorAttr().Get()
        _defaults["lights"][str(p.GetPath())] = (lt.GetIntensityAttr().Get(), Gf.Vec3f(col) if col is not None else Gf.Vec3f(1, 1, 1))
    _defaults["cube"] = UsdGeom.Gprim(st.GetPrimAtPath(pps.CUBE_PRIM_PATH)).GetDisplayColorAttr().Get()
    cam = UsdGeom.Xformable(st.GetPrimAtPath(pps.BASE_CAMERA_PRIM_PATH))
    _defaults["basecam_op"] = next(op for op in cam.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    _defaults["basecam"] = Gf.Vec3d(_defaults["basecam_op"].Get())
    g = _ground_prim(st)
    _defaults["ground_prim"] = g
    _defaults["ground_binding"] = UsdShade.MaterialBindingAPI(g).GetDirectBinding().GetMaterial()


def restore(scene):
    _capture(scene)
    st = scene.stage
    for path, (inten, col) in _defaults["lights"].items():
        lt = UsdLux.LightAPI(st.GetPrimAtPath(path))
        lt.GetIntensityAttr().Set(inten)
        lt.CreateColorAttr().Set(col)
    UsdGeom.Gprim(st.GetPrimAtPath(pps.CUBE_PRIM_PATH)).GetDisplayColorAttr().Set(_defaults["cube"])
    _defaults["basecam_op"].Set(_defaults["basecam"])
    g = _defaults["ground_prim"]
    api = UsdShade.MaterialBindingAPI(g)
    if _defaults["ground_binding"]:
        api.Bind(_defaults["ground_binding"], UsdShade.Tokens.strongerThanDescendants)
    else:
        api.UnbindDirectBinding()
    for child in Usd.PrimRange(g):
        cb = UsdShade.MaterialBindingAPI(child)
        if str(cb.GetDirectBindingRel().GetTargets()[0] if cb.GetDirectBindingRel() and cb.GetDirectBindingRel().GetTargets()
               else "") == GROUND_MATERIAL_PATH:
            cb.UnbindDirectBinding()
    scene.set_wrist_camera_flange_rotation(pps.WRIST_CAMERA_FLANGE_ROT_EULER, down_tilt_deg=pps.WRIST_CAMERA_DOWN_TILT_DEG)


def _ground_material(stage, rgb):
    mat = UsdShade.Material.Define(stage, GROUND_MATERIAL_PATH)
    sh = UsdShade.Shader.Define(stage, GROUND_MATERIAL_PATH + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def apply(scene, spec):
    """Applies `spec` on top of the defaults (restores first). Returns the parsed dict (for logging)."""
    p = parse(spec)
    restore(scene)
    st = scene.stage
    if "light" in p:
        for path, (inten, _) in _defaults["lights"].items():
            UsdLux.LightAPI(st.GetPrimAtPath(path)).GetIntensityAttr().Set(float(inten) * p["light"])
    if "tint" in p:
        for path in _defaults["lights"]:
            prim = st.GetPrimAtPath(path)
            if prim.IsA(UsdLux.DomeLight):
                UsdLux.LightAPI(prim).CreateColorAttr().Set(Gf.Vec3f(*p["tint"]))
    if "ground" in p:
        g = _defaults["ground_prim"]
        mat = _ground_material(st, p["ground"])
        UsdShade.MaterialBindingAPI.Apply(g).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        for child in Usd.PrimRange(g):          # the plane's own mesh may carry a direct binding of its own
            if child.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(child).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
    if "cube" in p:
        UsdGeom.Gprim(st.GetPrimAtPath(pps.CUBE_PRIM_PATH)).GetDisplayColorAttr().Set([Gf.Vec3f(*p["cube"])])
    if "basecam" in p:
        _defaults["basecam_op"].Set(_defaults["basecam"] + Gf.Vec3d(*p["basecam"]))
    if "wristtilt" in p:
        scene.set_wrist_camera_flange_rotation(pps.WRIST_CAMERA_FLANGE_ROT_EULER,
                                               down_tilt_deg=pps.WRIST_CAMERA_DOWN_TILT_DEG + p["wristtilt"])
    return p


def describe(p):
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "none"


__all__ = ["parse", "apply", "restore", "describe", "np"]
