"""One-factor visual perturbations of the pick-place scene for the visual-variation evaluation (README "visual variation").

The scene has never changed its look: one dome light (1500) + the ground plane's sphere light, the default grid ground,
a (0.8, 0.1, 0.1) cube, a green target marker, fixed cameras. A spec string changes one (or more) of these, for a whole run:

    light=0.5            all lights' intensity x 0.5 (dome + ground-plane sphere light)
    tint=1.0:0.8:0.6     dome light colour (r:g:b)
    ground=0.2:0.2:0.2   ground plane colour (a UsdPreviewSurface bound over its own material)
    cube=0.5:0.05:0.05   cube display colour
    basecam=0:0.03:0     base camera moved by (dx:dy:dz) m, orientation unchanged (a re-mounted camera, not re-aimed)
    wristtilt=5          wrist camera tilted down by this many degrees (set_wrist_camera_flange_rotation's down_tilt_deg)
    gridcolor=0.42:0.42:0.42   the grid ground's background colour (sRGB 0-1), grid lines kept: the grid material's own textures
                         (OmniPBR, Wireframe_blue.png = dark blue (16,44,76) + white lines, plus a glowing-line emission) are
                         recoloured, nothing else of the material changes. Separates "grid lines" from "ground colour".
    nogrid=1             the grid lines removed, colour kept: diffuse = flat background colour, line emission = 0.
                         `gridcolor` + `nogrid` = a flat ground of that colour through the same material.
    heldout=1            held-out evaluation ground k (isaac/ground_textures.py: 1 checker, 2 dots, 3 stripes; 0 = plain solid,
                         needs h0=r:g:b, the lab table colour) through the same material, emission off.
    randomize(scene, rng, exclude) draws one v4 training scene (ground family texture, light, tint, base camera shift,
                         wrist tilt), applies it, and returns a JSON-able dict for the episode log.

    e.g. "light=0.5" or "basecam=0:0.03:0,wristtilt=3". Empty / "none" = unchanged.
Applied after the scene's first reset (the cube prim exists then). All of these are USD attributes, which a world.reset()
(Stop+Play) does not touch, so they hold for every trial of the run. `restore(scene)` puts the defaults back (for the
image check that renders several variants in one process).
"""
import io
import os

import numpy as np
from PIL import Image
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
    unknown = set(out) - {"light", "tint", "ground", "cube", "basecam", "wristtilt", "gridcolor", "nogrid", "heldout", "h0"}
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
    sh = st.GetPrimAtPath(GRID_SHADER_PATH)
    _defaults["grid_tex"] = {n: UsdShade.Shader(sh).GetInput(n).Get() for n in GRID_TEX_INPUTS} if sh.IsValid() else {}


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
    if _defaults.get("grid_changed"):        # only if apply() swapped them; the authored relative paths ("./Materials/...")
        sh = UsdShade.Shader(st.GetPrimAtPath(GRID_SHADER_PATH))       # do not resolve when re-set from the root layer,
        for n, v in _defaults["grid_tex"].items():                     # so the resolved (absolute) originals go back in
            sh.GetInput(n).Set(Sdf.AssetPath(v.resolvedPath))
        sh.GetInput("enable_emission").Set(True)
        if sh.GetInput("texture_rotate"):
            sh.GetInput("texture_rotate").Set(0.0)
        _defaults["grid_changed"] = False


GRID_SHADER_PATH = "/World/defaultGroundPlane/Looks/theGrid/Shader"
GRID_TEX_INPUTS = ("diffuse_texture", "emissive_color_texture", "emissive_mask_texture")
GRID_BG = np.array([16, 44, 76], float)          # the only non-line colour of Wireframe_blue.png (checked: 2 unique colours)
TEX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval_runs", "ground_textures")   # git-ignored


def _read_original(name):
    """(PIL image, raw bytes) of an original grid texture."""
    import omni.client
    v = _defaults["grid_tex"][name]
    res, _, content = omni.client.read_file(v.resolvedPath)
    if res != omni.client.Result.OK:
        raise RuntimeError(f"cannot read {v.resolvedPath}: {res}")
    raw = memoryview(content).tobytes()
    return Image.open(io.BytesIO(raw)), raw


def _grid_textures(bg, lines):
    """Writes recoloured copies of the grid textures and returns {input: path}. bg: background sRGB 0-255 (3,); lines: keep
    them (True) or remove them (False). The line weight t of a pixel is its blend fraction from the blue background to
    white (exact 0/1 in the diffuse texture, blurred in the emission mask); the new pixel is bg*(1-t) + 255*t."""
    tag = f"bg{int(bg[0])}_{int(bg[1])}_{int(bg[2])}_{'lines' if lines else 'nolines'}"
    d = os.path.join(TEX_CACHE, tag)
    paths = {n: os.path.join(d, n + ".png") for n in GRID_TEX_INPUTS}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(d, exist_ok=True)
    for n in ("diffuse_texture", "emissive_mask_texture"):
        im, _ = _read_original(n)                 # 8-bit RGB / RGBA with an embedded sRGB ICC profile: both kept, since
        a = np.asarray(im).astype(float)           # the renderer's colour-space choice follows the file
        rgb = a[..., :3]
        t = np.clip(((rgb - GRID_BG) / (255.0 - GRID_BG)).mean(-1, keepdims=True), 0, 1) if lines else 0.0
        a[..., :3] = bg * (1 - t) + 255.0 * t
        Image.fromarray(np.round(a).astype(np.uint8), im.mode).save(paths[n], icc_profile=im.info.get("icc_profile"))
    em, raw = _read_original("emissive_color_texture")                  # 16-bit line glow, 0 off the lines
    if lines:
        open(paths["emissive_color_texture"], "wb").write(raw)          # unchanged, byte for byte
    else:
        Image.fromarray(np.zeros(np.asarray(em).shape, np.uint16)).save(paths["emissive_color_texture"])
    return paths


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
    if "gridcolor" in p or "nogrid" in p:
        if not _defaults["grid_tex"]:
            raise RuntimeError("no grid material at " + GRID_SHADER_PATH)
        bg = np.round(np.array(p["gridcolor"], float) * 255) if "gridcolor" in p else GRID_BG
        paths = _grid_textures(bg, lines=not p.get("nogrid", 0))
        sh = UsdShade.Shader(st.GetPrimAtPath(GRID_SHADER_PATH))
        for n, path in paths.items():
            sh.GetInput(n).Set(Sdf.AssetPath(os.path.abspath(path)))
        _defaults["grid_changed"] = True
    if "heldout" in p:
        import ground_textures as gt
        k = int(p["heldout"])
        img, rot, _ = gt.heldout_texture(k, p.get("h0"))
        tag = f"heldout{k}" + ("_" + "_".join(f"{x:.3f}" for x in p["h0"]) if k == 0 else "")
        set_ground_image(scene, img, rot, os.path.join(TEX_CACHE, tag + ".png"), reuse=True)
    return p


def set_ground_image(scene, img, rotate_deg, path, reuse=False):
    """Puts one procedural tile (ground_textures.py, float 0-1) on the grid material: diffuse texture, emission off (the
    original's glowing lines belong to its own texture), rotation by the material's texture_rotate (seamless tiling).
    A new `path` per image: the renderer caches textures by path."""
    import ground_textures as gt
    _capture(scene)
    if not (reuse and os.path.exists(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(gt.to_uint8(img)).save(path)
    sh = UsdShade.Shader(scene.stage.GetPrimAtPath(GRID_SHADER_PATH))
    sh.GetInput("diffuse_texture").Set(Sdf.AssetPath(os.path.abspath(path)))
    sh.GetInput("enable_emission").Set(False)
    (sh.GetInput("texture_rotate") or sh.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)).Set(float(rotate_deg))
    _defaults["grid_changed"] = True


_random_files = []
_rand_count = [0]


def randomize(scene, rng, exclude=None):
    """One v4 training scene (README "v4 ... design"): ground family texture + lights x log-U[0.5, 2] + dome tint U[0.75, 1]
    per channel + base camera shift U[-3, 3] cm per axis + wrist tilt U[-3, 3] deg. Returns the episode log dict."""
    import ground_textures as gt
    light = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    tint = rng.uniform(0.75, 1.0, 3)
    cam = rng.uniform(-0.03, 0.03, 3)
    tilt = float(rng.uniform(-3.0, 3.0))
    spec = (f"light={light:.4f},tint={tint[0]:.4f}:{tint[1]:.4f}:{tint[2]:.4f},"
            f"basecam={cam[0]:.4f}:{cam[1]:.4f}:{cam[2]:.4f},wristtilt={tilt:.3f}")
    apply(scene, spec)                                        # restores everything first, then the numeric factors
    img, rot, params = gt.random_texture(rng, exclude)
    _rand_count[0] += 1
    path = os.path.join(TEX_CACHE, "random", f"{os.getpid()}_{_rand_count[0]:05d}.png")
    set_ground_image(scene, img, rot, path)
    _random_files.append(path)
    while len(_random_files) > 3:                             # keep disk use bounded; the renderer has loaded older ones
        old = _random_files.pop(0)
        if os.path.exists(old):
            os.remove(old)
    return dict(spec=spec, ground=params)


def describe(p):
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "none"


__all__ = ["parse", "apply", "restore", "describe", "np"]
