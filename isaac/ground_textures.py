"""Procedural ground textures for v4 visual randomization and the held-out evaluation grounds (README "v4 ... design").

Pure numpy (no Isaac): every texture is ONE TILE of the ground, TILE_M x TILE_M metres, periodic so that tiles join without
seams. Orientation is not drawn into the image: it is applied by the material's own `texture_rotate` (degrees), which keeps
the tiling seamless. Colours are sRGB in 0-1 (the renderer decodes these files as sRGB: a 108 grey rendered like the
UsdPreviewSurface 0.15 linear grey, README "grid lines or ground colour?").

The scene's ground material (OmniPBR "theGrid") projects in world space with texture_scale 0.5 -> one tile = 2 m, so
RES px = 2 m (~1 mm / px at 2048). The original grid has 2 lines per tile (every 1 m, 2 cm wide).

Training families (`random_texture`, equal probability): solid, grid, wood, noise.
Held-out (`heldout_texture`, evaluation only, never drawn in training):
    H0 plain solid colour (the lab table colour; excluded from every training colour, see EXCLUDE_DIST)
    H1 checkerboard ~8.3 cm, beige / mid grey          (no checker family in training)
    H2 dots 3 cm on ~7.1 cm spacing, white on dark green (no dot family in training)
    H3 stripes 4 cm at 50 % duty (8 cm period), navy / orange, rotated 30 deg (grid lines are 2-10 mm wide)
"""
import colorsys

import numpy as np

TILE_M = 2.0
RES = 2048
EXCLUDE_DIST = 0.12          # sRGB Euclidean distance: a sampled colour this close to the held-out solid is redrawn
CUBE_RGB = np.array([0.8, 0.1, 0.1])
CUBE_HUE_EXCLUDE = 25 / 360  # ground colours with saturation > 0.45 and hue within this of red are redrawn (cube must stay visible)

FAMILIES = ("solid", "grid", "wood", "noise")
HELDOUT = {
    1: dict(kind="checker", n=24, c0=(0.85, 0.82, 0.75), c1=(0.45, 0.45, 0.45), rotate=0.0),
    2: dict(kind="dots", n=28, diameter_m=0.03, c0=(0.15, 0.35, 0.20), c1=(0.95, 0.95, 0.95), rotate=0.0),
    3: dict(kind="stripes", n=25, duty=0.5, c0=(0.10, 0.15, 0.35), c1=(0.95, 0.50, 0.10), rotate=30.0),
}


def _grid_coords(res=RES):
    u = (np.arange(res) + 0.5) / res                 # 0..1 across the tile
    return np.meshgrid(u, u, indexing="xy")


def _periodic_noise(rng, res, scale_m, slope=2.0):
    """Gaussian-filtered white noise via FFT (periodic by construction), normalised to 0-1. scale_m = feature size."""
    w = rng.standard_normal((res, res))
    f = np.fft.fftfreq(res, d=TILE_M / res)           # cycles per metre
    fx, fy = np.meshgrid(f, f, indexing="xy")
    r = np.sqrt(fx ** 2 + fy ** 2)
    filt = np.exp(-(r * scale_m) ** 2) * (1 + r) ** (-slope / 4)
    n = np.real(np.fft.ifft2(np.fft.fft2(w) * filt))
    n -= n.min()
    return n / max(n.max(), 1e-9)


def _ok_colour(rgb, exclude):
    rgb = np.asarray(rgb, float)
    if exclude is not None and np.linalg.norm(rgb - np.asarray(exclude, float)) < EXCLUDE_DIST:
        return False
    h, s, _ = colorsys.rgb_to_hsv(*rgb)
    if s > 0.45 and min(h, 1 - h) < CUBE_HUE_EXCLUDE:
        return False
    return True


def _colour(rng, exclude, s_max=0.8, v=(0.15, 0.95), hue=None, s=None):
    for _ in range(1000):
        h = rng.uniform(*hue) % 1.0 if hue else rng.uniform()
        rgb = np.array(colorsys.hsv_to_rgb(h, rng.uniform(*s) if s else rng.uniform(0, s_max), rng.uniform(*v)))
        if _ok_colour(rgb, exclude):
            return rgb
    raise RuntimeError("no admissible colour")


def _mix(c0, c1, t):
    t = np.asarray(t, float)[..., None]
    return np.asarray(c0, float) * (1 - t) + np.asarray(c1, float) * t


def solid(c, res=64):
    return np.broadcast_to(np.asarray(c, float), (res, res, 3)).copy()


def grid(bg, line, n, width_m, res=RES):
    u, v = _grid_coords(res)
    half = width_m / TILE_M / 2                       # half line width in tile units
    du = 1.0 / n / 2 - np.abs(((u * n) % 1.0) - 0.5) / n   # distance to the nearest line (lines at cell edges), tile units
    dv = 1.0 / n / 2 - np.abs(((v * n) % 1.0) - 0.5) / n
    on = (du < half) | (dv < half)
    return _mix(bg, line, on)


def checker(c0, c1, n, res=RES):
    u, v = _grid_coords(res)
    return _mix(c0, c1, ((np.floor(u * n) + np.floor(v * n)) % 2))


def dots(bg, dot, n, diameter_m, res=RES):
    u, v = _grid_coords(res)
    du = ((u * n) % 1.0) - 0.5
    dv = ((v * n) % 1.0) - 0.5
    r = np.sqrt(du ** 2 + dv ** 2) * TILE_M / n        # metres from the cell centre
    return _mix(bg, dot, r < diameter_m / 2)


def stripes(c0, c1, n, duty, res=RES):
    u, _ = _grid_coords(res)
    return _mix(c0, c1, ((u * n) % 1.0) < duty)


def wood(rng, dark, light, n_stripes, warp, res=RES):
    u, v = _grid_coords(res)
    nz = _periodic_noise(rng, res, scale_m=rng.uniform(0.05, 0.3))
    fine = _periodic_noise(rng, res, scale_m=0.004)
    g = np.sin(2 * np.pi * (n_stripes * u + warp * nz))   # integer stripes per tile -> periodic
    t = np.clip(0.5 + 0.5 * g, 0, 1) ** rng.uniform(0.6, 2.0)
    t = np.clip(0.85 * t + 0.15 * fine, 0, 1)
    return _mix(dark, light, t)


def noise(rng, c0, c1, scale_m, res=RES):
    t = _periodic_noise(rng, res, scale_m)
    sharp = rng.uniform(1.0, 6.0)
    t = 1 / (1 + np.exp(-sharp * (t - 0.5) * 4))
    return _mix(c0, c1, t)


def random_texture(rng, exclude=None):
    """(image HxWx3 float 0-1, rotate_deg, params dict) for one training episode."""
    fam = FAMILIES[rng.integers(len(FAMILIES))]
    rot = float(rng.uniform(0, 90))
    if fam == "solid":
        c = _colour(rng, exclude)
        return solid(c), 0.0, dict(family=fam, rgb=c.round(3).tolist())
    if fam == "grid":
        bg = _colour(rng, exclude)
        for _ in range(1000):
            line = _colour(rng, None)
            if abs(colorsys.rgb_to_hsv(*line)[2] - colorsys.rgb_to_hsv(*bg)[2]) >= 0.15:
                break
        spacing = float(rng.uniform(0.05, 0.40))
        n = max(1, int(round(TILE_M / spacing)))
        w = float(rng.uniform(0.002, 0.010))
        return grid(bg, line, n, w), rot, dict(family=fam, bg=bg.round(3).tolist(), line=line.round(3).tolist(),
                                                 spacing_m=round(TILE_M / n, 4), width_m=round(w, 4), rotate=round(rot, 1))
    if fam == "wood":
        hue = (20 / 360, 42 / 360)
        dark = _colour(rng, exclude, hue=hue, s=(0.35, 0.75), v=(0.15, 0.5))
        light = _colour(rng, exclude, hue=hue, s=(0.25, 0.6), v=(0.5, 0.9))
        period = float(rng.uniform(0.01, 0.08))
        n = max(1, int(round(TILE_M / period)))
        warp = float(rng.uniform(0.5, 4.0))
        return wood(rng, dark, light, n, warp), rot, dict(family=fam, dark=dark.round(3).tolist(), light=light.round(3).tolist(),
                                                            period_m=round(TILE_M / n, 4), warp=round(warp, 2), rotate=round(rot, 1))
    c0, c1 = _colour(rng, exclude), _colour(rng, exclude)
    scale = float(np.exp(rng.uniform(np.log(0.01), np.log(0.30))))
    return noise(rng, c0, c1, scale), rot, dict(family=fam, c0=c0.round(3).tolist(), c1=c1.round(3).tolist(),
                                                 scale_m=round(scale, 4), rotate=round(rot, 1))


def heldout_texture(k, h0_rgb=None):
    """(image, rotate_deg, params) of held-out ground k (0-3). H0 needs the table colour."""
    if k == 0:
        if h0_rgb is None:
            raise ValueError("H0 needs its colour (the lab table colour)")
        return solid(h0_rgb), 0.0, dict(heldout=0, kind="solid", rgb=list(h0_rgb))
    p = HELDOUT[k]
    if p["kind"] == "checker":
        img = checker(p["c0"], p["c1"], p["n"])
    elif p["kind"] == "dots":
        img = dots(p["c0"], p["c1"], p["n"], p["diameter_m"])
    else:
        img = stripes(p["c0"], p["c1"], p["n"], p["duty"])
    return img, p["rotate"], dict(heldout=k, **p)


def to_uint8(img):
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
