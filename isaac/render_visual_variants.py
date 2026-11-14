"""Image check for the visual-variation evaluation (no policy): renders what the policy would see at the home pose for each
visual perturbation (isaac/visual_perturbation.py), for a few eval seeds, plus the red-cube pixel count in each camera.
Output: <out>/variants.png (rows = conditions, columns = base/wrist per seed) and <out>/variants.json.

    GPU=0 scripts/isaac_py.sh render_visual_variants.py --out /path/to/dir --seeds 9200 9203 --conditions none "light=0.5" ...
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--x_range", type=float, nargs=2, default=[0.42, 0.48])
ap.add_argument("--y_range", type=float, nargs=2, default=[-0.10, 0.10])
ap.add_argument("--seeds", type=int, nargs="+", default=[9200, 9203])
ap.add_argument("--conditions", nargs="+", required=True)
ap.add_argument("--renders", type=int, default=30)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402
import visual_perturbation as vp  # noqa: E402


def red(img):
    im = img[..., :3].astype(int)
    return int(((im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)).sum())


try:
    from PIL import Image, ImageDraw
    os.makedirs(args.out, exist_ok=True)
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    scene = c.build_scene()
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    scene.reset()
    info, rows = [], []
    for cond in args.conditions:
        p = vp.apply(scene, cond)
        tiles = []
        for seed in args.seeds:
            scene._rng = np.random.default_rng(seed)
            scene.reset()
            for _ in range(args.renders):
                scene.world.render()
            o = scene.get_observation()
            b, w = np.asarray(o["base_rgb"])[..., :3], np.asarray(o["wrist_rgb"])[..., :3]
            info.append({"condition": cond, "seed": seed, "cube": np.round(scene.get_cube_position(), 4).tolist(),
                         "base_red_px": red(b), "wrist_red_px": red(w), "base_mean": float(b.mean()), "wrist_mean": float(w.mean())})
            print(json.dumps(info[-1]), flush=True)
            tiles += [b, w]
        row = Image.fromarray(np.concatenate(tiles, axis=1).astype(np.uint8))
        canvas = Image.new("RGB", (row.width + 170, row.height), "white")
        canvas.paste(row, (170, 0))
        ImageDraw.Draw(canvas).text((4, 4), vp.describe(p), fill="black")
        rows.append(canvas)
    vp.restore(scene)
    W = max(r.width for r in rows)
    mont = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        mont.paste(r, (0, y)); y += r.height
    mont.save(os.path.join(args.out, "variants.png"))
    json.dump(info, open(os.path.join(args.out, "variants.json"), "w"), indent=1)
    print("saved", os.path.join(args.out, "variants.png"), flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
