"""Render what the POLICY sees for given eval seeds: both camera images at the home pose with the cube at that seed's spawn.

Why (2026-11-13): failures sit at spawn Y > +0.10. The traces do not store the policy's input images, so this re-creates the scene
exactly as the bridge does (seed -> cube spawn in the given box), lets it settle and saves base_rgb / wrist_rgb (256x256, what the
bridge sends before the policy's resize) plus the cube's pixel bounding box in each camera, and a montage.

    GPU=0 scripts/isaac_py.sh render_spawn_images.py --x_range 0.38 0.48 --y_range -0.16 0.16 --seeds 8010 8015 8019 8003 8014 --out /tmp/spawn_imgs
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--x_range", type=float, nargs=2, required=True)
ap.add_argument("--y_range", type=float, nargs=2, required=True)
ap.add_argument("--seeds", type=int, nargs="*", default=[])
ap.add_argument("--xy", type=float, nargs="*", default=[], help="x1 y1 x2 y2 ...: spawn the cube exactly there (instead of seeds)")
ap.add_argument("--settle", type=int, default=10)
ap.add_argument("--renders", type=int, default=0, help="world.render() calls (no physics step) after the reset, before reading the images")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402


def red_box(img):
    im = img[..., :3].astype(int)
    m = (im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return {"px": int(m.sum()), "x0": int(xs.min()), "x1": int(xs.max()), "y0": int(ys.min()), "y1": int(ys.max()),
            "cx": round(float(xs.mean()), 1), "cy": round(float(ys.mean()), 1)}


try:
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    scene = c.build_scene()
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    info, tiles = [], []
    jobs = [(seed, None) for seed in args.seeds] + [(i, (args.xy[2 * i], args.xy[2 * i + 1])) for i in range(len(args.xy) // 2)]
    for seed, xy in jobs:
        scene._rng = np.random.default_rng(seed)
        if xy is not None:
            pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = (xy[0], xy[0]), (xy[1], xy[1])
        scene.reset()
        for _ in range(args.settle):
            scene.world.step(render=True)
        for _ in range(args.renders):
            scene.world.render()
        o = scene.get_observation()
        cube = scene.get_cube_position()
        b, w = np.asarray(o["base_rgb"]), np.asarray(o["wrist_rgb"])
        if xy is not None:
            seed = f"x{xy[0]:.2f}_y{xy[1]:+.2f}"
        row = {"seed": seed, "cube_xy": [round(float(cube[0]), 3), round(float(cube[1]), 3)], "base": red_box(b), "wrist": red_box(w)}
        Image.fromarray(b[..., :3].astype(np.uint8)).save(f"{args.out}/{seed}_base.png")
        info.append(row)
        tiles.append((seed, b[..., :3].astype(np.uint8), w[..., :3].astype(np.uint8)))
        print(row, flush=True)
    H = tiles[0][1].shape[0]
    mont = np.zeros((2 * H, H * len(tiles), 3), np.uint8)
    for i, (_, b, w) in enumerate(tiles):
        mont[:H, i * H:(i + 1) * H], mont[H:, i * H:(i + 1) * H] = b, w
    Image.fromarray(mont).save(f"{args.out}/montage.png")
    json.dump(info, open(f"{args.out}/info.json", "w"), indent=1)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
