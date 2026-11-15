"""Image check for v4 visual randomization (no policy): N random training scenes (visual_perturbation.randomize) and the
held-out grounds, rendered at the home pose as the policy would see them after the converged first frame.
Output: <out>/random_scenes.png (each tile = base | wrist, label = family) and <out>/random_scenes.json.

    GPU=0 scripts/isaac_py.sh render_random_scenes.py --out DIR --n 16 --exclude 0.9 0.9 0.88 [--heldout_h0 0.9 0.9 0.88]
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--exclude", type=float, nargs=3, default=None)
ap.add_argument("--heldout_h0", type=float, nargs=3, default=None)
ap.add_argument("--renders", type=int, default=30)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import visual_perturbation as vp  # noqa: E402

try:
    from PIL import Image, ImageDraw
    os.makedirs(args.out, exist_ok=True)
    scene = c.build_scene()
    scene.reset()
    rng = np.random.default_rng(args.seed)
    info, tiles = [], []

    def shot(label):
        for _ in range(args.renders):
            scene.world.render()
        o = scene.get_observation()
        b, w = np.asarray(o["base_rgb"])[..., :3], np.asarray(o["wrist_rgb"])[..., :3]
        t = Image.fromarray(np.concatenate([b, w], 1).astype(np.uint8))
        ImageDraw.Draw(t).text((3, 3), label, fill=(255, 255, 0))
        tiles.append(np.asarray(t))

    for i in range(args.n):
        scene._rng = np.random.default_rng(9200 + i)
        scene.reset()
        d = vp.randomize(scene, rng, args.exclude)
        info.append(d)
        print(json.dumps(d), flush=True)
        shot(f"{i} {d['ground']['family']}")
    for k in (1, 2, 3) + ((0,) if args.heldout_h0 else ()):
        scene._rng = np.random.default_rng(9206)
        scene.reset()
        spec = f"heldout={k}" + (",h0=" + ":".join(map(str, args.heldout_h0)) if k == 0 else "")
        vp.apply(scene, spec)
        info.append({"heldout": spec})
        shot(f"H{k}")
    vp.restore(scene)
    scene._rng = np.random.default_rng(9206)
    scene.reset()
    shot("restored (= none)")
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out, "random_scenes.png"))
    json.dump(info, open(os.path.join(args.out, "random_scenes.json"), "w"), indent=1)
    print("saved", os.path.join(args.out, "random_scenes.png"), flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
