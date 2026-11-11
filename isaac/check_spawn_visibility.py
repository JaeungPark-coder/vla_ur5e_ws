"""How many red-cube pixels do the base and wrist cameras see for a cube spawned at (x, y), with the arm at its home pose?

Why (2026-11-11): the spawn box was narrowed once because the wrist camera lost the cube in a wide box (comment above CUBE_X_RANGE
in pick_place_scene.py). Before evaluating a policy on a wider box, find where in the plane the cameras lose the cube. Samples
`--n` positions uniformly from a wide box (default X 0.34-0.54, Y -0.22-0.22), lets the scene settle `--settle` ticks, counts pixels
(PickPlaceScene.cube_pixels_visible), writes one JSON line per sample.

    GPU=0 scripts/isaac_py.sh check_spawn_visibility.py --n 200 --out /tmp/visibility.jsonl
"""
import argparse
import json
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--x_range", type=float, nargs=2, default=[0.34, 0.54])
ap.add_argument("--y_range", type=float, nargs=2, default=[-0.22, 0.22])
ap.add_argument("--settle", type=int, default=10)
ap.add_argument("--seed", type=int, default=5000)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402

try:
    scene = c.build_scene()
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    for k in range(args.n):
        scene._rng = np.random.default_rng(args.seed + k)
        scene.reset()
        for _ in range(args.settle):
            scene.world.step(render=True)
        o = scene.get_observation()
        cube = scene.get_cube_position()
        row = {"seed": args.seed + k, "x": round(float(cube[0]), 4), "y": round(float(cube[1]), 4),
               "base_px": scene.cube_pixels_visible(o["base_rgb"]), "wrist_px": scene.cube_pixels_visible(o["wrist_rgb"])}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if k % 20 == 0:
            print(k, row, flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
