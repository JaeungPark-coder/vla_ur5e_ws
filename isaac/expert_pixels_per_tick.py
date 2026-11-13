"""Per-tick red-cube pixels in the base and wrist cameras along the SCRIPTED EXPERT's episode, for eval seeds (expert_baseline.py
only keeps summaries). Proxy for what a policy sees early in an episode, before it deviates from the expert's motion.

    GPU=0 scripts/isaac_py.sh expert_pixels_per_tick.py --seeds 8000 8019 --x_range 0.38 0.48 --y_range -0.16 0.16 --out /tmp/px.npz
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs=2, required=True)
ap.add_argument("--x_range", type=float, nargs=2, default=None)
ap.add_argument("--y_range", type=float, nargs=2, default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402

try:
    if args.x_range:
        pps.CUBE_X_RANGE = tuple(args.x_range)
    if args.y_range:
        pps.CUBE_Y_RANGE = tuple(args.y_range)
    scene = c.build_scene()
    out = {}
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        c.record_expert(scene, seed)
        base_px, wrist_px = c.record_expert.last_pixels
        out[f"base_{seed}"] = np.asarray(base_px, dtype=np.int32)
        out[f"wrist_{seed}"] = np.asarray(wrist_px, dtype=np.int32)
        out[f"spawn_{seed}"] = np.asarray(scene.get_cube_position(), dtype=float)[:2]
        print(f"seed {seed}: ticks {len(base_px)}", flush=True)
        np.savez_compressed(args.out, **out)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
