"""Scripted-expert baseline on a cube spawn box: can the EXPERT do the task there, and does the camera see the cube?

Why (2026-11-11): a policy evaluated on a wider spawn box than it was trained on can fail because the expert's own method
(RMPflow reach, camera coverage -- the reasons the box was narrowed, see the comment above CUBE_X_RANGE in pick_place_scene.py)
fails there, not because of the policy. This runs the expert at the SAME seeds as a policy evaluation and writes one JSON line per
seed: placed/lifted/place error/max cube height, the wrist_3 statistics the policies are compared with (largest change in 30 ticks,
largest unwinding), and the red-cube pixels the base and wrist cameras see.

    GPU=0 scripts/isaac_py.sh expert_baseline.py --label base --seeds 8000 8019 --out /tmp/expert_base.jsonl
    ... --x_range 0.42 0.48 --y_range -0.13 0.13
(--seeds FIRST LAST inclusive; the evaluation's trial k uses seed S+k)
"""
import argparse
import json
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--seeds", type=int, nargs=2, required=True)
ap.add_argument("--x_range", type=float, nargs=2, default=None)
ap.add_argument("--y_range", type=float, nargs=2, default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402


def wrist3_stats(w):
    d30 = np.abs(w[30:] - w[:-30]).max() if len(w) > 30 else 0.0
    unwind = float(np.max(np.maximum.accumulate(w) - w))
    return float(d30), unwind


def main():
    if args.x_range:
        pps.CUBE_X_RANGE = tuple(args.x_range)
    if args.y_range:
        pps.CUBE_Y_RANGE = tuple(args.y_range)
    print(f"[{args.label}] spawn box X {pps.CUBE_X_RANGE} Y {pps.CUBE_Y_RANGE}", flush=True)
    scene = c.build_scene()
    t0 = time.time()
    for seed in range(args.seeds[0], args.seeds[1] + 1):
        states, actions, cmds, lifted, placed, place_err = c.record_expert(scene, seed)
        base_px, wrist_px = c.record_expert.last_pixels
        w = states[:, 5].astype(float)
        d30, unwind = wrist3_stats(w)
        row = {"label": args.label, "seed": seed, "ticks": len(actions), "lifted": bool(lifted), "placed": bool(placed),
               "place_error_mm": round(1000 * place_err, 1), "max_cube_z": round(c.record_expert.last_max_cube_z, 4),
               "wrist3_max_change_30": round(d30, 3), "wrist3_unwinding": round(unwind, 3),
               "base_px_min_first300": int(base_px[:300].min()), "base_px_median": int(np.median(base_px)),
               "wrist_px_min_150_450": int(wrist_px[150:450].min()), "wrist_px_median_150_450": int(np.median(wrist_px[150:450])),
               "wrist_px_zero_ticks_150_450": int((wrist_px[150:450] == 0).sum())}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{args.label}] seed {seed}: placed={placed} err={row['place_error_mm']}mm T={len(actions)} "
              f"wrist_px_min={row['wrist_px_min_150_450']} base_px_min={row['base_px_min_first300']} "
              f"({time.time() - t0:.0f}s)", flush=True)


try:
    main()
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
