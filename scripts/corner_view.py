#!/usr/bin/env python3
"""Is the cube visible to the policy in the near +Y corner (X < 0.42, Y > +0.11)? GPU-free, n=60 wide-box runs (images saved).

Red-cube pixels (as scripts/policy_input_pixels.py) in the base and wrist images the policy was given, per decision tick 0-550,
by spawn region, all three models. The mirror -Y corner and the other regions are the comparison.
    python3 scripts/corner_view.py [--json eval_results/corner_view.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def red(img):  # same threshold as scripts/policy_input_pixels.py (that module runs on import, so copied)
    im = img[..., :3].astype(int)
    return int(((im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)).sum())


RUNS = {"v2 3k": "n60_v2_3k", "v3 3k": "n60_v3_3k", "v3 6k": "n60_v3_6k"}
REGIONS = {
    "+Y corner X<0.42 Y>+0.11": lambda x, y: x < 0.42 and y > 0.11,
    "-Y corner X<0.42 Y<-0.11": lambda x, y: x < 0.42 and y < -0.11,
    "X<0.42 |Y|<=0.11": lambda x, y: x < 0.42 and abs(y) <= 0.11,
    "+Y edge X>=0.42 Y>+0.10": lambda x, y: x >= 0.42 and y > 0.10,
    "-Y edge X>=0.42 Y<-0.10": lambda x, y: x >= 0.42 and y < -0.10,
    "centre X>=0.42 |Y|<=0.10": lambda x, y: x >= 0.42 and abs(y) <= 0.10,
}
TICKS = list(range(0, 551, 50))


def main():
    seeds = json.load(open(os.path.join(ROOT, "eval_results", "n60_analysis.json")))["per_seed"]
    out = {}
    for m, run in RUNS.items():
        tdir = os.path.join(ROOT, "eval_runs", run, "trace")
        print(f"\n== {m}: median red-cube pixels base / wrist per decision tick; [n zero in both cameras]")
        print(f"{'region':28s} n ok | " + " ".join(f"{t:>11d}" for t in TICKS))
        out[m] = {}
        for rname, f in REGIONS.items():
            ss = [s for s, r in seeds.items() if f(r["x"], r["y"])]
            px = []
            for s in ss:
                z = np.load(os.path.join(tdir, f"trial_{int(s) - 7999:03d}_images.npz"))
                tick = list(z["tick"])
                px.append([(red(z["base"][tick.index(t)]), red(z["wrist"][tick.index(t)])) if t in tick else (np.nan, np.nan) for t in TICKS])
            px = np.array(px, dtype=float)  # (n, ticks, 2)
            ok = sum(seeds[s][m] == "placed" for s in ss)
            cells = [f"{np.nanmedian(px[:, k, 0]):4.0f}/{np.nanmedian(px[:, k, 1]):4.0f}[{int(((px[:, k, 0] == 0) & (px[:, k, 1] == 0)).sum())}]"
                     for k in range(len(TICKS))]
            print(f"{rname:28s}{len(ss):2d} {ok:2d} | " + " ".join(f"{c:>11s}" for c in cells))
            out[m][rname] = {"seeds": ss, "placed": ok, "base": px[:, :, 0].tolist(), "wrist": px[:, :, 1].tolist()}
    if "--json" in sys.argv:
        json.dump({"ticks": TICKS, **out}, open(sys.argv[sys.argv.index("--json") + 1], "w"))


if __name__ == "__main__":
    main()
