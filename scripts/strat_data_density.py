#!/usr/bin/env python3
"""GPU-free: v3 training-episode count per stratified region next to the v3 lateral miss (post-hoc check of the -Y asymmetry,
README "stratified evaluation results"). Episode position = grip point at the first close (FK of the recorded joints), as in
scripts/corner_expert_view.py. Per trial: number of v3 episodes within 30 mm of the spawn, and the rank correlation with the miss.
    python3 scripts/strat_data_density.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strat_analysis import REGIONS, load  # noqa: E402
from ur5e_fk import apply  # noqa: E402

FK_P = np.array([-np.pi, 0, 0, 0, 0, 0.12, 0])
BOX = {"centre": (0.42, 0.48, -0.10, 0.10), "posY_strip": (0.42, 0.48, 0.10, 0.16), "negY_strip": (0.42, 0.48, -0.16, -0.10),
       "posY_corner": (0.38, 0.42, 0.11, 0.16), "negY_corner": (0.38, 0.42, -0.16, -0.11), "mid_lowX": (0.38, 0.42, -0.11, 0.11)}


def episodes(repo):
    pts = []
    for f in sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/lerobot/Jaeung12/{repo}/data/*/*.parquet"))):
        d = pd.read_parquet(f, columns=["joints", "gripper", "episode_index"])
        for _, dd in d.groupby("episode_index"):
            q = np.stack(dd["joints"].values)[:, :6].astype(float)
            g = np.concatenate([np.atleast_1d(x) for x in dd["gripper"].values])
            c = np.nonzero(g > 0.5)[0]
            if len(c):
                gp = apply(FK_P, q)
                pts.append((gp[c[0], 0], gp[c[0], 1]))
    return np.array(pts)


def inbox(p, b):
    return (p[:, 0] >= b[0]) & (p[:, 0] < b[1]) & (p[:, 1] >= b[2]) & (p[:, 1] < b[3])


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    v2, v3 = episodes("ur5e_pick_place_v2"), episodes("ur5e_pick_place_v3")
    print(f"episodes with a close: v2 {len(v2)}, v3 {len(v3)}")
    res = {m: {g: load(f"strat_{m}_{g}", 9200) for g in REGIONS} for m in ("v3_3k", "v3_6k")}
    print("region        v2 eps  v3 eps | v3 3k placed, miss med | v3 6k placed, miss med | median v3 eps within 30 mm of spawn")
    allm = {m: [] for m in res}
    alln = []
    for g in REGIONS:
        b = BOX[g]
        cells = []
        nn = []
        for s in sorted(res["v3_3k"][g]):
            r = res["v3_3k"][g][s]
            nn.append(int((np.hypot(v3[:, 0] - r["x"], v3[:, 1] - r["y"]) < 0.03).sum()))
        for m in res:
            rows = res[m][g]
            ss = sorted(rows)
            cells.append(f"{sum(rows[s]['ok'] for s in ss):2d}/15 {np.nanmedian([rows[s]['miss_mm'] for s in ss]):5.0f}")
            allm[m] += [rows[s]["miss_mm"] for s in ss]
        alln += nn
        print(f"{g:12s} {inbox(v2, b).sum():6d} {inbox(v3, b).sum():7d} |        {cells[0]}         |        {cells[1]}         | {np.median(nn):.0f}")
    alln = np.array(alln)
    for m in res:
        mm = np.array(allm[m]); ok = np.isfinite(mm)
        print(f"{m}: Spearman(miss, v3 eps within 30 mm) over 90 = {spearman(mm[ok], alln[ok]):+.2f}  (+ = more data, less short)")


if __name__ == "__main__":
    main()
