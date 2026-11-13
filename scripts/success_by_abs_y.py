#!/usr/bin/env python3
"""Success rate against |spawn Y| (0.02 bins) over every saved evaluation of the final checkpoints -- cliff or slope? (GPU-free)

Spawn = the trace's first cube position. Runs: v2 seed 42 (`short_3k/2999`: training box K=50 trials of the 3-K run, step 1, step 2,
+Y/-Y strips), v2 seed 1 and v3 (training box, step 1, step 2). Per model and pooled, with Wilson intervals; X < 0.42 separated
because small X is a second difficulty.

    python3 scripts/success_by_abs_y.py [--png out.png]
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_eval import wilson  # noqa: E402

ROOT = os.path.dirname(HERE)
R, E = os.path.join(ROOT, "eval_results"), os.path.join(ROOT, "eval_runs")
RUNS = [  # model, csv, K filter, trace dir
    ("v2_s42", f"{R}/short_3k_2999_k10-25-50_seed8000.csv", 50, f"{E}/k3_2999/trace"),
    ("v2_s42", f"{R}/short_3k_2999_range_step1_k50_seed8000.csv", None, f"{E}/range_step1/trace"),
    ("v2_s42", f"{R}/short_3k_2999_range_step2_k50_seed8000.csv", None, f"{E}/range_step2/trace"),
    ("v2_s42", f"{R}/short_3k_2999_stripposY_k50_seed9000.csv", None, f"{E}/strip_posY/trace"),
    ("v2_s42", f"{R}/short_3k_2999_stripnegY_k50_seed9000.csv", None, f"{E}/strip_negY/trace"),
] + [(m, f"{R}/{pre}_{b}_k50_seed8000.csv", None, f"{E}/{d}_{b}/trace")
     for m, pre, d in (("v2_s1", "short_3k_seed1_2999", "seed1"), ("v3", "v3_3k_2999", "v3")) for b in ("train_box", "step1", "step2")]


def load():
    rows = []
    for model, path, keep_k, tdir in RUNS:
        for r in csv.DictReader(open(path)):
            if keep_k and int(r["execute_horizon"]) != keep_k:
                continue
            cube0 = np.load(os.path.join(tdir, f"trial_{int(r['trial']):03d}.npz"))["cube"][0]
            rows.append((model, float(cube0[0]), float(cube0[1]), r["success"] == "True"))
    return rows


def main():
    rows = load()
    edges = np.arange(0.0, 0.18, 0.02)
    models = ("v2_s42", "v2_s1", "v3")
    print(f"{len(rows)} attempts. placed/n per |Y| bin (X >= 0.42 only | all X); pooled Wilson interval on X >= 0.42")
    print("|Y| bin       " + "  ".join(f"{m:>12s}" for m in models) + "   pooled X>=0.42         pooled all X")
    curve = []
    for lo in edges[:-1]:
        hi = lo + 0.02
        sel = [r for r in rows if lo <= abs(r[2]) < hi]
        cells = []
        for m in models:
            a = [r for r in sel if r[0] == m and r[1] >= 0.42]
            cells.append(f"{sum(r[3] for r in a):>3d}/{len(a):<3d}" + f"|{sum(r[3] for r in sel if r[0] == m)}/{sum(1 for r in sel if r[0] == m)}")
        a = [r for r in sel if r[1] >= 0.42]
        k, n = sum(r[3] for r in a), len(a)
        lo_w, hi_w = wilson(k, n) if n else (float("nan"), float("nan"))
        curve.append((lo, k, n, lo_w, hi_w, sum(r[3] for r in sel), len(sel)))
        print(f"[{lo:.2f},{hi:.2f})  " + "  ".join(f"{c:>12s}" for c in cells) +
              f"   {k:3d}/{n:<3d} {k / n if n else float('nan'):.2f} [{lo_w:.2f},{hi_w:.2f}]   {sum(r[3] for r in sel)}/{len(sel)}")
    if "--png" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        c = [x for x in curve if x[2]]
        xs = [x[0] + 0.01 for x in c]
        ax.errorbar(xs, [x[1] / x[2] for x in c], yerr=[[x[1] / x[2] - x[3] for x in c], [x[4] - x[1] / x[2] for x in c]],
                    fmt="o-", color="#2a6fb0", capsize=3, label="pooled 3 models, X >= 0.42 (Wilson 95%)")
        for x in c:
            ax.annotate(f"{x[1]}/{x[2]}", (x[0] + 0.01, x[1] / x[2]), textcoords="offset points", xytext=(4, -12), fontsize=8)
        ax.axvline(0.10, color="#999", ls="--", lw=1)
        ax.text(0.101, 0.05, "training box edge", fontsize=8, color="#666")
        ax.set_xlabel("|spawn Y| (m)")
        ax.set_ylabel("placed fraction")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=130)


if __name__ == "__main__":
    main()
