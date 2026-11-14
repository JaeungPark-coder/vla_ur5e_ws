#!/usr/bin/env python3
"""Corner confirmation test analysis (GPU-free): v3 6k on the +Y corner (X 0.38-0.42, Y +0.11..+0.16) vs the -Y mirror, seeds 9100-9119,
converged first frame. Pre-registered (README "corner confirmation test"): primary = lateral miss at the first close (grip Y - spawn Y,
signed + = away from the centre line), +Y vs -Y by unpaired bootstrap of the medians; secondary = placed (Wilson), expert placed, results on
expert-success seeds; view: base-camera red pixels at t50/t100 (blind = 0 px in both cameras), wrist cube column and edge contact at t100.
    python3 scripts/corner_analysis.py [--json eval_results/corner_analysis.json]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from failure_overlap import analyse, classify  # noqa: E402
from n60_analysis import wilson  # noqa: E402

CORNERS = {"+Y": "corner_v3_6k_posY", "-Y": "corner_v3_6k_negY"}
EXPERT = {"+Y": "expert_corner_posY.jsonl", "-Y": "expert_corner_negY.jsonl"}


def mask(img):
    im = img[..., :3].astype(int)
    return (im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)


def load(run):
    d = os.path.join(ROOT, "eval_runs", run)
    rows = {}
    for r in csv.DictReader(open(os.path.join(d, "results.csv"))):
        t = int(r["trial"])
        a = analyse(os.path.join(d, "trace"), t)
        a["ok"] = r["success"] == "True"
        a["cls"] = classify(a, a["ok"])
        z = np.load(os.path.join(d, "trace", f"trial_{t:03d}.npz"))
        cube, grip, g = z["cube"].astype(float), z["grip_point"].astype(float), z["gripper"].astype(float)
        c = np.nonzero(g > 0.5)[0]
        a["x"], a["y"] = float(cube[0, 0]), float(cube[0, 1])
        a["miss_mm"] = float(1000 * (grip[c[0], 1] - a["y"]) * np.sign(a["y"])) if len(c) and np.isfinite(grip[c[0]]).all() else np.nan
        im = np.load(os.path.join(d, "trace", f"trial_{t:03d}_images.npz"))
        tick = list(im["tick"])
        for tt in (0, 50, 100):
            mb, mw = mask(im["base"][tick.index(tt)]), mask(im["wrist"][tick.index(tt)])
            a[f"base{tt}"], a[f"wrist{tt}"] = int(mb.sum()), int(mw.sum())
            if tt == 100:
                a["wrist100_col"] = float(np.nonzero(mw)[1].mean()) if mw.sum() else np.nan
                a["wrist100_edge"] = bool(mw.sum() and (mw[0].any() or mw[-1].any() or mw[:, 0].any() or mw[:, -1].any()))
        rows[9100 + t - 1] = a
    return rows


def main():
    res, exp, out = {}, {}, {}
    for k, run in CORNERS.items():
        res[k] = load(run)
        p = os.path.join(ROOT, "eval_results", EXPERT[k])
        exp[k] = {r["seed"]: r["placed"] for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    print("corner  placed (Wilson)        expert   on expert-success seeds | miss median [IQR] mm | base blind t50 (both cams 0) | base px t50/t100 med | wrist t100 col med, edge")
    for k, rows in res.items():
        ss = sorted(rows)
        ok = sum(rows[s]["ok"] for s in ss)
        lo, hi = wilson(ok, len(ss))
        e = exp[k]
        es = [s for s in ss if e.get(s)]
        miss = np.array([rows[s]["miss_mm"] for s in ss])
        blind = sum(rows[s]["base50"] == 0 and rows[s]["wrist50"] == 0 for s in ss)
        col = np.array([rows[s]["wrist100_col"] for s in ss])
        edge = sum(rows[s]["wrist100_edge"] for s in ss)
        cls = {}
        for s in ss:
            cls[rows[s]["cls"]] = cls.get(rows[s]["cls"], 0) + 1
        out[k] = {"placed": ok, "n": len(ss), "wilson": (lo, hi), "expert": (sum(e.values()), len(e)),
                  "placed_on_expert_success": (sum(rows[s]["ok"] for s in es), len(es)),
                  "miss_median": float(np.nanmedian(miss)), "miss_iqr": [float(v) for v in np.nanpercentile(miss, [25, 75])],
                  "blind50": blind, "base50_med": float(np.median([rows[s]["base50"] for s in ss])),
                  "base100_med": float(np.median([rows[s]["base100"] for s in ss])),
                  "wrist100_col_med": float(np.nanmedian(col)), "wrist100_edge": edge, "classes": cls,
                  "miss_placed": float(np.nanmedian([rows[s]["miss_mm"] for s in ss if rows[s]["ok"]] or [np.nan])),
                  "miss_failed": float(np.nanmedian([rows[s]["miss_mm"] for s in ss if not rows[s]["ok"]] or [np.nan])),
                  "per_seed": {s: {kk: rows[s][kk] for kk in ("x", "y", "ok", "cls", "miss_mm", "base0", "base50", "base100", "wrist100_col", "wrist100_edge")} | {"expert": e.get(s)} for s in ss}}
        o = out[k]
        print(f"{k:5s}  {ok:2d}/{len(ss)} [{lo:.2f},{hi:.2f}]   {o['expert'][0]:2d}/{o['expert'][1]:2d}    {o['placed_on_expert_success'][0]:2d}/{o['placed_on_expert_success'][1]:2d}"
              f"                 | {o['miss_median']:6.0f} [{o['miss_iqr'][0]:.0f},{o['miss_iqr'][1]:.0f}]      | {blind:2d}/{len(ss)}"
              f"                         | {o['base50_med']:4.0f} / {o['base100_med']:4.0f}          | {o['wrist100_col_med']:4.0f}, {edge}/{len(ss)}")
        print(f"       classes {cls}; miss median placed {o['miss_placed']:.0f} mm, failed {o['miss_failed']:.0f} mm; base px at t0 median {np.median([rows[s]['base0'] for s in ss]):.0f} (converged frame)")
    # primary: unpaired bootstrap of the difference of medians |miss| (+Y minus -Y); negative miss = toward the centre line
    rng = np.random.default_rng(0)
    a = np.array([res["+Y"][s]["miss_mm"] for s in res["+Y"]]); b = np.array([res["-Y"][s]["miss_mm"] for s in res["-Y"]])
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    bs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(10000)]
    d = float(np.median(a) - np.median(b)); lo, hi = np.percentile(bs, [2.5, 97.5])
    out["miss_diff_posY_minus_negY"] = (d, float(lo), float(hi))
    print(f"\nprimary: median miss +Y minus -Y = {d:+.0f} mm, 95 % [{lo:+.0f}, {hi:+.0f}] (unpaired bootstrap; negative = +Y misses more toward the centre)")
    print("\nper seed: seed x y | +Y: ok cls miss base50 base100 wristcol expert || -Y: same")
    for s in sorted(res["+Y"]):
        cells = []
        for k in ("+Y", "-Y"):
            r = res[k][s]
            cells.append(f"{r['x']:.3f} {r['y']:+.3f} {'ok' if r['ok'] else '--'} {r['cls'][:6]:6s} {r['miss_mm']:5.0f} {r['base50']:4d} {r['base100']:4d} {r['wrist100_col']:4.0f} {exp[k].get(s)}")
        print(f"  {s}  " + " || ".join(cells))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
