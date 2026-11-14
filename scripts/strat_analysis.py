#!/usr/bin/env python3
"""Stratified evaluation under the converged first frame (GPU-free). Pre-registered measures: README "stratified evaluation under the
converged first frame".
  Part A: v3 6k on the 7 n=60 +Y-corner seeds, converged frame, vs the stale-frame n=60 outcome on the same seeds.
  Part B: 6 regions x 15 seeds (9200-9214), v2 3k / v3 3k / v3 6k + expert:
    primary  - lateral miss at the first close (signed, + = away from the centre line), per region; paired v3 3k - v2 3k by bootstrap
               over seeds, per region and pooled over strips + corners; slope of grip Y at the close on spawn Y over all 90 (paired bootstrap)
    secondary- placed (Wilson), paired exact McNemar per region and overall, expert-success seeds, classes; views at t50/t100.
    python3 scripts/strat_analysis.py [--json eval_results/strat_analysis.json]
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
from n60_analysis import mcnemar, wilson  # noqa: E402

E = os.path.join(ROOT, "eval_runs")
MODELS = ["v2_3k", "v3_3k", "v3_6k"]
REGIONS = ["centre", "posY_strip", "negY_strip", "posY_corner", "negY_corner", "mid_lowX"]
EDGE = ["posY_strip", "negY_strip", "posY_corner", "negY_corner"]
A_SEEDS = [8007, 8012, 8015, 8019, 8022, 8033, 8036]


def mask(img):
    im = img[..., :3].astype(int)
    return (im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)


def load(run, seed0):
    d = os.path.join(E, run)
    p = os.path.join(d, "results.csv")
    if not os.path.exists(p):
        return {}
    rows = {}
    for r in csv.DictReader(open(p)):
        t = int(r["trial"])
        a = analyse(os.path.join(d, "trace"), t)
        a["ok"] = r["success"] == "True"
        a["cls"] = classify(a, a["ok"])
        z = np.load(os.path.join(d, "trace", f"trial_{t:03d}.npz"))
        cube, grip, g = z["cube"].astype(float), z["grip_point"].astype(float), z["gripper"].astype(float)
        c = np.nonzero(g > 0.5)[0]
        a["x"], a["y"] = float(cube[0, 0]), float(cube[0, 1])
        a["close_y"] = float(grip[c[0], 1]) if len(c) and np.isfinite(grip[c[0]]).all() else np.nan
        a["miss_mm"] = 1000 * (a["close_y"] - a["y"]) * (np.sign(a["y"]) or 1.0)
        ip = os.path.join(d, "trace", f"trial_{t:03d}_images.npz")
        if os.path.exists(ip):
            im = np.load(ip)
            tick = list(im["tick"])
            for tt in (0, 50, 100):
                if tt in tick:
                    mb, mw = mask(im["base"][tick.index(tt)]), mask(im["wrist"][tick.index(tt)])
                    a[f"base{tt}"], a[f"wrist{tt}"] = int(mb.sum()), int(mw.sum())
                    if tt == 100:
                        a["wcol"] = float(np.nonzero(mw)[1].mean()) if mw.sum() else np.nan
        rows[seed0 + t - 1] = a
    return rows


def boot_median_diff(a, b, rng, n=10000):
    """paired: median of (a - b) over seeds, 95 % bootstrap."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    bs = [np.median(rng.choice(d, len(d))) for _ in range(n)]
    return float(np.median(d)), *map(float, np.percentile(bs, [2.5, 97.5]))


def slope(rows):
    y = np.array([r["y"] for r in rows]); h = np.array([r["close_y"] for r in rows])
    ok = np.isfinite(h)
    return float(np.polyfit(y[ok], h[ok], 1)[0]) if ok.sum() >= 5 else np.nan


def main():
    out = {}
    rng = np.random.default_rng(0)
    # ---- Part A
    n60 = json.load(open(os.path.join(ROOT, "eval_results", "n60_analysis.json")))["per_seed"]
    a_rows = []
    for s in A_SEEDS:
        r = load(f"ffA_v3_6k_seed{s}", s).get(s)
        if r:
            a_rows.append((s, r["ok"], r["cls"], r["miss_mm"], n60[str(s)]["v3 6k"]))
    out["A"] = a_rows
    print(f"Part A (v3 6k, n=60 +Y-corner seeds): converged {sum(r[1] for r in a_rows)}/{len(a_rows)} vs stale "
          f"{sum(r[4] == 'placed' for r in a_rows)}/{len(a_rows)}")
    for s, ok, cls, miss, stale in a_rows:
        print(f"   {s}: converged {cls:9s} miss {miss:5.0f} mm | stale {stale}")
    # ---- Part B
    res = {m: {g: load(f"strat_{m}_{g}", 9200) for g in REGIONS} for m in MODELS}
    exp = {}
    for g in REGIONS:
        p = os.path.join(ROOT, "eval_results", f"expert_strat_{g}.jsonl")
        exp[g] = {r["seed"]: r["placed"] for r in map(json.loads, open(p))} if os.path.exists(p) else {}
    out["B"] = {}
    print("\nPart B: placed | miss median [IQR] mm | blind t50 | wrist col t100   per model;  expert placed")
    for g in REGIONS:
        line = [f"{g:12s}"]
        out["B"][g] = {}
        for m in MODELS:
            rows = res[m][g]
            if not rows:
                line.append(f"{m}: -"); continue
            ss = sorted(rows)
            k = sum(rows[s]["ok"] for s in ss)
            miss = np.array([rows[s]["miss_mm"] for s in ss])
            blind = sum(rows[s].get("base50", 1) == 0 and rows[s].get("wrist50", 1) == 0 for s in ss)
            wcol = np.nanmedian([rows[s].get("wcol", np.nan) for s in ss])
            cls = {}
            for s in ss:
                cls[rows[s]["cls"]] = cls.get(rows[s]["cls"], 0) + 1
            out["B"][g][m] = {"placed": k, "n": len(ss), "wilson": wilson(k, len(ss)), "miss_median": float(np.nanmedian(miss)),
                              "miss_iqr": [float(v) for v in np.nanpercentile(miss, [25, 75])], "blind50": blind, "wcol100": float(wcol), "classes": cls}
            line.append(f"{m} {k:2d}/{len(ss)} {np.nanmedian(miss):5.0f} [{np.nanpercentile(miss, 25):.0f},{np.nanpercentile(miss, 75):.0f}] b{blind:2d} c{wcol:3.0f}")
        e = exp[g]
        out["B"][g]["expert"] = (sum(e.values()), len(e))
        line.append(f"expert {sum(e.values())}/{len(e)}")
        print("  " + " | ".join(line))

    def paired(m1, m2, regions):
        ss = [(g, s) for g in regions for s in res[m1][g] if s in res[m2][g]]
        a = [res[m1][g][s]["miss_mm"] for g, s in ss]; b = [res[m2][g][s]["miss_mm"] for g, s in ss]
        oka = [res[m1][g][s]["ok"] for g, s in ss]; okb = [res[m2][g][s]["ok"] for g, s in ss]
        return boot_median_diff(a, b, rng), mcnemar(oka, okb), len(ss)

    print("\npaired comparisons: median of (miss first - miss second) [95 %] (+ = first closes nearer/beyond the cube) | McNemar first only / second only, p")
    out["paired"] = {}
    for m1, m2 in (("v3_3k", "v2_3k"), ("v3_6k", "v3_3k"), ("v3_6k", "v2_3k")):
        for lab, regs in [(g, [g]) for g in REGIONS] + [("strips+corners", EDGE), ("all 90", REGIONS)]:
            (d, lo, hi), (n10, n01, p), n = paired(m1, m2, regs)
            if n == 0:
                continue
            out["paired"][f"{m1} vs {m2} | {lab}"] = {"miss_diff": (d, lo, hi), "mcnemar": (n10, n01, p), "n": n}
            print(f"  {m1} vs {m2} {lab:15s} n={n:2d}  {d:+6.1f} [{lo:+.1f},{hi:+.1f}] mm | {n10:2d} / {n01:2d}  p={p:.4f}")

    print("\ntracking slope at the close over all attempts (paired bootstrap of the difference)")
    out["slope"] = {}
    keys = [(g, s) for g in REGIONS for s in res["v2_3k"][g]]
    for m in MODELS:
        rows = [res[m][g][s] for g, s in keys if s in res[m][g]]
        out["slope"][m] = slope(rows) if rows else np.nan
    for m1, m2 in (("v3_3k", "v2_3k"), ("v3_6k", "v3_3k")):
        kk = [(g, s) for g, s in keys if s in res[m1][g] and s in res[m2][g]]
        if len(kk) < 10:
            continue
        bs = []
        for _ in range(4000):
            idx = [kk[i] for i in rng.integers(0, len(kk), len(kk))]
            bs.append(slope([res[m1][g][s] for g, s in idx]) - slope([res[m2][g][s] for g, s in idx]))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        d = slope([res[m1][g][s] for g, s in kk]) - slope([res[m2][g][s] for g, s in kk])
        out["slope"][f"{m1}-{m2}"] = (d, float(lo), float(hi))
    print("  " + "  ".join(f"{k} {v:.2f}" if not isinstance(v, tuple) else f"{k} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for k, v in out["slope"].items()))

    print("\nexpert-success seeds only: placed per model")
    for g in REGIONS:
        es = [s for s, v in exp[g].items() if v]
        print(f"  {g:12s} n={len(es):2d} " + " ".join(f"{m} {sum(res[m][g][s]['ok'] for s in es if s in res[m][g])}" for m in MODELS if res[m][g]))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
