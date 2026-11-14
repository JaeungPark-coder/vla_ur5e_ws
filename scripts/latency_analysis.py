#!/usr/bin/env python3
"""Inference-latency evaluation (GPU-free analysis). Pre-registered measures: README "inference latency in simulation".
v3 6k, K=25, 3 regions x 15 seeds (9200-9214); conditions d0 / async10 / sync10 / async20 (+ the stratified K=50 d0 run as a reference).
  primary  - lateral miss at the first close (paired vs d0: median of differences, bootstrap) and the tracking slope (paired bootstrap)
  secondary- placed (McNemar vs d0), blow-ups, classes; chunk-boundary jump of the joint target (max over 6 joints, rad; from
             <run>/results_boundaries.csv, first chunk of a trial excluded) against the per-tick target step inside chunks (trace,
             hold ticks excluded); episode length in ticks (policy rows + holds) for placed trials.
    python3 scripts/latency_analysis.py [--json eval_results/latency_analysis.json]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from n60_analysis import mcnemar, wilson  # noqa: E402
from strat_analysis import boot_median_diff, load, slope  # noqa: E402

REGIONS = ["centre", "posY_strip", "negY_strip"]
CONDS = ["d0", "async10", "sync10", "async20"]
E = os.path.join(ROOT, "eval_runs")


def run_dir(cond, g):
    return f"strat_v3_6k_{g}" if cond == "K50" else f"lat_{cond}_{g}"


def extras(cond, g):
    """boundary jumps (excluding each trial's first chunk), within-chunk per-tick steps, ticks per trial."""
    d = os.path.join(E, run_dir(cond, g))
    out = {"jump": [], "within": [], "ticks": {}}
    p = os.path.join(d, "results_boundaries.csv")
    if os.path.exists(p):
        seen = set()
        for r in csv.DictReader(open(p)):
            t = int(r["trial"])
            if t in seen:
                out["jump"].append(float(r["jump_rad"]))
            seen.add(t)
    p = os.path.join(d, "results.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            t = int(r["trial"])
            out["ticks"][9200 + t - 1] = int(r["steps"]) + int(r.get("hold_ticks") or 0)
            z = np.load(os.path.join(d, "trace", f"trial_{t:03d}.npz"))
            if "target" in z:
                st = np.abs(np.diff(z["target"][:, :6], axis=0)).max(1)
                out["within"] += list(st[st > 0])     # holds repeat the target: step 0, dropped
    if cond == "K50" and os.path.exists(os.path.join(d, "trace")):   # K=50 d0: boundaries every 50 ticks in the trace
        out["jump"] = []
        for f in sorted(os.listdir(os.path.join(d, "trace"))):
            if f.endswith(".npz") and "images" not in f:
                st = np.abs(np.diff(np.load(os.path.join(d, "trace", f))["target"][:, :6], axis=0)).max(1)
                out["jump"] += list(st[49::50][1:])
    return out


def main():
    rng = np.random.default_rng(0)
    res = {c: {g: load(run_dir(c, g), 9200) for g in REGIONS} for c in ["K50"] + CONDS}
    ext = {c: {g: extras(c, g) for g in REGIONS} for c in ["K50"] + CONDS}
    out = {"per_region": {}, "pooled": {}}
    print("condition  region       placed  miss med [IQR] mm   blow-up  classes")
    for c in ["K50"] + CONDS:
        for g in REGIONS:
            rows = res[c][g]
            if not rows:
                continue
            ss = sorted(rows)
            k = sum(rows[s]["ok"] for s in ss)
            miss = np.array([rows[s]["miss_mm"] for s in ss])
            cls = {}
            for s in ss:
                cls[rows[s]["cls"]] = cls.get(rows[s]["cls"], 0) + 1
            out["per_region"][f"{c}|{g}"] = {"placed": k, "n": len(ss), "miss_median": float(np.nanmedian(miss)), "classes": cls}
            print(f"{c:9s}  {g:11s}  {k:2d}/{len(ss)}  {np.nanmedian(miss):5.0f} [{np.nanpercentile(miss, 25):.0f},{np.nanpercentile(miss, 75):.0f}]"
                  f"        {cls.get('blow-up', 0)}     {cls}")
    print("\npooled over 3 regions: placed (Wilson) | slope | vs d0 paired: miss diff [95%] mm (+ = closes further out), McNemar d0-only/cond-only p,"
          " slope diff [95%] | boundary jump med/p95/max rad | within-chunk step med/p95 | ticks of placed trials med")
    base = [(g, s) for g in REGIONS for s in res["d0"][g]]
    for c in ["K50"] + CONDS:
        keys = [(g, s) for g in REGIONS for s in res[c][g]]
        if not keys:
            continue
        rows = [res[c][g][s] for g, s in keys]
        k = sum(r["ok"] for r in rows)
        lo, hi = wilson(k, len(rows))
        sl = slope(rows)
        o = {"placed": k, "n": len(rows), "wilson": (lo, hi), "slope": sl}
        line = f"{c:9s} {k:2d}/{len(rows)} [{lo:.2f},{hi:.2f}] | {sl:.2f}"
        kk = [(g, s) for g, s in base if s in res[c][g]]
        if c != "d0" and len(kk) >= 10:
            d, dlo, dhi = boot_median_diff([res[c][g][s]["miss_mm"] for g, s in kk], [res["d0"][g][s]["miss_mm"] for g, s in kk], rng)
            n10, n01, p = mcnemar([res["d0"][g][s]["ok"] for g, s in kk], [res[c][g][s]["ok"] for g, s in kk])
            bs = []
            for _ in range(4000):
                idx = [kk[i] for i in rng.integers(0, len(kk), len(kk))]
                bs.append(slope([res[c][g][s] for g, s in idx]) - slope([res["d0"][g][s] for g, s in idx]))
            sd = slope([res[c][g][s] for g, s in kk]) - slope([res["d0"][g][s] for g, s in kk])
            slo, shi = np.nanpercentile(bs, [2.5, 97.5])
            o.update(miss_diff=(d, dlo, dhi), mcnemar=(n10, n01, p), slope_diff=(sd, float(slo), float(shi)))
            line += f" | {d:+5.1f} [{dlo:+.1f},{dhi:+.1f}]  {n10}/{n01} p={p:.3f}  {sd:+.2f} [{slo:+.2f},{shi:+.2f}]"
        else:
            line += " | (reference)" + " " * 40
        jump = np.concatenate([np.asarray(ext[c][g]["jump"], float) for g in REGIONS])
        jump = jump[np.isfinite(jump)]
        within = np.concatenate([np.asarray(ext[c][g]["within"], float) for g in REGIONS])
        ticks = [ext[c][g]["ticks"][s] for g, s in keys if res[c][g][s]["ok"] and s in ext[c][g]["ticks"]]
        if len(jump):
            o.update(jump=(float(np.median(jump)), float(np.percentile(jump, 95)), float(jump.max()), len(jump)))
            line += f" | {np.median(jump):.4f}/{np.percentile(jump, 95):.4f}/{jump.max():.4f}"
        if len(within):
            o.update(within=(float(np.median(within)), float(np.percentile(within, 95))))
            line += f" | {np.median(within):.4f}/{np.percentile(within, 95):.4f}"
        if ticks:
            o["ticks_placed_median"] = float(np.median(ticks))
            line += f" | {np.median(ticks):.0f}"
        out["pooled"][c] = o
        print(line)
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
