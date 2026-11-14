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


PHASES = ["approach", "contact", "carry"]
JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]


def phase_at(z, k):
    """approach: before the first close and grip-to-cube > 50 mm; contact: before the first close within 50 mm, or the
    30 ticks after it; carry: later (lift, transport, place, release). First close = first tick with gripper > 0.5."""
    g = z["gripper"].astype(float)
    c = np.nonzero(g > 0.5)[0]
    fc = c[0] if len(c) else len(g)
    if k >= fc + 30:
        return "carry"
    if k >= fc:
        return "contact"
    dist = np.linalg.norm(z["grip_point"][k].astype(float) - z["cube"][k].astype(float))
    return "contact" if dist <= 0.05 else "approach"


def boundaries(cond, g):
    """(jump rad, joint index, phase) for every chunk boundary except each trial's first, from the trace targets at the
    boundary ticks (latency runs: tick = step + hold_ticks from <run>/results_boundaries.csv; K=50 run: every 50 ticks)."""
    d = os.path.join(E, run_dir(cond, g))
    ticks = {}
    p = os.path.join(d, "results_boundaries.csv")
    if cond == "K50":
        for f in sorted(os.listdir(os.path.join(d, "trace"))):
            if f.endswith(".npz") and "images" not in f:
                n = len(np.load(os.path.join(d, "trace", f))["target"])
                ticks[int(f[6:9])] = list(range(100, n, 50))
    elif os.path.exists(p):
        for r in csv.DictReader(open(p)):
            ticks.setdefault(int(r["trial"]), []).append(int(r["step"]) + int(r["hold_ticks"]))
        ticks = {t: v[1:] for t, v in ticks.items()}
    out = []
    for t, ks in ticks.items():
        z = np.load(os.path.join(d, "trace", f"trial_{t:03d}.npz"))
        tg = z["target"][:, :6].astype(float)
        # added after the run (deviation, reported): from the first tick the measured joint state leaves the physical range
        # (|q| > 2 pi, a PhysX blow-up spinning a joint to hundreds of rad) the targets follow the state and their steps are
        # not policy boundaries. --raw keeps them (the pre-registered computation).
        wild = np.nonzero(np.abs(z["joints"][:, :6].astype(float)).max(1) > 2 * np.pi)[0]
        end = wild[0] if (len(wild) and "--raw" not in sys.argv) else len(tg)
        for k in ks:
            if 0 < k < end:
                dl = np.abs(tg[k] - tg[k - 1])
                out.append((float(dl.max()), int(dl.argmax()), phase_at(z, k)))
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
    print("\nboundary jumps by phase: n, median / p95 / max rad;  and which joint carries the large ones (> within-chunk p95 of the run)")
    out["phase"] = {}
    for c in ["K50"] + CONDS:
        bb = [b for g in REGIONS for b in boundaries(c, g)]
        if not bb:
            continue
        thr = out["pooled"].get(c, {}).get("within", (np.nan, np.nan))[1]
        cells, o = [], {}
        for ph in PHASES:
            j = np.array([b[0] for b in bb if b[2] == ph])
            if len(j):
                o[ph] = (len(j), float(np.median(j)), float(np.percentile(j, 95)), float(j.max()))
                cells.append(f"{ph} n={len(j):3d} {np.median(j):.4f}/{np.percentile(j, 95):.4f}/{j.max():.4f}")
        big = [b for b in bb if b[0] > thr]
        o["large_by_joint"] = {JOINTS[i]: sum(b[1] == i for b in big) for i in range(6)}
        o["large_by_phase"] = {ph: sum(b[2] == ph for b in big) for ph in PHASES}
        out["phase"][c] = o
        print(f"  {c:9s} " + " | ".join(cells))
        print(f"            large (> {thr:.4f}): {len(big)}/{len(bb)}  by phase {o['large_by_phase']}  by joint "
              + " ".join(f"{k} {v}" for k, v in o["large_by_joint"].items() if v))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
