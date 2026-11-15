#!/usr/bin/env python3
"""v4 (visual randomization) core evaluation, GPU-free. Decision rule and prediction: README "v4 amendment" (written before any
v4 data).
  A. base scene: 6 regions x seeds 9200-9214, K=50, v4 3k paired with v3 3k (and v3 6k for reference):
     placed (>= 74/90), paired signed lateral miss at the first close (+ = away from the centre line), median diff over the 90,
     bootstrap 95 % (upper end < +5 mm), McNemar.
  C. held-out H0 (plain, the lab table colour) on its own: v4 >= 11/15, McNemar vs v3 3k p < 0.05, no-descent <= 2;
     H1-H3 pooled: v4 >= 30/45 and not more than 3 below v3 3k.
  Ground conditions (ground_grey, ground_wood, nogrid): v4 3k paired with the v3 6k runs (prediction 12-15/15 each).
    python3 scripts/v4_analysis.py [--json eval_results/v4_analysis.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n60_analysis import mcnemar, wilson  # noqa: E402
from strat_analysis import REGIONS, boot_median_diff, load  # noqa: E402
from visual_analysis import failure_mode  # noqa: E402


def abs_err(r):
    return abs(r["close_y"] - r["y"]) * 1000


def close_z(cond, seed):
    z = np.load(os.path.join(os.path.dirname(HERE), "eval_runs", f"vis_{cond}", "trace", f"trial_{seed - 9200 + 1:03d}.npz"))
    c = np.nonzero(z["gripper"].astype(float) > 0.5)[0]
    return float(z["grip_point"][c[0], 2]) * 1000 if len(c) else np.nan


def summarize(cond, rows):
    ss = sorted(rows)
    fm = {}
    for s in ss:
        if not rows[s]["ok"]:
            m = failure_mode(cond, s, rows[s])
            fm[m] = fm.get(m, 0) + 1
    cz = [close_z(cond, s) for s in ss]
    return dict(placed=sum(rows[s]["ok"] for s in ss), n=len(ss), abs_err_median=float(np.nanmedian([abs_err(rows[s]) for s in ss])),
                failure_modes=fm, close_height_median_mm=float(np.nanmedian(cz)))


def main():
    rng = np.random.default_rng(0)
    out = {}

    # ---- A. base scene ----
    strat = {m: {g: load(f"strat_{m}_{g}", 9200) for g in REGIONS} for m in ("v4_3k", "v3_3k", "v3_6k")}
    print("A. base scene, stratified 6 x 15, K=50")
    print(f"{'region':12s} " + "  ".join(f"{m:>6s}" for m in strat) + "   v4-v3_3k miss med [95%]")
    for g in REGIONS:
        a, b = strat["v4_3k"][g], strat["v3_3k"][g]
        kk = sorted(set(a) & set(b))
        d = boot_median_diff([a[s]["miss_mm"] for s in kk], [b[s]["miss_mm"] for s in kk], rng)
        print(f"{g:12s} " + "  ".join(f"{sum(r['ok'] for r in strat[m][g].values()):2d}/{len(strat[m][g]):2d}" for m in strat)
              + f"   {d[0]:+5.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
    pa = {m: [(g, s, r) for g in REGIONS for s, r in strat[m][g].items()] for m in strat}
    tot = {m: sum(r["ok"] for _, _, r in pa[m]) for m in strat}
    res_a = {}
    for ref in ("v3_3k", "v3_6k"):
        kk = [(g, s) for g, s, _ in pa["v4_3k"] if s in strat[ref][g]]
        a = [strat["v4_3k"][g][s]["miss_mm"] for g, s in kk]
        b = [strat[ref][g][s]["miss_mm"] for g, s in kk]
        d = boot_median_diff(a, b, rng)
        mc = mcnemar([strat[ref][g][s]["ok"] for g, s in kk], [strat["v4_3k"][g][s]["ok"] for g, s in kk])
        res_a[ref] = dict(miss_diff=d, mcnemar_ref_only_v4_only_p=mc, n=len(kk))
        print(f"  v4 - {ref}: placed {tot['v4_3k']}/{len(pa['v4_3k'])} vs {tot[ref]}/{len(pa[ref])}; paired miss median {d[0]:+.1f} mm "
              f"[{d[1]:+.1f}, {d[2]:+.1f}]; McNemar {ref}-only/v4-only {mc[0]}/{mc[1]} p={mc[2]:.3f}")
    miss_med = {m: float(np.nanmedian([r["miss_mm"] for _, _, r in pa[m]])) for m in strat}
    cls = {m: {} for m in strat}
    for m in strat:
        for _, _, r in pa[m]:
            cls[m][r["cls"]] = cls[m].get(r["cls"], 0) + 1
    print("  miss median:", {m: round(v, 1) for m, v in miss_med.items()}, " classes:", cls)
    pass_a = tot["v4_3k"] >= 74 and res_a["v3_3k"]["miss_diff"][2] < 5
    print(f"  -> base scene no regression: {'PASS' if pass_a else 'FAIL'} (>= 74/90 and CI upper < +5 mm)")
    out["A"] = dict(placed=tot, n=len(pa["v4_3k"]), wilson_v4=wilson(tot["v4_3k"], len(pa["v4_3k"])), paired=res_a,
                    miss_median=miss_med, classes=cls, pass_=bool(pass_a))

    # ---- C. held-out grounds ----
    print("\nC. held-out grounds, centre, K=25")
    out["C"] = {}
    for k in range(4):
        v4, v3 = load(f"vis_v4_3k_heldout{k}", 9200), load(f"vis_v3_3k_heldout{k}", 9200)
        sv4, sv3 = summarize(f"v4_3k_heldout{k}", v4), summarize(f"v3_3k_heldout{k}", v3)
        kk = sorted(set(v4) & set(v3))
        mc = mcnemar([v3[s]["ok"] for s in kk], [v4[s]["ok"] for s in kk])
        d = boot_median_diff([abs_err(v4[s]) for s in kk], [abs_err(v3[s]) for s in kk], rng)
        out["C"][f"H{k}"] = dict(v4=sv4, v3_3k=sv3, mcnemar_v3only_v4only_p=mc, abs_err_diff=d)
        print(f"  H{k}: v4 {sv4['placed']:2d}/15 (|err| {sv4['abs_err_median']:5.1f}, close z {sv4['close_height_median_mm']:5.0f} mm, "
              f"fail {sv4['failure_modes']})  v3 3k {sv3['placed']:2d}/15 (|err| {sv3['abs_err_median']:5.1f}, close z "
              f"{sv3['close_height_median_mm']:5.0f}, fail {sv3['failure_modes']})  McNemar v3-only/v4-only {mc[0]}/{mc[1]} p={mc[2]:.4f}; "
              f"paired |err| v4-v3 {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
    h0 = out["C"]["H0"]
    pass_h0 = h0["v4"]["placed"] >= 11 and h0["mcnemar_v3only_v4only_p"][2] < 0.05 and h0["v4"]["failure_modes"].get("no-descent", 0) <= 2
    p4 = sum(out["C"][f"H{k}"]["v4"]["placed"] for k in (1, 2, 3))
    p3 = sum(out["C"][f"H{k}"]["v3_3k"]["placed"] for k in (1, 2, 3))
    pass_h13 = p4 >= 30 and p4 >= p3 - 3
    print(f"  -> H0 on its own: {'PASS' if pass_h0 else 'FAIL'} (>= 11/15, McNemar p < 0.05, no-descent <= 2)")
    print(f"  -> H1-H3 pooled: v4 {p4}/45 vs v3 3k {p3}/45: {'PASS' if pass_h13 else 'FAIL'} (>= 30 and not > 3 below v3 3k)")
    out["C"].update(pass_h0=bool(pass_h0), h13_pooled=dict(v4=p4, v3_3k=p3), pass_h13=bool(pass_h13))

    # ---- ground conditions ----
    print("\nGround conditions, centre, K=25: v4 3k vs v3 6k (same seeds)")
    out["ground"] = {}
    for c in ("ground_grey", "ground_wood", "nogrid"):
        v4, v36 = load(f"vis_v4_3k_{c}", 9200), load(f"vis_{c}", 9200)
        sv4, sv36 = summarize(f"v4_3k_{c}", v4), summarize(c, v36)
        kk = sorted(set(v4) & set(v36))
        mc = mcnemar([v36[s]["ok"] for s in kk], [v4[s]["ok"] for s in kk])
        out["ground"][c] = dict(v4=sv4, v3_6k=sv36, mcnemar_v36only_v4only_p=mc)
        print(f"  {c:12s} v4 {sv4['placed']:2d}/15 (|err| {sv4['abs_err_median']:5.1f}, close z {sv4['close_height_median_mm']:4.0f}, "
              f"fail {sv4['failure_modes']})  v3 6k {sv36['placed']:2d}/15 (fail {sv36['failure_modes']})  McNemar {mc[0]}/{mc[1]} p={mc[2]:.4f}")
    effect = pass_a and pass_h0 and pass_h13
    print(f"\nEFFECT (all three): {effect}")
    out["effect"] = bool(effect)
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
