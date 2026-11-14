#!/usr/bin/env python3
"""Visual-variation evaluation (GPU-free analysis). Pre-registered measures: README "visual variation".
v3 6k, K=25, d=0, centre seeds 9200-9214; one visual factor per run, paired with the unperturbed rerun `none` on the same seeds
(and with eval_runs/lat_d0_centre, the earlier identical run, as a run-to-run noise reference).
  primary  - paired |lateral error| at the first close (|grip Y - spawn Y|, mm): median of (condition - none), bootstrap 95 %;
             signed world-Y error (grip Y - spawn Y) median, for the camera shifts
  secondary- placed (McNemar vs none), classes
  rule     - a factor is WEAK if placed <= 11/15, or the paired |error| increase is > 10 mm with the CI above 0
    python3 scripts/visual_analysis.py [--json eval_results/visual_analysis.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n60_analysis import mcnemar  # noqa: E402
from strat_analysis import boot_median_diff, load  # noqa: E402

CONDS = ["none", "light05", "light2", "tint", "ground_grey", "ground_wood", "cube_dark", "cube_orange", "basecam3", "basecam6",
         "wrist3", "wrist8"]


def failure_mode(cond, seed, row):
    """Failed trials only (added before the run, reviewer request): 'no-descent' = before the first close (gripper > 0.5; the
    whole trial if it never closed) the grip point never came below 60 mm (cube top 40 mm; the smoke trial hovered at ~150 mm,
    then closed in the air at ~100 mm and went to the target empty -- its lowest point was AT the target, hence 'before the
    first close'); 'lateral-miss' = it descended but closed > 15 mm sideways from the cube or never closed; 'later' = descended
    and closed within 15 mm, failed afterwards (slip, drop, place error)."""
    run = "lat_d0_centre" if cond.startswith("lat_d0") else f"vis_{cond}"
    z = np.load(os.path.join(os.path.dirname(HERE), "eval_runs", run, "trace", f"trial_{seed - 9200 + 1:03d}.npz"))
    grip = z["grip_point"].astype(float)
    c = np.nonzero(z["gripper"].astype(float) > 0.5)[0]
    upto = c[0] + 1 if len(c) else len(grip)
    if np.nanmin(grip[:upto, 2]) > 0.06:
        return "no-descent"
    err = abs(row["close_y"] - row["y"]) if np.isfinite(row["close_y"]) else np.inf
    return "lateral-miss" if err > 0.015 else "later"


def main():
    rng = np.random.default_rng(0)
    res = {c: load(f"vis_{c}", 9200) for c in CONDS}
    res["lat_d0 (earlier run)"] = load("lat_d0_centre", 9200)
    base = res["none"]
    out = {}
    print("condition              placed  |err| med  signed Y err med | paired |err| diff vs none [95%] | McNemar none-only/cond-only p | classes | verdict")
    for c, rows in res.items():
        if not rows:
            continue
        ss = sorted(rows)
        err = np.array([abs(rows[s]["close_y"] - rows[s]["y"]) * 1000 for s in ss])
        sgn = np.array([(rows[s]["close_y"] - rows[s]["y"]) * 1000 for s in ss])
        k = sum(rows[s]["ok"] for s in ss)
        cls = {}
        for s in ss:
            cls[rows[s]["cls"]] = cls.get(rows[s]["cls"], 0) + 1
        fm = {}
        for s in ss:
            if not rows[s]["ok"]:
                fm[failure_mode(c, s, rows[s])] = fm.get(failure_mode(c, s, rows[s]), 0) + 1
        o = {"placed": k, "n": len(ss), "abs_err_median": float(np.nanmedian(err)), "signed_err_median": float(np.nanmedian(sgn)), "classes": cls,
             "failure_modes": fm}
        cls = {**cls, **{f"fail:{m}": v for m, v in fm.items()}}
        line = f"{c:22s} {k:2d}/{len(ss)}   {np.nanmedian(err):6.1f}    {np.nanmedian(sgn):+7.1f}        "
        kk = [s for s in ss if s in base]
        if c != "none" and base and len(kk) >= 10:
            a = [abs(rows[s]["close_y"] - rows[s]["y"]) * 1000 for s in kk]
            b = [abs(base[s]["close_y"] - base[s]["y"]) * 1000 for s in kk]
            d, lo, hi = boot_median_diff(a, b, rng)
            n10, n01, p = mcnemar([base[s]["ok"] for s in kk], [rows[s]["ok"] for s in kk])
            weak = k <= 11 or (d > 10 and lo > 0)
            o.update(abs_err_diff=(d, lo, hi), mcnemar=(n10, n01, p), weak=bool(weak))
            line += f"| {d:+6.1f} [{lo:+.1f},{hi:+.1f}]            | {n10}/{n01} p={p:.3f}  | {cls} | {'WEAK' if weak else 'ok'}"
        else:
            line += f"| (reference)                         |                | {cls}"
        out[c] = o
        print(line)
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
