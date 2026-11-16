#!/usr/bin/env python3
"""K=25 additional evaluation (README "K=25 additional evaluation", written before the runs), GPU-free.
  E1 v4 3k / E2 v3 3k on the original ground, E3 v4 3k on held-out H0; stratified 6 regions x seeds 9200-9214, K=25.
  primary  |lateral miss| at the first close, paired median difference v4 - v3 (bootstrap 95 %)
  rule     E1 vs E2: v4 placed >= v3 - 4 AND |miss| diff CI upper < +5 mm
           E3: >= 74/90 AND no region < 10/15 AND no-descent <= 3/90
    python3 scripts/k25_analysis.py [--json eval_results/k25_analysis.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n60_analysis import mcnemar, wilson  # noqa: E402
from strat_analysis import REGIONS, boot_median_diff, load  # noqa: E402

E = os.path.join(os.path.dirname(HERE), "eval_runs")
SETS = {"E1 v4 3k orig": "strat_k25_v4_3k", "E2 v3 3k orig": "strat_k25_v3_3k", "E3 v4 3k H0": "strat_h0_k25_v4_3k",
        "(K=50) v4 3k orig": "strat_v4_3k", "(K=50) v3 3k orig": "strat_v3_3k"}


def extra(run, seed, r):
    z = np.load(os.path.join(E, run, "trace", f"trial_{seed - 9200 + 1:03d}.npz"))
    grip = z["grip_point"].astype(float)
    c = np.nonzero(z["gripper"].astype(float) > 0.5)[0]
    upto = c[0] + 1 if len(c) else len(grip)
    r["no_descent"] = bool(np.nanmin(grip[:upto, 2]) > 0.06)
    r["close_z_mm"] = float(grip[c[0], 2]) * 1000 if len(c) else np.nan
    r["abs_miss"] = abs(r["miss_mm"])
    return r


def main():
    rng = np.random.default_rng(0)
    data = {}
    for lab, pre in SETS.items():
        data[lab] = {g: {s: extra(f"{pre}_{g}", s, r) for s, r in load(f"{pre}_{g}", 9200).items()} for g in REGIONS}
    out = {}
    print(f"{'region':12s} " + " ".join(f"{k:>18s}" for k in SETS))
    for g in REGIONS:
        print(f"{g:12s} " + " ".join(f"{sum(r['ok'] for r in data[k][g].values()):>12d}/{len(data[k][g]):2d}   " for k in SETS))
    for lab in SETS:
        rows = [r for g in REGIONS for r in data[lab][g].values()]
        if not rows:
            continue
        cls = {}
        for r in rows:
            cls[r["cls"]] = cls.get(r["cls"], 0) + 1
        k = sum(r["ok"] for r in rows)
        out[lab] = dict(placed=k, n=len(rows), wilson=wilson(k, len(rows)), abs_miss_median=float(np.nanmedian([r["abs_miss"] for r in rows])),
                        no_descent=sum(r["no_descent"] for r in rows), close_z_median=float(np.nanmedian([r["close_z_mm"] for r in rows])),
                        classes=cls, per_region={g: sum(r["ok"] for r in data[lab][g].values()) for g in REGIONS})
        print(f"{lab:20s} {k:2d}/{len(rows)} |miss| med {out[lab]['abs_miss_median']:5.1f} mm  no-descent {out[lab]['no_descent']}  "
              f"close z {out[lab]['close_z_median']:4.0f}  {cls}")

    def paired(a, b):
        kk = [(g, s) for g in REGIONS for s in data[a][g] if s in data[b][g]]
        d = boot_median_diff([data[a][g][s]["abs_miss"] for g, s in kk], [data[b][g][s]["abs_miss"] for g, s in kk], rng)
        mc = mcnemar([data[b][g][s]["ok"] for g, s in kk], [data[a][g][s]["ok"] for g, s in kk])
        return d, mc, len(kk)

    print()
    for a, b in [("E1 v4 3k orig", "E2 v3 3k orig"), ("E1 v4 3k orig", "(K=50) v4 3k orig"), ("E2 v3 3k orig", "(K=50) v3 3k orig"),
                 ("E3 v4 3k H0", "E1 v4 3k orig")]:
        if a in out and b in out:
            d, mc, n = paired(a, b)
            out[f"{a} - {b}"] = dict(abs_miss_diff=d, mcnemar_b_only_a_only_p=mc, n=n)
            print(f"{a} - {b}: |miss| {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] mm; McNemar {b}-only/{a}-only {mc[0]}/{mc[1]} p={mc[2]:.3f} (n={n})")
    if "E1 v4 3k orig" in out and "E2 v3 3k orig" in out:
        d = out["E1 v4 3k orig - E2 v3 3k orig"]["abs_miss_diff"]
        p1 = out["E1 v4 3k orig"]["placed"] >= out["E2 v3 3k orig"]["placed"] - 4 and d[2] < 5
        out["pass_E1"] = bool(p1)
        print(f"-> E1 vs E2 no harm at K=25: {'PASS' if p1 else 'FAIL'}")
    if "E3 v4 3k H0" in out:
        e3 = out["E3 v4 3k H0"]
        p3 = e3["placed"] >= 74 and min(e3["per_region"].values()) >= 10 and e3["no_descent"] <= 3
        out["pass_E3"] = bool(p3)
        print(f"-> E3 table-like ground across the box: {'PASS' if p3 else 'FAIL'} (>= 74, no region < 10, no-descent <= 3)")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
