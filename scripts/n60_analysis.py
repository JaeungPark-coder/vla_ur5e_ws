"""n=60 wide-box analysis (GPU-free): v2 3k / v3 3k / v3 6k + expert on seeds 8000-8059, X 0.38-0.48 x Y +-0.16.

Pre-registered measures (README "2026-11-14: n=60 wide box"): placed per model with Wilson 95 % intervals; paired exact McNemar
v3 3k vs v2 3k and v3 6k vs v3 3k; by region (edge X>=0.42 & |Y|>0.10, X<0.42, centre X>=0.42 & |Y|<=0.10); on expert-success
seeds; tracking slope at the first close (OLS of grip Y on cube spawn Y, paired bootstrap of the difference); outcome classes.
    python3 scripts/n60_analysis.py [--json eval_results/n60_analysis.json]
"""
import csv
import json
import os
import sys
from math import comb, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from failure_overlap import analyse, classify  # noqa: E402

MODELS = {"v2 3k": "n60_v2_3k", "v3 3k": "n60_v3_3k", "v3 6k": "n60_v3_6k"}
EXPERT_FILES = ["expert_baseline_n60_a.jsonl", "expert_baseline_n60_a2.jsonl", "expert_baseline_n60_a3.jsonl", "expert_baseline_n60_b.jsonl"]
REGIONS = {
    "edge X>=0.42 & |Y|>0.10": lambda x, y: x >= 0.42 and abs(y) > 0.10,
    "centre X>=0.42 & |Y|<=0.10": lambda x, y: x >= 0.42 and abs(y) <= 0.10,
    "X<0.42": lambda x, y: x < 0.42,
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(a, b):
    """exact two-sided McNemar on paired booleans: returns (a only, b only, p)."""
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def load_model(run):
    d = os.path.join(ROOT, "eval_runs", run)
    out = {}
    for r in csv.DictReader(open(os.path.join(d, "results.csv"))):
        t = int(r["trial"])
        a = analyse(os.path.join(d, "trace"), t)
        a["ok"] = r["success"] == "True"
        a["cls"] = classify(a, a["ok"])
        z = np.load(os.path.join(d, "trace", f"trial_{t:03d}.npz"))
        grip, g, cube = z["grip_point"].astype(float), z["gripper"].astype(float), z["cube"].astype(float)
        c = np.nonzero(g > 0.5)[0]
        a["close_y"] = float(grip[c[0], 1]) if len(c) and np.isfinite(grip[c[0]]).all() else np.nan
        a["x"], a["y"] = float(cube[0, 0]), float(cube[0, 1])
        out[8000 + t - 1] = a
    return out


def slope(rows):
    y = np.array([r["y"] for r in rows]); h = np.array([r["close_y"] for r in rows])
    ok = np.isfinite(h)
    if ok.sum() < 5:
        return np.nan
    return float(np.polyfit(y[ok], h[ok], 1)[0])


def main():
    res = {m: load_model(r) for m, r in MODELS.items()}
    seeds = sorted(set.intersection(*(set(v) for v in res.values())))
    # the cube spawn is a function of the seed: check the three models agree
    for s in seeds:
        xy = np.array([[res[m][s]["x"], res[m][s]["y"]] for m in res])
        assert np.ptp(xy, axis=0).max() < 2e-3, (s, xy)
    expert = {}
    for f in EXPERT_FILES:
        p = os.path.join(ROOT, "eval_results", f)
        if os.path.exists(p):
            for r in map(json.loads, open(p)):
                expert[r["seed"]] = r["placed"]
    out = {"seeds": len(seeds), "expert_seeds": len(expert)}
    print(f"{len(seeds)} paired seeds; expert outcome on {len(expert)} of them ({sum(expert.values())} placed)")

    def table(title, sel):
        ss = [s for s in seeds if sel(s)]
        line = {}
        cells = []
        for m in res:
            k = sum(res[m][s]["ok"] for s in ss)
            lo, hi = wilson(k, len(ss))
            line[m] = (k, len(ss))
            cells.append(f"{m} {k}/{len(ss)} [{lo:.2f},{hi:.2f}]")
        ex = [expert[s] for s in ss if s in expert]
        line["expert"] = (sum(ex), len(ex))
        cells.append(f"expert {sum(ex)}/{len(ex)}")
        print(f"  {title:32s} " + " | ".join(cells))
        return line

    print("\nplaced (Wilson 95 %)")
    out["placed"] = {"all": table("all", lambda s: True)}
    for name, f in REGIONS.items():
        out["placed"][name] = table(name, lambda s, f=f: f(res["v2 3k"][s]["x"], res["v2 3k"][s]["y"]))
    out["placed"]["expert-success seeds"] = table("expert-success seeds", lambda s: expert.get(s) is True)

    print("\npaired exact McNemar (first only / second only / p)")
    out["mcnemar"] = {}
    for a, b in (("v3 3k", "v2 3k"), ("v3 6k", "v3 3k"), ("v3 6k", "v2 3k")):
        for rname, f in [("all", lambda x, y: True)] + list(REGIONS.items()):
            ss = [s for s in seeds if f(res["v2 3k"][s]["x"], res["v2 3k"][s]["y"])]
            n10, n01, p = mcnemar([res[a][s]["ok"] for s in ss], [res[b][s]["ok"] for s in ss])
            out["mcnemar"][f"{a} vs {b} | {rname}"] = (n10, n01, p)
            print(f"  {a} vs {b:6s} {rname:28s} {n10:2d} / {n01:2d}  p={p:.4f}")

    print("\ntracking slope at the first close (grip Y on spawn Y); paired bootstrap 95 % of the difference, 4000 resamples")
    rng = np.random.default_rng(0)
    out["slope"] = {}
    for rname, f in [("all", lambda x, y: True), ("|Y|>0.10", lambda x, y: abs(y) > 0.10), ("|Y|<=0.10", lambda x, y: abs(y) <= 0.10)]:
        ss = np.array([s for s in seeds if f(res["v2 3k"][s]["x"], res["v2 3k"][s]["y"])])
        b = {m: slope([res[m][s] for s in ss]) for m in res}
        out["slope"][rname] = b
        diffs = {}
        for a1, a2 in (("v3 3k", "v2 3k"), ("v3 6k", "v3 3k")):
            bs = []
            for _ in range(4000):
                idx = rng.choice(ss, len(ss))
                bs.append(slope([res[a1][s] for s in idx]) - slope([res[a2][s] for s in idx]))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            diffs[f"{a1}-{a2}"] = (b[a1] - b[a2], float(lo), float(hi))
        out["slope"][rname + " diff"] = diffs
        print(f"  {rname:10s} n={len(ss):2d} " + "  ".join(f"{m} {v:.2f}" for m, v in b.items()) + "   " +
              "  ".join(f"{k} {d:+.2f} [{lo:+.2f},{hi:+.2f}]" for k, (d, lo, hi) in diffs.items()))

    print("\nlateral miss at the close of FAILED attempts (grip Y - spawn Y, mm, + = away from the centre line), median")
    for m in res:
        e = [1000 * (res[m][s]["close_y"] - res[m][s]["y"]) * np.sign(res[m][s]["y"]) for s in seeds
             if not res[m][s]["ok"] and np.isfinite(res[m][s]["close_y"]) and abs(res[m][s]["y"]) > 0.02]
        print(f"  {m}: n={len(e)} median {np.median(e) if e else np.nan:.0f} mm")

    print("\noutcome classes")
    out["classes"] = {}
    for m in res:
        cnt = {}
        for s in seeds:
            cnt[res[m][s]["cls"]] = cnt.get(res[m][s]["cls"], 0) + 1
        out["classes"][m] = cnt
        print(f"  {m}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))

    print("\nper seed (x, y, v2 3k / v3 3k / v3 6k / expert; . placed, letter = class of the failure)")
    short = {"placed": ".", "blow-up": "B", "runaway": "R", "lifted": "L", "untouched": "U", "pushed": "P"}
    out["per_seed"] = {}
    for s in seeds:
        x, y = res["v2 3k"][s]["x"], res["v2 3k"][s]["y"]
        e = expert.get(s)
        tag = " ".join(short[res[m][s]["cls"]] for m in res) + " " + ("." if e else ("x" if e is False else "?"))
        out["per_seed"][s] = {"x": x, "y": y, **{m: res[m][s]["cls"] for m in res}, "expert": e}
        print(f"  {s} {x:.3f} {y:+.3f}  {tag}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
