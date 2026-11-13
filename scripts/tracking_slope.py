#!/usr/bin/env python3
"""Tracking slope: how far does the hand follow the cube sideways, tick by tick? (GPU-free, saved traces + the training dataset.)

Why (2026-11-14): failures at the edge close the hand ~80 mm toward the centre line. If the policy follows the cube only partially
(hand Y = a + b * cube Y with b < 1 at the close), the miss grows with |Y| and the 8-15 mm error inside the training box and the edge
failures are the same effect at two sizes. b is fitted by OLS of grip-point Y on cube spawn Y over the attempts of one model in one
region, at decision ticks 50, 100, ... and at the first close (gripper > 0.5).

Expert: the dataset has no cube position, so the expert's grip-point Y at its first close stands in for cube Y (the expert closes within
a few mm of the cube; its slope at the close is ~1 by construction, and the ticks before show how its approach develops).

    python3 scripts/tracking_slope.py [--png out.png] [--json out.json] [--extra name=run_dir ...]
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from failure_overlap import RUNS  # noqa: E402
from ur5e_fk import apply  # noqa: E402

E = os.path.join(ROOT, "eval_runs")
TICKS = [50, 100, 150, 200, 250, 300, 350, 400]
LEROBOT = os.path.expanduser("~/.cache/huggingface/lerobot/Jaeung12")
# FK placement of the bridge's grip point (fitted with scripts/ur5e_fk.fit on measured data, residual 0.000 mm): yaw pi, no translation, tool 0.12 m along flange y
FK_P = np.array([-np.pi, 0, 0, 0, 0, 0.12, 0])


def policy_runs(extra):
    """(model, region, results csv, keep_k, trace dir) for every saved evaluation with traces."""
    out = []
    for model, boxes in RUNS.items():
        for box, (path, keep_k, tdir) in boxes.items():
            out.append((model, box, path, keep_k, tdir))
    out += [("v2_s42", "strips", f"{E}/strip_{b}/results.csv", None, f"{E}/strip_{b}/trace") for b in ("posY", "negY")]
    for name, d in extra:
        for sub, region in (("train_box", "train"), ("step2", "step2"), ("strip_posY", "strips"), ("strip_negY", "strips")):
            p = f"{d}_{sub}/results.csv"
            if os.path.exists(p):
                out.append((name, region, p, None, f"{d}_{sub}/trace"))
    return out


def attempt(tdir, t):
    z = np.load(os.path.join(tdir, f"trial_{t:03d}.npz"))
    cube, grip, g = z["cube"].astype(float), z["grip_point"].astype(float), z["gripper"].astype(float)
    row = {"spawn_y": cube[0, 1], "spawn_x": cube[0, 0]}
    for k in TICKS:
        row[k] = grip[k, 1] if k < len(grip) and np.isfinite(grip[k]).all() else np.nan
    c = np.nonzero(g > 0.5)[0]
    row["close"] = grip[c[0], 1] if len(c) and np.isfinite(grip[c[0]]).all() else np.nan
    row["close_tick"] = int(c[0]) if len(c) else -1
    row["cube_close"] = cube[c[0], 1] if len(c) and np.isfinite(cube[c[0]]).all() else np.nan
    return row


def expert_rows(version):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(f"{LEROBOT}/ur5e_pick_place_{version}/data/*/*.parquet")):
        d = pd.read_parquet(f, columns=["joints", "gripper", "episode_index"])
        for _, dd in d.groupby("episode_index"):
            q = np.stack(dd["joints"].values)[:, :6].astype(float)
            g = np.concatenate([np.atleast_1d(x) for x in dd["gripper"].values]).astype(float)
            gp = apply(FK_P, q)
            c = np.nonzero(g > 0.5)[0]
            if not len(c):
                continue
            row = {"spawn_y": gp[c[0], 1], "spawn_x": gp[c[0], 0], "close": gp[c[0], 1], "close_tick": int(c[0])}
            for k in TICKS:
                row[k] = gp[k, 1] if k < len(gp) else np.nan
            rows.append(row)
    return rows


def slope(rows, key):
    y = np.array([r["spawn_y"] for r in rows]); h = np.array([r[key] for r in rows], dtype=float)
    ok = np.isfinite(h)
    if ok.sum() < 5 or np.ptp(y[ok]) < 0.02:
        return np.nan, np.nan, int(ok.sum())
    b, a = np.polyfit(y[ok], h[ok], 1)
    return float(b), float(a), int(ok.sum())


def main():
    extra = [a.split("=", 1) for i, a in enumerate(sys.argv) if i and sys.argv[i - 1] == "--extra"]
    groups = {}
    for model, region, path, keep_k, tdir in policy_runs(extra):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path)):
            if keep_k and int(r["execute_horizon"]) != keep_k:
                continue
            a = attempt(tdir, int(r["trial"]))
            a["ok"] = r["success"] == "True"
            groups.setdefault(model, []).append(dict(a, region=region))
    exp = {"expert_v2": expert_rows("v2"), "expert_v3": expert_rows("v3")}
    regions = {"train box |Y|<=0.10": lambda r: abs(r["spawn_y"]) <= 0.10,
               "edge |Y|>0.10": lambda r: abs(r["spawn_y"]) > 0.10,
               "all": lambda r: True}
    keys = TICKS + ["close"]
    res = {}
    print("OLS slope b of grip-point Y on cube Y (hand follows the cube fully: b = 1).  n = attempts with a value at that tick")
    print("model      region               " + " ".join(f"{('t' + str(k)) if k != 'close' else 'close':>6}" for k in keys) + "    n(close)  median close tick")
    for name, rows in list(exp.items()) + list(groups.items()):
        for rl, f in regions.items():
            rs = [r for r in rows if f(r)]
            if len(rs) < 5:
                continue
            bs = [slope(rs, k) for k in keys]
            res.setdefault(name, {})[rl] = {str(k): b[0] for k, b in zip(keys, bs)}
            ct = [r["close_tick"] for r in rs if r["close_tick"] >= 0]
            print(f"{name:10s} {rl:20s} " + " ".join(f"{b[0]:6.2f}" for b in bs) + f"    {bs[-1][2]:4d}      {np.median(ct):.0f}")
    # lateral error at the close against cube Y (policy), binned by |Y|: is the train-box error the small end of the edge miss?
    print("\nlateral error at the close (grip Y - cube spawn Y, mm, signed so that + = beyond the cube away from the centre line), by |Y| bin; all policy models pooled")
    pol = [r for rows in groups.values() for r in rows if np.isfinite(r["close"])]
    bins = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
    for lo, hi in zip(bins[:-1], bins[1:]):
        rs = [r for r in pol if lo <= abs(r["spawn_y"]) < hi]
        if not rs:
            continue
        e = np.array([1000 * (r["close"] - r["spawn_y"]) * np.sign(r["spawn_y"]) for r in rs])
        ok = np.array([r["ok"] for r in rs])
        m = np.array([1000 * (r["cube_close"] - r["spawn_y"]) * np.sign(r["spawn_y"]) for r in rs])
        print(f"  |Y| {lo:.2f}-{hi:.2f}: n={len(rs):3d}  all median {np.median(e):+6.1f}  placed median {np.median(e[ok]) if ok.any() else np.nan:+6.1f} (n={ok.sum()})"
              f"  failed median {np.median(e[~ok]) if (~ok).any() else np.nan:+6.1f} (n={(~ok).sum()})   cube moved before the close: median {np.nanmedian(m):+5.1f}")
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    if "--png" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        xs = list(range(len(keys)))
        for name in ["expert_v2", "v2_s42", "v2_s1", "v3"] + [n for n, _ in extra]:
            if name in res and "all" in res[name]:
                ax[0].plot(xs, [res[name]["all"][str(k)] for k in keys], marker="o", label=name)
        ax[0].set_xticks(xs, [str(k) for k in TICKS] + ["close"]); ax[0].axhline(1, color="grey", lw=0.8, ls="--")
        ax[0].set_xlabel("tick"); ax[0].set_ylabel("slope of hand Y on cube Y"); ax[0].set_title("Tracking slope, all regions"); ax[0].legend()
        for name, rows in groups.items():
            rs = [r for r in rows if np.isfinite(r["close"])]
            ax[1].scatter([r["spawn_y"] for r in rs], [r["close"] for r in rs], s=10, label=name, alpha=0.6)
        ax[1].plot([-0.17, 0.17], [-0.17, 0.17], color="grey", lw=0.8, ls="--", label="hand on cube")
        ax[1].axvspan(-0.10, 0.10, color="0.92", zorder=0)
        ax[1].set_xlabel("cube spawn Y (m)"); ax[1].set_ylabel("grip-point Y at first close (m)"); ax[1].set_title("Where the hand closes (grey band: v2 training box)")
        ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(sys.argv[sys.argv.index("--png") + 1], dpi=110)


if __name__ == "__main__":
    main()
