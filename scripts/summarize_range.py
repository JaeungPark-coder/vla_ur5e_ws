#!/usr/bin/env python3
"""A policy evaluation on a spawn box, reported the way the range experiments need it: on ALL seeds and on the seeds where the
scripted expert succeeded (so the policy is not blamed for what the expert cannot do), with blow-ups, wrist_3 and placement error.

    scripts/summarize_range.py RESULTS.csv TRACE_DIR EXPERT.jsonl [FIRST_SEED=8000] [K]

RESULTS.csv/TRACE_DIR: one evaluation (eval_checkpoint.sh output: trial k = seed FIRST_SEED+k-1; with the 3-K run pass K to keep
only that horizon: trial i -> seed FIRST_SEED+(i-1)//3). EXPERT.jsonl: isaac/expert_baseline.py output for the same box and seeds.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_eval import wilson  # noqa: E402


def main():
    csv_path, trace_dir, expert_path = sys.argv[1:4]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    keep_k = int(sys.argv[5]) if len(sys.argv) > 5 else None
    expert = {r["seed"]: r for r in map(json.loads, open(expert_path))}
    rows = []
    for r in csv.DictReader(open(csv_path)):
        k = int(r["execute_horizon"]) if r.get("execute_horizon") else None
        if keep_k and k != keep_k:
            continue
        t = int(r["trial"])
        seed = first + ((t - 1) // 3 if keep_k else t - 1)
        z = np.load(os.path.join(trace_dir, f"trial_{t:03d}.npz"))
        meta = json.load(open(os.path.join(trace_dir, f"trial_{t:03d}.json")))
        w = z["joints"][:, 5].astype(float)
        fin = np.isfinite(w) & (np.abs(w) < 1e4)           # up to the tick where the joint values diverge (numerical explosion)
        w = w[:int(np.argmax(~fin))] if (~fin).any() else w
        d30 = float(np.abs(w[30:] - w[:-30]).max()) if len(w) > 30 else 0.0
        unwind = float(np.max(np.maximum.accumulate(w) - w))
        # wrist_3 "runaway": turns by more than 3 rad against its running maximum, or leaves [-1, 5.5] (the demonstrations stay in
        # [-0.05, 4.5] and unwind at most 1.4 rad) -- counted over ALL trials, blow-up flagged or not
        runaway = bool(unwind > 3.0 or w.min() < -1.0 or w.max() > 5.5 or (~fin).any())
        spawn = z["cube"][0, :2]
        cube_end = z["cube"][-1, :2]
        err = 1000 * float(np.linalg.norm(cube_end - np.array([0.45, 0.30])))
        rows.append({"trial": t, "seed": seed, "ok": r["success"] == "True", "blowup": bool(meta["blowup"]) or r["blowup"] == "1",
                     "d30": d30, "unwind": unwind, "runaway": runaway, "err": err, "x": float(spawn[0]), "y": float(spawn[1])})

    def report(label, rs):
        n = len(rs)
        if not n:
            print(f"{label}: no trials")
            return
        ok = sum(r["ok"] for r in rs)
        lo, hi = wilson(ok, n)
        bl = sum(r["blowup"] for r in rs)
        clean = [r for r in rs if not r["runaway"]]
        runaway = sum(r["runaway"] for r in rs)
        placed = np.array([r["err"] for r in rs if r["ok"]])
        d30 = np.array([r["d30"] for r in clean]) if clean else np.zeros(1)
        un = np.array([r["unwind"] for r in clean]) if clean else np.zeros(1)
        print(f"{label:34s} n={n:2d} placed {ok:2d} ({ok / n:.2f}, Wilson {lo:.2f}-{hi:.2f}) blow-ups {bl} | "
              f"placed err median {np.median(placed) if len(placed) else float('nan'):.1f} mm | "
              f"wrist_3 RUNAWAY in {runaway}/{n} | without runaway: 30-tick max {d30.max():.2f}, unwinding >0.5 rad in {int((un > 0.5).sum())}/{len(clean)}")

    report("all seeds", rows)
    good = [r for r in rows if expert.get(r["seed"], {}).get("placed")]
    report("seeds where the expert succeeded", good)
    bad = [r for r in rows if not expert.get(r["seed"], {}).get("placed", True)]
    report("seeds where the expert failed", bad)
    # where do the failures sit? (trends only: n per band is 3-10)
    bands = (("X < 0.42", lambda r: r["x"] < 0.42), ("0.42 <= X <= 0.45", lambda r: 0.42 <= r["x"] <= 0.45), ("X > 0.45", lambda r: r["x"] > 0.45))
    print("by spawn X band (placed/n, all seeds):", "; ".join(
        f"{lab}: {sum(r['ok'] for r in rows if f(r))}/{sum(1 for r in rows if f(r))}" for lab, f in bands))
    ybands = (("|Y| <= 0.10", lambda r: abs(r["y"]) <= 0.10), ("|Y| > 0.10", lambda r: abs(r["y"]) > 0.10))
    print("by spawn |Y| band:", "; ".join(f"{lab}: {sum(r['ok'] for r in rows if f(r))}/{sum(1 for r in rows if f(r))}" for lab, f in ybands))
    inbox = [r for r in rows if 0.42 <= r["x"] <= 0.48 and abs(r["y"]) <= 0.10]
    print(f"inside the OLD training box (X 0.42-0.48, |Y| <= 0.10): {sum(r['ok'] for r in inbox)}/{len(inbox)}; outside it: "
          f"{sum(r['ok'] for r in rows if r not in inbox)}/{len(rows) - len(inbox)}")
    print("expert failed on seeds:", sorted(s for s, e in expert.items() if not e["placed"]))


if __name__ == "__main__":
    main()
