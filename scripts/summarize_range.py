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
        fin = np.isfinite(w)
        w = w[:int(np.argmax(~fin))] if (~fin).any() else w
        d30 = float(np.abs(w[30:] - w[:-30]).max()) if len(w) > 30 else 0.0
        unwind = float(np.max(np.maximum.accumulate(w) - w))
        cube_end = z["cube"][-1, :2]
        err = 1000 * float(np.linalg.norm(cube_end - np.array([0.45, 0.30])))
        rows.append({"trial": t, "seed": seed, "ok": r["success"] == "True", "blowup": bool(meta["blowup"]) or r["blowup"] == "1",
                     "d30": d30, "unwind": unwind, "err": err})

    def report(label, rs):
        n = len(rs)
        if not n:
            print(f"{label}: no trials")
            return
        ok = sum(r["ok"] for r in rs)
        lo, hi = wilson(ok, n)
        bl = sum(r["blowup"] for r in rs)
        clean = [r for r in rs if not r["blowup"]]
        placed = np.array([r["err"] for r in rs if r["ok"]])
        d30 = np.array([r["d30"] for r in clean]) if clean else np.zeros(1)
        un = np.array([r["unwind"] for r in clean]) if clean else np.zeros(1)
        print(f"{label:34s} n={n:2d} placed {ok:2d} ({ok / n:.2f}, Wilson {lo:.2f}-{hi:.2f}) blow-ups {bl} | "
              f"placed err median {np.median(placed) if len(placed) else float('nan'):.1f} mm | "
              f"wrist_3 30-tick max {d30.max():.2f}, unwinding >0.5 rad in {int((un > 0.5).sum())}/{len(clean)} (max {un.max():.2f}) [non-blow-up trials]")

    report("all seeds", rows)
    good = [r for r in rows if expert.get(r["seed"], {}).get("placed")]
    report("seeds where the expert succeeded", good)
    bad = [r for r in rows if not expert.get(r["seed"], {}).get("placed", True)]
    report("seeds where the expert failed", bad)
    print("expert failed on seeds:", sorted(s for s, e in expert.items() if not e["placed"]))


if __name__ == "__main__":
    main()
