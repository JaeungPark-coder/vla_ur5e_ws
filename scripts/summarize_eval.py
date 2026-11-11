#!/usr/bin/env python3
"""Summarise a closed-loop evaluation: success rate per execute_horizon (Wilson 95% interval), blow-ups counted
SEPARATELY, and the paired view (same seed = same cube spawn under every K).

    scripts/summarize_eval.py RESULTS.csv [TRACE_DIR]

RESULTS.csv: vla_policy_client's results_csv_path. TRACE_DIR: the bridge's VLA_BRIDGE_TRACE_DIR (trial_NNN.json, the
blow-up verdict by trial_trace.py). Trial i of the csv is trial i of the trace; seeds are NOT in the csv -- trials are
grouped by seed as `(i-1) // K_count` (the schedule cycles K inside each seed).
"""
import csv
import json
import math
import os
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(csv_path, trace_dir=None):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r = dict(r)
            r["trial"] = int(r["trial"])
            r["success"] = r["success"] in ("True", "1")
            r["execute_horizon"] = int(r.get("execute_horizon") or 0)
            r["client_blowup"] = bool(int(r.get("blowup") or 0))
            r["trace_blowup"] = None
            r["trace_reasons"] = {}
            if trace_dir:
                p = os.path.join(trace_dir, f"trial_{r['trial']:03d}.json")
                if os.path.exists(p):
                    t = json.load(open(p))
                    r["trace_blowup"] = t["blowup"]
                    r["trace_reasons"] = t["reasons"]
            r["blowup"] = r["client_blowup"] or bool(r["trace_blowup"])
            rows.append(r)
    return rows


def summarize(rows):
    out = {}
    for K in sorted({r["execute_horizon"] for r in rows}):
        rs = [r for r in rows if r["execute_horizon"] == K]
        n = len(rs)
        ok = sum(r["success"] for r in rs)
        bl = sum(r["blowup"] for r in rs)
        clean = [r for r in rs if not r["blowup"]]
        okc = sum(r["success"] for r in clean)
        out[K] = {"n": n, "success": ok, "ci": wilson(ok, n), "blowups": bl,
                  "n_no_blowup": len(clean), "success_no_blowup": okc, "ci_no_blowup": wilson(okc, len(clean))}
    return out


def paired(rows, n_k):
    """Per seed group: which K succeeded. Only complete groups (n_k trials) are used."""
    groups = {}
    for r in rows:
        groups.setdefault((r["trial"] - 1) // n_k, []).append(r)
    return {g: {r["execute_horizon"]: r["success"] for r in rs} for g, rs in groups.items() if len(rs) == n_k}


def main():
    rows = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    s = summarize(rows)
    print(f"{len(rows)} trials")
    print(f"{'K':>4} {'n':>3} {'success':>8} {'rate':>6} {'95% Wilson':>14} {'blow-ups':>9}   without blow-ups")
    for K, v in s.items():
        print(f"{K:>4} {v['n']:>3} {v['success']:>8} {v['success'] / max(v['n'], 1):>6.2f} "
              f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] {v['blowups']:>9}   "
              f"{v['success_no_blowup']}/{v['n_no_blowup']} [{v['ci_no_blowup'][0]:.2f}, {v['ci_no_blowup'][1]:.2f}]")
    ks = sorted(s)
    pr = paired(rows, len(ks))
    if pr:
        print(f"\npaired by seed ({len(pr)} complete groups), S = success:")
        for g, d in sorted(pr.items()):
            print(f"  seed+{g:<3} " + "  ".join(f"K={k}:{'S' if d.get(k) else '-'}" for k in ks))
    reasons = {}
    for r in rows:
        for k in r["trace_reasons"]:
            reasons[k] = reasons.get(k, 0) + 1
    if reasons:
        print("\nblow-up criteria hit (trials):", reasons)
    dis = [r["trial"] for r in rows if r["trace_blowup"] is not None and r["client_blowup"] != r["trace_blowup"]]
    if dis:
        print("trials where the client's z>0.25 flag and the bridge's trace verdict disagree:", dis)


if __name__ == "__main__":
    main()
