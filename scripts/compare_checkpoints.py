#!/usr/bin/env python3
"""Compare closed-loop runs of several checkpoints on the SAME seeds, mainly by the placement error itself.

    scripts/compare_checkpoints.py LABEL=RESULTS.csv:TRACE_DIR[:K] ...   # first one is the reference

Per trial the placement error is the cube's final xy distance to the target (from the per-tick trace; for placed trials it
equals the client's xy_error_mm). A run is reduced to the trials of horizon K (default: all) and ordered by seed: the n-th
trial of that K is seed base+n for a run that used VLA_BRIDGE_SEED_REPEAT=1, and the n-th seed group when K was cycled
(the 3-K run: trial i -> seed (i-1)//3; keep only the K you want and the order is the same). Failed trials count with their
final cube position (a cube never lifted is far from the target), so they are not silently dropped.
"""
import csv
import os
import sys

import numpy as np

TARGET_XY = np.array([0.45, 0.30])


def load(spec):
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    csv_path, trace_dir = parts[0], parts[1]
    k = int(parts[2]) if len(parts) > 2 else None
    err, ok, trial = [], [], []
    for r in csv.DictReader(open(csv_path)):
        if k is not None and int(r["execute_horizon"]) != k:
            continue
        z = np.load(os.path.join(trace_dir, f"trial_{int(r['trial']):03d}.npz"))["cube"]
        err.append(1000 * float(np.linalg.norm(z[-1, :2] - TARGET_XY)))
        ok.append(r["success"] == "True")
        trial.append(int(r["trial"]))
    return label, np.array(err), np.array(ok)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    runs = [load(s) for s in sys.argv[1:]]
    n = min(len(r[1]) for r in runs)
    print(f"{n} paired seeds\n")
    print(f"{'run':<10}{'placed(30mm)':>13}{'<=10mm':>8}{'median mm':>11}{'p90 mm':>9}{'mean mm':>9}   (errors over all trials, failures at their final cube position)")
    for label, err, ok in runs:
        e = err[:n]
        print(f"{label:<10}{int(ok[:n].sum()):>9}/{n:<3}{int((e <= 10).sum()):>5}/{n:<2}{np.median(e):>11.1f}{np.percentile(e, 90):>9.1f}{e.mean():>9.1f}")
    ref_label, ref, _ = runs[0]
    for label, err, _ in runs[1:]:
        d = err[:n] - ref[:n]
        print(f"\n{label} minus {ref_label}, per seed: median {np.median(d):+.1f} mm, mean {d.mean():+.1f} mm; "
              f"better in {int((d < 0).sum())}, worse in {int((d > 0).sum())}, tie {int((d == 0).sum())} of {n}")
        # exact two-sided sign test
        from math import comb
        k, m = int((d < 0).sum()), int((d != 0).sum())
        if m:
            tail = sum(comb(m, i) for i in range(0, min(k, m - k) + 1)) / 2 ** m
            print(f"  sign test p = {min(1.0, 2 * tail):.3f}")
        # paired bootstrap of the median difference
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, (4000, n))
        meds = np.median(d[idx], axis=1)
        print(f"  bootstrap 95% interval of the median difference: [{np.percentile(meds, 2.5):+.1f}, {np.percentile(meds, 97.5):+.1f}] mm")


if __name__ == "__main__":
    main()
