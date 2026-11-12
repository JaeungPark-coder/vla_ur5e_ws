#!/usr/bin/env python3
"""Paired comparison of two evaluation runs on the SAME seeds: outcome table, exact McNemar test (paired), Fisher exact test (unpaired,
the conservative one to quote), Wilson intervals, per-seed list.

    scripts/paired_success.py A=results_a.csv[:K] B=results_b.csv[:K]    # trial i of the file (restricted to horizon K if given) = seed first+i

Run A is the reference (e.g. v2 2999), B the candidate (v3). Trials are matched by their order (seed 8000+k) -- both runs must have used the
same VLA_BRIDGE_SEED and the same number of trials; the 3-K run needs ':50' so that only its K=50 trials are used.
"""
import csv
import sys
from math import comb

from scipy.stats import fisher_exact

HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, HERE)
from summarize_eval import wilson  # noqa: E402


def load(spec):
    label, rest = spec.split("=", 1)
    path, _, k = rest.partition(":")
    out = []
    for r in csv.DictReader(open(path)):
        if k and int(r["execute_horizon"]) != int(k):
            continue
        out.append(r["success"] == "True")
    return label, out


def main():
    (la, a), (lb, b) = load(sys.argv[1]), load(sys.argv[2])
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(f"WARNING: {la} has {len(a)} trials, {lb} has {len(b)}; using the first {n}")
    a, b = a[:n], b[:n]
    both = sum(x and y for x, y in zip(a, b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    neither = n - both - only_a - only_b
    ka, kb = sum(a), sum(b)
    print(f"{la}: {ka}/{n} Wilson [{wilson(ka, n)[0]:.2f}, {wilson(ka, n)[1]:.2f}]    {lb}: {kb}/{n} Wilson [{wilson(kb, n)[0]:.2f}, {wilson(kb, n)[1]:.2f}]")
    print(f"paired: both {both}, only {la} {only_a}, only {lb} {only_b}, neither {neither}")
    m = only_a + only_b
    p_mc = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** m)
    print(f"exact McNemar p = {p_mc:.3f} (paired); Fisher exact p = {fisher_exact([[ka, n - ka], [kb, n - kb]])[1]:.3f} (unpaired, quote this one)")
    print("seeds where they differ:", [(i, "A" if x else "B") for i, (x, y) in enumerate(zip(a, b)) if x != y])


if __name__ == "__main__":
    main()
