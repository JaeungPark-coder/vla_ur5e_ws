#!/usr/bin/env python3
"""Signed offset of the grip point from the cube at the moment the policy first closes the gripper (GPU-free, from the saved traces).

Why (2026-11-13): failed attempts at spawn Y > +0.10 close the hand ~5 cm beside the cube. Is the miss toward the training centre
(regression to the mean of the data), or random? offset = grip_point - cube (world frame, metres) at the first tick with gripper > 0.5.

    python3 scripts/grasp_offset.py
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from failure_overlap import RUNS, load_box  # noqa: E402


def close_tick(gripper):
    idx = np.nonzero(gripper > 0.5)[0]
    return int(idx[0]) if len(idx) else None


def main():
    rows = []
    for box in ("train", "step1", "step2"):
        res, _ = load_box(box)
        for model, boxes in RUNS.items():
            path, keep_k, tdir = boxes[box]
            for r in csv.DictReader(open(path)):
                if keep_k and int(r["execute_horizon"]) != keep_k:
                    continue
                t = int(r["trial"])
                seed = 8000 + ((t - 1) // 3 if keep_k else t - 1)
                z = np.load(os.path.join(tdir, f"trial_{t:03d}.npz"))
                cube, grip, g = z["cube"].astype(float), z["grip_point"].astype(float), z["gripper"].astype(float)
                k = close_tick(g)
                if k is None or not np.isfinite(cube[k]).all():
                    continue
                spawn = cube[0]
                rows.append({"box": box, "seed": seed, "model": model, "ok": r["success"] == "True", "cls": res[seed][model]["cls"],
                             "sx": spawn[0], "sy": spawn[1], "tick": k, "dx": (grip[k] - cube[k])[0] * 1000,
                             "dy": (grip[k] - cube[k])[1] * 1000, "dz": (grip[k] - cube[k])[2] * 1000})
    def line(lab, rs):
        if not rs:
            print(f"{lab}: n=0"); return
        a = np.array([[r["dx"], r["dy"], r["dz"]] for r in rs])
        print(f"{lab:34s} n={len(rs):3d}  dx(fwd) median {np.median(a[:,0]):+6.1f} [{np.percentile(a[:,0],25):+.0f},{np.percentile(a[:,0],75):+.0f}]  "
              f"dy(lateral) median {np.median(a[:,1]):+6.1f} [{np.percentile(a[:,1],25):+.0f},{np.percentile(a[:,1],75):+.0f}]  dz median {np.median(a[:,2]):+5.1f}  "
              f"|dxy| median {np.median(np.hypot(a[:,0],a[:,1])):.1f}")
    print("offset = grip point - cube at the first tick the gripper closes, mm; [25th,75th percentile]; +y = away from the robot's centre line toward the place target side\n")
    for lab, f in (("Y < -0.10", lambda r: r["sy"] < -0.10), ("|Y| <= 0.10", lambda r: abs(r["sy"]) <= 0.10), ("Y > +0.10", lambda r: r["sy"] > 0.10)):
        line(f"{lab} all", [r for r in rows if f(r)])
        line(f"{lab} placed", [r for r in rows if f(r) and r["ok"]])
        line(f"{lab} failed", [r for r in rows if f(r) and not r["ok"]])
    print("\nper model, Y > +0.10:")
    for m in RUNS:
        line(f"  {m}", [r for r in rows if r["sy"] > 0.10 and r["model"] == m])
    print("\nlateral miss vs spawn Y (all attempts, Pearson r of dy on spawn y):", f"{np.corrcoef([r['sy'] for r in rows], [r['dy'] for r in rows])[0,1]:+.2f}")
    print("same, failed attempts only:", f"{np.corrcoef([r['sy'] for r in rows if not r['ok']], [r['dy'] for r in rows if not r['ok']])[0,1]:+.2f}")
    json.dump(rows, open(os.path.join(os.path.dirname(HERE), "eval_results", "grasp_offset_seed_repeat.json"), "w"))


if __name__ == "__main__":
    main()
