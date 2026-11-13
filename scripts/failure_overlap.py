#!/usr/bin/env python3
"""Do the same seeds fail in every model? (GPU-free: reads the saved evaluation CSVs and per-tick traces.)

Why (2026-11-13): after the seed repeat the three models (v2 seed 42, v2 seed 1, v3) score alike on every spawn box. If the same
seeds fail in all of them the cause sits in that cube position (camera view, reach), not in training luck; if the failures differ per
model it is training variation. One table per box: per seed the cube spawn, the expert's camera pixels, and per model the outcome
class and a short trajectory summary of the policy's attempt.

    python3 scripts/failure_overlap.py [--json out.json]

Outcome classes (mutually exclusive, first match): placed | blow-up (trace verdict) | runaway (wrist_3 runaway, no blow-up) |
lifted (cube reached >0.05 m but was not released within 30 mm) | untouched (cube never moved more than 5 mm from its spawn) |
pushed (cube moved but was never lifted).
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "eval_results")
E = os.path.join(ROOT, "eval_runs")
PLACE_XY = np.array([0.45, 0.30])

# model -> box -> (results csv, K filter or None, trace dir)
RUNS = {
    "v2_s42": {"train": (f"{R}/short_3k_2999_k10-25-50_seed8000.csv", 50, f"{E}/k3_2999/trace"),
               "step1": (f"{R}/short_3k_2999_range_step1_k50_seed8000.csv", None, f"{E}/range_step1/trace"),
               "step2": (f"{R}/short_3k_2999_range_step2_k50_seed8000.csv", None, f"{E}/range_step2/trace")},
    "v2_s1": {b: (f"{R}/short_3k_seed1_2999_{d}_k50_seed8000.csv", None, f"{E}/seed1_{d}/trace")
              for b, d in (("train", "train_box"), ("step1", "step1"), ("step2", "step2"))},
    "v3": {b: (f"{R}/v3_3k_2999_{d}_k50_seed8000.csv", None, f"{E}/v3_{d}/trace")
           for b, d in (("train", "train_box"), ("step1", "step1"), ("step2", "step2"))},
}
EXPERT = {"train": f"{R}/expert_baseline_base.jsonl", "step1": f"{R}/expert_baseline_step1.jsonl",
          "step2": f"{R}/expert_baseline_step2.jsonl"}


def runaway_of(w):
    fin = np.isfinite(w) & (np.abs(w) < 1e4)
    bad = bool((~fin).any())
    w = w[:int(np.argmax(~fin))] if bad else w
    unwind = float(np.max(np.maximum.accumulate(w) - w)) if len(w) else 0.0
    return bool(unwind > 3.0 or (len(w) and (w.min() < -1.0 or w.max() > 5.5)) or bad)


def analyse(trace_dir, trial):
    z = np.load(os.path.join(trace_dir, f"trial_{trial:03d}.npz"))
    meta = json.load(open(os.path.join(trace_dir, f"trial_{trial:03d}.json")))
    cube, grip, joints, gripper = (z[k].astype(float) for k in ("cube", "grip_point", "joints", "gripper"))
    ok_rows = np.isfinite(cube).all(1) & np.isfinite(grip).all(1) & (np.abs(joints).max(1) < 1e4)
    cube, grip, joints, gripper = cube[ok_rows], grip[ok_rows], joints[ok_rows], gripper[ok_rows]
    spawn = cube[0]
    moved = float(np.linalg.norm(cube[:, :2] - spawn[:2], axis=1).max())
    dist = np.linalg.norm(grip - spawn, axis=1)
    last = joints[-min(600, len(joints)):]
    return {"blowup": bool(meta["blowup"]), "runaway": runaway_of(joints[:, 5]), "max_z": float(cube[:, 2].max()),
            "cube_moved_mm": 1000 * moved, "min_grip_to_spawn_mm": 1000 * float(dist.min()),
            "tick_of_min": int(dist.argmin()), "closed_ever": bool((gripper > 0.5).any()),
            "closed_frac": float((gripper > 0.5).mean()),
            "joint_range_last600": float(np.abs(last.max(0) - last.min(0)).max()),
            "end_grip_to_spawn_mm": 1000 * float(dist[-1]), "ticks": int(len(cube)),
            "spawn": spawn[:2].tolist(), "end_cube_to_place_mm": 1000 * float(np.linalg.norm(cube[-1, :2] - PLACE_XY))}


def classify(a, ok):
    if ok:
        return "placed"
    if a["blowup"]:
        return "blow-up"
    if a["runaway"]:
        return "runaway"
    if a["max_z"] > 0.05:
        return "lifted"
    if a["cube_moved_mm"] < 5.0:
        return "untouched"
    return "pushed"


def load_box(box):
    expert = {r["seed"]: r for r in map(json.loads, open(EXPERT[box]))}
    out = {}
    for model, boxes in RUNS.items():
        path, keep_k, tdir = boxes[box]
        for r in csv.DictReader(open(path)):
            if keep_k and int(r["execute_horizon"]) != keep_k:
                continue
            t = int(r["trial"])
            seed = 8000 + ((t - 1) // 3 if keep_k else t - 1)
            a = analyse(tdir, t)
            a["ok"] = r["success"] == "True"
            a["cls"] = classify(a, a["ok"])
            out.setdefault(seed, {})[model] = a
    return out, expert


def main():
    dump = {}
    short = {"placed": "ok", "blow-up": "BLOW", "runaway": "RUN", "lifted": "LIFT", "untouched": "UNT", "pushed": "PUSH"}
    for box in ("train", "step1", "step2"):
        res, expert = load_box(box)
        print(f"\n===== {box} box: outcome class per seed (ok placed | UNT never touched | LIFT lifted, not placed | RUN wrist_3 runaway |"
              f" BLOW blow-up | PUSH moved, not lifted)")
        print("seed   spawn X,Y        expert  expert-cam px (base med / wrist min)   v2_s42  v2_s1   v3")
        rows = []
        for seed in sorted(res):
            m = res[seed]
            ex = expert.get(seed, {})
            sp = m["v2_s1"]["spawn"]
            fails = [mm for mm in RUNS if m[mm]["cls"] != "placed"]
            rows.append((seed, fails))
            print(f"{seed}  ({sp[0]:.3f},{sp[1]:+.3f})   {'ok ' if ex.get('placed') else 'FAIL'}    "
                  f"{ex.get('base_px_median', '-'):>5} / {ex.get('wrist_px_min_150_450', '-'):>4}"
                  f"                         " + "  ".join(f"{short[m[mm]['cls']]:6s}" for mm in RUNS) + ("   <- fails in all 3" if len(fails) == 3 else ""))
        n_all = sum(1 for _, f in rows if len(f) == 3)
        cnt = {k: sum(1 for _, f in rows if len(f) == k) for k in range(4)}
        print(f"seeds failing in 0/1/2/3 of the models: {cnt[0]}/{cnt[1]}/{cnt[2]}/{cnt[3]}; "
              f"failures per model: " + ", ".join(f"{mm} {sum(1 for _, f in rows if mm in f)}" for mm in RUNS))
        # how many seeds would fail in all three by chance, given each model's own failure count (independent shuffles)?
        p = [sum(1 for _, f in rows if mm in f) / len(rows) for mm in RUNS]
        exp_all = len(rows) * p[0] * p[1] * p[2]
        exp_two = len(rows) * (p[0] * p[1] + p[0] * p[2] + p[1] * p[2] - 3 * p[0] * p[1] * p[2])
        print(f"  if the models failed independently: ~{exp_all:.1f} seeds in all 3, ~{exp_two:.1f} in exactly 2 (observed {cnt[3]}, {cnt[2]})")
        # class totals per model
        for mm in RUNS:
            c = {}
            for s in res:
                c[res[s][mm]["cls"]] = c.get(res[s][mm]["cls"], 0) + 1
            print(f"  {mm:7s} classes: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        # trajectory summary of the untouched attempts
        unt = [(s, mm, res[s][mm]) for s in sorted(res) for mm in RUNS if res[s][mm]["cls"] == "untouched"]
        if unt:
            print("  untouched attempts: seed model | closest the grip point came to the cube spawn (mm, at tick) | gripper ever closed |"
                  " arm still moving in the last 600 ticks (max joint range rad) | grip point to spawn at the end (mm)")
            for s, mm, a in unt:
                print(f"    {s} {mm:7s}| {a['min_grip_to_spawn_mm']:6.0f} @ {a['tick_of_min']:4d} | closed {'yes' if a['closed_ever'] else 'no '} "
                      f"({100 * a['closed_frac']:.0f}% of ticks) | {a['joint_range_last600']:.2f} | {a['end_grip_to_spawn_mm']:.0f}")
        dump[box] = {str(s): {mm: {k: v for k, v in a.items()} for mm, a in res[s].items()} for s in res}
    if "--json" in sys.argv:
        json.dump(dump, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
