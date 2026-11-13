#!/usr/bin/env python3
"""First-frame evaluation follow-up (GPU-free): outcome classes, lateral offset at the close, and commanded vs measured hand height.

Why (2026-11-14): the converged-first-frame runs log the commanded joint targets per tick (`target`). For every attempt this prints
  - class (as scripts/failure_overlap.py), lateral offset grip-cube at the first close (as scripts/grasp_offset.py)
  - the lowest measured grip-point height z_meas, and the commanded grip-point height (FK of the target joints, scripts/ur5e_fk.py)
    at that tick and the lowest commanded height in the attempt
  - for runaways, the tick where wrist_3 first moves faster than 0.1 rad/tick, the measured and commanded height and gripper there.
If the hand was commanded low (z_cmd ~ z_meas) the policy asked for it; if the command stayed high while the hand went low, it is
control/physics.

    python3 scripts/ff_height_analysis.py eval_runs/ff_posY eval_runs/ff_negY eval_runs/ff_train_box [--json out.json]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from failure_overlap import analyse, classify  # noqa: E402
from ur5e_fk import apply, fit  # noqa: E402


def onset(w, thr=0.1):
    d = np.abs(np.diff(w))
    idx = np.nonzero(~np.isfinite(d) | (d > thr))[0]
    return int(idx[0]) + 1 if len(idx) else None


def main():
    runs = [a for a in sys.argv[1:] if os.path.isdir(a)]
    # fit the FK placement once on measured data of the first run (exact: the bridge computes grip_point the same way)
    qs, gs = [], []
    for t in range(2, 6):
        z = np.load(os.path.join(runs[0], "trace", f"trial_{t:03d}.npz"))
        qs.append(z["joints"][::10].astype(float)); gs.append(z["grip_point"][::10].astype(float))
    p, med, mx = fit(np.concatenate(qs), np.concatenate(gs))
    print(f"FK fit: median residual {1000 * med:.3f} mm, max {1000 * mx:.3f} mm\n")
    out = []
    for run in runs:
        print(f"===== {run}")
        print("trial  spawn X,Y        class     dy@close  z_meas_min (tick)  z_cmd there  z_cmd_min | runaway onset: tick  z_meas  z_cmd  gripper")
        for r in csv.DictReader(open(os.path.join(run, "results.csv"))):
            t = int(r["trial"])
            tdir = os.path.join(run, "trace")
            a = analyse(tdir, t)
            ok = r["success"] == "True"
            cls = classify(a, ok)
            z = np.load(os.path.join(tdir, f"trial_{t:03d}.npz"))
            q, g, cube, grip_cmd = (z[k].astype(float) for k in ("joints", "grip_point", "cube", "gripper"))
            tgt = z["target"].astype(float)
            good = np.isfinite(g).all(1) & (np.abs(q).max(1) < 1e4)
            n = int(np.argmin(good)) if not good.all() else len(good)
            zc = apply(p, tgt[:n, :6])[:, 2]
            zm = g[:n, 2]
            closes = np.nonzero(grip_cmd[:n] > 0.5)[0]
            k = int(closes[0]) if len(closes) else None
            dy = 1000 * (g[k, 1] - cube[k, 1]) if k is not None else float("nan")
            lo = int(np.argmin(zm))
            row = {"run": os.path.basename(run), "trial": t, "spawn": a["spawn"], "ok": ok, "cls": cls, "dy_close_mm": dy,
                   "z_meas_min_mm": 1000 * zm[lo], "tick_low": lo, "z_cmd_at_low_mm": 1000 * zc[lo], "z_cmd_min_mm": 1000 * zc.min(),
                   "gripper_at_low": float(grip_cmd[lo])}
            s = (f"{t:3d}   ({a['spawn'][0]:.3f},{a['spawn'][1]:+.3f})   {cls:8s}  {dy:+7.1f}   {1000 * zm[lo]:6.1f} ({lo:4d})      "
                 f"{1000 * zc[lo]:6.1f}     {1000 * zc.min():6.1f}")
            if cls in ("runaway", "blow-up"):
                o = onset(q[:, 5])
                if o is not None and o < len(q):
                    oo = min(o, n - 1)
                    row.update({"onset": o, "z_meas_onset_mm": 1000 * zm[oo], "z_cmd_onset_mm": 1000 * zc[oo],
                                "gripper_onset": float(grip_cmd[oo])})
                    s += f"   | {o:4d}  {1000 * zm[oo]:6.1f}  {1000 * zc[oo]:6.1f}  {grip_cmd[oo]:.2f}"
            print(s)
            out.append(row)
        print()
    def med(rs, key):
        v = [r[key] for r in rs if key in r and np.isfinite(r[key])]
        return f"{np.median(v):6.1f} (n={len(v)})" if v else "  -   (n=0)"
    print("medians (mm)                     z_meas_min       z_cmd_at_low      z_cmd_min        |dy|@close")
    for lab, f in (("placed", lambda r: r["ok"]), ("failed", lambda r: not r["ok"]),
                   ("runaway/blow-up", lambda r: r["cls"] in ("runaway", "blow-up"))):
        rs = [r for r in out if f(r)]
        for r in rs:
            r["absdy"] = abs(r["dy_close_mm"])
        print(f"  {lab:16s}  {med(rs, 'z_meas_min_mm')}  {med(rs, 'z_cmd_at_low_mm')}  {med(rs, 'z_cmd_min_mm')}  {med(rs, 'absdy')}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
