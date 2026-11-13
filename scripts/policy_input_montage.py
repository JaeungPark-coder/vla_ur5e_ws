#!/usr/bin/env python3
"""Montage of the images the policy was actually given (trace/trial_NNN_images.npz, VLA_BRIDGE_SAVE_IMAGES=1) for chosen trials:
one row per trial, base camera then wrist camera at the given ticks; row label = trial, outcome, spawn. GPU-free.

    python3 scripts/policy_input_montage.py RUN_DIR OUT.png --trials 1 2 3 --ticks 50 100 150 200
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--trials", type=int, nargs="+", required=True)
ap.add_argument("--ticks", type=int, nargs="+", default=[50, 100, 150, 200])
args = ap.parse_args()
res = {int(r["trial"]): r for r in csv.DictReader(open(os.path.join(args.run, "results.csv")))}
S, L = 192, 150
canvas = Image.new("RGB", (L + 2 * S * len(args.ticks), S * len(args.trials) + 18), "white")
d = ImageDraw.Draw(canvas)
for j, t in enumerate(args.ticks):
    d.text((L + j * S + 4, 2), f"base t={t}", fill="black")
    d.text((L + (len(args.ticks) + j) * S + 4, 2), f"wrist t={t}", fill="black")
for i, tr in enumerate(args.trials):
    z = np.load(os.path.join(args.run, "trace", f"trial_{tr:03d}_images.npz"))
    cube0 = np.load(os.path.join(args.run, "trace", f"trial_{tr:03d}.npz"))["cube"][0]
    y0 = 18 + i * S
    d.text((4, y0 + 4), f"trial {tr}\n{'PLACED' if res[tr]['success'] == 'True' else 'failed'}\nX {cube0[0]:.3f}\nY {cube0[1]:+.3f}", fill="black")
    ticks = list(z["tick"])
    for j, t in enumerate(args.ticks):
        if t not in ticks:
            continue
        k = ticks.index(t)
        canvas.paste(Image.fromarray(z["base"][k]).resize((S, S)), (L + j * S, y0))
        canvas.paste(Image.fromarray(z["wrist"][k]).resize((S, S)), (L + (len(args.ticks) + j) * S, y0))
canvas.save(args.out)
print("saved", args.out)
