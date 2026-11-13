#!/usr/bin/env python3
"""Red-cube pixels in the images the policy was actually given (trace/trial_NNN_images.npz), per decision tick, placed vs failed.
GPU-free.  python3 scripts/policy_input_pixels.py RUN_DIR [RUN_DIR ...]"""
import csv
import os
import sys

import numpy as np


def red(img):
    im = img[..., :3].astype(int)
    return int(((im[..., 0] > 120) & (im[..., 0] > im[..., 1] + 60) & (im[..., 0] > im[..., 2] + 60)).sum())


for run in sys.argv[1:]:
    res = {int(r["trial"]): r["success"] == "True" for r in csv.DictReader(open(os.path.join(run, "results.csv")))}
    per = {}
    for t, ok in res.items():
        z = np.load(os.path.join(run, "trace", f"trial_{t:03d}_images.npz"))
        for k, tick in enumerate(z["tick"][:6]):
            per.setdefault(int(tick), []).append((ok, red(z["base"][k]), red(z["wrist"][k])))
    print(f"== {run}: median red-cube pixels in the policy's inputs (n placed / failed = {sum(res.values())}/{len(res) - sum(res.values())})")
    print("tick   base placed  base failed  | wrist placed  wrist failed | trials with cube in NEITHER camera")
    for tick in sorted(per):
        a = per[tick]
        med = lambda sel: f"{np.median(sel):6.0f}" if sel else "     -"
        print(f"{tick:4d}   {med([b for o, b, w in a if o])}       {med([b for o, b, w in a if not o])}       |  {med([w for o, b, w in a if o])}        "
              f"{med([w for o, b, w in a if not o])}     | {sum(1 for o, b, w in a if b == 0 and w == 0)}/{len(a)}")
