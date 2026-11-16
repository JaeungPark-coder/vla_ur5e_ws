#!/usr/bin/env python3
"""The sim workspace in the UR base frame (what the teach pendant shows), for laying out the real table: cube spawn boxes,
the evaluation regions, the place target, heights.  Isaac world = UR base rotated 180 deg about z (x, y -> -x, -y; z kept;
fitted on the bridge traces with 0.000 mm residual, README 2026-11-16), so a world box [x0, x1] x [y0, y1] is the UR box
[-x1, -x0] x [-y1, -y0].

Constants are read from isaac/pick_place_scene.py without Isaac (camera_framing.read_constants, VLA_SIM_PROFILE picks the
profile); the collection boxes and evaluation regions are the ones in scripts/v4_pipeline.sh and scripts/v4_6k_driver.sh.
    python3 scripts/workspace_ur_frame.py [--json eval_results/workspace_ur_frame.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'isaac'))
from camera_framing import read_constants  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--json', default=None)
a = ap.parse_args()

c = read_constants(os.path.join(HERE, '..', 'isaac', 'pick_place_scene.py'))
regions_line = re.search(r'^REGIONS="([^"]+)"', open(os.path.join(HERE, 'v4_6k_driver.sh')).read(), re.M).group(1)


def box_ur(xr, yr):
    return dict(x=[round(-xr[1], 4), round(-xr[0], 4)], y=[round(-yr[1], 4), round(-yr[0], 4)])


boxes = {'training_box (v1-v4 default, "old")': ((0.42, 0.48), (-0.10, 0.10)),
         'wide_box (v3/v4 collection, half of v4)': ((0.38, 0.48), (-0.16, 0.16))}
assert tuple(c['CUBE_X_RANGE']) == (0.42, 0.48) and tuple(c['CUBE_Y_RANGE']) == (-0.10, 0.10), 'scene box changed'
regions = {}
for item in regions_line.split():
    name, xs, ys = item.split(':')
    xr, yr = tuple(map(float, xs.split(','))), tuple(map(float, ys.split(',')))
    regions[name] = dict(world=dict(x=list(xr), y=list(yr)), ur=box_ur(xr, yr))
place_w = np.asarray(c['PLACE_TARGET_POSITION'], float)
table_z = float(c['TABLE_Z_M'])
out = dict(
    frame='UR base (pendant); world = UR base rotated 180 deg about z',
    table_z_ur=table_z, cube_size_m=float(c['CUBE_SIZE_M']), cube_centre_z_ur=float(c['CUBE_Z']),
    spawn_boxes={k: dict(world=dict(x=list(v[0]), y=list(v[1])), ur=box_ur(*v)) for k, v in boxes.items()},
    eval_regions=regions,
    place_target=dict(world=place_w.round(4).tolist(), ur=[round(-place_w[0], 4), round(-place_w[1], 4), round(place_w[2], 4)],
                      success_radius_m=0.03),
    lift_threshold_z_ur=float(c['LIFT_Z_THRESHOLD']),
    base_camera_v4=dict(world=list(c['BASE_CAMERA_POSITION']), ur=[-c['BASE_CAMERA_POSITION'][0], -c['BASE_CAMERA_POSITION'][1],
                                                                  c['BASE_CAMERA_POSITION'][2]],
                        aim_world=list(c['BASE_CAMERA_AIM_POINT']),
                        aim_ur=[-c['BASE_CAMERA_AIM_POINT'][0], -c['BASE_CAMERA_AIM_POINT'][1], c['BASE_CAMERA_AIM_POINT'][2]]),
)
print(json.dumps(out, indent=1))
if a.json:
    with open(a.json, 'w') as f:
        json.dump(out, f, indent=1)
