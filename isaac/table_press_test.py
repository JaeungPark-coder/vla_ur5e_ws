"""Table-press test (README 2026-11-17, reviewer's request, written before the run): does pressing the OPEN fingers into
the table / onto the cube's edge make the simulated cube explode, deterministically and without a policy?

Why: every v5 pilot failure (5/15) and the v4 6k blow-ups look the same -- the open hand comes down beside the cube,
lower than the demonstrations, and the cube is flung 0.7-1.6 m.  On the real robot a finger pressing the table stops
the arm (protective stop) or nudges the cube; in the sim it ends as a PhysX blow-up.  If this script reproduces that
with a fixed motion, the contact settings can be tuned against it before the v5 data is collected.

Each case: cube spawned at CUBE_XY, hand tool-down with the gripper OPEN (target 0.0), driven by the expert's own
Cartesian controller (scene.step_towards, RMPflow, the v5 joint-speed cap):
  approach  to (target xy, 0.15 m)                 APPROACH ticks
  descend   to (target xy, target z)               DESCEND ticks
  hold      stay there                             HOLD ticks
  drag      +DRAG_M along y, staying low           DRAG ticks
  lift      back to 0.15 m                         LIFT ticks
Target = cube centre + (dx, dy) for the grip point (fingertip, sim_profile formula); z relative to the table top.
Every tick is judged with the evaluation's blow-up criteria (vla_bridge/blowup_criteria.py via trial_trace.TrialTrace).
    GPU=0 VLA_SIM_PROFILE=v5 scripts/isaac_py.sh table_press_test.py --out eval_results/table_press/v5
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="all", help="comma-separated case names, or all")
args = ap.parse_args()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(args.out):
    args.out = os.path.join(REPO, args.out)
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)

CUBE_XY = (0.45, 0.0)
APPROACH, DESCEND, HOLD, DRAG, LIFT = 200, 150, 180, 120, 120
DRAG_M = 0.06
# name: (dx, dy, z above table) for the grip point, metres
CASES = {
    "table_beside_-10": (-0.07, 0.0, -0.010),    # fingers press the table 7 cm from the cube centre, cube clear
    "table_beside_-20": (-0.07, 0.0, -0.020),
    "pilot_x_+20": (-0.022, 0.0, 0.020),         # the pilot failures: 21-25 mm short in x, fingertip 0.021-0.027 m
    "pilot_x_0": (-0.022, 0.0, 0.000),
    "pilot_x_-10": (-0.022, 0.0, -0.010),
    "pilot_y_+20": (0.0, -0.022, 0.020),         # the same offset along y (finger opening axis unknown a priori)
    "pilot_y_-10": (0.0, -0.022, -0.010),
    "straddle_-10": (0.0, 0.0, -0.010),          # a normal grasp descent, but 1 cm too deep, fingers open
    "straddle_-20": (0.0, 0.0, -0.020),
}

try:
    import pick_place_scene as pps
    import sim_profile
    from scripted_pick_place import DOWNWARD_ROTVEC
    from trial_trace import TrialTrace
    print(sim_profile.describe(), flush=True)
    os.makedirs(args.out, exist_ok=True)
    pps.CUBE_X_RANGE = (CUBE_XY[0], CUBE_XY[0])
    pps.CUBE_Y_RANGE = (CUBE_XY[1], CUBE_XY[1])
    scene = c.build_scene()
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    results = []
    for name in names:
        dx, dy, zt = CASES[name]
        scene.reset()
        cube0 = np.asarray(scene.get_cube_position(), float)
        tz = float(pps.TABLE_Z_M)
        tgt = np.array([cube0[0] + dx, cube0[1] + dy, tz + zt])
        above = np.array([tgt[0], tgt[1], tz + 0.15])
        start = np.asarray(scene.grip_point_world(), float)
        trace = TrialTrace(None, table_z=tz)
        trace.start(0, cube=cube0)
        plan = [("approach", start, above, APPROACH), ("descend", above, tgt, DESCEND), ("hold", tgt, tgt, HOLD),
                ("drag", tgt, tgt + [0, DRAG_M, 0], DRAG), ("lift", tgt + [0, DRAG_M, 0], above + [0, DRAG_M, 0], LIFT)]
        tick, phase_of_tick, min_grip_z, phase_stats = 0, [], np.inf, {}
        for phase, a, b, n in plan:
            p0 = len(trace.cube)
            for i in range(1, n + 1):
                pos = a + (b - a) * (i / n)
                scene.step_towards(pos, DOWNWARD_ROTVEC, 0.0)
                cube = np.asarray(scene.get_cube_position(), float)
                grip = np.asarray(scene.grip_point_world(), float)
                prop = scene.get_proprioception()
                trace.add(tick, cube, grip, prop["joints"], float(prop["gripper"][0]))
                phase_of_tick.append(phase)
                min_grip_z = min(min_grip_z, float(grip[2] - tz))
                tick += 1
            cs = np.asarray(trace.cube[p0:], float)
            phase_stats[phase] = dict(cube_xy_moved_mm=float(np.linalg.norm(cs[:, :2] - cube0[:2], axis=1).max() * 1000),
                                      cube_max_z_m=float(cs[:, 2].max() - tz))
        s = trace.summary()
        ft = s["first_trip"]
        cube_end = np.asarray(trace.cube[-1], float)
        r = dict(case=name, dx=dx, dy=dy, target_z=zt, blowup=s["blowup"], reasons=s["reasons"],
                 first_trip_tick=None if ft is None else ft["tick"],
                 first_trip_phase=None if ft is None else phase_of_tick[ft["tick"]],
                 max_one_tick_jump_mm=1000 * s["max_one_tick_jump_m"], max_cube_z_m=s["max_cube_z"],
                 cube_end=np.round(cube_end, 4).tolist(),
                 cube_end_moved_mm=float(np.linalg.norm(cube_end[:2] - cube0[:2]) * 1000),
                 min_grip_z_above_table_m=round(min_grip_z, 4), phases=phase_stats)
        results.append(r)
        print("CASE", json.dumps(r), flush=True)
    summary = dict(profile=sim_profile.PROFILE, cube_xy=CUBE_XY, ticks=dict(approach=APPROACH, descend=DESCEND, hold=HOLD,
                   drag=DRAG, lift=LIFT), drag_m=DRAG_M, blowups=sum(r["blowup"] for r in results), cases=len(results))
    json.dump(dict(summary=summary, cases=results), open(os.path.join(args.out, "table_press.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
