"""Per-tick trace of the SCRIPTED EXPERT (joints, gripper, grip point, cube) for given seeds -- to compare expert wrist_3 runaways
with the policy's (hand height and gripper state at the onset). expert_baseline.py keeps only summaries.

    GPU=0 scripts/isaac_py.sh expert_trace.py --seeds 8008 8016 --x_range 0.38 0.48 --y_range -0.16 0.16 --out /tmp/expert_trace.npz
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--x_range", type=float, nargs=2, required=True)
ap.add_argument("--y_range", type=float, nargs=2, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import pick_place_scene as pps  # noqa: E402

try:
    pps.CUBE_X_RANGE, pps.CUBE_Y_RANGE = tuple(args.x_range), tuple(args.y_range)
    scene = c.build_scene()
    rec = []
    orig_step = scene.world.step

    def step(*a, **k):
        r = orig_step(*a, **k)
        rec.append(np.concatenate([np.asarray(scene.robot.get_joint_positions())[:6], [scene.gripper_position() if hasattr(scene, "gripper_position") else np.nan],
                                   scene.grip_point_world(), scene.get_cube_position()]))
        return r
    scene.world.step = step
    out = {}
    for seed in args.seeds:
        rec.clear()
        states, actions, cmds, lifted, placed, err = c.record_expert(scene, seed)
        a = np.asarray(rec, dtype=float)
        out[f"rec_{seed}"] = a                      # per physics step: 6 joints, gripper(nan if n/a), grip point xyz, cube xyz
        out[f"states_{seed}"] = np.asarray(states)   # per recorded frame (joints + gripper as in the dataset)
        out[f"placed_{seed}"] = np.asarray(placed)
        print(f"seed {seed}: steps {len(a)} frames {len(states)} placed {placed}", flush=True)
        np.savez_compressed(args.out, **out)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
