"""How much does the simulator setting change the tolerance of the serving recipe to
policy-like noise -- and what does it cost the expert?

Why (2026-10-05): the final serving recipe (delta rows relative to a block start, velocity
feed-forward, execute_horizon K) tolerates ~0.005 rad of row noise but not 0.02, sometimes
with a PhysX blow-up (check_action_replay.py, README). The suspicion: the grasp targets the
cube's top face with zero clearance and the arm drives are stiff, so a few mm of error
becomes a large contact force.

Design (the data's actions are MEASURED next positions, so they contain the drive's
dynamics -- changing the drive only at serving time would break tracking before noise
matters):
  * one process = ONE setting: arm drive stiffness x --stiffness_scale (damping untouched),
    scripted_pick_place.GRASP_HEIGHT = --grasp_height. Only those two knobs, never both at once
    in the same comparison, so a difference can be attributed.
  * the scripted expert is RE-RECORDED under that setting (so the recording carries that
    setting's dynamics); its success rate over the attempts is reported -- a setting that is
    robust to noise but makes the expert fail is useless.
  * each successful recording is replayed through the bridge's physics path with the serving
    recipe (blocks of K rows relative to the block-start state, velocities from the noisy rows
    via vla_bridge.feedforward.chunk_velocities) at several AR(1) noise levels.

The conclusion it can support: "re-collecting under THIS setting improves noise tolerance by
this much" -- not "serving changes fix it".

    ISAAC_ENV=/home/icrs/bigdisk/conda_envs/env_isaaclab
    LD_PRELOAD=$ISAAC_ENV/lib/libstdc++.so.6 $ISAAC_ENV/bin/python check_noise_robustness.py \
        --name baseline --episodes 10 --out /tmp/noise_baseline.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--stiffness_scale", type=float, default=1.0)
ap.add_argument("--grasp_height", type=float, default=None, help="default: the repo's value")
ap.add_argument("--episodes", type=int, default=10, help="successful expert recordings to replay")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.005, 0.01])
ap.add_argument("--block", type=int, default=25)
ap.add_argument("--window", type=int, default=11)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]  # check_action_replay parses argv in its own main(), which is not used here

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
import scripted_pick_place as sp  # noqa: E402
from isaac_sim_common import get_joint_drive_gains, set_joint_drive_gains  # noqa: E402

BLOWUP_Z = 0.25  # the cube never gets this high in a normal episode (~0.16 m); collect_demos filters on it too


def log(rec):
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def apply_setting(scene):
    robot_prim = scene.stage.GetPrimAtPath(c.ROBOT_PRIM_PATH)
    before = get_joint_drive_gains(robot_prim)
    if args.stiffness_scale != 1.0:
        for name, gains in before.items():
            if gains is not None and gains[0] is not None:
                set_joint_drive_gains(robot_prim, stiffness=gains[0] * args.stiffness_scale, joint_names=(name,))
    if args.grasp_height is not None:
        sp.GRASP_HEIGHT = args.grasp_height
    scene.reset()  # the USD drive schema is only re-parsed at the next Stop+Play
    after = get_joint_drive_gains(robot_prim)
    live = None
    try:
        live = [np.round(np.asarray(g), 1).tolist() for g in scene.robot.get_articulation_controller().get_gains()]
    except Exception:
        pass
    print(f"[{args.name}] stiffness_scale={args.stiffness_scale} grasp_height={sp.GRASP_HEIGHT}", flush=True)
    print(f"[{args.name}] USD drive gains (stiffness, damping) before: "
          f"{ {k: (None if v is None else tuple(round(x, 2) for x in v)) for k, v in before.items()} }", flush=True)
    print(f"[{args.name}] USD drive gains after:  "
          f"{ {k: (None if v is None else tuple(round(x, 2) for x in v)) for k, v in after.items()} }", flush=True)
    print(f"[{args.name}] live controller (kps, kds): {live}", flush=True)


def is_blowup(r):
    pr = r["path_ratio"]
    return bool(r["max_cube_z"] > BLOWUP_Z or pr is None or not np.isfinite(pr) or pr > 5)


def main():
    scene = c.build_scene()
    apply_setting(scene)
    t0 = time.time()
    attempts = successes = 0
    cap = 2 * args.episodes + 2
    while successes < args.episodes and attempts < cap:
        seed = args.seed + attempts
        attempts += 1
        states, actions, cmds, lifted, placed, place_err = c.record_expert(scene, seed)
        log({"kind": "expert", "setting": args.name, "seed": seed, "placed": bool(placed), "lifted": bool(lifted),
             "place_error_mm": round(1000 * place_err, 1), "ticks": len(actions)})
        print(f"[{args.name}] expert seed={seed}: placed={placed} err={1000 * place_err:.1f}mm T={len(actions)} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not placed:
            continue
        successes += 1
        for sigma in args.noise:
            r = c.replay(scene, seed, states, actions, cmds, f"chunkvn:w{args.window}", float(args.block), "raw",
                         (0.45, 0.25), noise_sigma=sigma, cvels=None)
            r.update(kind="replay", setting=args.name, seed=seed, noise=sigma, blowup=is_blowup(r))
            log(r)
            print(f"[{args.name}] seed={seed} sigma={sigma}: placed={r['placed']} blowup={r['blowup']} "
                  f"zmax={r['max_cube_z']:.3f} err={r['place_error_mm']}mm ({time.time() - t0:.0f}s)", flush=True)
    log({"kind": "summary", "setting": args.name, "expert_attempts": attempts, "expert_successes": successes})
    print(f"[{args.name}] done: expert {successes}/{attempts} placed", flush=True)


try:
    main()
except BaseException:
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
