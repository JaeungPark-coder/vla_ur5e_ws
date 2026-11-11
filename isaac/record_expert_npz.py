"""Record scripted-expert episodes at given seeds and save each as an .npz (states, actions, outcome).

For the fake-policy check of the evaluation harness (test/fakes/openpi_client): the bridge started with
VLA_BRIDGE_SEED=S spawns the cube of trial k at default_rng(S + k), exactly what record_expert(scene, S + k) does, so an
episode recorded here replays in the bridge on the same cube. The stored max_cube_z / placed are the ground truth the
evaluation's per-tick cube height is compared with.

    scripts/isaac_py.sh record_expert_npz.py --seeds 7000 7002 --out /tmp/expert_
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--out", required=True, help="path prefix; writes <prefix><seed>.npz")
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)

scene = c.build_scene()
for seed in args.seeds:
    states, actions, cmds, lifted, placed, place_err = c.record_expert(scene, seed)
    z = c.record_expert.last_max_cube_z
    path = f"{args.out}{seed}.npz"
    np.savez(path, states=states, actions=actions, seed=seed, lifted=bool(lifted), placed=bool(placed),
             max_cube_z=z, place_error_m=place_err)
    print(f"RECORDED seed={seed} T={len(actions)} lifted={lifted} placed={placed} max_cube_z={z:.3f} "
          f"place_err={1000 * place_err:.1f}mm -> {path}", flush=True)
c.simulation_app.close()
