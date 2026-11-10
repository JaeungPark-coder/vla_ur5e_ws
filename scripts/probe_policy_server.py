"""Offline check of a SERVED policy against the dataset: latency, error vs the recorded chunk, and jitter.

Run with the openpi venv (it has lerobot and openpi_client), against a running serve_policy.py:

  # terminal 1 (note: --port goes BEFORE policy:checkpoint)
  cd $OPENPI && uv run --no-sync scripts/serve_policy.py --port 8765 policy:checkpoint \
      --policy.config=pi0_ur5e_pick_place --policy.dir=checkpoints/pi0_ur5e_pick_place/short_3k/3000
  # terminal 2
  cd $OPENPI && .venv/bin/python <repo>/scripts/probe_policy_server.py --port 8765 --out /tmp/probe.json

It sends the same observation format the ROS client node sends (uint8 RGB x2, joints, gripper, prompt). The episodes are
TRAINING episodes (openpi has no validation split), so errors are optimistic; the closed-loop simulator evaluation is
the unbiased number. Metrics: src/vla_bridge/vla_bridge/policy_metrics.py (same-t pairs; the jerk ratio only over windows with enough recorded motion).
The first request takes ~20 s (JAX compile) and is excluded from the latency statistics.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "vla_bridge"))
from vla_bridge.policy_metrics import chunk_comparison, summarize_windows  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--repo_id", default="Jaeung12/ur5e_pick_place_v2")
ap.add_argument("--episodes", type=int, nargs="*", default=[0, 25, 50, 75, 100, 125, 150])
ap.add_argument("--stride", type=int, default=100, help="ticks between probed windows")
ap.add_argument("--min_motion", type=float, default=5e-4, help="only windows whose recorded step is >= this (rad/tick) enter the jitter summary")
ap.add_argument("--out", default=None)
args = ap.parse_args()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

ds = LeRobotDataset(args.repo_id)
pol = WebsocketClientPolicy(host=args.host, port=args.port)
print("server metadata:", pol.get_server_metadata(), flush=True)

rows, lat = [], []
first = True
for e in args.episodes:
    ep = ds.meta.episodes[e]
    a0, b0 = ep["dataset_from_index"], ep["dataset_to_index"]
    acts = np.stack([np.asarray(ds[i]["actions"]) for i in range(a0, b0)]).astype(float)
    for t in range(0, len(acts) - 50, args.stride):
        s = ds[a0 + t]
        obs = {"joints": np.asarray(s["joints"], np.float32), "gripper": np.asarray(s["gripper"], np.float32).reshape(1),
               "base_rgb": (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
               "wrist_rgb": (s["wrist_image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "prompt": s["task"]}
        t0 = time.time()
        out = pol.infer(obs)
        dt = time.time() - t0
        if first:
            print(f"first request: {dt:.1f} s (compile), excluded", flush=True)
            first = False
        else:
            lat.append(dt)
        r = chunk_comparison(np.asarray(out["actions"]), acts[t:t + 50])
        r.update(episode=e, t=t)
        rows.append(r)

summary = {**summarize_windows(rows, args.min_motion), "latency_ms_median": 1000 * float(np.median(lat)) if lat else None}
print(json.dumps(summary, indent=1))
print("compare with the noise study: it used AR(1) row noise of 0.002 / 0.005 / 0.01 rad (per joint, independent); "
      "err_rms / err_row0 here are the policy's own deviation from the recording (training data -> optimistic).")
if args.out:
    json.dump({"summary": summary, "windows": rows}, open(args.out, "w"), indent=1)
