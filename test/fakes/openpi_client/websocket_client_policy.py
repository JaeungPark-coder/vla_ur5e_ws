"""A fake policy for exercising vla_policy_client end to end WITHOUT a trained model or
an openpi server.

`vla_policy_client` imports `openpi_client.websocket_client_policy.WebsocketClientPolicy`
lazily in __init__, so putting this directory first on PYTHONPATH swaps it in. It plays a
"perfect policy": it finds where in a recorded dataset episode the arm currently is
(nearest recorded state, searched forward from the last match) and returns the next
`chunk_len` recorded actions as ABSOLUTE joint rows plus the gripper -- the shape and
meaning of what the real server returns after AbsoluteActions.

It cannot succeed at the task (the cube is somewhere else than in the recording); what it
checks is the client + bridge plumbing: chunk execution, step counting, velocity
feed-forward, lockstep acks, the per-trial reset. Every call is appended to the file named
by $FAKE_POLICY_LOG as one JSON line (match index and how far the arm was from the
recorded state at that index).

    PYTHONPATH=test/fakes:... ; FAKE_POLICY_EPISODE=0 FAKE_POLICY_LOG=/tmp/fake_policy.jsonl
"""
import glob
import json
import os

import numpy as np

_DATASET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/vla_ur5e_ws/ur5e_pick_place_v2/data/chunk-*/*.parquet")


class WebsocketClientPolicy:
    def __init__(self, host="localhost", port=8000, chunk_len=50):
        import pandas as pd
        episode = int(os.environ.get("FAKE_POLICY_EPISODE", "0"))
        frames = [pd.read_parquet(p, columns=["episode_index", "joints", "gripper", "actions"])
                  for p in sorted(glob.glob(_DATASET_GLOB))]
        df = pd.concat(frames)
        ep = df[df["episode_index"] == episode]
        self.states = np.concatenate(
            [np.stack(ep["joints"].values), np.stack(ep["gripper"].values).reshape(len(ep), -1)], 1).astype(float)
        self.actions = np.stack(ep["actions"].values).astype(float)
        self.chunk_len = chunk_len
        self.last = 0
        self.calls = 0
        self.log_path = os.environ.get("FAKE_POLICY_LOG")

    def infer(self, obs):
        q = np.concatenate([np.asarray(obs["joints"], dtype=float), np.asarray(obs["gripper"], dtype=float)])
        window = self.states[self.last:self.last + 400, :6]
        d = np.abs(window - q[:6]).max(1)
        idx = self.last + int(np.argmin(d))
        dist = float(d.min())
        if dist > 0.3:   # the scene was reset (or the arm is lost): search the whole episode
            d_all = np.abs(self.states[:, :6] - q[:6]).max(1)
            idx, dist = int(np.argmin(d_all)), float(d_all.min())
        self.last = idx
        rows = self.actions[idx:idx + self.chunk_len]
        if len(rows) < self.chunk_len:
            rows = np.concatenate([rows, np.repeat(self.actions[-1:], self.chunk_len - len(rows), 0)])
        self.calls += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"call": self.calls, "match_index": idx, "arm_vs_recorded_max_rad": round(dist, 4),
                                    "prompt": obs.get("prompt"),
                                    "base_shape": list(np.asarray(obs["base_rgb"]).shape),
                                    "wrist_shape": list(np.asarray(obs["wrist_rgb"]).shape)}) + "\n")
        return {"actions": rows}
