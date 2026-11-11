"""A fake policy for exercising vla_policy_client end to end WITHOUT a trained model or
an openpi server.

`vla_policy_client` imports `openpi_client.websocket_client_policy.WebsocketClientPolicy`
lazily in __init__, so putting this directory first on PYTHONPATH swaps it in. It plays a
"perfect policy": it finds where in a recorded episode the arm currently is (nearest recorded
state, searched forward from the last match) and returns the next `chunk_len` recorded
actions as ABSOLUTE joint rows plus the gripper -- the shape and meaning of what the real
server returns after AbsoluteActions.

Modes (environment variables):
  default                       replay dataset episode $FAKE_POLICY_EPISODE (cannot succeed at the task: the cube is
                                somewhere else than in the recording; checks the plumbing only)
  FAKE_POLICY_NPZ=a.npz,b.npz   replay episodes recorded by isaac/record_expert_npz.py instead, one per "lift" trial
  FAKE_POLICY_PLAN=lift,hold,.. per eval trial: "lift" replays the next npz episode (the cube is lifted and, for a good
                                recording, placed), "hold" returns the current joint state again and again (the arm never
                                moves, the cube is never touched). Default: all lift.
  FAKE_POLICY_HOLD_CALLS=N      how many infer() calls a "hold" trial lasts (= ceil(max_steps / execute_horizon)); the
                                client gives a hold trial no other end than max_steps
Trial changes are detected from the arm: after a lift trial the scene reset puts the arm back at the home pose.
Every call is appended to $FAKE_POLICY_LOG as one JSON line.
"""
import glob
import json
import os

import numpy as np

_DATASET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/vla_ur5e_ws/ur5e_pick_place_v2/data/chunk-*/*.parquet")


def _dataset_episode(episode):
    import pandas as pd
    frames = [pd.read_parquet(p, columns=["episode_index", "joints", "gripper", "actions"])
              for p in sorted(glob.glob(_DATASET_GLOB))]
    ep = pd.concat(frames)
    ep = ep[ep["episode_index"] == episode]
    states = np.concatenate(
        [np.stack(ep["joints"].values), np.stack(ep["gripper"].values).reshape(len(ep), -1)], 1).astype(float)
    return states, np.stack(ep["actions"].values).astype(float)


class WebsocketClientPolicy:
    def __init__(self, host="localhost", port=8000, chunk_len=50):
        npz = [p for p in os.environ.get("FAKE_POLICY_NPZ", "").split(",") if p]
        if npz:
            self.episodes = []
            for p in npz:
                z = np.load(p)
                self.episodes.append((z["states"].astype(float), z["actions"].astype(float)))
        else:
            self.episodes = [_dataset_episode(int(os.environ.get("FAKE_POLICY_EPISODE", "0")))]
        self.plan = [m for m in os.environ.get("FAKE_POLICY_PLAN", "lift").split(",") if m]
        self.hold_calls = int(os.environ.get("FAKE_POLICY_HOLD_CALLS", "0"))
        self.chunk_len = chunk_len
        self.home = self.episodes[0][0][0, :6].copy()
        self.trial = 0
        self.lift_count = 0       # lift trials started so far (selects the npz episode)
        self.last = 0
        self.hold_seen = 0
        self.calls = 0
        self.log_path = os.environ.get("FAKE_POLICY_LOG")

    def _mode(self):
        return self.plan[min(self.trial, len(self.plan) - 1)] if len(self.plan) > 1 else self.plan[0]

    def _next_trial(self):
        self.trial += 1
        self.last = 0
        self.hold_seen = 0

    def infer(self, obs):
        q = np.concatenate([np.asarray(obs["joints"], dtype=float), np.asarray(obs["gripper"], dtype=float)])
        at_home = bool(np.abs(q[:6] - self.home).max() < 0.03)
        mode = self._mode()
        if mode == "lift" and self.last > 100 and at_home:        # the scene was reset after a lift trial
            self._next_trial()
            mode = self._mode()
        if mode == "hold" and self.hold_calls and self.hold_seen >= self.hold_calls:
            self._next_trial()
            mode = self._mode()

        dist = 0.0
        if mode == "hold":
            self.hold_seen += 1
            rows = np.repeat(np.concatenate([q[:6], [0.0]])[None, :], self.chunk_len, 0)
            idx = -1
        else:
            states, actions = self.episodes[min(self.lift_count_for_trial(), len(self.episodes) - 1)]
            idx, dist = self._match(states, q, self.last, 400)
            if dist > 0.3:
                idx, dist = self._match(states, q, 0, len(states))
            self.last = idx
            rows = actions[idx:idx + self.chunk_len]
            if len(rows) < self.chunk_len:
                rows = np.concatenate([rows, np.repeat(actions[-1:], self.chunk_len - len(rows), 0)])
        self.calls += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"call": self.calls, "trial": self.trial, "mode": mode, "match_index": idx,
                                    "arm_vs_recorded_max_rad": round(dist, 4), "prompt": obs.get("prompt"),
                                    "base_shape": list(np.asarray(obs["base_rgb"]).shape),
                                    "wrist_shape": list(np.asarray(obs["wrist_rgb"]).shape)}) + "\n")
        return {"actions": rows}

    @staticmethod
    def _match(states, q, start, span):
        """Where in the recording is the arm? Distance = max over the 6 joints and the gripper. The recordings contain
        stretches where the arm waits (the expert dwells for 90+ ticks while the gripper closes): all those frames look
        alike, and returning the FIRST of them makes the next 50 rows a standstill again, forever. Among near-ties
        (within 2 mrad of the best) in the first 150 frames of the tie run, return the LAST one."""
        win = states[start:start + span]
        d = np.maximum(np.abs(win[:, :6] - q[:6]).max(1), np.abs(win[:, 6] - q[6]))
        dmin = float(d.min())
        ties = np.nonzero(d <= dmin + 0.002)[0]
        ties = ties[ties <= ties[0] + 150]
        return start + int(ties.max()), dmin

    def lift_count_for_trial(self):
        """Index of the npz episode for the current trial = number of lift trials before it."""
        return sum(1 for m in self.plan[:self.trial] if m == "lift") if len(self.plan) > 1 else self.trial

    def get_server_metadata(self):
        return {}
