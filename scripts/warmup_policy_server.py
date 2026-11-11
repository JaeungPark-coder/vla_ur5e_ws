"""Sends 3 dummy requests to a served policy so the ~22 s JAX compile happens before the first real trial.
Run with the openpi venv: (cd $OPENPI && uv run --no-sync python <repo>/scripts/warmup_policy_server.py)"""
import time, numpy as np
from openpi_client import websocket_client_policy as w
p = w.WebsocketClientPolicy("localhost", 8000)
obs = {"joints": np.zeros(6, np.float32), "gripper": np.zeros(1, np.float32),
       "base_rgb": np.zeros((256, 256, 3), np.uint8), "wrist_rgb": np.zeros((256, 256, 3), np.uint8),
       "prompt": "pick up the cube and place it in the target zone"}
for i in range(3):
    t = time.time(); r = p.infer(obs); print(i, round(time.time() - t, 2), np.asarray(r["actions"]).shape, flush=True)
