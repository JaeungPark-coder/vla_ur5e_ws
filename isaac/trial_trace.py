"""Per-tick trace and blow-up verdict for one evaluation trial -- pure Python/numpy, no Isaac or ROS import
(unit-tested in test/test_trial_trace.py).

Why (2026-11-11): the closed-loop evaluation of a trained policy must say, for every trial, not only "placed or
not" but "was it a PhysX contact blow-up", and a later diagnosis (which contact pair, how deep, where the gripper
was) must not need the trial to be run again. The bridge therefore records EVERY sim tick of a trial -- cube
position, grip-point position, joints, gripper -- and judges it with criteria fixed BEFORE the run:

  the criteria live in src/vla_bridge/vla_bridge/blowup_criteria.py (2026-11-17: one definition shared with the client's
  results.csv column): z_high, jump, fell, far, nan.  Heights are judged relative to the table top (table_z).

`blowup` = any of them. The verdict carries the first tick that tripped, the context at that tick (cube, grip point,
gripper, joints, previous cube position) and the largest one-tick cube displacement of the whole trial.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "vla_bridge"))
from vla_bridge.blowup_criteria import (  # noqa: E402,F401  (re-exported: tests and older scripts import them from here)
    BLOWUP_Z_M, FAR_M, FELL_Z_M, JUMP_M, TASK_CENTRE_XY, judge_tick, reasons_to_mask)

SAVE_EVERY_TICKS = 600


class TrialTrace:
    def __init__(self, out_dir=None, table_z=0.0):
        self.out_dir = out_dir
        self.table_z = float(table_z)   # criteria heights are above the table top
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.epoch = 0
        self._dirty = False
        self.start(0)

    def start(self, epoch, cube=None, grip=None, joints=None, gripper=0.0):
        """Begin a trial (epoch = the bridge's reset counter). Optionally records the initial state as tick 0's
        predecessor so the first real tick's displacement is measured against it."""
        self.epoch = int(epoch)
        self.request_id, self.cube, self.grip, self.joints, self.gripper = [], [], [], [], []
        self.reasons_seen = {}
        self.img_tick, self.img_base, self.img_wrist = [], [], []
        self.target = []
        self.first_trip = None
        self.max_jump = 0.0
        self.max_z = -np.inf
        self._prev = None if cube is None else np.asarray(cube, dtype=float)
        self._dirty = False

    @property
    def ticks(self):
        return len(self.cube)

    def add(self, request_id, cube, grip, joints, gripper, target=None):
        cube = np.asarray(cube, dtype=float)
        jump = 0.0 if self._prev is None else float(np.linalg.norm(cube - self._prev))
        if np.isfinite(jump):
            self.max_jump = max(self.max_jump, jump)
        if np.isfinite(cube[2]):
            self.max_z = max(self.max_z, float(cube[2]))
        reasons = judge_tick(cube - [0.0, 0.0, self.table_z], None if self._prev is None else self._prev - [0.0, 0.0, self.table_z])
        for r in reasons:
            self.reasons_seen[r] = self.reasons_seen.get(r, 0) + 1
        if reasons and self.first_trip is None:
            self.first_trip = {
                "tick": self.ticks, "request_id": int(request_id), "reasons": reasons,
                "cube": cube.tolist(), "prev_cube": None if self._prev is None else self._prev.tolist(),
                "grip_point": np.asarray(grip, dtype=float).tolist(),
                "gripper": float(gripper), "joints": np.asarray(joints, dtype=float).tolist(),
                "jump_m": jump,
            }
        self.request_id.append(int(request_id))
        self.cube.append(cube)
        self.grip.append(np.asarray(grip, dtype=float))
        self.joints.append(np.asarray(joints, dtype=float))
        self.gripper.append(float(gripper))
        if target is not None:   # the commanded joint targets (6) + gripper target of this tick, to tell command from physics
            self.target.append(np.asarray(target, dtype=float))
        self._prev = cube
        self._dirty = True
        if self.out_dir and self.ticks % SAVE_EVERY_TICKS == 0:
            self.save()

    def add_images(self, tick, base_rgb, wrist_rgb):
        """Keep the two camera images the policy was given at this tick (uint8 HxWx3, the bridge's own arrays). Call it on the ticks
        that publish images (every K-th with execute_horizon K = exactly what the policy saw); saved to trial_NNN_images.npz."""
        self.img_tick.append(int(tick))
        self.img_base.append(np.array(base_rgb, dtype=np.uint8))
        self.img_wrist.append(np.array(wrist_rgb, dtype=np.uint8))
        self._dirty = True

    @property
    def reasons_mask(self):
        return reasons_to_mask(self.reasons_seen)

    def summary(self):
        return {
            "epoch": self.epoch, "trial": self.epoch + 1, "ticks": self.ticks,
            "blowup": bool(self.reasons_seen), "reasons": self.reasons_seen,
            "max_cube_z": None if not np.isfinite(self.max_z) else round(float(self.max_z), 4),
            "max_one_tick_jump_m": round(self.max_jump, 5), "first_trip": self.first_trip,
        }

    def save(self):
        """Overwrites trial_<epoch>.npz and .json (idempotent: safe to call whenever)."""
        self._dirty = False
        if not self.out_dir or not self.ticks:
            return None
        stem = os.path.join(self.out_dir, f"trial_{self.epoch + 1:03d}")
        np.savez_compressed(stem + ".npz", request_id=np.asarray(self.request_id), cube=np.asarray(self.cube),
                            grip_point=np.asarray(self.grip), joints=np.asarray(self.joints),
                            gripper=np.asarray(self.gripper),
                            **({"target": np.asarray(self.target)} if len(self.target) == len(self.cube) else {}))
        with open(stem + ".json", "w") as f:
            json.dump(self.summary(), f)
        if self.img_tick:
            np.savez_compressed(stem + "_images.npz", tick=np.asarray(self.img_tick), base=np.asarray(self.img_base),
                                wrist=np.asarray(self.img_wrist))
        return stem

    def flush_if_dirty(self):
        if self._dirty:
            self.save()
