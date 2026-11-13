"""Per-tick trace and blow-up verdict for one evaluation trial -- pure Python/numpy, no Isaac or ROS import
(unit-tested in test/test_trial_trace.py).

Why (2026-11-11): the closed-loop evaluation of a trained policy must say, for every trial, not only "placed or
not" but "was it a PhysX contact blow-up", and a later diagnosis (which contact pair, how deep, where the gripper
was) must not need the trial to be run again. The bridge therefore records EVERY sim tick of a trial -- cube
position, grip-point position, joints, gripper -- and judges it with criteria fixed BEFORE the run:

  z_high   cube centre above BLOWUP_Z_M       (a normal episode peaks near 0.16 m; collect_demos drops > 0.25 m)
  jump     cube moved more than JUMP_M in ONE tick (1.2 m/s at 60 Hz; the arm carries it at 2-4 mm/tick and the
           cube's maxDepenetrationVelocity is capped at 0.5 m/s = 8 mm/tick, so anything above is not a push)
  fell     cube centre below FELL_Z_M         (rests at 0.02 m; below the table top = off the edge / through it)
  far      cube centre farther than FAR_M in the table plane from the task's centre (everything the task uses lies
           within ~0.3 m of it)
  nan      a non-finite cube coordinate

`blowup` = any of them. The verdict carries the first tick that tripped, the context at that tick (cube, grip point,
gripper, joints, previous cube position) and the largest one-tick cube displacement of the whole trial.
"""
import json
import os

import numpy as np

BLOWUP_Z_M = 0.25
JUMP_M = 0.02
FELL_Z_M = 0.0
FAR_M = 0.5
TASK_CENTRE_XY = (0.45, 0.10)
SAVE_EVERY_TICKS = 600


def judge_tick(cube, prev_cube):
    """Reasons ([] = fine) that this tick's cube position trips a criterion."""
    cube = np.asarray(cube, dtype=float)
    reasons = []
    if not np.all(np.isfinite(cube)):
        return ["nan"]
    if cube[2] > BLOWUP_Z_M:
        reasons.append("z_high")
    if cube[2] < FELL_Z_M:
        reasons.append("fell")
    if float(np.hypot(cube[0] - TASK_CENTRE_XY[0], cube[1] - TASK_CENTRE_XY[1])) > FAR_M:
        reasons.append("far")
    if prev_cube is not None and float(np.linalg.norm(cube - np.asarray(prev_cube, dtype=float))) > JUMP_M:
        reasons.append("jump")
    return reasons


class TrialTrace:
    def __init__(self, out_dir=None):
        self.out_dir = out_dir
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
        self.first_trip = None
        self.max_jump = 0.0
        self.max_z = -np.inf
        self._prev = None if cube is None else np.asarray(cube, dtype=float)
        self._dirty = False

    @property
    def ticks(self):
        return len(self.cube)

    def add(self, request_id, cube, grip, joints, gripper):
        cube = np.asarray(cube, dtype=float)
        jump = 0.0 if self._prev is None else float(np.linalg.norm(cube - self._prev))
        if np.isfinite(jump):
            self.max_jump = max(self.max_jump, jump)
        if np.isfinite(cube[2]):
            self.max_z = max(self.max_z, float(cube[2]))
        reasons = judge_tick(cube, self._prev)
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
                            gripper=np.asarray(self.gripper))
        with open(stem + ".json", "w") as f:
            json.dump(self.summary(), f)
        if self.img_tick:
            np.savez_compressed(stem + "_images.npz", tick=np.asarray(self.img_tick), base=np.asarray(self.img_base),
                                wrist=np.asarray(self.img_wrist))
        return stem

    def flush_if_dirty(self):
        if self._dirty:
            self.save()
