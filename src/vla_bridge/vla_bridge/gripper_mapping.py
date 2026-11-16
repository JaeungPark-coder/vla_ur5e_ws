"""Sim gripper scale (0..1) <-> Robotiq 2F-85 position counts (0..255), piecewise linear through calibration points.

The sim's 0..1 is finger_joint 0..0.80 rad (isaac_sim_common.GRIPPER_CLOSED_POS; hard stop 0.8203).  The Robotiq's counts are
roughly linear in that same joint angle, so the default -- (0, 0) and (1, 255), i.e. the old linear map -- is the right first
guess.  But the policy is fed the gripper reading as state and was trained on the sim's reading while HOLDING the cube (0.63-0.76
on the v4 data), so the point that matters is the holding one: add a calibration point (sim value holding the cube, Robotiq
counts holding the real cube of the same size), measured on both sides, and the map passes through it exactly.
"""
import numpy as np


class GripperMap:
    def __init__(self, points=((0.0, 0), (1.0, 255))):
        """points: (sim_fraction, counts) pairs, both strictly increasing."""
        pts = sorted((float(f), float(c)) for f, c in points)
        self.f = np.array([p[0] for p in pts])
        self.c = np.array([p[1] for p in pts])
        if len(pts) < 2 or np.any(np.diff(self.f) <= 0) or np.any(np.diff(self.c) <= 0):
            raise ValueError(f'gripper calibration points must be >= 2 and strictly increasing on both axes: {points}')

    @classmethod
    def from_flat(cls, flat):
        """[f0, c0, f1, c1, ...] -- the shape a ROS double-array parameter can carry."""
        flat = [float(v) for v in flat]
        if len(flat) < 4 or len(flat) % 2:
            raise ValueError(f'expected [f0, c0, f1, c1, ...], got {flat}')
        return cls(list(zip(flat[0::2], flat[1::2])))

    def to_counts(self, fraction):
        return int(round(float(np.interp(np.clip(fraction, self.f[0], self.f[-1]), self.f, self.c))))

    def to_fraction(self, counts):
        return float(np.interp(np.clip(counts, self.c[0], self.c[-1]), self.c, self.f))
