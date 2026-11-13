#!/usr/bin/env python3
"""Numpy UR5e forward kinematics (standard DH), with the world placement and tool offset fitted to the bridge's measured grip point.

Why (2026-11-14): the traces log the commanded joint targets (`target`) next to the measured joints and grip point. To ask "was the
hand commanded too low, or commanded right but measured low?" we need the grip-point height of the *commanded* joints, which Isaac's
Lula solver could give but only inside Kit. This is GPU-free: fit(world yaw, translation, tool offset in the flange frame) once on measured
(joints, grip_point) pairs, check the residual, then apply the same map to the commanded joints.
"""
import numpy as np
from scipy.optimize import least_squares

D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]


def flange(q):
    """q (N,6) -> flange pose (N,4,4) in the DH base frame."""
    q = np.atleast_2d(q)
    T = np.tile(np.eye(4), (len(q), 1, 1))
    for i in range(6):
        ct, st = np.cos(q[:, i]), np.sin(q[:, i])
        ca, sa = np.cos(ALPHA[i]), np.sin(ALPHA[i])
        M = np.zeros((len(q), 4, 4))
        M[:, 0, 0], M[:, 0, 1], M[:, 0, 2], M[:, 0, 3] = ct, -st * ca, st * sa, A[i] * ct
        M[:, 1, 0], M[:, 1, 1], M[:, 1, 2], M[:, 1, 3] = st, ct * ca, -ct * sa, A[i] * st
        M[:, 2, 1], M[:, 2, 2], M[:, 2, 3] = sa, ca, D[i]
        M[:, 3, 3] = 1.0
        T = T @ M
    return T


def apply(p, q):
    """p = (yaw, tx, ty, tz, tool_x, tool_y, tool_z) -> grip point in world (N,3); tool offset is in the flange frame."""
    T = flange(q)
    tip = T[:, :3, 3] + T[:, :3, :3] @ np.asarray(p[4:7])
    c, s = np.cos(p[0]), np.sin(p[0])
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return tip @ R.T + p[1:4]


def fit(q, grip):
    ok = np.isfinite(q).all(1) & np.isfinite(grip).all(1) & (np.abs(q).max(1) < 20)
    q, grip = q[ok], grip[ok]
    best = None
    for yaw0 in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        r = least_squares(lambda p: (apply(p, q) - grip).ravel(), [yaw0, 0, 0, 0, 0, 0, 0.15])
        if best is None or r.cost < best.cost:
            best = r
    res = np.linalg.norm(apply(best.x, q) - grip, axis=1)
    return best.x, float(np.median(res)), float(res.max())
