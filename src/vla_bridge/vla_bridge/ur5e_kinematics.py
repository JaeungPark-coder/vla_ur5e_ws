"""UR5e forward kinematics (standard DH, the controller's own convention), numpy only.

The same parameters as scripts/ur5e_fk.py.  URSim 5.26 reported the flange pose at exactly these values on 5 joint vectors
(0.00 mm, 0.00 deg; README "blocker confirmed on URSim").  Frames are the UR base frame (what getActualTCPPose reports),
not the Isaac world: Isaac world = UR base rotated 180 deg about z.
"""
import numpy as np

D = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
A = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)

# UR5e datasheet: every joint 180 deg/s.
MAX_JOINT_SPEED_RAD_S = np.full(6, np.pi)


def flange(q):
    """q (6,) or (N,6) -> flange (tool0, TCP offset 0) pose (N,4,4) in the UR base frame."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
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


def tool_point(q, offset_in_flange=(0.0, 0.0, 0.16)):
    """Position (N,3), UR base frame, of a point fixed in the flange frame.  A real gripper on the tool flange is coaxial:
    offset (0, 0, TCP length).  The v1-v4 simulated gripper sat at (0, 0.12, 0), 90 deg off (README BLOCKER)."""
    T = flange(q)
    return T[:, :3, 3] + T[:, :3, :3] @ np.asarray(offset_in_flange, dtype=float)
