#!/usr/bin/env python3
"""Ready-pose candidates for v5 and the real robot: tool pointing straight down, elbow bent up, away from the three UR
singularities, above the work area -- and whether every grasp and the place point are reachable from it on the same
IK branch (no elbow or wrist flip on the way).  Optionally checks each candidate on URSim (moveJ there, read back).

Why (2026-11-16): v1-v4 start every episode at q = 0, the arm stretched out horizontally (a shoulder/elbow singularity, and
not a pose anyone would start a real robot in).  v5 starts where the real robot will: a ready pose taught on the pendant.

Frames: UR base frame (the pendant's).  Isaac world = UR base rotated 180 deg about z.  Workspace (scene constants, in the UR
base frame): spawn box X -0.48..-0.38, Y -0.16..0.16 (wide box); place (-0.45, -0.30); table z = sim_profile TABLE_Z_M.

    python3 scripts/ready_pose_candidates.py [--tcp_m 0.16] [--table_z 0] [--json out.json]
    <python with ur_rtde> scripts/ready_pose_candidates.py --ursim 127.0.0.1 ...     # also move URSim to each
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'vla_bridge'))
from vla_bridge.ur5e_kinematics import flange  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tcp_m', type=float, default=0.16, help='flange to fingertip (sim 2F-85 0.12; real gripper+coupling longer)')
ap.add_argument('--table_z', type=float, default=0.0)
ap.add_argument('--cube_m', type=float, default=0.04)
ap.add_argument('--ursim', default=None, help='URSim host: moveJ to each candidate and its grasp/place poses, read back')
ap.add_argument('--speed', type=float, default=0.6, help='URSim moveJ speed, rad/s')
ap.add_argument('--json', default=None)
a = ap.parse_args()

DOWN = np.array([0.0, 0.0, -1.0])
SPAWN_X, SPAWN_Y = (-0.48, -0.38), (-0.16, 0.16)        # wide box in the UR base frame (world X 0.38..0.48, Y +-0.16)
PLACE = np.array([-0.45, -0.30])
# (name, fingertip x, y, height of the fingertip above the table)
CANDIDATES = [('A_over_spawn', -0.45, 0.00, 0.25), ('B_spawn_to_place', -0.42, -0.12, 0.25), ('C_pulled_back', -0.32, -0.08, 0.30)]


def rot_err(F, R):
    return 0.5 * sum(np.cross(F[:3, i], R[:, i]) for i in range(3))


def jac(q):
    F = flange(q)[0]
    J = np.zeros((6, 6))
    for j in range(6):
        dq = np.zeros(6)
        dq[j] = 1e-6
        G = flange(q + dq)[0]
        J[:3, j] = (G[:3, 3] - F[:3, 3]) / 1e-6
        Rd = G[:3, :3] @ F[:3, :3].T
        J[3:, j] = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / 2e-6
    return F, J


def ik(p, R, seed, iters=300, damping=1e-6):
    q = np.array(seed, float)
    for _ in range(iters):
        F, J = jac(q)
        e = np.concatenate([p - F[:3, 3], rot_err(F, R)])
        if np.linalg.norm(e) < 1e-9:
            break
        q = q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
    F = flange(q)[0]
    return q, float(np.linalg.norm(p - F[:3, 3])), float(np.linalg.norm(rot_err(F, R)))


def down_rotation(yaw_x):
    """Tool z straight down, tool x along yaw_x (projected to horizontal)."""
    x = np.array([yaw_x[0], yaw_x[1], 0.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(DOWN, x), DOWN])


def flange_target(tip_xy, tip_h):
    return np.array([tip_xy[0], tip_xy[1], a.table_z + tip_h + a.tcp_m])


def describe(q):
    """Singularity distances and branch of a joint vector."""
    T = flange(q)[0]
    wrist_centre = T[:3, 3] - D6 * T[:3, 2]
    _, J = jac(q)
    return dict(axis_from_down_deg=float(np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))),
                elbow_sin=float(np.sin(q[2])), wrist_sin_q5=float(np.sin(q[4])),
                shoulder_radius_m=float(np.hypot(wrist_centre[0], wrist_centre[1])),
                manipulability=float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0))),
                elbow_up=bool(q[2] > 0), branch=(int(np.sign(q[2])), int(np.sign(np.sin(q[4])))))


D6 = 0.0996
out = dict(tcp_m=a.tcp_m, table_z=a.table_z, frame='UR base', candidates=[])
for name, x, y, h in CANDIDATES:
    q1 = np.arctan2(-y, -x) - np.arctan2(0.133, np.hypot(x, y))      # face the point; 0.133 = the lateral wrist offset
    seed = np.array([q1, -1.8, 1.9, -np.pi / 2 - (-1.8 + 1.9), -np.pi / 2, 0.0])
    R = down_rotation(flange(seed)[0][:3, 0])                          # yaw of the canonical pose: q6 stays near 0
    q, perr, rerr = ik(flange_target((x, y), h), R, seed)
    rec = dict(name=name, fingertip_target=[x, y, a.table_z + h], q=np.round(q, 4).tolist(), ik_pos_err_m=perr,
               ik_rot_err=rerr, **describe(q))
    # every grasp corner (fingertip at the cube centre) and the place point (fingertip one cube above the table), same yaw
    reach = []
    cz = a.cube_m / 2
    targets = [(f'spawn({sx:+.2f},{sy:+.2f})', (sx, sy), cz) for sx in SPAWN_X for sy in SPAWN_Y] + \
              [('spawn_centre', (np.mean(SPAWN_X), 0.0), cz), ('place', tuple(PLACE), a.cube_m + 0.01)]
    for tname, xy, th in targets:
        qg, pe, re = ik(flange_target(xy, th), R, q)
        d = describe(qg)
        reach.append(dict(target=tname, q=np.round(qg, 4).tolist(), ok=bool(pe < 1e-4 and re < 1e-4),
                          same_branch=d['branch'] == rec['branch'], max_joint_move_rad=float(np.abs(qg - q).max()),
                          time_at_pi_over_2_s=float(np.abs(qg - q).max() / (np.pi / 2)),
                          elbow_sin=d['elbow_sin'], wrist_sin_q5=d['wrist_sin_q5'], shoulder_radius_m=d['shoulder_radius_m']))
    rec['reach'] = reach
    rec['checks'] = dict(
        tool_down=rec['axis_from_down_deg'] < 0.5, elbow_bent_up=rec['elbow_up'] and abs(rec['elbow_sin']) > 0.5,
        wrist_clear=abs(rec['wrist_sin_q5']) > 0.5, shoulder_clear=rec['shoulder_radius_m'] > 0.20,
        all_targets_reachable=all(r['ok'] for r in reach), no_branch_flip=all(r['same_branch'] for r in reach),
        targets_clear_of_singularities=all(abs(r['wrist_sin_q5']) > 0.5 and abs(r['elbow_sin']) > 0.2
                                           and r['shoulder_radius_m'] > 0.15 for r in reach))
    rec['pass'] = all(rec['checks'].values())
    out['candidates'].append(rec)

if a.ursim:
    import time
    import rtde_control
    import rtde_receive
    from scipy.spatial.transform import Rotation as Rot
    ctrl = rtde_control.RTDEControlInterface(a.ursim)
    recv = rtde_receive.RTDEReceiveInterface(a.ursim)
    ctrl.setTcp([0, 0, 0, 0, 0, 0])
    for rec in out['candidates']:
        rows = []
        for label, qt in [('ready', rec['q'])] + [(r['target'], r['q']) for r in rec['reach']] + [('ready_again', rec['q'])]:
            ok = ctrl.moveJ(list(qt), a.speed, 0.8)
            time.sleep(0.2)
            qa = np.array(recv.getActualQ())
            pose = np.array(recv.getActualTCPPose())
            z = Rot.from_rotvec(pose[3:]).as_matrix()[:, 2]
            fk = flange(qa)[0]
            rows.append(dict(at=label, moveJ_ok=bool(ok), joint_err_rad=float(np.abs(qa - qt).max()),
                             flange_vs_fk_mm=float(np.linalg.norm(pose[:3] - fk[:3, 3]) * 1000),
                             axis_from_down_deg=float(np.degrees(np.arccos(np.clip(-z[2], -1, 1)))),
                             safety_mode=int(recv.getSafetyMode()), robot_mode=int(recv.getRobotMode())))
            if (rec['name'], label) == (out['candidates'][0]['name'], 'ready'):
                pass
        rec['ursim'] = rows
        rec['ursim_pass'] = all(r['moveJ_ok'] and r['joint_err_rad'] < 1e-3 and r['flange_vs_fk_mm'] < 0.5
                                and r['axis_from_down_deg'] < 0.5 and r['safety_mode'] == 1 for r in rows)
    ctrl.stopScript()

for rec in out['candidates']:
    print(f"{rec['name']}: q={rec['q']}  pass={rec['pass']}" + (f"  ursim_pass={rec['ursim_pass']}" if 'ursim' in rec else ''))
    print('   ', {k: v for k, v in rec['checks'].items()})
    print(f"    axis {rec['axis_from_down_deg']:.3f} deg, sin(q3) {rec['elbow_sin']:.2f}, sin(q5) {rec['wrist_sin_q5']:.2f}, "
          f"shoulder radius {rec['shoulder_radius_m']:.3f} m, manipulability {rec['manipulability']:.3f}")
    print('    to targets: max joint move', round(max(r['max_joint_move_rad'] for r in rec['reach']), 3), 'rad,',
          'min sin(q5)', round(min(abs(r['wrist_sin_q5']) for r in rec['reach']), 2),
          'min shoulder radius', round(min(r['shoulder_radius_m'] for r in rec['reach']), 3))
if a.json:
    with open(a.json, 'w') as f:
        json.dump(out, f, indent=1)
