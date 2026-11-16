#!/usr/bin/env python3
"""Joint-convention and tool-axis check against URSim (Universal Robots' official controller simulator), no real robot.

Sends joint vectors taken from the simulator's demonstrations/evaluations (home and the grasp moments) to a URSim UR5e,
reads back the controller's own tool-flange pose (TCP offset 0), and compares it with
  * scripts/ur5e_fk.py (standard UR5e DH): same zero and direction conventions if the flange positions agree;
  * the Isaac grip point at the same joints: where the simulated fingertips were;
  * a coaxial real gripper: fingertips on the flange z axis at --tcp_m.
URSim:  docker run -d --name ursim_check -e ROBOT_MODEL=UR5 -p 127.0.0.1:29999:29999 -p 127.0.0.1:30001-30004:30001-30004 \
          universalrobots/ursim_e-series ;  dashboard: "power on", "brake release"
Run with a Python that has ur_rtde:   <venv>/bin/python scripts/ursim_kinematics_check.py [--host 127.0.0.1] [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ur5e_fk import flange  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--tcp_m", type=float, default=0.16, help="coaxial real gripper: fingertip distance along the flange z")
ap.add_argument("--n_grasps", type=int, default=4)
ap.add_argument("--json", default=None)
args = ap.parse_args()

import socket  # noqa: E402
import time  # noqa: E402

import rtde_receive  # noqa: E402

WORLD_FROM_BASE = np.diag([-1.0, -1.0, 1.0])   # Isaac world = UR base rotated 180 deg about z (fitted, residual 0.000 mm)


def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


poses = [("home q=0 (every episode starts here)", np.zeros(6), None)]
root = os.path.join(os.path.dirname(HERE), "eval_runs", "strat_k25_v3_3k_centre", "trace")
for p in [p for p in sorted(glob.glob(os.path.join(root, "trial_0*.npz"))) if "images" not in p]:
    z = np.load(p)
    q, g, grip = z["joints"][:, :6].astype(float), z["gripper"].astype(float), z["grip_point"].astype(float)
    c = np.nonzero(g > 0.5)[0]
    if len(c) and np.abs(q[c[0]]).max() < 7:
        poses.append((f"grasp, {os.path.basename(p)} tick {c[0]}", q[c[0]], grip[c[0]]))
    if len(poses) > args.n_grasps:
        break

recv = rtde_receive.RTDEReceiveInterface(args.host)


def urscript(line):
    """URScript over the secondary interface (30002): works with Remote Control off, where RTDEControlInterface cannot start."""
    s = socket.create_connection((args.host, 30002), timeout=3)
    s.sendall((line + "\n").encode())
    s.close()


urscript("set_tcp(p[0,0,0,0,0,0])")
time.sleep(0.5)
out = []
for name, q, sim_grip in poses:
    urscript("movej([" + ",".join(f"{v:.6f}" for v in q) + "], a=1.4, v=1.0)")
    t0 = time.time()
    while time.time() - t0 < 20:
        time.sleep(0.3)
        if np.abs(np.array(recv.getActualQ()) - q).max() < 1e-3 and np.abs(recv.getActualQd()).max() < 1e-4:
            break
    ok = np.abs(np.array(recv.getActualQ()) - q).max() < 1e-3
    qa = np.array(recv.getActualQ())
    tcp = np.array(recv.getActualTCPPose())
    R = rotvec_to_R(tcp[3:])
    T = flange(q)[0]
    z_ur, z_dh = R[:, 2], T[:3, 2]
    tip_coax_world = WORLD_FROM_BASE @ (tcp[:3] + args.tcp_m * z_ur)
    flange_world = WORLD_FROM_BASE @ tcp[:3]
    z_world = WORLD_FROM_BASE @ z_ur
    r = dict(name=name, q=q.round(4).tolist(), moved=bool(ok), q_err_max=float(np.abs(qa - q).max()),
             flange_base_ursim=tcp[:3].round(4).tolist(), flange_base_dh=T[:3, 3].round(4).tolist(),
             flange_pos_diff_mm=float(np.linalg.norm(tcp[:3] - T[:3, 3]) * 1000),
             flange_axes_diff_deg=float(np.degrees(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1)))),
             tool_axis_world=z_world.round(3).tolist(), tool_axis_from_down_deg=float(np.degrees(np.arccos(np.clip(-z_world[2], -1, 1)))),
             flange_world=flange_world.round(4).tolist(), coaxial_tip_world=tip_coax_world.round(4).tolist())
    if sim_grip is not None:
        r["isaac_fingertip_world"] = sim_grip.round(4).tolist()
        r["coaxial_tip_vs_isaac_mm"] = float(np.linalg.norm(tip_coax_world - sim_grip) * 1000)
    out.append(r)
    print(f"\n{name}\n  q = {np.round(q, 3).tolist()}  (reached, max err {r['q_err_max']:.4f} rad)"
          f"\n  flange (UR base): URSim {r['flange_base_ursim']}  DH {r['flange_base_dh']}  -> {r['flange_pos_diff_mm']:.2f} mm, "
          f"axes {r['flange_axes_diff_deg']:.2f} deg"
          f"\n  tool axis (flange z) in Isaac world {r['tool_axis_world']}: {r['tool_axis_from_down_deg']:.1f} deg from straight down")
    if sim_grip is not None:
        print(f"  Isaac fingertips {r['isaac_fingertip_world']}  vs  coaxial real gripper ({args.tcp_m} m) {r['coaxial_tip_world']}"
              f"  -> {r['coaxial_tip_vs_isaac_mm']:.0f} mm apart")
if args.json:
    json.dump(dict(ursim_host=args.host, tcp_m=args.tcp_m, poses=out), open(args.json, "w"), indent=1)
