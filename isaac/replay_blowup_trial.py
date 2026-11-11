"""Replay the MEASURED joint trajectory of an evaluated trial through the simulator with PhysX contact reporting on,
to see what touched the cube at a blow-up (README 2026-11-11, learning-curve section).

Why a replay and not a re-run: the policy samples, so running it again gives another trajectory. The bridge's per-tick trace
(isaac/trial_trace.py) holds the joints and gripper the arm actually had at every tick; sending them back as targets with the
velocity feed-forward (= the "absvel" mode of check_action_replay.py, which reproduced recorded episodes to 0.0002 rad)
re-creates the arm motion, and the cube spawn is re-created from the trial's seed (VLA_BRIDGE_SEED + trial - 1).

What it reports per replay:
  * did the blow-up come back (first tick of a >2 cm one-tick cube move, versus the original trace's tick) -- if not, the
    replay is not evidence about the original;
  * every body that touched the cube: first tick, number of ticks in contact, largest impulse, deepest penetration
    (negative contact separation);
  * the per-tick contacts in a window around the first jump.

    GPU=0 scripts/isaac_py.sh replay_blowup_trial.py --out /tmp/replay.json \
        --trial eval_runs/curve_2000/trace:4:8003  --trial eval_runs/k3_2999/trace:3:8000:900
        (--trial TRACE_DIR:TRIAL:SEED[:MAX_TICKS])
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="append", required=True)
ap.add_argument("--after", type=int, default=60, help="ticks to keep replaying after the first jump")
ap.add_argument("--window", type=int, default=6, help="ticks before the first jump whose contacts are listed")
ap.add_argument("--snap", type=int, nargs="*", default=[], help="ticks at which to record the hand pose (tilt, heights)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)
from trial_trace import TrialTrace  # noqa: E402

from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import PhysxSchema  # noqa: E402
from omni.physx.scripts.physicsUtils import PhysicsSchemaTools  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from pick_place_scene import prim_world_pose, TOOL_LINK_PRIM_PATH  # noqa: E402

CUBE = "/World/cube"
state = {"tick": 0, "events": []}


def on_contact(headers, data):
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        if CUBE not in (a0, a1):
            continue
        other = a1 if a0 == CUBE else a0
        col = str(PhysicsSchemaTools.intToSdfPath(h.collider1 if a0 == CUBE else h.collider0))
        imp, sep, pts = 0.0, 1.0, []
        for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
            d = data[i]
            imp = max(imp, float(np.linalg.norm([d.impulse[0], d.impulse[1], d.impulse[2]])))
            sep = min(sep, float(d.separation))
            pts.append([float(d.position[0]), float(d.position[1]), float(d.position[2])])
        state["events"].append((state["tick"], other, col, imp, sep, str(h.type), pts))


FINGERS = {"L": "/World/ur5e/Gripper/Robotiq_2F_85/left_inner_finger", "R": "/World/ur5e/Gripper/Robotiq_2F_85/right_inner_finger"}


def hand_pose(scene, cube):
    pos, quat = prim_world_pose(scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
    zax = Rot.from_quat(quat).apply(np.array([0.0, 0.0, 1.0]))
    d = {"tilt_deg": round(float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1)))), 1), "flange": np.round(pos, 3).tolist(),
         "grip_point": np.round(scene.grip_point_world(), 3).tolist(), "cube": np.round(cube, 3).tolist()}
    for k, path in FINGERS.items():
        d["finger_" + k] = np.round(prim_world_pose(scene.stage.GetPrimAtPath(path))[0], 3).tolist()
    return d


def replay_one(scene, spec):
    parts = spec.split(":")
    trace_dir, trial, seed = parts[0], int(parts[1]), int(parts[2])
    max_ticks = int(parts[3]) if len(parts) > 3 else None
    z = np.load(os.path.join(trace_dir, f"trial_{trial:03d}.npz"))
    meta = json.load(open(os.path.join(trace_dir, f"trial_{trial:03d}.json")))
    joints, grip, cube0 = z["joints"].astype(float), z["gripper"].astype(float), z["cube"].astype(float)
    orig_trip = None if meta["first_trip"] is None else meta["first_trip"]["tick"]

    scene._rng = np.random.default_rng(seed)
    scene.reset()
    spawn = np.asarray(scene.get_cube_position(), dtype=float)
    state["events"].clear()
    trace = TrialTrace(None)
    trace.start(0, cube=spawn)
    dt = scene.physics_dt
    q_prev = np.asarray(scene.get_observation()["joints"], dtype=float)
    n = len(joints) if max_ticks is None else min(len(joints), max_ticks)
    first_jump = None
    at_jump = None
    snaps = {}
    for i in range(n):
        state["tick"] = i
        vel = (joints[i] - q_prev) / dt
        scene.robot.apply_action(ArticulationAction(joint_positions=joints[i], joint_velocities=vel,
                                                    joint_indices=np.arange(6)))
        scene.gripper.set_target(float(np.clip(grip[i], 0.0, 1.0)))
        scene.world.step(render=True)
        q_prev = joints[i]
        cube = scene.get_cube_position()
        trace.add(i + 1, cube, scene.grip_point_world(), np.zeros(6), grip[i])
        if i in args.snap:
            snaps[i] = hand_pose(scene, cube)
        if first_jump is None and trace.first_trip is not None:
            first_jump = i
            pos, quat = prim_world_pose(scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
            zax = Rot.from_quat(quat).apply(np.array([0.0, 0.0, 1.0]))
            at_jump = {"tool_flange_pos": np.round(pos, 4).tolist(), "tool_axis": np.round(zax, 3).tolist(),
                       "tilt_from_vertical_deg": round(float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1)))), 1),
                       "cube_before": np.round(trace.first_trip["prev_cube"], 4).tolist(),
                       "grip_point": np.round(trace.first_trip["grip_point"], 4).tolist(), "hand_pose": hand_pose(scene, cube)}
        if first_jump is not None and i >= first_jump + args.after:
            break

    by_body = {}
    for tick, other, col, imp, sep, typ, pts in state["events"]:
        b = by_body.setdefault(other, {"first_tick": tick, "ticks": set(), "max_impulse": 0.0, "min_separation_m": 1.0,
                                       "colliders": set()})
        b["ticks"].add(tick)
        b["max_impulse"] = max(b["max_impulse"], imp)
        b["min_separation_m"] = min(b["min_separation_m"], sep)
        b["colliders"].add(col)
    bodies = {k: {"first_tick": v["first_tick"], "ticks_in_contact": len(v["ticks"]),
                  "max_impulse": round(v["max_impulse"], 4), "min_separation_m": round(v["min_separation_m"], 5),
                  "colliders": sorted(v["colliders"])[:4]} for k, v in by_body.items()}
    window = []
    if first_jump is not None:
        lo = first_jump - args.window
        for tick, other, col, imp, sep, typ, pts in state["events"]:
            if lo <= tick <= first_jump + 1:
                window.append({"tick": tick, "body": other, "collider": col, "impulse": round(imp, 4),
                               "separation_m": round(sep, 5), "type": typ,
                               "points": np.round(np.mean(pts, axis=0), 4).tolist() if pts else None})
    out = {"trial": trial, "trace": trace_dir, "seed": seed, "ticks_replayed": trace.ticks,
           "spawn_matches_original_mm": round(1000 * float(np.linalg.norm(spawn[:2] - cube0[0, :2])), 1),
           "original_first_trip_tick": orig_trip, "original_reasons": None if meta["first_trip"] is None else meta["first_trip"]["reasons"],
           "replay_first_jump_tick": first_jump, "replay_summary": trace.summary(),
           "touched_by": bodies, "snapshots": snaps, "contacts_around_first_jump": window}
    if first_jump is not None:
        out["replay_first_trip"] = trace.first_trip
        out["at_first_jump"] = at_jump
    return out


def main():
    scene = c.build_scene()
    cube_prim = scene.stage.GetPrimAtPath(CUBE)
    api = PhysxSchema.PhysxContactReportAPI.Apply(cube_prim)
    api.CreateThresholdAttr().Set(0.0)
    scene.reset()   # the schema is picked up at the next Stop+Play
    sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)  # noqa: F841 (keep alive)
    results = []
    t0 = time.time()
    for spec in args.trial:
        r = replay_one(scene, spec)
        results.append(r)
        bodies = ", ".join(f"{k.split('/')[-1]}({v['ticks_in_contact']}t)" for k, v in r["touched_by"].items())
        print(f"[{spec}] spawn off {r['spawn_matches_original_mm']} mm; original trip tick {r['original_first_trip_tick']}, "
              f"replay jump tick {r['replay_first_jump_tick']}; touched by: {bodies} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)


try:
    main()
except BaseException:
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
