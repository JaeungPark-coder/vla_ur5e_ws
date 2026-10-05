"""End-to-end check of the lockstep ROS2 path, with a scripted stand-in for the
policy: no openpi, no GPU for the client, no policy server.

Runs the REAL client-side interface (vla_bridge.isaac_robot_interface with
lockstep=True) against a running `pick_place_scene_bridge.py --lockstep`, and
plays the first N ticks of a recorded episode's absolute joint targets through it.

Checks (each printed PASS/FAIL):
  1. every request is acknowledged with its own id, in order;
  2. the images carry the id of the tick they belong to (header.stamp.sec);
  3. the arm follows the recorded trajectory (the approach phase involves no
     contact, so it does not depend on where the cube is). With position targets only
     this FAILS by design (~0.3 rad lag, README 2026-10-05); it passes with the
     velocity feed-forward the script sends by default (--no_velocity to see the lag);
  4. the gripper command sent in the same message closes the gripper;
  5. the request rate (how many sim ticks/s the round trip sustains).
Then it exits; read the bridge's own log for the world step index: after N
requests it must have advanced by exactly N, and by 0 while idle.

Start the bridge and this script in the SAME ROS environment, Isaac's bundled Humble,
not the system ROS 2 (mixing them crashes the bridge in FastDDS discovery -- README
2026-09-23). See the commands in README (2026-10-05).

    python check_lockstep_e2e.py --ticks 300
"""
import argparse
import glob
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "vla_bridge"))

import rclpy  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from vla_bridge.feedforward import chunk_velocities  # noqa: E402
from vla_bridge.isaac_robot_interface import IsaacSimRobotInterface  # noqa: E402

DATASET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/lerobot/vla_ur5e_ws/ur5e_pick_place_v2/data/chunk-*/*.parquet")


def load_first_episode():
    import pandas as pd
    for path in sorted(glob.glob(DATASET_GLOB)):
        df = pd.read_parquet(path, columns=["episode_index", "joints", "gripper", "actions"])
        ep = df[df["episode_index"] == df["episode_index"].min()]
        if len(ep):
            joints = np.stack(ep["joints"].values).astype(float)
            actions = np.stack(ep["actions"].values).astype(float)
            return joints, actions
    raise RuntimeError(f"no parquet under {DATASET_GLOB}")


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--close_ticks", type=int, default=60)
    ap.add_argument("--no_velocity", action="store_true",
                    help="send positions only (the old serving behaviour) instead of position + velocity feed-forward")
    args = ap.parse_args()

    joints_rec, actions_rec = load_first_episode()
    n = min(args.ticks, len(actions_rec))
    velocities = None if args.no_velocity else chunk_velocities(actions_rec[:n + 1, :6], 1.0 / 60.0)

    rclpy.init()
    node = Node("lockstep_e2e_client")
    group = ReentrantCallbackGroup()
    robot = IsaacSimRobotInterface(node, callback_group=group, lockstep=True, lockstep_timeout_s=20.0)
    stamps = {"base": -1, "wrist": -1}

    def on_img(key):
        def cb(msg):
            stamps[key] = int(msg.header.stamp.sec)
        return cb

    node.create_subscription(Image, "/vla/base_image", on_img("base"), 10, callback_group=group)
    node.create_subscription(Image, "/vla/wrist_image", on_img("wrist"), 10, callback_group=group)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    t0 = time.time()
    while robot.get_joint_positions() is None or stamps["base"] < 0 or stamps["wrist"] < 0:
        if time.time() - t0 > 60:
            print("[FAIL] never received an observation from the bridge (is it running with --lockstep, "
                  "same ROS_DOMAIN_ID / RMW?)", flush=True)
            sys.exit(1)
        time.sleep(0.05)
    q_start = robot.get_joint_positions()
    print(f"first observation received; ack id {robot.acked_request_id()}, "
          f"start joints vs recording: max |diff| = {np.max(np.abs(q_start - joints_rec[0])):.4f} rad", flush=True)

    # --- 1-3, 5: play the recorded arm targets, gripper open ----------------------
    ok_acks, errs, gaps = True, [], []
    t_play = time.time()
    for t in range(n):
        robot.set_gripper(0.0)
        if not robot.move_joints(actions_rec[t, :6], joint_velocities=None if velocities is None else velocities[t]):
            ok_acks = False
            print(f"request {t + 1}: no ack within the timeout", flush=True)
            break
        if robot.acked_request_id() != robot.last_request_id():
            ok_acks = False
        q = robot.get_joint_positions()
        errs.append(np.abs(q - actions_rec[t, :6]).max())
        if t in (0, 1, 2, 10, 50, 100, 200) or t == n - 1:
            print(f"  t={t:3d} |q-target|max={errs[-1]:.4f}  q={np.round(q, 3).tolist()}  "
                  f"target={np.round(actions_rec[t, :6], 3).tolist()}", flush=True)
        gaps.append(robot.last_request_id() - robot.acked_request_id())
    rate = n / (time.time() - t_play)
    time.sleep(0.5)  # let the last images arrive
    last_id = robot.last_request_id()
    errs = np.asarray(errs)
    report("1. every request acknowledged with its own id", ok_acks and max(gaps) == 0,
           f"({last_id} requests)")
    report("2. images carry the tick id", stamps["base"] == last_id and stamps["wrist"] == last_id,
           f"(base {stamps['base']}, wrist {stamps['wrist']}, last request {last_id})")
    report("3. arm follows the recorded trajectory", errs.max() < 0.05,
           f"(max |q - target| over {len(errs)} ticks = {errs.max():.4f} rad, mean {errs.mean():.4f})")
    print(f"5. sustained {rate:.1f} requests (= sim ticks) per second", flush=True)

    # --- 4: gripper command rides in the same message ------------------------------
    g_before = robot.get_gripper_state().position
    hold = actions_rec[n - 1, :6]
    for _ in range(args.close_ticks):
        robot.set_gripper(1.0)
        robot.move_joints(hold)
    g_after = robot.get_gripper_state().position
    report("4. gripper closes from the command in the joint-target message", g_after - g_before > 0.3,
           f"(measured {g_before:.2f} -> {g_after:.2f})")

    total = robot.last_request_id()
    print(f"sent {total} requests in total; the bridge's world step index must have advanced by exactly "
          f"{total} (compare with its log)", flush=True)
    executor.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
