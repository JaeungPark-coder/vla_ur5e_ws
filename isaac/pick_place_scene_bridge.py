"""Standalone Isaac Sim scene for VLA policy inference/evaluation -- bridges
PickPlaceScene to ROS2 topics so vla_policy_client.py (via
isaac_robot_interface.IsaacSimRobotInterface, in the vla_bridge ROS2
package) can drive it exactly like it will eventually drive the real UR5e.

NOT a ROS2 node run via `ros2 run` -- launched directly with Isaac Sim's own
Kit Python runtime, same as potato_drill_ws's isaac_scene.py:

    <isaac-sim-install-dir>/python.sh pick_place_scene_bridge.py

Deliberately applies joint targets DIRECTLY (no RMPflow/Cartesian IK) --
unlike collect_demos.py's scripted demonstrator, which works in Cartesian
waypoint space to generate the expert trajectory, the trained openpi UR5e
policy already outputs joint-space actions (see
../openpi_integration/ur5e_pick_place_policy.py), so no IK is needed here at
inference time.

Written and reasoned about WITHOUT the ability to run Isaac Sim/ROS2 in the
environment this was authored in -- treat as a solid first draft, not
verified to run.
"""
import argparse
import os
import time

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "0") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")  # makes rclpy importable/usable in this process, same as isaac_scene.py

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState, Image  # noqa: E402
from std_msgs.msg import Float32, Float32MultiArray, Empty  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, ROBOT_PRIM_PATH, GRIPPER_FINGER_KP, GRIPPER_FINGER_KD,
    GRIPPER_FINGER_EFFORT_LIMIT)
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, set_joint_max_force, zero_follower_joint_drives)
from lockstep_protocol import LockstepGate  # noqa: E402
from trial_trace import TrialTrace  # noqa: E402

JOINT_TARGET_TOPIC = "/vla/joint_target"
GRIPPER_TARGET_TOPIC = "/vla/gripper_target"
JOINT_STATE_TOPIC = "/vla/joint_state"
BASE_IMAGE_TOPIC = "/vla/base_image"
WRIST_IMAGE_TOPIC = "/vla/wrist_image"
# Evaluation-only signals: ground-truth cube position (for offline
# success/failure judging) and a trial-reset trigger. Consumed only by
# vla_policy_client.py's eval_mode -- NOT part of the policy's own
# observation (see residual_rl_train_env.py's docstring on why privileged
# sim state stays out of anything the policy itself sees).
EVAL_CUBE_POSITION_TOPIC = "/vla/eval/cube_position"
EVAL_RESET_TOPIC = "/vla/eval/reset"
# Lockstep only: [epoch, max cube z since the previous message] as Float32MultiArray, the maximum taken over EVERY tick
# (see vla_bridge/eval_tracking.py). Lets eval_mode see a blow-up that happens inside one executed block.
EVAL_CUBE_MAX_Z_TOPIC = "/vla/eval/cube_max_z"


class PickPlaceSceneBridge(Node):
    def __init__(self, lockstep=False):
        super().__init__("pick_place_scene_bridge")
        self.bridge = CvBridge()
        # See lockstep_protocol.py: when True the sim advances exactly one tick
        # per request instead of free-running. Default False = unchanged behaviour.
        self.lockstep = lockstep
        self.gate = LockstepGate()
        # Counts scene resets; carried in header.stamp.nanosec of every lockstep observation so a
        # client that asked for a reset can tell the post-reset observation from the pre-reset
        # ones still being re-published while the reset runs.
        self.reset_count = 0

        self.joint_state_pub = self.create_publisher(JointState, JOINT_STATE_TOPIC, 10)
        self.base_image_pub = self.create_publisher(Image, BASE_IMAGE_TOPIC, 10)
        self.wrist_image_pub = self.create_publisher(Image, WRIST_IMAGE_TOPIC, 10)
        self.eval_cube_position_pub = self.create_publisher(Point, EVAL_CUBE_POSITION_TOPIC, 10)
        self.eval_cube_max_z_pub = self.create_publisher(Float32MultiArray, EVAL_CUBE_MAX_Z_TOPIC, 10)

        self.latest_joint_target = None
        self.latest_gripper_target = 0.0
        self.reset_requested = False
        self.create_subscription(JointState, JOINT_TARGET_TOPIC, self._on_joint_target, 10)
        self.create_subscription(Float32, GRIPPER_TARGET_TOPIC, self._on_gripper_target, 10)
        self.create_subscription(Empty, EVAL_RESET_TOPIC, self._on_reset_request, 10)

    def _on_joint_target(self, msg: JointState):
        if self.lockstep:
            position = np.array(msg.position, dtype=float)
            gripper = float(position[6]) if len(position) >= 7 else None
            velocities = np.array(msg.velocity, dtype=float) if len(msg.velocity) >= 6 else None
            self.gate.submit(msg.header.stamp.sec, position[:6], gripper,
                             None if velocities is None else velocities[:6],
                             publish_images=(msg.header.frame_id != "noimg"))
            return
        self.latest_joint_target = np.array(msg.position, dtype=float)

    def _on_gripper_target(self, msg: Float32):
        self.latest_gripper_target = float(msg.data)

    def _on_reset_request(self, msg: Empty):
        self.reset_requested = True

    def publish_cube_position(self, scene):
        pos = scene.get_cube_position()
        self.eval_cube_position_pub.publish(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))

    def publish_cube_max_z(self, max_z):
        self.eval_cube_max_z_pub.publish(Float32MultiArray(data=[float(self.reset_count), float(max_z)]))

    def publish_joint_state(self, obs, ack_id):
        """joint_state only (lockstep ticks that need no images)."""
        js = JointState()
        js.position = [float(p) for p in obs["joints"]] + [float(obs["gripper"][0])]
        js.header.stamp.sec = int(ack_id)
        js.header.stamp.nanosec = self.reset_count
        self.joint_state_pub.publish(js)

    def publish_observation(self, obs, ack_id=None):
        """ack_id (lockstep only): id of the last request executed, carried in
        header.stamp.sec on all three messages so the client can tell which
        tick an observation belongs to. None = leave the stamps untouched."""
        js = JointState()
        js.position = [float(p) for p in obs["joints"]] + [float(obs["gripper"][0])]
        if ack_id is not None:
            js.header.stamp.sec = int(ack_id)
            js.header.stamp.nanosec = self.reset_count
        self.joint_state_pub.publish(js)

        base_rgb = np.ascontiguousarray(np.asarray(obs["base_rgb"])[..., :3])
        wrist_rgb = np.ascontiguousarray(np.asarray(obs["wrist_rgb"])[..., :3])
        base_msg = self.bridge.cv2_to_imgmsg(base_rgb, encoding="rgb8")
        wrist_msg = self.bridge.cv2_to_imgmsg(wrist_rgb, encoding="rgb8")
        if ack_id is not None:
            base_msg.header.stamp.sec = int(ack_id)
            wrist_msg.header.stamp.sec = int(ack_id)
            base_msg.header.stamp.nanosec = self.reset_count
            wrist_msg.header.stamp.nanosec = self.reset_count
        self.base_image_pub.publish(base_msg)
        self.wrist_image_pub.publish(wrist_msg)


def _lockstep_requested():
    """`--lockstep` or VLA_BRIDGE_LOCKSTEP=1 (the env var is the fallback in case
    Kit objects to an unknown argv entry)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lockstep", action="store_true")
    args, _ = parser.parse_known_args()
    return args.lockstep or os.environ.get("VLA_BRIDGE_LOCKSTEP", "0") == "1"


def main():
    # 2026-09-29: this bridge is what vla_policy_client.py's eval_mode (and
    # the eventual real UR5e serving path) actually grasps against -- it
    # was still constructing PickPlaceScene with no gripper override at
    # all, silently defaulting to the as-shipped-fixed 20000/500/no-limit
    # config that has never once produced a successful lift, even after
    # 2026-09-28's collect_demos.py fix (see pick_place_scene.py's own
    # comment on GRIPPER_FINGER_KP for the full story and why the three
    # values below go together). Whatever pi0 policy gets evaluated here
    # was going to fail on the gripper alone, independent of how good the
    # policy itself is.
    scene = PickPlaceScene(finger_kp=GRIPPER_FINGER_KP, finger_kd=GRIPPER_FINGER_KD)

    # Same ordering requirement as collect_demos.py -- must run BEFORE the
    # first scene.reset() below. set_joint_max_force on finger_joint's prim
    # reverts its LIVE controller gain (the finger_kp/finger_kd just
    # requested above) back to the raw USD default as a side effect;
    # scene.reset()'s reapply_drive_gains() is what restores it, and this
    # bridge already calls scene.reset() again on every eval trial
    # (bridge.reset_requested below), so this block only needs to run once
    # here.
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    set_joint_max_force(robot_prim, GRIPPER_FINGER_EFFORT_LIMIT, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
    follower_names = zero_follower_joint_drives(robot_prim)
    if not follower_names:
        raise RuntimeError(
            "zero_follower_joint_drives found no follower/passive gripper joints to zero -- "
            "check resolve_gripper_follower_joint_names, the grasp fix depends on this.")

    # Reproducible, PAIRED evaluation: with VLA_BRIDGE_SEED=S the cube spawn of eval trial k (k = 0, 1, ...) is the one
    # numpy.random.default_rng(S + k) draws, in every run that uses the same S -- so two runs that differ only in the policy
    # settings (e.g. execute_horizon) face identical scenes. Lockstep mode only (that is where eval resets are handled
    # with an epoch); unset = random spawns as before.
    base_seed = os.environ.get("VLA_BRIDGE_SEED")
    base_seed = None if base_seed in (None, "") else int(base_seed)
    # VLA_BRIDGE_SEED_REPEAT=R: R consecutive trials share one spawn (trial k uses default_rng(S + k // R)). The client
    # cycles execute_horizon over those R trials, so one seed is evaluated under every K before the next seed starts and
    # an interrupted run still leaves complete pairs.
    seed_repeat = max(1, int(os.environ.get("VLA_BRIDGE_SEED_REPEAT") or 1))
    if base_seed is not None:
        scene._rng = np.random.default_rng(base_seed)
        print(f"VLA_BRIDGE_SEED={base_seed}: trial k spawns the cube with default_rng({base_seed}+k//{seed_repeat})", flush=True)
    # VLA_BRIDGE_CUBE_X_RANGE="0.42,0.48" / VLA_BRIDGE_CUBE_Y_RANGE="-0.1,0.1": spawn box of the cube for this run (the scene's
    # default is the box the demonstrations were collected in). Evaluation of a wider box: README 2026-11-11.
    import pick_place_scene as _pps
    for _name, _attr in (("VLA_BRIDGE_CUBE_X_RANGE", "CUBE_X_RANGE"), ("VLA_BRIDGE_CUBE_Y_RANGE", "CUBE_Y_RANGE")):
        _v = os.environ.get(_name)
        if _v:
            setattr(_pps, _attr, tuple(float(t) for t in _v.split(",")))
            print(f"{_name}: {_attr} = {getattr(_pps, _attr)}", flush=True)
    scene.reset()
    if base_seed is not None:
        print(f"seeded reset 0: cube at {np.round(scene.cube_position, 4).tolist()}", flush=True)

    rclpy.init()
    lockstep = _lockstep_requested()
    bridge = PickPlaceSceneBridge(lockstep=lockstep)

    print("pick_place_scene_bridge.py running -- publishing joint_state/base_image/wrist_image, "
          "subscribed to joint_target/gripper_target. Ctrl+C (or close the Isaac Sim window) to stop. "
          "Run vla_policy_client.py (robot_backend:=isaac_sim) in another terminal.")
    if lockstep:
        print("LOCKSTEP mode: the simulator advances exactly one tick per /vla/joint_target request "
              "(client must run with lockstep:=true). See lockstep_protocol.py. Prefer "
              "ISAAC_PICK_PLACE_HEADLESS=1: the idle wait deliberately does not render, so the "
              "window will not refresh between requests.", flush=True)
    # Every tick of every trial -> VLA_BRIDGE_TRACE_DIR/trial_NNN.npz + .json (trial_trace.py: cube/grip point/joints/gripper
    # per tick and the blow-up verdict). Unset = no trace.
    trace = TrialTrace(os.environ.get("VLA_BRIDGE_TRACE_DIR") or None)
    # 2026-11-13: scene.reset() renders once; that first frame still shows the cube at its authored default pose (a faint sliver at
    # one fixed image position, every trial -- README 2026-11-13 late). N render-only passes fix it (30 = clean, 10 leaves a ghost).
    reset_renders = int(os.environ.get("VLA_BRIDGE_RESET_RENDERS", "0") or 0)
    save_images = bool(trace.out_dir) and os.environ.get("VLA_BRIDGE_SAVE_IMAGES", "") == "1"   # + trial_NNN_images.npz (~3 MB/trial)
    last_obs = scene.get_observation() if lockstep else None
    last_republish = time.time()
    tick_max_z = 0.0  # lockstep: max cube height over the ticks since the last published message
    if lockstep:
        # Diagnostics for check_lockstep_e2e.py: the world step index must advance by
        # exactly the number of requests executed, and not at all while idle.
        print(f"lockstep: world step index after reset = {scene.world.current_time_step_index}", flush=True)

    try:
        while simulation_app.is_running():
            if lockstep:
                # One tick per request; between requests nothing steps and nothing
                # renders (an extra render would change the image-vs-state lag the
                # demonstrations were recorded with). The last observation is
                # re-published every 0.5 s so a client that connects late, or
                # right after a reset, still gets a starting observation.
                rclpy.spin_once(bridge, timeout_sec=0.005)
                if bridge.reset_requested:
                    trace.save()   # the trial that just ended
                    if base_seed is not None:
                        scene._rng = np.random.default_rng(base_seed + (bridge.reset_count + 1) // seed_repeat)
                    obs = scene.reset()
                    if reset_renders:   # converge the renderer (no physics step) so the first policy input shows the real cube
                        for _ in range(reset_renders):
                            scene.world.render()
                        obs = scene.get_observation()
                    bridge.reset_requested = False
                    bridge.reset_count += 1
                    if base_seed is not None:
                        print(f"seeded reset {bridge.reset_count}: cube at {np.round(scene.cube_position, 4).tolist()}", flush=True)
                    bridge.gate.reset()
                    bridge.latest_gripper_target = 0.0
                    last_obs = obs if isinstance(obs, dict) else scene.get_observation()
                    tick_max_z = float(scene.get_cube_position()[2])
                    trace.start(bridge.reset_count, cube=scene.get_cube_position())
                    if save_images:   # the trial's FIRST policy input (one render after the reset -- possibly a stale frame)
                        trace.add_images(0, np.asarray(last_obs["base_rgb"])[..., :3], np.asarray(last_obs["wrist_rgb"])[..., :3])
                    # cube messages BEFORE the observation: the client treats the observation (joint_state) as
                    # the "tick done" signal and reads the cube values right after it
                    bridge.publish_cube_position(scene)
                    bridge.publish_cube_max_z(tick_max_z)
                    bridge.publish_observation(last_obs, ack_id=bridge.gate.last_executed_id)
                    last_republish = time.time()
                request = bridge.gate.take()
                if request is not None:
                    request_id, joint_target, gripper_target, joint_velocity, publish_images = request
                    from isaacsim.core.utils.types import ArticulationAction
                    # velocity feed-forward when the client sent one (the expert's own
                    # commands carried a velocity target every tick; positions alone lag)
                    scene.robot.apply_action(
                        ArticulationAction(
                            joint_positions=np.asarray(joint_target, dtype=float),
                            joint_velocities=(None if joint_velocity is None
                                              else np.asarray(joint_velocity, dtype=float)),
                            joint_indices=np.arange(6)))
                    if gripper_target is None:
                        gripper_target = bridge.latest_gripper_target
                    scene.gripper.set_target(gripper_target)
                    scene.world.step(render=True)  # exactly one tick
                    cube_now = scene.get_cube_position()
                    tick_max_z = max(tick_max_z, float(cube_now[2]))
                    prop = scene.get_proprioception()
                    trace.add(request_id, cube_now, scene.grip_point_world(), prop["joints"], prop["gripper"][0],
                              target=np.concatenate([np.asarray(joint_target, dtype=float)[:6], [float(gripper_target)]]))
                    if request_id % 100 == 0:
                        print(f"lockstep: request {request_id} executed, world step index "
                              f"{scene.world.current_time_step_index}", flush=True)
                    if publish_images:
                        last_obs = scene.get_observation()
                        if save_images:   # the policy's own inputs of this trial, to see later what it saw
                            trace.add_images(trace.ticks, np.asarray(last_obs["base_rgb"])[..., :3], np.asarray(last_obs["wrist_rgb"])[..., :3])
                        bridge.publish_cube_position(scene)
                        bridge.publish_cube_max_z(tick_max_z)
                        tick_max_z = float(scene.get_cube_position()[2])  # start the next window at the current height
                        bridge.publish_observation(last_obs, ack_id=request_id)
                    else:
                        bridge.publish_joint_state(scene.get_proprioception(), ack_id=request_id)
                        last_obs = None  # nothing complete to re-publish while idle until the next image tick
                    last_republish = time.time()
                elif last_obs is not None and time.time() - last_republish > 0.5:
                    trace.flush_if_dirty()   # the last trial of a run gets no reset after it
                    bridge.publish_cube_position(scene)
                    bridge.publish_cube_max_z(tick_max_z)   # repeat only: the window is not cleared here
                    bridge.publish_observation(last_obs, ack_id=bridge.gate.last_executed_id)
                    last_republish = time.time()
                continue

            rclpy.spin_once(bridge, timeout_sec=0.0)

            if bridge.reset_requested:
                scene.reset()
                bridge.reset_requested = False
                bridge.latest_joint_target = None
                bridge.latest_gripper_target = 0.0

            if bridge.latest_joint_target is not None:
                # scene.robot is a SingleArticulation (unbatched) -- see
                # pick_place_scene.py -- so targets go through apply_action(),
                # not a vectorized set_joint_position_targets(). joint_indices
                # must be explicit: this articulation is 12-DOF (6 arm +
                # gripper; scene.robot.num_dof was 12 from tick 1, confirmed
                # live, never changed mid-session), and a joint_positions
                # array shorter than that with joint_indices=None (the
                # default) requires it to match ALL dofs -- ValueError:
                # shape mismatch, (1,6) into (1,12). CONFIRMED 2026-09-23
                # live via a real eval_mode run (this is what silently ended
                # every eval_mode session so far, not scene.reset() or the
                # ROS2 bridge extension as first suspected).
                from isaacsim.core.utils.types import ArticulationAction
                scene.robot.apply_action(
                    ArticulationAction(
                        joint_positions=np.asarray(bridge.latest_joint_target[:6], dtype=float),
                        joint_indices=np.arange(6))
                )
            scene.gripper.set_target(bridge.latest_gripper_target)
            scene.world.step(render=True)

            bridge.publish_observation(scene.get_observation())
            bridge.publish_cube_position(scene)
    except KeyboardInterrupt:
        pass
    except BaseException:
        # 2026-09-23: this used to be an unhandled exception that Kit
        # swallowed without a trace in headless mode (--installSignalHandlers=0,
        # --no-window), so every eval_mode session that hit the apply_action
        # shape-mismatch bug just above looked like a silent, crash-free
        # shutdown -- see the joint_indices fix's comment for what it
        # actually was. Print it instead of letting that happen again.
        import traceback
        print("pick_place_scene_bridge.py: unhandled exception, shutting down:\n"
              + traceback.format_exc(), flush=True)
    finally:
        trace.save()
        if lockstep:
            print(f"lockstep: shutting down after {bridge.gate.last_executed_id} requests, world step index "
                  f"{scene.world.current_time_step_index}", flush=True)
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
