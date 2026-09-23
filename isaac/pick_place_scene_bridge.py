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
import os

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
from std_msgs.msg import Float32, Empty  # noqa: E402
from geometry_msgs.msg import Point  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402

from pick_place_scene import PickPlaceScene  # noqa: E402

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


class PickPlaceSceneBridge(Node):
    def __init__(self):
        super().__init__("pick_place_scene_bridge")
        self.bridge = CvBridge()

        self.joint_state_pub = self.create_publisher(JointState, JOINT_STATE_TOPIC, 10)
        self.base_image_pub = self.create_publisher(Image, BASE_IMAGE_TOPIC, 10)
        self.wrist_image_pub = self.create_publisher(Image, WRIST_IMAGE_TOPIC, 10)
        self.eval_cube_position_pub = self.create_publisher(Point, EVAL_CUBE_POSITION_TOPIC, 10)

        self.latest_joint_target = None
        self.latest_gripper_target = 0.0
        self.reset_requested = False
        self.create_subscription(JointState, JOINT_TARGET_TOPIC, self._on_joint_target, 10)
        self.create_subscription(Float32, GRIPPER_TARGET_TOPIC, self._on_gripper_target, 10)
        self.create_subscription(Empty, EVAL_RESET_TOPIC, self._on_reset_request, 10)

    def _on_joint_target(self, msg: JointState):
        self.latest_joint_target = np.array(msg.position, dtype=float)

    def _on_gripper_target(self, msg: Float32):
        self.latest_gripper_target = float(msg.data)

    def _on_reset_request(self, msg: Empty):
        self.reset_requested = True

    def publish_cube_position(self, scene):
        pos = scene.get_cube_position()
        self.eval_cube_position_pub.publish(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))

    def publish_observation(self, obs):
        js = JointState()
        js.position = [float(p) for p in obs["joints"]] + [float(obs["gripper"][0])]
        self.joint_state_pub.publish(js)

        base_rgb = np.ascontiguousarray(np.asarray(obs["base_rgb"])[..., :3])
        wrist_rgb = np.ascontiguousarray(np.asarray(obs["wrist_rgb"])[..., :3])
        self.base_image_pub.publish(self.bridge.cv2_to_imgmsg(base_rgb, encoding="rgb8"))
        self.wrist_image_pub.publish(self.bridge.cv2_to_imgmsg(wrist_rgb, encoding="rgb8"))


def main():
    scene = PickPlaceScene()
    scene.reset()

    rclpy.init()
    bridge = PickPlaceSceneBridge()

    print("pick_place_scene_bridge.py running -- publishing joint_state/base_image/wrist_image, "
          "subscribed to joint_target/gripper_target. Ctrl+C (or close the Isaac Sim window) to stop. "
          "Run vla_policy_client.py (robot_backend:=isaac_sim) in another terminal.")

    try:
        while simulation_app.is_running():
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
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
