"""Gymnasium env for training a Residual RL correction policy on top of the
frozen, already-fine-tuned pi0_ur5e_pick_place policy: the RL agent only
outputs a small correction added to pi0's own proposed joint targets, not a
replacement action. Freezing the VLA and training a lightweight residual
policy over it is one of the two RL post-training directions currently
gaining traction for closing the sim/real success-rate gap (per the ICLR
2026 VLA survey this project's README Phase 4 section cites) -- "no method
has yet established dominance" there, so whether it actually helps on this
task is a genuine open question, not a known result.

NOT a ROS2 node -- run only via `<isaac-sim-install-dir>/python.sh` (needs
Isaac Sim's own Kit runtime). Not imported directly; see
train_residual_policy.py. Requires an openpi policy server already running
and serving the `pi0_ur5e_pick_place` checkpoint (see ../README.md Phase 3)
-- every env.step() makes a live network call to it.

Deliberately slow (a live policy-server round-trip AND a real Isaac Sim
physics step, every single RL step) -- same "training realism over training
speed" tradeoff already accepted elsewhere in this session
(quadruped_parkour_ws's CIGR training, this project's own collect_demos.py).

Written and reasoned about WITHOUT the ability to run Isaac Sim or a live
openpi server in the environment this was authored in -- treat as a solid
first draft, not verified to run.
"""
import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_RESIDUAL_ENV_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION  # noqa: E402

OBS_DIM = 14   # joints(6) + gripper(1) + pi0's own proposed action(7) -- see module docstring in ../README.md Phase 4 on why NOT cube/target position
ACTION_DIM = 6  # residual correction on the 6 arm joints only -- gripper stays under pi0's direct control

SUCCESS_XY_TOLERANCE_M = 0.03  # cube-to-target planar distance counted as "placed"
HOLDING_GRIPPER_THRESHOLD = 0.5  # gripper value above this counts as "closed enough to be holding the cube"


class IsaacResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, policy_host="localhost", policy_port=8000,
                 prompt="pick up the cube and place it in the target zone",
                 residual_scale=0.05, max_episode_steps=150):
        super().__init__()
        self.simulation_app = simulation_app
        self.prompt = prompt
        self.residual_scale = residual_scale
        self.max_episode_steps = max_episode_steps

        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.policy = WebsocketClientPolicy(host=policy_host, port=policy_port)

        self.scene = PickPlaceScene()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._steps = 0
        self._was_holding = False
        self._base_action = None  # cached each step so reward/obs can reuse it without re-querying

    def _query_base_policy(self, obs):
        result = self.policy.infer({
            "joints": obs["joints"],
            "gripper": obs["gripper"],
            "base_rgb": obs["base_rgb"],
            "wrist_rgb": obs["wrist_rgb"],
            "prompt": self.prompt,
        })
        action_chunk = np.asarray(result["actions"])
        action = action_chunk[0] if action_chunk.ndim == 2 else action_chunk
        return action.astype(np.float32)  # (7,): 6 joint targets + gripper

    def _build_obs(self, scene_obs, base_action):
        return np.concatenate([scene_obs["joints"], scene_obs["gripper"], base_action]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        scene_obs = self.scene.reset()
        self._base_action = self._query_base_policy(scene_obs)
        self._steps = 0
        self._was_holding = False
        return self._build_obs(scene_obs, self._base_action), {}

    def step(self, residual_action):
        residual_action = np.clip(np.asarray(residual_action, dtype=float), -1.0, 1.0)

        base_action = self._base_action  # queried at the END of the previous step (or reset) against the CURRENT scene state
        target_joints = base_action[:6] + self.residual_scale * residual_action
        target_gripper = float(np.clip(base_action[6], 0.0, 1.0))

        self.scene.apply_joint_targets(target_joints, target_gripper)
        scene_obs = self.scene.get_observation()

        reward, terminated, is_holding = self._compute_reward(scene_obs, target_gripper, residual_action)
        self._was_holding = is_holding

        self._steps += 1
        truncated = self._steps >= self.max_episode_steps

        # Query pi0 again against the state we just reached, for the NEXT
        # step's base_action -- cached now so step() only makes one policy
        # call per tick (not two).
        self._base_action = self._query_base_policy(scene_obs)

        obs = self._build_obs(scene_obs, self._base_action)
        info = {"is_holding": is_holding}
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, scene_obs, target_gripper, residual_action):
        """Training-only -- uses ground-truth cube/target position from the
        sim (not part of the policy's observation, see module docstring).
        Dense distance-to-target while holding the cube, a one-off success
        bonus on releasing it inside the target zone, and a small penalty
        on the residual's own magnitude so corrections stay "residual"
        rather than a second policy fighting pi0's own action."""
        cube_pos = self.scene.get_cube_position()
        is_holding = target_gripper >= HOLDING_GRIPPER_THRESHOLD and cube_pos[2] > 0.03  # closed AND lifted off the table

        xy_dist_to_target = float(np.linalg.norm(cube_pos[:2] - PLACE_TARGET_POSITION[:2]))
        reward = 0.0
        terminated = False

        if is_holding:
            reward += 1.0 - np.tanh(xy_dist_to_target / 0.2)  # dense shaping toward the target while carrying

        released_here = self._was_holding and not is_holding
        if released_here and xy_dist_to_target <= SUCCESS_XY_TOLERANCE_M:
            reward += 10.0
            terminated = True  # task solved -- end the episode

        reward -= 0.01 * float(np.linalg.norm(residual_action))  # keep corrections small
        return reward, terminated, is_holding

    def close(self):
        self.simulation_app.close()
