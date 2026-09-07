"""Trains the Residual RL correction policy (residual_rl_train_env.py) on
top of the already-fine-tuned, frozen `pi0_ur5e_pick_place` policy.

NOT a ROS2 node -- launched directly with Isaac Sim's own Kit Python
runtime, same as isaac_scene.py-style scripts elsewhere in this session:

    <isaac-sim-install-dir>/python.sh train_residual_policy.py --policy_host <gpu-host> --policy_port 8000

Prerequisites:
  1. A `pi0_ur5e_pick_place` LoRA checkpoint already exists (README Phase 2)
     and is being served (`uv run scripts/serve_policy.py --config
     pi0_ur5e_pick_place --checkpoint <path>` in your openpi checkout,
     README Phase 3) -- this script queries that server every RL step.
  2. `pip install openpi-client` (or the equivalent editable install from
     your openpi checkout) in whatever Python environment runs this script.

Run a tiny --total_timesteps first (e.g. 200-500) to confirm the env steps
end-to-end (policy server reachable, shapes line up) before committing to a
real training run -- every step here is a live network round-trip plus a
real physics step, so this is slow by design; treat TOTAL_TIMESTEPS as a
starting point to iterate on, not a guaranteed-converged budget, same
caveat as this session's other Isaac-Sim RL training scripts.
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from residual_rl_train_env import IsaacResidualEnv

CHECKPOINT_EVERY = 1_000
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "rl_logs", "residual")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", type=str, default="localhost")
    parser.add_argument("--policy_port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="pick up the cube and place it in the target zone")
    parser.add_argument("--residual_scale", type=float, default=0.05)
    parser.add_argument("--total_timesteps", type=int, default=50_000)
    args = parser.parse_args()

    env = IsaacResidualEnv(
        policy_host=args.policy_host, policy_port=args.policy_port,
        prompt=args.prompt, residual_scale=args.residual_scale,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=os.path.join(MODELS_DIR, "residual_checkpoints"),
        name_prefix="residual_policy",
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
    finally:
        model.save(os.path.join(MODELS_DIR, "residual_policy_final"))
        env.close()

    print(f"done -- final model saved to {os.path.join(MODELS_DIR, 'residual_policy_final.zip')}")
    print("set vla_policy_client's use_residual_policy:=true and rl_model_path to that file to use it.")


if __name__ == "__main__":
    main()
