"""Train a PPO agent to pick a block with the 6-DOF robotic arm."""
from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.arm_env import ArmPickEnv


def make_env(seed: int, vision_device: str | None):
    def _init():
        env = ArmPickEnv(render_mode=None, vision_device=vision_device)
        env.reset(seed=seed)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total PPO training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Policy learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--vision-device",
        type=str,
        default=None,
        help="Device for the OpenCLIP model (cuda, cpu). Defaults to auto-detect.",
    )
    parser.add_argument("--save-dir", type=str, default="models", help="Directory for checkpoints")
    parser.add_argument("--total-callback-interval", type=int, default=50_000, help="Steps between checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device for the PPO policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.seed, args.vision_device)])
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=args.device,
        tensorboard_log=os.path.join(args.save_dir, "tensorboard"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.total_callback_interval // env.num_envs,
        save_path=args.save_dir,
        name_prefix="ppo_arm_pick",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_callback])
    model.save(os.path.join(args.save_dir, "ppo_arm_pick_final"))
    env.close()


if __name__ == "__main__":
    main()
