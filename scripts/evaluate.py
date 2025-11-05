"""Evaluate a trained policy in the arm environment."""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from envs.arm_env import ArmPickEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=str, help="Path to the saved PPO model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation rollouts")
    parser.add_argument("--vision-device", type=str, default=None, help="Device for the OpenCLIP model")
    parser.add_argument("--render", action="store_true", help="Render the environment using the GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None
    env = ArmPickEnv(render_mode=render_mode, vision_device=args.vision_device)
    model = PPO.load(args.model, env=env)

    rewards = []
    successes = 0
    for episode in range(args.episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                successes += 1
        rewards.append(ep_reward)
        print(f"Episode {episode + 1}: reward={ep_reward:.3f}, success={terminated}")

    print(f"Average reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Success rate: {successes / args.episodes:.2%}")
    env.close()


if __name__ == "__main__":
    main()
