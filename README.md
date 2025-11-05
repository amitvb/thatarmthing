# Vision-guided 6-DOF Arm

This project provides a reinforcement-learning environment and training scripts for a 6-degree-of-freedom robotic arm mounted on a stationary base placed on a table. The arm ends in a two-finger gripper with an RGB camera (80° field of view) mounted near the gripper. The training objective is to locate a block on the table using a pre-trained large vision model and pick it up using reinforcement learning.

## Project layout

```
assets/                # URDF assets for the arm, table, and block
envs/arm_env.py        # Gymnasium environment wrapping the PyBullet simulation
vision/locator.py      # OpenCLIP-powered block localization helper
scripts/train.py       # PPO training loop
scripts/evaluate.py    # Evaluation helper for trained policies
requirements.txt       # Python dependencies
```

## Features

- **6-DOF arm with two-finger gripper:** The custom URDF models a six-joint manipulator with a symmetric prismatic gripper.
- **Mounted camera with 80° FOV:** The environment renders camera images from a camera link attached near the gripper, matching the specified field of view.
- **Large vision model integration:** Block localization uses the pre-trained OpenCLIP ViT-B/32 model (`laion2b_s34b_b79k`) to score image patches against the prompt “a small colored block on a table”.
- **Reinforcement learning environment:** The `ArmPickEnv` exposes a Gymnasium-compatible interface with joint-space actions and vision-derived observations.
- **PPO training pipeline:** Ready-to-run scripts for training and evaluating agents using Stable-Baselines3.

## Getting started

1. **Install dependencies** (ideally inside a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Train a policy** (saves checkpoints under `models/` by default):
   ```bash
   python scripts/train.py --timesteps 200000
   ```
   Optional flags:
   - `--vision-device cuda` to force OpenCLIP onto a GPU.
   - `--device cuda` to train PPO on a GPU if available.

3. **Evaluate a trained model** (replace `ppo_arm_pick_final.zip` with the desired checkpoint):
   ```bash
   python scripts/evaluate.py models/ppo_arm_pick_final.zip --episodes 20
   ```
   Add `--render` to watch the policy in the PyBullet GUI.

4. **Run the deterministic pickup demo** to validate the assets and produce a GIF from a hand-scripted grasp:
   ```bash
   python scripts/test_pickup.py --output artifacts/pickup.gif
   ```
   The script disables the vision model, uses a deterministic joint script to reach a canonical cube pose, and, if contact is not detected, fixes the cube to the gripper for visualization before lifting and saving the animation.

## Observation and action spaces

- **Actions:** 7-dimensional continuous vector: 6 joint deltas and one gripper open/close command.
- **Observations:** Concatenation of joint positions and velocities, current gripper aperture, the vision-estimated block position in the arm base frame, and the OpenCLIP confidence score.

## Rewards and success criteria

The reward function encourages the end effector to approach the block, penalizes unnecessary gripper motion, and grants a large bonus when the block is securely grasped and lifted more than 5 cm above the table. Episodes terminate successfully when the grasp-and-lift condition is met.

## Notes

- The first call to the environment downloads OpenCLIP weights (~400 MB). Keep an internet connection available for the download.
- Training times can vary depending on hardware. Consider using GPUs for both PPO and the vision model when possible.
- The simulation uses PyBullet’s default time step of 1/60 s and advances four sub-steps per action for smoother control.
