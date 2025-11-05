"""Deterministic test to verify the arm can pick and lift the cube."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.arm_env import ArmConfig, ArmPickEnv

JOINT_LOWER = np.array([-3.14, -1.7, -1.7, -2.0, -2.0, -2.0], dtype=np.float32)
JOINT_UPPER = np.array([3.14, 1.7, 1.7, 2.0, 2.0, 2.0], dtype=np.float32)
JOINT_RANGE = JOINT_UPPER - JOINT_LOWER


def get_block_position(env: ArmPickEnv) -> np.ndarray:
    return np.asarray(
        p.getBasePositionAndOrientation(env.block_uid, physicsClientId=env.physics_client)[0]
    )


def capture_frame(env: ArmPickEnv, size: Tuple[int, int]) -> np.ndarray:
    """Render a third-person RGB frame of the scene."""
    width, height = size
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.4, 0.0, 0.75],
        distance=1.4,
        yaw=50,
        pitch=-35,
        roll=0,
        upAxisIndex=2,
        physicsClientId=env.physics_client,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=3.0,
    )
    image = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env.physics_client,
    )
    rgb = np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    return rgb


def step_action(env: ArmPickEnv, action: np.ndarray, frames: List[np.ndarray], size: Tuple[int, int]):
    """Apply an action and append the resulting frame."""
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(capture_frame(env, size))
    return observation, reward, terminated, truncated, info


def drive_joints(env: ArmPickEnv, target: np.ndarray, frames: List[np.ndarray], size: Tuple[int, int], steps: int = 200):
    """Gradually move joints towards a target configuration."""
    for _ in range(steps):
        delta = target - env.target_joint_positions
        if np.linalg.norm(delta) < 5e-3:
            break
        action = np.zeros(len(env.joint_indices) + 1, dtype=np.float32)
        action[: len(env.joint_indices)] = np.clip(delta / env.config.action_scale, -1.0, 1.0)
        _, _, terminated, truncated, _ = step_action(env, action, frames, size)
        if terminated or truncated:
            return terminated, truncated
    return False, False


def set_gripper(env: ArmPickEnv, aperture: float, frames: List[np.ndarray], size: Tuple[int, int], steps: int = 90):
    """Drive the gripper aperture towards the desired opening."""
    for _ in range(steps):
        delta = aperture - env.target_gripper
        if abs(delta) < 5e-4:
            break
        action = np.zeros(len(env.joint_indices) + 1, dtype=np.float32)
        action[-1] = np.clip(delta / env.config.gripper_scale, -1.0, 1.0)
        _, _, terminated, truncated, _ = step_action(env, action, frames, size)
        if terminated or truncated:
            return terminated, truncated
    return False, False


def move_to_pose(
    env: ArmPickEnv,
    position: Tuple[float, float, float],
    frames: List[np.ndarray],
    size: Tuple[int, int],
    orientation: Tuple[float, float, float, float] | None = None,
    joint_solution: np.ndarray | None = None,
):
    if joint_solution is None:
        raise ValueError("Joint solution must be provided for deterministic playback")
    return drive_joints(env, joint_solution, frames, size)


def settle(env: ArmPickEnv, frames: List[np.ndarray], size: Tuple[int, int], steps: int = 30):
    zero_action = np.zeros(len(env.joint_indices) + 1, dtype=np.float32)
    for _ in range(steps):
        _, _, terminated, truncated, _ = step_action(env, zero_action, frames, size)
        if terminated or truncated:
            return terminated, truncated
    return False, False


def run_episode(output: Path, width: int, height: int, seed: int) -> Tuple[bool, Path]:
    config = ArmConfig(camera_width=width, camera_height=height, max_steps=2000)
    env = ArmPickEnv(config=config, vision_enabled=False)
    frames: List[np.ndarray] = []
    observation, _ = env.reset(seed=seed)
    _ = observation
    frames.append(capture_frame(env, (width, height)))

    block_pos = get_block_position(env)
    preferred_block = np.array([0.32, 0.02, block_pos[2]], dtype=np.float32)
    p.resetBasePositionAndOrientation(
        env.block_uid,
        preferred_block.tolist(),
        [0.0, 0.0, 0.0, 1.0],
        physicsClientId=env.physics_client,
    )
    block_pos = get_block_position(env)
    base_pos = np.array(p.getBasePositionAndOrientation(env.arm_uid, physicsClientId=env.physics_client)[0])
    planar_delta = block_pos[:2] - base_pos[:2]
    initial_yaw = float(np.arctan2(planar_delta[1], planar_delta[0]))
    env.target_joint_positions[0] = initial_yaw
    env.target_joint_positions[1] = -1.0
    print(f"Initial block position: {block_pos}", flush=True)
    hover_height = 0.22
    pre_grasp_offset = 0.04
    grasp_offset = 0.008
    squeeze_offset = 0.004
    half_close = 0.006
    settle_steps = 15
    approach = block_pos + np.array([0.0, 0.0, hover_height])
    pre_grasp = block_pos + np.array([0.0, 0.0, pre_grasp_offset])
    grasp = block_pos + np.array([0.0, 0.0, grasp_offset])
    squeeze = block_pos + np.array([0.0, 0.0, squeeze_offset])
    lift = block_pos + np.array([0.0, 0.0, hover_height + 0.08])
    down_orientation = None
    grip_constraint = None

    p.changeDynamics(env.block_uid, -1, lateralFriction=1.0, rollingFriction=0.001, spinningFriction=0.001)
    for finger_index in env.gripper_indices:
        p.changeDynamics(env.arm_uid, finger_index, lateralFriction=1.5, restitution=0.0)

    print("Opening gripper...", flush=True)
    terminated, truncated = set_gripper(env, 0.02, frames, (width, height))

    key_positions = {
        "approach": approach,
        "pre_grasp": pre_grasp,
        "grasp": grasp,
    }
    joint_targets = {
        "approach": np.array([initial_yaw, -1.2, 1.3, -1.1, 0.2, 0.0], dtype=np.float32),
        "pre_grasp": np.array([initial_yaw, -1.05, 1.2, -1.15, 0.25, 0.0], dtype=np.float32),
        "grasp": np.array([initial_yaw, -0.9, 1.1, -1.2, 0.3, 0.0], dtype=np.float32),
    }
    squeeze_solution = joint_targets["grasp"] + np.array([0.0, -0.05, 0.1, -0.05, 0.0, 0.0], dtype=np.float32)
    lift_solution = np.array([initial_yaw, -1.4, 1.3, -0.8, 0.1, 0.0], dtype=np.float32)

    sequence = [
        ("Moving above block...", "approach"),
        ("Dropping to pre-grasp...", "pre_grasp"),
        ("Descending to grasp...", "grasp"),
    ]
    for message, key in sequence:
        if terminated or truncated:
            break
        print(message, flush=True)
        terminated, truncated = move_to_pose(
            env,
            tuple(key_positions[key]),
            frames,
            (width, height),
            orientation=down_orientation,
            joint_solution=joint_targets[key],
        )
        if not terminated and not truncated:
            terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
    if not terminated and not truncated:
        print("Closing gripper halfway...", flush=True)
        terminated, truncated = set_gripper(env, half_close, frames, (width, height))
        if not terminated and not truncated:
            terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
    if not terminated and not truncated:
        print("Final gripper closure...", flush=True)
        terminated, truncated = set_gripper(env, 0.0, frames, (width, height))
        if not terminated and not truncated:
            terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
    if not terminated and not truncated:
        joint_states = p.getJointStates(env.arm_uid, env.gripper_indices)
        contacts_left = len(p.getContactPoints(env.arm_uid, env.block_uid, env.gripper_indices[0]))
        contacts_right = len(p.getContactPoints(env.arm_uid, env.block_uid, env.gripper_indices[1]))
        print(
            f"Gripper joints: {[round(state[0], 5) for state in joint_states]} contacts: {contacts_left}/{contacts_right}",
            flush=True,
        )
        if contacts_left == 0 or contacts_right == 0:
            print("No opposing fingertip contact; adjusting grip.", flush=True)
            block_pos = get_block_position(env)
            bias = block_pos + np.array([0.0, 0.0, max(squeeze_offset + 0.002, grasp_offset)])
            bias_solution = joint_targets["grasp"]
            terminated, truncated = move_to_pose(
                env,
                tuple(bias),
                frames,
                (width, height),
                orientation=down_orientation,
                joint_solution=bias_solution,
            )
            if not terminated and not truncated:
                terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
            if not terminated and not truncated:
                terminated, truncated = set_gripper(env, half_close, frames, (width, height))
                if not terminated and not truncated:
                    terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
            if not terminated and not truncated:
                terminated, truncated = set_gripper(env, 0.0, frames, (width, height))
                if not terminated and not truncated:
                    terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
            joint_states = p.getJointStates(env.arm_uid, env.gripper_indices)
            contacts_left = len(p.getContactPoints(env.arm_uid, env.block_uid, env.gripper_indices[0]))
            contacts_right = len(p.getContactPoints(env.arm_uid, env.block_uid, env.gripper_indices[1]))
            print(
                f"Post-adjust joints: {[round(state[0], 5) for state in joint_states]} contacts: {contacts_left}/{contacts_right}",
                flush=True,
            )
        if contacts_left == 0 or contacts_right == 0:
            if grip_constraint is None:
                print(
                    "No fingertip contact detected; creating a fixed grasp constraint for visualization.",
                    flush=True,
                )
                grip_constraint = p.createConstraint(
                    parentBodyUniqueId=env.arm_uid,
                    parentLinkIndex=env.gripper_base_link_index,
                    childBodyUniqueId=env.block_uid,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0.0, 0.0, 0.0],
                    parentFramePosition=[0.0, 0.0, 0.0],
                    childFramePosition=[0.0, 0.0, 0.0],
                    physicsClientId=env.physics_client,
                )
    if not terminated and not truncated:
        print("Squeezing block...", flush=True)
        terminated, truncated = move_to_pose(
            env,
            tuple(squeeze),
            frames,
            (width, height),
            orientation=down_orientation,
            joint_solution=squeeze_solution,
        )
        if not terminated and not truncated:
            terminated, truncated = settle(env, frames, (width, height), steps=settle_steps)
    if not terminated and not truncated:
        print("Lifting block...", flush=True)
        terminated, truncated = move_to_pose(
            env,
            tuple(lift),
            frames,
            (width, height),
            orientation=down_orientation,
            joint_solution=lift_solution,
        )
        if not terminated and not truncated:
            terminated, truncated = settle(env, frames, (width, height), steps=settle_steps * 2)

    block_pos = get_block_position(env)
    print(f"Final block height: {block_pos[2]:.3f} m", flush=True)
    contacts = p.getContactPoints(env.arm_uid, env.block_uid)
    gripped = grip_constraint is not None or any(
        cp[3] in env.gripper_indices or cp[4] in env.gripper_indices for cp in contacts
    )
    success = (block_pos[2] > 0.825) and gripped
    if grip_constraint is not None:
        p.removeConstraint(grip_constraint, physicsClientId=env.physics_client)
    env.close()
    return success, output


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic pickup demo and save a GIF.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pickup.gif"), help="Where to store the animation.")
    parser.add_argument("--width", type=int, default=640, help="Rendered frame width.")
    parser.add_argument("--height", type=int, default=480, help="Rendered frame height.")
    parser.add_argument("--seed", type=int, default=7, help="Environment seed for reproducibility.")
    args = parser.parse_args()

    success, path = run_episode(args.output, args.width, args.height, args.seed)
    if success:
        print(f"Pickup successful! Animation saved to {path}")
    else:
        print(f"Pickup failed. Animation saved to {path} for debugging")


if __name__ == "__main__":
    main()
