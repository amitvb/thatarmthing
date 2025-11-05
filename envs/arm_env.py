"""Gymnasium environment for a 6-DOF arm with vision-guided block picking."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from vision.locator import BlockDetection, VisionBlockLocator


@dataclass
class ArmConfig:
    """Configuration of the robotic arm environment."""

    time_step: float = 1.0 / 60.0
    action_scale: float = 0.04
    gripper_scale: float = 0.01
    max_steps: int = 240
    camera_width: int = 224
    camera_height: int = 224
    camera_fov: float = 80.0
    near: float = 0.02
    far: float = 3.0


class ArmPickEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reinforcement learning environment for picking a block from a table."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        config: Optional[ArmConfig] = None,
        vision_device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.config = config or ArmConfig()
        self.vision_locator = VisionBlockLocator(device=vision_device)

        self.physics_client = None
        self.step_counter = 0
        self.assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        self.arm_uid = None
        self.block_uid = None
        self.table_uid = None
        self.joint_indices = list(range(6))
        self.gripper_indices = [6, 7]
        self.camera_link_index = 8  # camera_mount link index (after fingers)
        self.target_joint_positions = np.zeros(len(self.joint_indices))
        self.target_gripper = 0.015
        self.block_detection: Optional[BlockDetection] = None

        obs_dim = len(self.joint_indices) * 2 + 2 + 3 + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_indices) + 1,), dtype=np.float32)

        self._connect()
        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self._connect()
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        _ = plane_id  # silence lint
        table_path = os.path.join(self.assets_dir, "environment", "table.urdf")
        self.table_uid = p.loadURDF(table_path, basePosition=[0.4, 0.0, 0.0], useFixedBase=True)
        arm_path = os.path.join(self.assets_dir, "arm", "arm6dof.urdf")
        self.arm_uid = p.loadURDF(
            arm_path,
            basePosition=[0.0, -0.3, 0.75],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )
        block_path = os.path.join(self.assets_dir, "objects", "block.urdf")
        block_pos = self._sample_block_position()
        self.block_uid = p.loadURDF(block_path, block_pos, useFixedBase=False)

        self.target_joint_positions = np.zeros(len(self.joint_indices))
        self.target_gripper = 0.018
        self.step_counter = 0

        # Let things settle
        for _ in range(10):
            p.stepSimulation()

        observation = self._get_observation()
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.target_joint_positions += action[: len(self.joint_indices)] * self.config.action_scale
        self.target_joint_positions = np.clip(self.target_joint_positions, -math.pi, math.pi)
        self.target_gripper += action[-1] * self.config.gripper_scale
        self.target_gripper = float(np.clip(self.target_gripper, 0.0, 0.02))

        for idx, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_uid,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(self.target_joint_positions[idx]),
                force=120,
                positionGain=0.3,
                velocityGain=0.5,
            )

        # Symmetric gripper motion
        for i, gripper_index in enumerate(self.gripper_indices):
            target = self.target_gripper if i == 0 else self.target_gripper
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_uid,
                jointIndex=gripper_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=30,
                positionGain=0.8,
                velocityGain=1.0,
            )

        for _ in range(4):
            p.stepSimulation()

        self.step_counter += 1
        observation = self._get_observation()
        reward, terminated = self._compute_reward_and_termination()
        truncated = self.step_counter >= self.config.max_steps
        info = {"block_detection_confidence": self.block_detection.confidence if self.block_detection else 0.0}
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            raise RuntimeError("Rendering is only available in human mode")
        return None

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        if self.physics_client is None:
            connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT
            self.physics_client = p.connect(connection_mode)

    def _sample_block_position(self):
        x = self.np_random.uniform(0.2, 0.6)
        y = self.np_random.uniform(-0.15, 0.15)
        z = 0.775
        return [float(x), float(y), float(z)]

    def _get_observation(self) -> np.ndarray:
        joint_states = p.getJointStates(self.arm_uid, self.joint_indices + self.gripper_indices)
        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)
        gripper_pos = joint_positions[-2:]

        camera_rgb, camera_depth = self._render_camera()
        self.block_detection = self.vision_locator.locate(camera_rgb)
        block_world = self._project_detection_to_world(self.block_detection, camera_depth)
        if block_world is None or not np.all(np.isfinite(block_world)):
            block_world = np.array(p.getBasePositionAndOrientation(self.block_uid)[0])
        self.block_detection.world_position = block_world
        block_in_base = self._world_to_base(block_world)

        obs = np.concatenate(
            [
                joint_positions[:-2],
                joint_velocities[:-2],
                [gripper_pos.mean()],
                block_in_base,
                [self.block_detection.confidence],
            ]
        ).astype(np.float32)
        return obs

    def _render_camera(self):
        cam_state = p.getLinkState(self.arm_uid, self.camera_link_index, computeForwardKinematics=True)
        cam_pos = np.array(cam_state[0])
        cam_orn = np.array(cam_state[1])
        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot @ np.array([0, 0, 1])
        up = rot @ np.array([0, 1, 0])
        right = rot @ np.array([1, 0, 0])
        view_matrix = p.computeViewMatrix(cam_pos, cam_pos + forward, up)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=float(self.config.camera_width) / float(self.config.camera_height),
            nearVal=self.config.near,
            farVal=self.config.far,
        )
        width = self.config.camera_width
        height = self.config.camera_height
        image = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)
        depth_buffer = np.reshape(image[3], (height, width))
        return rgb, depth_buffer

    def _project_detection_to_world(self, detection: BlockDetection, depth_buffer: np.ndarray) -> np.ndarray:
        width = self.config.camera_width
        height = self.config.camera_height
        px, py = detection.pixel
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        depth_value = float(depth_buffer[py, px])
        distance = VisionBlockLocator.depth_buffer_to_meters(depth_value, self.config.near, self.config.far)

        cam_state = p.getLinkState(self.arm_uid, self.camera_link_index, computeForwardKinematics=True)
        cam_pos = np.array(cam_state[0])
        cam_orn = np.array(cam_state[1])
        rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot @ np.array([0, 0, 1])
        up = rot @ np.array([0, 1, 0])
        right = rot @ np.array([1, 0, 0])
        clipped_pixel = (px, py)
        ray_dir = VisionBlockLocator.pixel_to_ray(
            clipped_pixel,
            width,
            height,
            self.config.camera_fov,
            right,
            up,
            forward,
        )
        if not np.isfinite(distance):
            distance = 0.3
        world_pos = cam_pos + ray_dir * distance
        detection.pixel = clipped_pixel
        return world_pos

    def _world_to_base(self, point: np.ndarray) -> np.ndarray:
        if point is None:
            return np.zeros(3, dtype=np.float32)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.arm_uid)
        inv_pos, inv_orn = p.invertTransform(base_pos, base_orn)
        local_pos, _ = p.multiplyTransforms(inv_pos, inv_orn, point.tolist(), [0, 0, 0, 1])
        return np.array(local_pos, dtype=np.float32)

    def _compute_reward_and_termination(self):
        block_pos, _ = p.getBasePositionAndOrientation(self.block_uid)
        ee_state = p.getLinkState(self.arm_uid, self.camera_link_index, computeForwardKinematics=True)
        ee_pos = np.array(ee_state[0])
        dist = np.linalg.norm(np.array(block_pos) - ee_pos)
        reward = -dist

        gripper_states = p.getJointStates(self.arm_uid, self.gripper_indices)
        gripper_width = sum(state[0] for state in gripper_states) / len(gripper_states)
        reward -= 5.0 * abs(gripper_width - self.target_gripper)

        success = False
        table_height = 0.775
        block_height = block_pos[2]
        contacts_left = p.getContactPoints(self.arm_uid, self.block_uid, self.gripper_indices[0])
        contacts_right = p.getContactPoints(self.arm_uid, self.block_uid, self.gripper_indices[1])
        if block_height > table_height + 0.05 and contacts_left and contacts_right:
            reward += 15.0
            success = True
        reward += -0.01
        return reward, success
