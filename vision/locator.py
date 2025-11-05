"""Vision-based block localization using a pre-trained OpenCLIP model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class BlockDetection:
    """Detected block location in image and 3D space."""

    pixel: Tuple[int, int]
    confidence: float
    world_position: Optional[np.ndarray]


class VisionBlockLocator:
    """Use a large pre-trained vision-language model to locate the block.

    The locator samples a set of overlapping patches from the RGB camera frame,
    scores them with OpenCLIP using the text prompt "a small colored block on a
    table", and returns the center of the highest-scoring patch. The
    world-space coordinate is filled in by the environment after projecting the
    pixel through the depth map.
    """

    def __init__(self, device: Optional[str] = None, patch_size: int = 96, stride: int = 48) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text = ["a small colored block on a table"]
            tokens = self.tokenizer(text).to(self.device)
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def locate(self, rgb: np.ndarray) -> BlockDetection:
        """Locate the block in an RGB image.

        Args:
            rgb: HxWx3 uint8 numpy array in RGB order.

        Returns:
            A :class:`BlockDetection` with pixel coordinates and confidence.
        """

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB image must be HxWx3")

        pil = Image.fromarray(rgb)
        width, height = pil.size
        patches = []
        centers = []

        # Cover the image with a dense grid of candidate patches.
        for top in range(0, height - self.patch_size + 1, self.stride):
            for left in range(0, width - self.patch_size + 1, self.stride):
                box = (left, top, left + self.patch_size, top + self.patch_size)
                patch = pil.crop(box)
                patches.append(self.preprocess(patch).to(self.device))
                centers.append((left + self.patch_size // 2, top + self.patch_size // 2))

        if not patches:
            # Fallback: resize entire image if the frame is smaller than patch size.
            resized = pil.resize((self.patch_size, self.patch_size))
            patches.append(self.preprocess(resized).to(self.device))
            centers.append((width // 2, height // 2))

        image_tensor = torch.stack(patches)
        image_features = self.model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ self.text_features.T
        best_idx = int(torch.argmax(sims).item())
        confidence = float(sims[best_idx].item())
        best_center = centers[best_idx]
        return BlockDetection(pixel=best_center, confidence=confidence, world_position=None)

    @staticmethod
    def depth_buffer_to_meters(depth_buffer: float, near: float, far: float) -> float:
        """Convert PyBullet depth buffer values to metric distances."""

        depth_buffer = float(depth_buffer)
        return (2.0 * near * far) / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))

    @staticmethod
    def pixel_to_ray(
        pixel: Tuple[int, int],
        width: int,
        height: int,
        fov_degrees: float,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
    ) -> np.ndarray:
        """Convert a pixel to a unit ray direction in world coordinates."""

        u, v = pixel
        aspect = width / height
        fov = math.radians(fov_degrees)
        image_plane_height = 2.0 * math.tan(fov / 2.0)
        image_plane_width = image_plane_height * aspect
        nx = ((u + 0.5) / width - 0.5) * image_plane_width
        ny = (0.5 - (v + 0.5) / height) * image_plane_height
        direction = forward + nx * right + ny * up
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return forward
        return direction / norm
