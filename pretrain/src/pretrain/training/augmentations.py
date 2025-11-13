"""Data augmentation utilities for IMU windows."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass
class AugmentationSettings:
    sampling_rate: int
    time_jitter_seconds: float = 0.0
    noise_std: float = 0.0
    rotation_degrees: float = 0.0
    drop_channels_prob: float = 0.0


class IMUAugmentor:
    def __init__(self, settings: AugmentationSettings):
        self.settings = settings
        self.max_shift = int(round(settings.time_jitter_seconds * settings.sampling_rate))

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        # window shape: (C, T)
        if self.max_shift > 0:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                window = torch.roll(window, shifts=shift, dims=-1)
        if self.settings.noise_std > 0:
            noise = torch.randn_like(window) * self.settings.noise_std
            window = window + noise
        if self.settings.rotation_degrees > 0:
            window = self._apply_rotation(window)
        if self.settings.drop_channels_prob > 0:
            mask = torch.rand(window.shape[0], device=window.device) < self.settings.drop_channels_prob
            window = window.clone()
            window[mask] = 0.0
        return window

    def _apply_rotation(self, window: torch.Tensor) -> torch.Tensor:
        max_rad = math.radians(self.settings.rotation_degrees)
        angles = torch.empty(3, device=window.device).uniform_(-max_rad, max_rad)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)

        rot_x = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=window.dtype, device=window.device)
        rot_y = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=window.dtype, device=window.device)
        rot_z = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=window.dtype, device=window.device)
        rotation = rot_z @ rot_y @ rot_x

        rotated = window.clone()
        if window.shape[0] >= 3:
            rotated[:3] = rotation @ window[:3]
        if window.shape[0] >= 6:
            rotated[3:6] = rotation @ window[3:6]
        return rotated


__all__ = ["IMUAugmentor", "AugmentationSettings"]
