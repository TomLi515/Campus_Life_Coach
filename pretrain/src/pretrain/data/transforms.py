"""Signal processing helpers for resampling and windowing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def resample_signal(sequence: np.ndarray, original_rate: int, target_rate: int, method: str = "resample_poly") -> np.ndarray:
    """Resample along the last axis to the target rate."""

    if original_rate == target_rate:
        return sequence

    if method == "resample_poly":
        gcd = np.gcd(original_rate, target_rate)
        up = target_rate // gcd
        down = original_rate // gcd
        return signal.resample_poly(sequence, up=up, down=down, axis=-1)
    if method == "fft":
        target_length = int(round(sequence.shape[-1] * target_rate / original_rate))
        return signal.resample(sequence, num=target_length, axis=-1)
    if method == "decimate":
        factor = int(round(original_rate / target_rate))
        return signal.decimate(sequence, factor, axis=-1, ftype="fir", zero_phase=True)
    raise ValueError(f"Unsupported resample method: {method}")


def sliding_window(sequence: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Generate sliding windows along time axis (last dimension)."""

    if sequence.shape[-1] < window_size:
        raise ValueError("Sequence shorter than window size")
    num_steps = 1 + (sequence.shape[-1] - window_size) // step_size
    shape = sequence.shape[:-1] + (num_steps, window_size)
    strides = sequence.strides[:-1] + (step_size * sequence.strides[-1], sequence.strides[-1])
    return np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)


def zscore_normalize(sequence: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (sequence - mean[..., None]) / (std[..., None] + eps)


def compute_channel_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std for data shaped (N, C, T)."""

    mean = data.mean(axis=(0, 2))
    std = data.std(axis=(0, 2))
    return mean, std


__all__ = [
    "resample_signal",
    "sliding_window",
    "zscore_normalize",
    "compute_channel_stats",
]
