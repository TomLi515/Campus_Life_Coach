"""PyTorch datasets for pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentations import AugmentationSettings, IMUAugmentor


class IMUWindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray, augmentor: Optional[IMUAugmentor] = None):
        if windows.ndim != 3:
            raise ValueError("windows must have shape (N, C, T)")
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.augmentor = augmentor

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int):
        window = self.windows[idx]
        label = self.labels[idx]
        if self.augmentor is not None:
            window = self.augmentor(window)
        return window, label


def load_splits(npz_path: Path, train_split: str, val_split: str, test_split: str) -> Dict[str, Dict[str, np.ndarray]]:
    with np.load(npz_path, allow_pickle=True) as data:
        splits = {}
        for split in [train_split, val_split, test_split]:
            windows_key = f"{split}_windows"
            if windows_key not in data:
                continue
            splits[split] = {
                "windows": data[windows_key].astype(np.float32),
                "labels": data[f"{split}_labels"].astype(np.int64),
            }
        splits["label_names"] = data["label_names"].tolist()
    return splits


def create_dataloaders(
    npz_path: Path,
    batch_size: int,
    num_workers: int,
    augmentation: Optional[AugmentationSettings],
    train_split: str,
    val_split: str,
    test_split: str,
) -> Dict[str, DataLoader]:
    splits = load_splits(npz_path, train_split, val_split, test_split)
    loaders: Dict[str, DataLoader] = {}

    augmentor = IMUAugmentor(augmentation) if augmentation else None

    train_data = splits.get(train_split)
    if train_data:
        dataset = IMUWindowDataset(train_data["windows"], train_data["labels"], augmentor=augmentor)
        loaders["train"] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    for split_name in [val_split, test_split]:
        split = splits.get(split_name)
        if not split:
            continue
        dataset = IMUWindowDataset(split["windows"], split["labels"], augmentor=None)
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    loaders["label_names"] = splits.get("label_names", [])
    return loaders


__all__ = ["IMUWindowDataset", "create_dataloaders", "AugmentationSettings"]
