"""UCI-HAR dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy import signal

from ..config import LabelSpace, SignalSpec
from ..utils.logging import get_logger
from .base_dataset import BaseDatasetBuilder, DatasetBuildResult, SlidingWindowDataset
from .utils import download_file, ensure_directory, extract_archive

LOGGER = get_logger(__name__)

_UCI_ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


@dataclass
class UCIHARMetadata:
    source: str
    window_length_seconds: float
    original_sampling_rate: int
    target_sampling_rate: int
    num_subjects_train: int
    num_subjects_test: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class UCIHARBuilder(BaseDatasetBuilder):
    name = "uci_har"

    def prepare(self) -> SlidingWindowDataset:
        if not self.config.get("enabled", True):
            raise RuntimeError("UCI-HAR builder invoked but disabled in config")

        dataset_root = self.raw_dir / "uci_har"
        ensure_directory(dataset_root)
        archive_path = dataset_root / "uci_har.zip"
        download_file(self.config["url"], archive_path)
        extract_archive(archive_path, dataset_root)

        extracted_dir = dataset_root / self.config.get("extract_dirname", "UCI HAR Dataset")
        if not extracted_dir.exists():
            zipped_alt = extracted_dir.with_suffix('.zip')
            if zipped_alt.exists():
                LOGGER.info("Found nested archive %s; extracting", zipped_alt)
                extract_archive(zipped_alt, dataset_root)
        if not extracted_dir.exists():
            raise FileNotFoundError(f"Expected extracted directory {extracted_dir} not found")
        LOGGER.info("Preparing UCI-HAR from %s", extracted_dir)

        train_data = self._load_split(extracted_dir, "train")
        test_data = self._load_split(extracted_dir, "test")

        unique_train_subjects = np.unique(train_data["subjects"])
        val_ratio = float(self.config.get("val_split_ratio", 0.15))
        val_count = max(1, int(round(len(unique_train_subjects) * val_ratio)))
        val_subjects = set(unique_train_subjects[-val_count:])
        train_subjects = [s for s in unique_train_subjects if s not in val_subjects]

        train_mask = np.isin(train_data["subjects"], list(train_subjects))
        val_mask = np.isin(train_data["subjects"], list(val_subjects))

        train_result = DatasetBuildResult(
            windows=train_data["windows"][train_mask],
            labels=train_data["labels"][train_mask],
            label_names=train_data["label_names"],
            subjects=train_data["subjects"][train_mask],
            metadata={"dataset": "UCI-HAR", "split": "train"},
        )
        val_result = DatasetBuildResult(
            windows=train_data["windows"][val_mask],
            labels=train_data["labels"][val_mask],
            label_names=train_data["label_names"],
            subjects=train_data["subjects"][val_mask],
            metadata={"dataset": "UCI-HAR", "split": "val"},
        )
        test_result = DatasetBuildResult(
            windows=test_data["windows"],
            labels=test_data["labels"],
            label_names=test_data["label_names"],
            subjects=test_data["subjects"],
            metadata={"dataset": "UCI-HAR", "split": "test"},
        )

        metadata = UCIHARMetadata(
            source="UCI Machine Learning Repository",
            window_length_seconds=2.56,
            original_sampling_rate=50,
            target_sampling_rate=self.signal_spec.target_sampling_rate,
            num_subjects_train=int(len(unique_train_subjects)),
            num_subjects_test=int(len(np.unique(test_data["subjects"]))),
            notes="Original 2.56 s windows resampled to 3.0 s (150 samples) via Fourier resampling.",
        ).to_dict()

        return SlidingWindowDataset(
            splits={"train": train_result, "val": val_result, "test": test_result},
            label_names=train_data["label_names"],
            metadata=metadata,
        )

    def _load_split(self, root: Path, split: str) -> Dict[str, Any]:
        inertial_dir = (root / split / "Inertial Signals").resolve()
        subject_path = (root / split / f"subject_{split}.txt").resolve()
        label_path = (root / split / f"y_{split}.txt").resolve()
        if not subject_path.exists():
            raise FileNotFoundError(f"{subject_path} not found. Check that UCI-HAR was extracted correctly.")
        if not label_path.exists():
            raise FileNotFoundError(f"{label_path} not found. Check that UCI-HAR was extracted correctly.")

        subjects = np.loadtxt(subject_path, dtype=int)
        y = np.loadtxt(label_path, dtype=int)

        def _load_channel(filename: str) -> np.ndarray:
            path = (inertial_dir / filename).resolve()
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Verify UCI-HAR Inertial Signals are present.")
            return np.loadtxt(path)

        acc_x = _load_channel(f"total_acc_x_{split}.txt")
        acc_y = _load_channel(f"total_acc_y_{split}.txt")
        acc_z = _load_channel(f"total_acc_z_{split}.txt")
        gyro_x = _load_channel(f"body_gyro_x_{split}.txt")
        gyro_y = _load_channel(f"body_gyro_y_{split}.txt")
        gyro_z = _load_channel(f"body_gyro_z_{split}.txt")

        windows = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], axis=1)  # (N, C, 128)
        windows = self._resample_windows(windows)
        windows = self._standardize_per_subject(windows, subjects)

        aggregated_labels = self._map_labels(y)
        encoded_labels = self._encode_labels(aggregated_labels)

        return {
            "windows": windows,
            "labels": encoded_labels,
            "label_names": list(self.label_space.all_labels()),
            "subjects": subjects,
        }

    def _resample_windows(self, windows: np.ndarray) -> np.ndarray:
        target_length = self.signal_spec.window_size_samples()
        if windows.shape[-1] == target_length:
            return windows.astype(np.float32)
        resampled = np.zeros((windows.shape[0], windows.shape[1], target_length), dtype=np.float32)
        for idx in range(windows.shape[0]):
            resampled[idx] = signal.resample(windows[idx], target_length, axis=-1)
        return resampled

    def _map_labels(self, raw_labels: Sequence[int]) -> List[str]:
        mapping: Mapping[str, Sequence[str]] = self.config.get("mapping", {})
        available_labels = self.label_space.all_labels()
        aggregated: List[str] = []
        for label in raw_labels:
            raw_name = _UCI_ACTIVITY_LABELS[int(label)]
            mapped = None
            for target_label, source_labels in mapping.items():
                if raw_name in source_labels:
                    mapped = target_label
                    break
            if mapped is None:
                if "Other" in available_labels:
                    mapped = "Other"
                else:
                    continue
            aggregated.append(mapped)
        return aggregated

    def _standardize_per_subject(self, windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        out = windows.astype(np.float32)
        unique_sub = np.unique(subjects)
        for subject in unique_sub:
            mask = subjects == subject
            subject_windows = out[mask]
            mean = subject_windows.mean(axis=(0, 2), keepdims=True)
            std = subject_windows.std(axis=(0, 2), keepdims=True)
            std[std < 1e-6] = 1.0
            out[mask] = (subject_windows - mean) / std
        return out
