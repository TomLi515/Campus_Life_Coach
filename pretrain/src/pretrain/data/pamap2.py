"""PAMAP2 wrist sensor dataset preparation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base_dataset import BaseDatasetBuilder, DatasetBuildResult, SlidingWindowDataset
from .transforms import resample_signal
from .utils import download_file, ensure_directory, extract_archive

LOGGER = get_logger(__name__)

PAMAP2_ACTIVITY_LABELS = {
    0: "other",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

COLUMN_NAMES = [
    "timestamp", "activity_id", "heart_rate",
    "hand_temp",
    "hand_acc_16g_x", "hand_acc_16g_y", "hand_acc_16g_z",
    "hand_acc_6g_x", "hand_acc_6g_y", "hand_acc_6g_z",
    "hand_gyro_x", "hand_gyro_y", "hand_gyro_z",
    "hand_mag_x", "hand_mag_y", "hand_mag_z",
    "hand_orient_w", "hand_orient_x", "hand_orient_y", "hand_orient_z",
    "chest_temp",
    "chest_acc_16g_x", "chest_acc_16g_y", "chest_acc_16g_z",
    "chest_acc_6g_x", "chest_acc_6g_y", "chest_acc_6g_z",
    "chest_gyro_x", "chest_gyro_y", "chest_gyro_z",
    "chest_mag_x", "chest_mag_y", "chest_mag_z",
    "chest_orient_w", "chest_orient_x", "chest_orient_y", "chest_orient_z",
    "ankle_temp",
    "ankle_acc_16g_x", "ankle_acc_16g_y", "ankle_acc_16g_z",
    "ankle_acc_6g_x", "ankle_acc_6g_y", "ankle_acc_6g_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "ankle_orient_w", "ankle_orient_x", "ankle_orient_y", "ankle_orient_z",
]

HAND_COLUMNS = [
    "hand_acc_6g_x",
    "hand_acc_6g_y",
    "hand_acc_6g_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
]


@dataclass
class PAMAP2Metadata:
    source: str
    original_sampling_rate: int
    target_sampling_rate: int
    window_size_seconds: float
    window_step_seconds: float
    num_subjects: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class PAMAP2Builder(BaseDatasetBuilder):
    name = "pamap2"

    def prepare(self) -> SlidingWindowDataset:
        if not self.config.get("enabled", True):
            raise RuntimeError("PAMAP2 builder invoked but disabled in config")

        dataset_root = self.raw_dir / "pamap2"
        ensure_directory(dataset_root)
        archive_path = dataset_root / "pamap2.zip"
        download_file(self.config["url"], archive_path)
        extract_archive(archive_path, dataset_root)

        extracted_dir = dataset_root / self.config.get("extract_dirname", "PAMAP2_Dataset")
        if not extracted_dir.exists():
            nested_archive = extracted_dir.with_suffix(".zip")
            if nested_archive.exists():
                LOGGER.info("Extracting nested PAMAP2 archive %s", nested_archive)
                extract_archive(nested_archive, dataset_root)
            elif (nested_archive := dataset_root / "PAMAP2_Dataset.zip").exists():
                LOGGER.info("Extracting nested PAMAP2 archive %s", nested_archive)
                extract_archive(nested_archive, dataset_root)
            if not extracted_dir.exists():
                raise FileNotFoundError(f"Expected extracted directory {extracted_dir} not found after extraction.")

        protocol_dir = extracted_dir / "Protocol"
        if not protocol_dir.exists():
            # Some distributions use lowercase directory names
            alt_protocol_dir = extracted_dir / "protocol"
            if alt_protocol_dir.exists():
                protocol_dir = alt_protocol_dir
            else:
                raise FileNotFoundError(f"Protocol directory missing: {protocol_dir}")

        dat_files = sorted(protocol_dir.glob("subject*.dat"))
        if not dat_files:
            raise FileNotFoundError(f"No subject .dat files found in {protocol_dir}")

        windows, labels, subjects = self._build_windows(dat_files)
        windows = self._standardize_per_subject(windows, subjects)
        encoded_labels = self._encode_labels(labels)

        train_subjects = [str(s) for s in self.config.get("train_subjects", [])]
        val_subjects = [str(s) for s in self.config.get("val_subjects", [])]
        test_subjects = [str(s) for s in self.config.get("test_subjects", [])]

        subj_str = np.array([str(s) for s in subjects])
        train_mask = np.isin(subj_str, train_subjects)
        val_mask = np.isin(subj_str, val_subjects)
        test_mask = np.isin(subj_str, test_subjects)

        train_result = DatasetBuildResult(
            windows=windows[train_mask],
            labels=encoded_labels[train_mask],
            label_names=list(self.label_space.all_labels()),
            subjects=subjects[train_mask],
            metadata={"dataset": "PAMAP2", "split": "train"},
        )
        val_result = DatasetBuildResult(
            windows=windows[val_mask],
            labels=encoded_labels[val_mask],
            label_names=list(self.label_space.all_labels()),
            subjects=subjects[val_mask],
            metadata={"dataset": "PAMAP2", "split": "val"},
        )
        test_result = DatasetBuildResult(
            windows=windows[test_mask],
            labels=encoded_labels[test_mask],
            label_names=list(self.label_space.all_labels()),
            subjects=subjects[test_mask],
            metadata={"dataset": "PAMAP2", "split": "test"},
        )

        metadata = PAMAP2Metadata(
            source="PAMAP2 Physical Activity Monitoring",
            original_sampling_rate=100,
            target_sampling_rate=self.signal_spec.target_sampling_rate,
            window_size_seconds=self.signal_spec.window_size_seconds,
            window_step_seconds=self.signal_spec.window_step_seconds,
            num_subjects=len(np.unique(subjects)),
            notes="Hand IMU (acc 6g + gyro) resampled from 100 Hz to 50 Hz.",
        ).to_dict()

        return SlidingWindowDataset(
            splits={"train": train_result, "val": val_result, "test": test_result},
            label_names=list(self.label_space.all_labels()),
            metadata=metadata,
        )

    def _build_windows(self, dat_files: Sequence[Path]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        window_size = self.signal_spec.window_size_samples()
        step_size = self.signal_spec.window_step_samples()
        mapping: Mapping[str, Sequence[str]] = self.config.get("mapping", {})
        available_labels = self.label_space.all_labels()

        windows: List[np.ndarray] = []
        labels: List[str] = []
        subjects: List[int] = []

        for dat_path in dat_files:
            subject_id = self._infer_subject(dat_path)
            df = pd.read_csv(dat_path, delim_whitespace=True, header=None, names=COLUMN_NAMES)
            df = df.replace({"nan": np.nan}).dropna(subset=HAND_COLUMNS + ["activity_id"])
            df = df.reset_index(drop=True)
            activities = df["activity_id"].astype(int).to_numpy()
            segments = self._segment_indices(activities)
            for start, end in segments:
                activity_id = int(activities[start])
                raw_label = PAMAP2_ACTIVITY_LABELS.get(activity_id, "other")
                target_label = self._map_activity(raw_label, mapping, available_labels)
                if target_label is None:
                    continue
                segment = df.iloc[start:end]
                sequence = segment[HAND_COLUMNS].to_numpy(dtype=np.float32).T  # (C, T)
                if sequence.shape[1] < window_size:
                    continue
                resampled = resample_signal(sequence, original_rate=100, target_rate=self.signal_spec.target_sampling_rate, method=self.config.get("resample_method", "resample_poly"))
                for idx in range(0, resampled.shape[1] - window_size + 1, step_size):
                    window = resampled[:, idx : idx + window_size]
                    windows.append(window)
                    labels.append(target_label)
                    subjects.append(subject_id)

        if not windows:
            raise RuntimeError("No windows extracted from PAMAP2 dataset")

        return np.stack(windows, axis=0), labels, np.array(subjects, dtype=int)

    def _segment_indices(self, activity_array: np.ndarray) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        start = 0
        for idx in range(1, len(activity_array)):
            if activity_array[idx] != activity_array[idx - 1]:
                segments.append((start, idx))
                start = idx
        segments.append((start, len(activity_array)))
        return segments

    def _map_activity(self, raw_label: str, mapping: Mapping[str, Sequence[str]], labels: Sequence[str]) -> str | None:
        normalized = raw_label.lower()
        for target_label, source_labels in mapping.items():
            if normalized in [s.lower() for s in source_labels]:
                return target_label
        if "Other" in labels:
            return "Other"
        return None

    def _infer_subject(self, path: Path) -> int:
        match = re.search(r"subject(\d+)", path.stem)
        if not match:
            raise ValueError(f"Cannot infer subject id from {path}")
        return int(match.group(1))

    def _standardize_per_subject(self, windows: np.ndarray, subjects: np.ndarray) -> np.ndarray:
        windows = windows.astype(np.float32)
        for subject in np.unique(subjects):
            mask = subjects == subject
            mean = windows[mask].mean(axis=(0, 2), keepdims=True)
            std = windows[mask].std(axis=(0, 2), keepdims=True)
            std[std < 1e-6] = 1.0
            windows[mask] = (windows[mask] - mean) / std
        return windows
