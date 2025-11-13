"""MotionSense dataset preparation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .base_dataset import BaseDatasetBuilder, DatasetBuildResult, SlidingWindowDataset
from .utils import download_file, ensure_directory, extract_archive

LOGGER = get_logger(__name__)

_ACTIVITY_PATTERN = re.compile(r"(walking|jogging|sitting|standing|upstairs|downstairs|lying|lyingdown)", re.IGNORECASE)
_ACTIVITY_CODE_PATTERN = re.compile(r"^(dws|ups|wlk|jog|sit|std|lyi)_", re.IGNORECASE)
_ACTIVITY_CODE_TO_NAME = {
    "dws": "downstairs",
    "ups": "upstairs",
    "wlk": "walking",
    "jog": "jogging",
    "sit": "sitting",
    "std": "standing",
    "lyi": "lying",
}
_SUBJECT_PATTERN = re.compile(r"sub_(\d+)")


@dataclass
class MotionSenseMetadata:
    source: str
    sampling_rate_hz: int
    window_size_seconds: float
    window_step_seconds: float
    num_subjects: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class MotionSenseBuilder(BaseDatasetBuilder):
    name = "motionsense"

    def prepare(self) -> SlidingWindowDataset:
        if not self.config.get("enabled", True):
            raise RuntimeError("MotionSense builder invoked but disabled in config")

        local_dir = self.config.get("local_dir")
        if local_dir:
            extracted_dir = Path(local_dir).expanduser().resolve()
            if not extracted_dir.exists():
                raise FileNotFoundError(f"Configured MotionSense local_dir {extracted_dir} does not exist.")
            LOGGER.info("Using MotionSense data from local directory %s", extracted_dir)
        else:
            dataset_root = self.raw_dir / "motionsense"
            ensure_directory(dataset_root)
            archive_path = dataset_root / "motionsense.zip"
            download_file(self.config["url"], archive_path)
            extract_archive(archive_path, dataset_root)
            extracted_dir = dataset_root / self.config.get("extract_dirname", "MotionSense")
            if not extracted_dir.exists():
                raise FileNotFoundError(f"Extracted MotionSense directory {extracted_dir} not found.")
        csv_files = sorted(extracted_dir.glob("**/*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {extracted_dir}")

        windows, labels, subjects = self._build_windows(csv_files)
        windows, subjects = self._standardize_per_subject(windows, subjects)
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
            metadata={"dataset": "MotionSense", "split": "train"},
        )
        val_result = DatasetBuildResult(
            windows=windows[val_mask],
            labels=encoded_labels[val_mask],
            label_names=list(self.label_space.all_labels()),
            subjects=subjects[val_mask],
            metadata={"dataset": "MotionSense", "split": "val"},
        )
        test_result = DatasetBuildResult(
            windows=windows[test_mask],
            labels=encoded_labels[test_mask],
            label_names=list(self.label_space.all_labels()),
            subjects=subjects[test_mask],
            metadata={"dataset": "MotionSense", "split": "test"},
        )

        metadata = MotionSenseMetadata(
            source="Kaggle MotionSense",
            sampling_rate_hz=50,
            window_size_seconds=self.signal_spec.window_size_seconds,
            window_step_seconds=self.signal_spec.window_step_seconds,
            num_subjects=len(np.unique(subjects)),
            notes="UserAcceleration (acc) + rotationRate (gyro) used; lying activity absent.",
        ).to_dict()

        return SlidingWindowDataset(
            splits={"train": train_result, "val": val_result, "test": test_result},
            label_names=list(self.label_space.all_labels()),
            metadata=metadata,
        )

    def _build_windows(self, csv_files: Sequence[Path]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        window_size = self.signal_spec.window_size_samples()
        step_size = self.signal_spec.window_step_samples()
        label_mapping: Mapping[str, Sequence[str]] = self.config.get("mapping", {})
        available_labels = self.label_space.all_labels()

        window_list: List[np.ndarray] = []
        label_list: List[str] = []
        subject_list: List[int] = []

        for csv_path in csv_files:
            if any(part.startswith("__MACOSX") for part in csv_path.parts):
                continue
            if csv_path.name.startswith("._"):
                continue
            activity = self._infer_activity(csv_path)
            if activity is None:
                continue
            target_label = self._map_activity(activity, label_mapping, available_labels)
            if target_label is None:
                continue
            subject_id = self._infer_subject(csv_path)
            df = self._read_csv(csv_path)
            first_col = df.columns[0]
            if isinstance(first_col, str) and first_col.lower().startswith("unnamed"):
                df = df.drop(columns=first_col)
            for col in ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z", "rotationRate.x", "rotationRate.y", "rotationRate.z"]:
                if col not in df.columns:
                    raise KeyError(f"Expected column {col} in {csv_path}")
            sequence = df[[
                "userAcceleration.x",
                "userAcceleration.y",
                "userAcceleration.z",
                "rotationRate.x",
                "rotationRate.y",
                "rotationRate.z",
            ]].to_numpy(dtype=np.float32).T  # (C, T)
            if sequence.shape[1] < window_size:
                continue
            for start in range(0, sequence.shape[1] - window_size + 1, step_size):
                window = sequence[:, start : start + window_size]
                window_list.append(window)
                label_list.append(target_label)
                subject_list.append(subject_id)

        if not window_list:
            raise RuntimeError("No windows extracted from MotionSense dataset")


        windows = np.stack(window_list, axis=0)
        subjects = np.array(subject_list, dtype=int)
        return windows, label_list, subjects

    def _read_csv(self, path: Path) -> pd.DataFrame:
        encodings = ["utf-8", "utf-8-sig", "latin1"]
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as err:
                last_error = err
        if last_error is not None:
            raise last_error
        # Fallback, should not happen
        return pd.read_csv(path)

    def _map_activity(self, activity: str, mapping: Mapping[str, Sequence[str]], labels: Sequence[str]) -> str | None:
        normalized_activity = activity.lower()
        for target_label, source_labels in mapping.items():
            if normalized_activity in [s.lower() for s in source_labels]:
                return target_label
        if "Other" in labels:
            return "Other"
        return None

    def _infer_activity(self, path: Path) -> str | None:
        # Check parent directory name first (Kaggle format: dws_11, wlk_1, etc.)
        parent_name = path.parent.name.lower()
        code_match = _ACTIVITY_CODE_PATTERN.match(parent_name)
        if code_match:
            code = code_match.group(1).lower()
            return _ACTIVITY_CODE_TO_NAME.get(code)
        # Fall back to checking file stem
        stem = path.stem.lower()
        match = _ACTIVITY_PATTERN.search(stem)
        if match:
            return match.group(1).lower()
        return None

    def _infer_subject(self, path: Path) -> int:
        match = _SUBJECT_PATTERN.search(path.stem)
        if not match:
            raise ValueError(f"Cannot infer subject from path {path}")
        return int(match.group(1))

    def _standardize_per_subject(self, windows: np.ndarray, subjects: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        windows = windows.astype(np.float32)
        for subject in np.unique(subjects):
            mask = subjects == subject
            mean = windows[mask].mean(axis=(0, 2), keepdims=True)
            std = windows[mask].std(axis=(0, 2), keepdims=True)
            std[std < 1e-6] = 1.0
            windows[mask] = (windows[mask] - mean) / std
        return windows, subjects
