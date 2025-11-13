"""Base classes and helpers for inertial dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ..config import LabelSpace, SignalSpec
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DatasetBuildResult:
    """Holds windowed sensor data for a given split."""

    windows: np.ndarray  # shape (N, C, T)
    labels: np.ndarray  # label indices
    label_names: List[str]
    subjects: np.ndarray  # subject identifiers aligned with windows
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.windows.ndim != 3:
            raise ValueError(f"windows must have shape (N, C, T); got {self.windows.shape}")
        if self.labels.shape[0] != self.windows.shape[0]:
            raise ValueError("labels must align with windows")
        if self.subjects.shape[0] != self.windows.shape[0]:
            raise ValueError("subjects must align with windows")

    def num_samples(self) -> int:
        return int(self.windows.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "windows": self.windows,
            "labels": self.labels,
            "label_names": np.array(self.label_names),
            "subjects": self.subjects,
            "metadata": self.metadata,
        }

    @staticmethod
    def concat(results: Iterable["DatasetBuildResult"], label_names: Sequence[str]) -> "DatasetBuildResult":
        windows_list, labels_list, subjects_list, metadata_list = [], [], [], []
        for result in results:
            windows_list.append(result.windows)
            labels_list.append(result.labels)
            subjects_list.append(result.subjects)
            metadata_list.append(result.metadata)
        windows = np.concatenate(windows_list, axis=0) if windows_list else np.empty((0, 0, 0))
        labels = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,))
        subjects = np.concatenate(subjects_list, axis=0) if subjects_list else np.empty((0,), dtype=np.int64)
        merged_metadata: Dict[str, Any] = {f"dataset_{idx}": meta for idx, meta in enumerate(metadata_list)}
        return DatasetBuildResult(windows=windows, labels=labels, label_names=list(label_names), subjects=subjects, metadata=merged_metadata)


@dataclass
class SlidingWindowDataset:
    """Container for all dataset splits."""

    splits: Dict[str, DatasetBuildResult]
    label_names: List[str]
    metadata: Dict[str, Any]

    def save_npz(self, path: Path) -> None:
        LOGGER.info("Saving window dataset to %s", path)
        np.savez_compressed(
            path,
            **{f"{split}_windows": result.windows for split, result in self.splits.items()},
            **{f"{split}_labels": result.labels for split, result in self.splits.items()},
            **{f"{split}_subjects": result.subjects for split, result in self.splits.items()},
            label_names=np.array(self.label_names),
            metadata=np.array([self.metadata], dtype=object),
        )

    @classmethod
    def from_npz(cls, path: Path) -> "SlidingWindowDataset":
        data = np.load(path, allow_pickle=True)
        label_names = data["label_names"].tolist()
        metadata = data["metadata"].item()
        splits: Dict[str, DatasetBuildResult] = {}
        for split_prefix in ["train", "val", "test"]:
            windows_key = f"{split_prefix}_windows"
            if windows_key not in data:
                continue
            splits[split_prefix] = DatasetBuildResult(
                windows=data[windows_key],
                labels=data[f"{split_prefix}_labels"],
                label_names=label_names,
                subjects=data[f"{split_prefix}_subjects"],
            )
        return cls(splits=splits, label_names=label_names, metadata=metadata)


class BaseDatasetBuilder:
    """Interface for dataset preparation builders."""

    name: str

    def __init__(self, signal_spec: SignalSpec, label_space: LabelSpace, root_dir: Path, raw_dir: Path, config: Mapping[str, Any]):
        self.signal_spec = signal_spec
        self.label_space = label_space
        self.root_dir = Path(root_dir)
        self.raw_dir = Path(raw_dir)
        self.config = dict(config)
        self.label_to_index = {label: idx for idx, label in enumerate(label_space.all_labels())}

    def prepare(self) -> SlidingWindowDataset:
        raise NotImplementedError

    def _encode_labels(self, labels: Sequence[str]) -> np.ndarray:
        unknown_labels = set(labels) - set(self.label_to_index)
        if unknown_labels:
            raise KeyError(f"Encountered unknown labels: {sorted(unknown_labels)}")
        return np.array([self.label_to_index[label] for label in labels], dtype=np.int64)

    def _label_index(self, label: str) -> int:
        return self.label_to_index[label]


def subject_split(subjects: Sequence[Any], train_ids: Sequence[Any], val_ids: Sequence[Any], test_ids: Sequence[Any]) -> np.ndarray:
    """Return mask array encoding split per subject."""

    subject_to_split = {}
    for sid in train_ids:
        subject_to_split[str(sid)] = "train"
    for sid in val_ids:
        subject_to_split[str(sid)] = "val"
    for sid in test_ids:
        subject_to_split[str(sid)] = "test"

    splits = []
    for s in subjects:
        splits.append(subject_to_split.get(str(s), "train"))
    return np.array(splits)


__all__ = [
    "DatasetBuildResult",
    "SlidingWindowDataset",
    "BaseDatasetBuilder",
    "subject_split",
]
