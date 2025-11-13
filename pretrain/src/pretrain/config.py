"""Configuration helpers for data preparation and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml


@dataclass(frozen=True)
class SignalSpec:
    """Unified inertial signal specification."""

    target_sampling_rate: int
    window_size_seconds: float
    window_step_seconds: float
    channels: List[str]
    channel_order_comment: str | None = None

    def window_size_samples(self) -> int:
        return int(round(self.target_sampling_rate * self.window_size_seconds))

    def window_step_samples(self) -> int:
        return int(round(self.target_sampling_rate * self.window_step_seconds))


@dataclass(frozen=True)
class LabelSpace:
    primary_labels: List[str]
    include_other_class: bool = True

    def all_labels(self) -> List[str]:
        labels = list(self.primary_labels)
        if self.include_other_class and "Other" not in labels:
            labels.append("Other")
        return labels


@dataclass
class DatasetMapping:
    name: str
    raw_config: Dict[str, Any]
    mapping: Dict[str, List[str]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            **self.raw_config,
            "mapping": self.mapping,
        }


@dataclass
class ArtifactsConfig:
    phone_dataset_filename: str
    watch_dataset_filename: str
    metadata_filename: str


@dataclass
class DataPrepConfig:
    seed: int
    output_root: Path
    raw_root: Path
    signal_spec: SignalSpec
    label_space: LabelSpace
    datasets: Dict[str, DatasetMapping]
    artifacts: ArtifactsConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "DataPrepConfig":
        config = _load_yaml(path)
        signal = SignalSpec(**config["signal_spec"])
        label_space = LabelSpace(**config["label_space"])
        artifacts = ArtifactsConfig(**config["artifacts"])

        dataset_cfgs: Dict[str, DatasetMapping] = {}
        for key, value in config.items():
            if key in {"signal_spec", "label_space", "seed", "output_root", "raw_root", "artifacts"}:
                continue
            if not isinstance(value, Mapping):
                continue
            dataset_cfgs[key] = DatasetMapping(name=key, raw_config=dict(value), mapping=dict(value.get("mapping", {})))
        return cls(
            seed=int(config.get("seed", 42)),
            output_root=Path(config["output_root"]),
            raw_root=Path(config["raw_root"]),
            signal_spec=signal,
            label_space=label_space,
            datasets=dataset_cfgs,
            artifacts=artifacts,
        )


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float = 0.0
    betas: Optional[List[float]] = None
    eps: Optional[float] = None


@dataclass
class SchedulerConfig:
    name: str
    warmup_epochs: int = 0


@dataclass
class TrainingLoopConfig:
    epochs: int
    batch_size: int
    num_workers: int
    grad_clip_norm: Optional[float] = None
    class_weighting: Optional[str] = None


@dataclass
class AugmentationConfig:
    time_jitter_seconds: float = 0.0
    add_noise_std: float = 0.0
    random_rotation_degrees: float = 0.0
    drop_channels_prob: float = 0.0


@dataclass
class ModelConfig:
    backbone: str
    input_channels: int
    sequence_length: int
    embedding_dim: int
    dropout: float


@dataclass
class HeadsConfig:
    num_classes: int
    include_other: bool = True


@dataclass
class DatasetSplitConfig:
    path: Path
    metadata_path: Path
    train_split: str
    val_split: str
    test_split: str


@dataclass
class MetricsConfig:
    compute_per_class_f1: bool = True
    compute_confusion_matrix: bool = False


@dataclass
class LoggingConfig:
    project: str
    run_name: str
    log_interval_steps: int
    checkpoint_dir: Path
    save_top_k: int = 1


@dataclass
class TrainingConfig:
    seed: int
    device: str
    logging: LoggingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingLoopConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    heads: HeadsConfig
    dataset: DatasetSplitConfig
    metrics: MetricsConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingConfig":
        config = _load_yaml(path)
        return cls(
            seed=int(config.get("seed", 42)),
            device=config.get("device", "auto"),
            logging=LoggingConfig(**_with_path(config["logging"])),
            optimizer=OptimizerConfig(**config["optimizer"]),
            scheduler=SchedulerConfig(**config["scheduler"]),
            training=TrainingLoopConfig(**config["training"]),
            augmentation=AugmentationConfig(**config.get("augmentation", {})),
            model=ModelConfig(**config["model"]),
            heads=HeadsConfig(**config["heads"]),
            dataset=DatasetSplitConfig(**_with_path(config["dataset"])),
            metrics=MetricsConfig(**config.get("metrics", {})),
        )


def _load_yaml(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _with_path(config: MutableMapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in config.items():
        if "path" in key or key.endswith("dir"):
            out[key] = Path(value)
        else:
            out[key] = value
    return out


__all__ = [
    "SignalSpec",
    "LabelSpace",
    "DatasetMapping",
    "ArtifactsConfig",
    "DataPrepConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingLoopConfig",
    "AugmentationConfig",
    "ModelConfig",
    "HeadsConfig",
    "DatasetSplitConfig",
    "MetricsConfig",
    "LoggingConfig",
    "TrainingConfig",
]
