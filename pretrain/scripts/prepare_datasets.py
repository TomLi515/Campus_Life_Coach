"""Prepare public datasets for phone and watch encoder pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pretrain.config import DataPrepConfig
from pretrain.data import MotionSenseBuilder, PAMAP2Builder, UCIHARBuilder
from pretrain.data.base_dataset import DatasetBuildResult, SlidingWindowDataset
from pretrain.data.utils import ensure_directory
from pretrain.utils import get_logger, set_global_seed

LOGGER = get_logger(__name__)


def combine_datasets(datasets: Dict[str, SlidingWindowDataset], label_names: List[str]) -> SlidingWindowDataset:
    splits: Dict[str, DatasetBuildResult] = {}
    metadata = {name: ds.metadata for name, ds in datasets.items()}
    for split in ["train", "val", "test"]:
        split_results = []
        for ds in datasets.values():
            if split in ds.splits:
                split_results.append(ds.splits[split])
        if split_results:
            concat_result = DatasetBuildResult.concat(split_results, label_names)
            splits[split] = concat_result
    return SlidingWindowDataset(splits=splits, label_names=label_names, metadata={"sources": metadata})


def summarize_split(result: DatasetBuildResult) -> Dict[str, int]:
    counts = np.bincount(result.labels, minlength=len(result.label_names))
    return {result.label_names[idx]: int(count) for idx, count in enumerate(counts)}


def build_metadata_json(phone_dataset: SlidingWindowDataset, watch_dataset: SlidingWindowDataset, output_path: Path) -> None:
    metadata = {
        "phone_dataset": {
            "label_names": phone_dataset.label_names,
            "splits": {split: summarize_split(result) for split, result in phone_dataset.splits.items()},
            "sources": phone_dataset.metadata.get("sources", {}),
        },
        "watch_dataset": {
            "label_names": watch_dataset.label_names,
            "splits": {split: summarize_split(result) for split, result in watch_dataset.splits.items()},
            "sources": watch_dataset.metadata,
        },
    }
    output_path.write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Wrote metadata to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public datasets for motion encoder pretraining")
    parser.add_argument("--config", type=Path, required=True, help="Path to data preparation YAML config")
    args = parser.parse_args()

    cfg = DataPrepConfig.from_yaml(args.config)
    set_global_seed(cfg.seed)

    ensure_directory(cfg.output_root)
    ensure_directory(cfg.raw_root)

    label_names = cfg.label_space.all_labels()

    dataset_results: Dict[str, SlidingWindowDataset] = {}

    for key, dataset_cfg in cfg.datasets.items():
        if not dataset_cfg.raw_config.get("enabled", True):
            LOGGER.info("Skipping disabled dataset %s", key)
            continue
        LOGGER.info("Preparing dataset %s", key)
        if key == "uci_har":
            builder = UCIHARBuilder(cfg.signal_spec, cfg.label_space, cfg.output_root, cfg.raw_root, dataset_cfg.raw_config)
        elif key == "motionsense":
            builder = MotionSenseBuilder(cfg.signal_spec, cfg.label_space, cfg.output_root, cfg.raw_root, dataset_cfg.raw_config)
        elif key == "pamap2":
            builder = PAMAP2Builder(cfg.signal_spec, cfg.label_space, cfg.output_root, cfg.raw_root, dataset_cfg.raw_config)
        else:
            LOGGER.warning("Unknown dataset key %s; skipping", key)
            continue
        dataset_results[key] = builder.prepare()

    phone_sources = {name: dataset_results[name] for name in ["uci_har", "motionsense"] if name in dataset_results}
    if not phone_sources:
        raise RuntimeError("No phone datasets prepared; enable at least one of uci_har or motionsense")

    phone_dataset = combine_datasets(phone_sources, label_names)
    phone_output = cfg.output_root / cfg.artifacts.phone_dataset_filename
    phone_dataset.save_npz(phone_output)

    if "pamap2" not in dataset_results:
        raise RuntimeError("PAMAP2 dataset must be enabled for watch encoder pretraining")
    watch_dataset = dataset_results["pamap2"]
    watch_output = cfg.output_root / cfg.artifacts.watch_dataset_filename
    watch_dataset.save_npz(watch_output)

    metadata_output = cfg.output_root / cfg.artifacts.metadata_filename
    build_metadata_json(phone_dataset, watch_dataset, metadata_output)

    LOGGER.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
