"""Train the phone encoder using prepared datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretrain.config import TrainingConfig
from pretrain.training import Trainer, create_dataloaders
from pretrain.training.augmentations import AugmentationSettings
from pretrain.utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phone encoder")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config)

    augmentation_cfg = cfg.augmentation
    augmentation = AugmentationSettings(
        sampling_rate=50,
        time_jitter_seconds=augmentation_cfg.time_jitter_seconds,
        noise_std=augmentation_cfg.add_noise_std,
        rotation_degrees=augmentation_cfg.random_rotation_degrees,
        drop_channels_prob=augmentation_cfg.drop_channels_prob,
    )

    loaders = create_dataloaders(
        npz_path=cfg.dataset.path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        augmentation=augmentation,
        train_split=cfg.dataset.train_split,
        val_split=cfg.dataset.val_split,
        test_split=cfg.dataset.test_split,
    )
    label_names = loaders.pop("label_names", [])

    trainer = Trainer(cfg, label_names=label_names)
    metrics = trainer.train(loaders)

    metrics_path = cfg.logging.checkpoint_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
