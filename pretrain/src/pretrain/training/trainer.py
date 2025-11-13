"""Training loop for encoder pretraining."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..models.model import EncoderWithHead
from ..utils import get_logger, set_global_seed
from .metrics import compute_metrics

LOGGER = get_logger(__name__)


class Trainer:
    def __init__(self, config: TrainingConfig, label_names: List[str]):
        self.config = config
        self.label_names = label_names
        set_global_seed(config.seed)
        self.device = self._resolve_device(config.device)
        self.checkpoint_dir = config.logging.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            LOGGER.warning("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self) -> EncoderWithHead:
        model_cfg = self.config.model
        heads_cfg = self.config.heads
        model = EncoderWithHead(
            backbone_name=model_cfg.backbone,
            input_channels=model_cfg.input_channels,
            embedding_dim=model_cfg.embedding_dim,
            num_classes=heads_cfg.num_classes,
            dropout=model_cfg.dropout,
        )
        return model.to(self.device)

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        opt_cfg = self.config.optimizer
        if opt_cfg.name.lower() != "adamw":
            raise ValueError("Only AdamW optimizer is implemented in this trainer")
        betas = tuple(float(b) for b in opt_cfg.betas) if opt_cfg.betas else (0.9, 0.999)
        eps = float(opt_cfg.eps) if opt_cfg.eps is not None else 1e-8
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
            betas=betas,
            eps=eps,
        )

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> torch.optim.lr_scheduler.LambdaLR:
        sched_cfg = self.config.scheduler
        total_steps = self.config.training.epochs * max(steps_per_epoch, 1)
        warmup_steps = sched_cfg.warmup_epochs * max(steps_per_epoch, 1)

        def lr_lambda(step: int) -> float:
            if total_steps == 0:
                return 1.0
            if step < warmup_steps and warmup_steps > 0:
                return max(1e-3, step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _compute_class_weights(self, loader: DataLoader) -> torch.Tensor | None:
        if self.config.training.class_weighting != "balanced":
            return None
        label_counts = torch.zeros(len(self.label_names))
        for _, labels in loader:
            unique, counts = labels.unique(return_counts=True)
            label_counts[unique] += counts
        class_weights = torch.reciprocal(label_counts.clamp_min(1.0))
        class_weights = class_weights / class_weights.sum() * len(self.label_names)
        return class_weights.to(self.device)

    def train(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Dict[str, float]]:
        train_loader = dataloaders.get("train")
        val_loader = dataloaders.get(self.config.dataset.val_split)
        test_loader = dataloaders.get(self.config.dataset.test_split)

        if train_loader is None or val_loader is None or test_loader is None:
            raise ValueError("Train, val, and test loaders are required")

        model = self.build_model()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer, steps_per_epoch=len(train_loader))

        class_weights = self._compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_f1 = -1.0
        best_state = None
        global_step = 0

        for epoch in range(1, self.config.training.epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_idx, (windows, labels) in enumerate(train_loader, start=1):
                windows = windows.to(self.device)
                labels = labels.to(self.device)

                output = model(windows)
                loss = criterion(output.logits, labels)

                optimizer.zero_grad()
                loss.backward()
                if self.config.training.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.grad_clip_norm)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                global_step += 1

                if batch_idx % self.config.logging.log_interval_steps == 0:
                    avg_loss = running_loss / self.config.logging.log_interval_steps
                    LOGGER.info("Epoch %d Step %d Loss %.4f", epoch, batch_idx, avg_loss)
                    running_loss = 0.0

            val_metrics = self.evaluate(model, val_loader)
            LOGGER.info("Epoch %d validation metrics: %s", epoch, val_metrics)
            val_f1 = val_metrics.get("macro_f1", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": asdict(self.config),
                    "label_names": self.label_names,
                }
                self._save_checkpoint(best_state)

        if best_state is not None:
            model.load_state_dict(best_state["model_state"])

        test_metrics = self.evaluate(model, test_loader, compute_confusion=self.config.metrics.compute_confusion_matrix)
        LOGGER.info("Test metrics: %s", test_metrics)
        return {"val": best_state.get("val_metrics", {}) if best_state else {}, "test": test_metrics}

    def evaluate(self, model: EncoderWithHead, loader: DataLoader, compute_confusion: bool = False) -> Dict[str, object]:
        model.eval()
        logits_list = []
        label_list = []
        with torch.no_grad():
            for windows, labels in loader:
                windows = windows.to(self.device)
                labels = labels.to(self.device)
                output = model(windows)
                logits_list.append(output.logits.cpu())
                label_list.append(labels.cpu())
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        return compute_metrics(logits, labels, self.label_names, compute_confusion)

    def _save_checkpoint(self, state: Dict[str, object]) -> None:
        path = self.checkpoint_dir / "best.ckpt"
        torch.save(state, path)
        LOGGER.info("Saved checkpoint to %s", path)
