"""Metric computation helpers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, label_names: List[str], compute_confusion: bool = False) -> Dict[str, object]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    metrics = {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro")),
    }
    per_class_f1 = f1_score(labels_np, preds, average=None, labels=np.arange(len(label_names)))
    metrics["per_class_f1"] = {label_names[idx]: float(score) for idx, score in enumerate(per_class_f1)}
    if compute_confusion:
        metrics["confusion_matrix"] = confusion_matrix(labels_np, preds, labels=np.arange(len(label_names))).tolist()
    return metrics


__all__ = ["compute_metrics"]
