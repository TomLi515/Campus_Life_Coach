"""Classification head for encoders."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["ClassificationHead"]
