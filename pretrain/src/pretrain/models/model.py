"""Encoder wrapper combining backbone and classification head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .backbones import build_backbone
from .heads import ClassificationHead


@dataclass
class EncoderOutput:
    embeddings: torch.Tensor
    logits: torch.Tensor


class EncoderWithHead(nn.Module):
    def __init__(self, backbone_name: str, input_channels: int, embedding_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = build_backbone(backbone_name, input_channels=input_channels, embedding_dim=embedding_dim, dropout=dropout)
        self.head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        embeddings = self.backbone(x)
        logits = self.head(embeddings)
        return EncoderOutput(embeddings=embeddings, logits=logits)


__all__ = ["EncoderWithHead", "EncoderOutput"]
