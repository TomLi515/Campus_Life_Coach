"""Backbone architectures for motion encoder pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = out + identity
        return torch.relu(out)


class ConvNetBackbone(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        channels = [input_channels, 64, 128, 256, 256]
        blocks = []
        for idx in range(len(channels) - 1):
            stride = 1 if idx == 0 else 2
            blocks.append(ConvBlock(channels[idx], channels[idx + 1], kernel_size=5, stride=stride, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = self.pool(features).squeeze(-1)
        return self.proj(pooled)


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: Tuple[int, int, int] = (3, 5, 7), bottleneck_channels: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        branches = []
        for kernel in kernel_sizes:
            padding = kernel // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(bottleneck_channels, out_channels, kernel_size=kernel, padding=padding, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.pool_proj = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(x)
        outputs = [branch(bottleneck) for branch in self.branches]
        outputs.append(self.pool_proj(x))
        return torch.cat(outputs, dim=1)


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.inception1 = InceptionModule(64, 32)
        self.inception2 = InceptionModule(32 * 4, 64)
        self.reduction = nn.Sequential(
            nn.Conv1d(64 * 4, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.reduction(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_channels, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.proj(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        pooled = self.pool(x).squeeze(-1)
        return pooled


def build_backbone(name: str, input_channels: int, embedding_dim: int, dropout: float = 0.0) -> nn.Module:
    name = name.lower()
    if name == "convnet_small":
        return ConvNetBackbone(input_channels=input_channels, embedding_dim=embedding_dim, dropout=dropout)
    if name == "inception_like":
        return InceptionBackbone(input_channels=input_channels, embedding_dim=embedding_dim, dropout=dropout)
    if name == "transformer_small":
        return TransformerBackbone(input_channels=input_channels, embedding_dim=embedding_dim, dropout=dropout)
    raise ValueError(f"Unknown backbone name: {name}")


__all__ = [
    "build_backbone",
    "ConvNetBackbone",
    "InceptionBackbone",
    "TransformerBackbone",
]
