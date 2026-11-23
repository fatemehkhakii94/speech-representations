"""Minimal probe architectures."""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.network = nn.Linear(input_size, output_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MLPProbe(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def create_probe(
    probe_type: str,
    input_size: int,
    output_size: int,
    hidden_size: int = 256,
    dropout: float = 0.1,
) -> nn.Module:
    if probe_type == "linear":
        return LinearProbe(input_size, output_size)
    if probe_type == "mlp":
        return MLPProbe(input_size, output_size, hidden_size, dropout)
    raise ValueError(f"Unsupported probe type: {probe_type}")
