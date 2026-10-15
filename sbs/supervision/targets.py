"""Detached hard targets used only by the training objective."""

from dataclasses import dataclass
import torch


@dataclass
class RelayTargets:
    mask: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor

    @classmethod
    def empty(cls, labels, ignore_index):
        return cls(
            torch.zeros_like(labels, dtype=torch.bool),
            torch.full_like(labels, ignore_index),
            torch.zeros_like(labels, dtype=torch.float32),
            torch.zeros_like(labels, dtype=torch.float32),
        )
