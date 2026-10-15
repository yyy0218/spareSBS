"""Paper-run AdamW grouping and warmup/cosine schedule.

These small helpers implement the recorded grouping and scheduler equations;
there is no vendored optimizer library or unused optimizer registry.
"""

import math
import torch


def build_optimizer(model, config):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    return torch.optim.AdamW(
        [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": config["weight_decay"]},
        ],
        lr=config["learning_rate"],
    )


def learning_rate(step, total_steps, warmup_steps, peak):
    if total_steps <= 0:
        raise ValueError("Training loader is empty")
    if warmup_steps > 0 and step < warmup_steps:
        return peak / 20 + (peak - peak / 20) * step / warmup_steps
    # Warmup is part of the cosine horizon (no warmup-prefix shift).
    minimum = peak / 10000
    return minimum + 0.5 * (peak - minimum) * (
        1 + math.cos(math.pi * min(step, total_steps) / total_steps)
    )
