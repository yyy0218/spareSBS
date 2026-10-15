"""Global Semantic Confirmation from real and locally accepted anchors.

The recorded training implementation constructs references over the collated
batch. This scope is explicit and is not silently changed to per-scene here.
"""

import math
import torch
from torch.nn import functional as F
from .targets import RelayTargets


class GlobalSemanticConfirmation:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def __call__(self, logits, features, labels, local, boundary, num_classes, ignore_index):
        cfg = self.config
        result = RelayTargets.empty(labels, ignore_index)
        features = F.normalize(features.float(), dim=-1, eps=1e-6)
        confidence, prediction = logits.float().softmax(-1).max(-1)
        anchor_mask = labels != ignore_index
        anchor_labels = labels.clone()
        weights = anchor_mask.float()
        if local is not None:
            anchor_mask = anchor_mask | local.mask
            anchor_labels = torch.where(local.mask, local.labels, anchor_labels)
            weights = torch.where(local.mask, local.weights.clamp(0, 1), weights)
        if not anchor_mask.any():
            return result
        prototypes = features.new_zeros((num_classes, features.shape[1]))
        valid = torch.zeros(num_classes, dtype=torch.bool, device=labels.device)
        for c in range(num_classes):
            mask = anchor_mask & (anchor_labels == c)
            if int(mask.sum()) < cfg["min_anchors"]:
                continue
            w = weights[mask, None].clamp_min(1e-4)
            prototypes[c] = F.normalize((features[mask] * w).sum(0) / w.sum(0), dim=0, eps=1e-6)
            valid[c] = True
        if not valid.any():
            return result
        similarity = features @ prototypes.T
        similarity = similarity.masked_fill(~valid[None], -1e4)
        best, classes = similarity.max(-1)
        candidates = (
            (labels == ignore_index)
            & (classes == prediction)
            & (best >= cfg["similarity_threshold"])
            & (confidence >= cfg["confidence_threshold"])
            & (boundary <= cfg["boundary_threshold"])
        )
        if local is not None:
            candidates = candidates & ~local.mask
        selected = torch.zeros_like(candidates)
        for c in range(num_classes):
            indices = (candidates & (classes == c)).nonzero().flatten()
            if len(indices) == 0:
                continue
            keep = max(1, math.ceil(len(indices) * cfg["per_class_retention"]))
            selected[indices[best[indices].topk(min(keep, len(indices))).indices]] = True
        weight = (
            (best - cfg["similarity_threshold"]) / max(1e-6, 1 - cfg["similarity_threshold"])
        ).clamp(0, 1)
        return RelayTargets(
            selected,
            torch.where(selected, classes, result.labels),
            weight * selected,
            best * selected,
        )
