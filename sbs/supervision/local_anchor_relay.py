"""Local Anchor Relay: conservative, direction-supported label inheritance."""

import math
import torch
from torch.nn import functional as F
from .targets import RelayTargets


class LocalAnchorRelay:
    def __init__(self, config):
        self.config = config

    @torch.no_grad()
    def __call__(self, labels, support, boundary, num_classes, ignore_index):
        result = RelayTargets.empty(labels, ignore_index)
        cfg = self.config
        knn = support["knn"].long()
        knn = knn[:, 1:] if knn.shape[1] > 1 else knn
        if knn.numel() == 0 or not (labels == ignore_index).any():
            return result
        feature = F.normalize(support["features"].float(), dim=-1, eps=1e-6)
        xyz = support["xyz"].float()
        neighbors = labels[knn]
        valid = neighbors != ignore_index
        similarity = (feature[:, None] * feature[knn]).sum(-1).clamp(-1, 1).clamp_min(0)
        distance = (xyz[knn] - xyz[:, None]).norm(dim=-1)
        spatial = torch.exp(-(distance / distance.mean(1, keepdim=True).clamp_min(1e-4)).square())
        gate = (1 - torch.maximum(boundary[:, None], boundary[knn])).clamp(0, 1)
        edge = similarity * spatial * gate * valid.float()
        scores = feature.new_zeros((len(labels), num_classes))
        safe = torch.where(valid, neighbors, torch.zeros_like(neighbors)).clamp(0, num_classes - 1)
        scores.scatter_add_(1, safe, edge)
        top = scores.topk(min(2, num_classes), dim=-1)
        best = top.values[:, 0]
        second = top.values[:, 1] if num_classes > 1 else torch.zeros_like(best)
        selected = (
            (labels == ignore_index)
            & (best >= cfg["score_threshold"])
            & ((best - second) >= cfg["margin"])
            & (boundary <= cfg["boundary_threshold"])
        )
        eligible = selected.nonzero().flatten()
        ratio = cfg["eligible_retention"]
        if len(eligible) == 0 or ratio <= 0:
            return result
        keep = max(1, math.ceil(len(eligible) * ratio))
        if keep < len(eligible):
            retained = eligible[best[eligible].topk(keep).indices]
            selected.zero_()
            selected[retained] = True
        weight = ((best - cfg["score_threshold"]) / max(1e-6, 1 - cfg["score_threshold"])).clamp(
            0, 1
        )
        return RelayTargets(
            selected,
            torch.where(selected, top.indices[:, 0], result.labels),
            weight * selected,
            best * selected,
        )
