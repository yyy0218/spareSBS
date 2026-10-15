"""Parameter-free boundary risk used to construct ARL targets."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def boundary_risk(support):
    knn = support["knn"].long()
    knn = knn[:, 1:] if knn.shape[1] > 1 else knn
    xyz = support["xyz"].float()
    features = F.normalize(support["features"].float(), dim=-1, eps=1e-6)
    if knn.numel() == 0:
        return xyz.new_zeros(len(xyz))
    similarity = (features[:, None] * features[knn]).sum(-1).clamp(-1, 1)
    distance = (xyz[knn] - xyz[:, None]).norm(dim=-1)
    scaled = (distance / distance.mean(1, keepdim=True).clamp_min(1e-4) - 1).clamp_min(0)
    scaled = scaled / scaled.max(1, keepdim=True).values.clamp_min(1)
    return (0.7 * (1 - similarity).clamp(0, 1) + 0.3 * scaled).mean(1).clamp(0, 1)
