"""Complete pre-backbone Directional Encoding (DE).

Template matching produces v_i; signature projection, normalization and local
contextualization produce g_i. All of these steps belong to this one module.
No label, pseudo-label constructor, or segmentation head is part of DE.
"""

from collections.abc import Mapping, Sequence
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from .directional_templates import DirectionalTemplateMatching
from .layers import ResidualGeometryBlock


class DirectionalEncoding(nn.Module):
    """Map point observations and their neighbors to early support features g_i.

    Args:
        config: Shared model settings. The first hierarchy level defines DE's
            channels, residual depth, neighbor count and coordinate scale.
        distances: One-dimensional distance statistics for the finest level.

    The returned tensor remains differentiable. The training system, rather
    than DE itself, detaches a copy when it constructs ARL supervision targets.
    """

    def __init__(self, config: Mapping[str, Any], distances: torch.Tensor):
        super().__init__()
        width = config["widths"][0]
        momentum = config["bn_momentum"]
        self.output_dim = width
        self.coordinate_scale = 1 / config["coordinate_scales"][0]
        # Preserve the audited construction and optimizer-group order.
        self.signature_norm = nn.BatchNorm1d(width, momentum=momentum)
        nn.init.constant_(self.signature_norm.weight, 0.8)
        self.signature_projection = nn.Sequential(
            nn.BatchNorm1d(width, momentum=momentum),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width, bias=False),
        )
        self.template_matching = DirectionalTemplateMatching(width, distances, config["input_mode"])
        self.local_contextualization = ResidualGeometryBlock(
            width,
            config["depths"][0],
            config["drop_paths"][0],
            config["mlp_ratio"],
            momentum,
            distances,
        )

    def forward(
        self,
        xyz: torch.Tensor,
        attributes: torch.Tensor,
        neighbors: torch.Tensor,
        counts: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return N-by-width support features; do not consume hierarchy lists."""
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("DE coordinates must have shape [N, 3]")
        if attributes.ndim != 2 or len(attributes) != len(xyz):
            raise ValueError("DE attributes must have shape [N, F]")
        if neighbors.ndim != 2 or len(neighbors) != len(xyz):
            raise ValueError("DE neighbors must have shape [N, K]")
        neighbors = neighbors.long()
        scaled_xyz = xyz * self.coordinate_scale
        signature = self.template_matching(scaled_xyz, neighbors, attributes)
        projected = self.signature_norm(self.signature_projection(signature))
        neighborhood = (
            neighbors.unsqueeze(0).int(),
            F.pad(scaled_xyz, (0, 1)).unsqueeze(0),
        )
        support_features = self.local_contextualization(
            projected.unsqueeze(0), neighborhood, counts
        ).squeeze(0)
        return support_features
