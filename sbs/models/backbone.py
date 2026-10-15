"""Hierarchical backbone operating on the complete DE output g_i.

There is no raw-attribute stem and no hidden first-stage DE block here.
Coarser stages retain the audited spatial-template primitive and multi-scale
fusion. Coordinate regularization is an auxiliary training loss, not ARL.
"""

import torch
from torch import nn
from torch.nn import functional as F
from sbs.ops import directional_response
from .directional_templates import DirectionalTemplateMatching
from .layers import DirectionalAggregation, ResidualGeometryBlock, StochasticDepth


class GeometryStage(nn.Module):
    """One coarser hierarchy level; depth starts at one, after explicit DE."""

    def __init__(self, cfg, distances, depth=1):
        super().__init__()
        if depth < 1:
            raise ValueError("The finest-level DE is separate from GeometryStage")
        self.depth = depth
        self.last = depth == len(cfg["depths"]) - 1
        dim = cfg["widths"][depth]
        base = cfg["widths"][0]
        momentum = cfg["bn_momentum"]
        self.k = cfg["neighbors"][depth]
        self.coordinate_scale = 1 / cfg["coordinate_scales"][depth]
        self.signature_norm = nn.BatchNorm1d(dim, momentum=momentum)
        nn.init.constant_(self.signature_norm.weight, 0.2)
        self.signature_projection = nn.Sequential(
            nn.BatchNorm1d(base, momentum=momentum),
            nn.Linear(base, 2 * base),
            nn.GELU(),
            nn.Linear(2 * base, dim, bias=False),
        )
        if depth == 1:
            self.spatial_templates = DirectionalTemplateMatching(base, distances[1], "spatial")
        previous_dim = cfg["widths"][depth - 1]
        self.down_aggregation = DirectionalAggregation(
            previous_dim, dim, momentum, distances[1], 0.3
        )
        self.down_projection = nn.Sequential(
            nn.Linear(previous_dim, dim, bias=False), nn.BatchNorm1d(dim, momentum=momentum)
        )
        nn.init.constant_(self.down_projection[1].weight, 0.3)
        self.block = ResidualGeometryBlock(
            dim,
            cfg["depths"][depth],
            cfg["drop_paths"][depth],
            cfg["mlp_ratio"],
            momentum,
            distances[1],
        )
        self.output_drop = StochasticDepth(cfg["head_drops"][depth])
        self.output_projection = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=momentum), nn.Linear(dim, cfg["head_dim"], bias=False)
        )
        nn.init.constant_(self.output_projection[0].weight, (base / dim) ** 0.5)
        self.coordinate_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.GELU(),
            nn.Linear(32, 3, bias=False),
        )
        if not self.last:
            self.next_stage = GeometryStage(cfg, distances, depth + 1)

    def forward(self, x, xyz, previous, indices, counts, shared_weights=None):
        ids = indices.pop()
        xyz = xyz[ids]
        x = (
            self.down_projection(x)[ids]
            + self.down_aggregation(x.unsqueeze(0), previous).squeeze(0)[ids]
        )
        knn = indices.pop().long()
        scaled = xyz * self.coordinate_scale
        if self.depth == 1:
            signature = self.spatial_templates(scaled, knn)
            shared_weights = self.spatial_templates.spatial_weights()
        else:
            signature = (
                directional_response(
                    F.pad(scaled, (0, 1)).unsqueeze(0),
                    knn.unsqueeze(0).int(),
                    shared_weights,
                    "spatial",
                    self.training,
                )
                .squeeze(0)
                .sqrt()
            )
        x = self.signature_norm(self.signature_projection(signature)) + x
        neighborhood = (knn.unsqueeze(0).int(), F.pad(scaled, (0, 1)).unsqueeze(0))
        stage_counts = counts.pop() if counts is not None else None
        x = self.block(x.unsqueeze(0), neighborhood, stage_counts).squeeze(0)
        deeper, coordinate_loss = (
            (None, None)
            if self.last
            else self.next_stage(x, xyz, neighborhood, indices, counts, shared_weights)
        )
        if self.training:
            selected = torch.randint(self.k, (len(x), 1), device=x.device)
            selected = torch.gather(knn, 1, selected).flatten()
            expected = (xyz[selected] - xyz) * self.coordinate_scale
            term = F.mse_loss(self.coordinate_head(x[selected] - x), expected)
            coordinate_loss = term if coordinate_loss is None else coordinate_loss + term
        x = self.output_projection(x)
        x = x[indices[self.depth - 1].long()]
        x = self.output_drop(x)
        return (x if deeper is None else deeper + x), coordinate_loss


class GeometryBackbone(nn.Module):
    """Consume DE features and fuse them with the coarser hierarchy outputs."""

    def __init__(self, cfg, distances):
        super().__init__()
        width = cfg["widths"][0]
        momentum = cfg["bn_momentum"]
        self.input_dim = width
        self.output_dim = cfg["head_dim"]
        self.k = cfg["neighbors"][0]
        self.coordinate_scale = 1 / cfg["coordinate_scales"][0]
        self.input_drop = StochasticDepth(cfg["head_drops"][0])
        self.input_projection = nn.Sequential(
            nn.BatchNorm1d(width, momentum=momentum),
            nn.Linear(width, cfg["head_dim"], bias=False),
        )
        nn.init.constant_(self.input_projection[0].weight, 1.0)
        self.coordinate_head = nn.Sequential(
            nn.Linear(width, 32, bias=False),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.GELU(),
            nn.Linear(32, 3, bias=False),
        )
        self.stages = GeometryStage(cfg, distances, 1) if len(cfg["depths"]) > 1 else None

    def forward(self, xyz, support_features, neighbors, indices, counts=None):
        """Return semantic features h_i and training-only coordinate loss.

        `indices` and `counts` describe coarser levels only; the model has
        already consumed the finest-level neighbors/counts when invoking DE.
        Caller-owned lists and tensors are not modified.
        """
        if support_features.ndim != 2 or support_features.shape != (len(xyz), self.input_dim):
            raise ValueError("Backbone expects DE support features [N, model.widths[0]]")
        knn = neighbors.long()
        neighborhood = (
            knn.unsqueeze(0).int(),
            F.pad(xyz * self.coordinate_scale, (0, 1)).unsqueeze(0),
        )
        if self.stages is None:
            deeper = coordinate_loss = None
        else:
            deeper, coordinate_loss = self.stages(
                support_features,
                xyz,
                neighborhood,
                list(indices),
                None if counts is None else list(counts),
            )
        if self.training:
            selected = torch.randint(self.k, (len(support_features), 1), device=xyz.device)
            selected = torch.gather(knn, 1, selected).flatten()
            expected = (xyz[selected] - xyz) * self.coordinate_scale
            term = F.mse_loss(
                self.coordinate_head(support_features[selected] - support_features), expected
            )
            coordinate_loss = term if coordinate_loss is None else coordinate_loss + term
        fine = self.input_drop(self.input_projection(support_features))
        semantic = fine if deeper is None else deeper + fine
        return semantic, coordinate_loss
