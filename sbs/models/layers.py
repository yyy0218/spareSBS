"""Pointwise residual layers and directional local aggregation."""

import torch
from torch import nn
from sbs.ops import directional_pool
from .directional_templates import sample_template_centers


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth; mechanism follows timm (Apache-2.0)."""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep) / keep
        return x * mask


class DirectionalAggregation(nn.Module):
    def __init__(self, input_dim, output_dim, momentum, distances, initial_scale=0.0):
        super().__init__()
        if output_dim % 4:
            raise ValueError("Directional aggregation width must be divisible by four")
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.normalization = nn.BatchNorm1d(output_dim, momentum=momentum)
        self.centers = nn.Parameter(sample_template_centers(output_dim // 4, distances).flatten())
        self.bandwidth = nn.Parameter(torch.full((output_dim // 4,), 0.1))
        nn.init.constant_(self.normalization.weight, initial_scale)

    def forward(self, x, neighborhood):
        neighbors, xyz = neighborhood
        b, n, _ = x.shape
        weight = torch.cat([self.centers.reshape(-1, 3).T, self.bandwidth.square().reshape(1, -1)])
        pooled = directional_pool(self.projection(x), xyz, neighbors, weight, self.training)
        return self.normalization(pooled.reshape(b * n, -1)).reshape(b, n, -1)


class PointMLP(nn.Module):
    def __init__(self, dim, ratio, momentum, initial_scale=0.0):
        super().__init__()
        hidden = round(dim * ratio)
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
            nn.BatchNorm1d(dim, momentum=momentum),
        )
        nn.init.constant_(self.layers[-1].weight, initial_scale)

    def forward(self, x):
        b, n, c = x.shape
        return self.layers(x.reshape(b * n, c)).reshape(b, n, c)


class ResidualGeometryBlock(nn.Module):
    def __init__(self, dim, depth, drop_rates, ratio, momentum, distances):
        super().__init__()
        self.depth = depth
        self.aggregations = nn.ModuleList(
            [DirectionalAggregation(dim, dim, momentum, distances) for _ in range(depth)]
        )
        self.input_mlp = PointMLP(dim, ratio, momentum, 0.2)
        self.mlps = nn.ModuleList([PointMLP(dim, ratio, momentum) for _ in range(depth // 2)])
        self.drop_paths = nn.ModuleList([StochasticDepth(rate) for rate in drop_rates])

    def drop(self, x, index, counts):
        if counts is None:
            return self.drop_paths[index](x)
        return torch.cat([self.drop_paths[index](v) for v in torch.split(x, counts, 1)], 1)

    def forward(self, x, neighborhood, counts):
        x = x + self.drop(self.input_mlp(x), 0, counts)
        for i, aggregation in enumerate(self.aggregations):
            x = x + self.drop(aggregation(x, neighborhood), i, counts)
            if i % 2 == 1:
                x = x + self.drop(self.mlps[i // 2](x), i, counts)
        return x
