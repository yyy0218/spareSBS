"""SparseSBS: explicit Directional Encoding -> backbone -> segmentation head.

ARL belongs to the training system and is neither imported nor registered here.
"""

from torch import nn
from .directional_encoding import DirectionalEncoding
from .backbone import GeometryBackbone


def _initialize_linear(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SparseSBS(nn.Module):
    """Support Before Supervision with the same abbreviation as the paper.

    The three registered top-level modules expose the complete prediction path.
    The DE output g_i is a trainable feature, not a detached pseudo-label.
    """

    def __init__(self, model_config: dict, distances):
        super().__init__()
        if not model_config["widths"] or len(distances) < min(2, len(model_config["widths"])):
            raise ValueError("Model hierarchy and its DE statistics must be provided")
        self.directional_encoding = DirectionalEncoding(model_config, distances[0])
        self.backbone = GeometryBackbone(model_config, distances)
        # Match the audited linear initialization order without any old-model
        # object, state-key remapping, or import-time compatibility machinery.
        self.directional_encoding.apply(_initialize_linear)
        self.backbone.apply(_initialize_linear)
        dim = model_config["head_dim"]
        self.segmentation_head = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=model_config["bn_momentum"]),
            nn.GELU(),
            nn.Linear(dim, model_config["num_classes"]),
        )

    def encode(self, xyz, attributes, indices, counts=None, return_support=False):
        """Run DE and backbone without touching the segmentation head.

        The trainer uses this path for geometry-only BN warmup. Its input
        hierarchy layout is identical to the ordinary forward/predict path.
        """
        hierarchy = list(indices)
        if not hierarchy:
            raise ValueError("The point hierarchy must contain finest-level neighbors")
        neighbors = hierarchy.pop().long()
        remaining_counts = None if counts is None else list(counts)
        finest_counts = remaining_counts.pop() if remaining_counts is not None else None
        support_features = self.directional_encoding(xyz, attributes, neighbors, finest_counts)
        semantic, coordinate = self.backbone(
            xyz, support_features, neighbors, hierarchy, remaining_counts
        )
        support = None
        if return_support:
            support = {
                "knn": neighbors.detach(),
                "xyz": xyz.detach(),
                "features": support_features.detach(),
            }
        return semantic, coordinate, support

    def forward(self, xyz, features, indices, counts=None, return_support=False):
        semantic, coordinate, support = self.encode(xyz, features, indices, counts, return_support)
        return {
            "logits": self.segmentation_head(semantic),
            "semantic_features": semantic,
            "coordinate_loss": coordinate,
            "support": support,
        }

    def predict(self, xyz, features, indices, counts=None):
        if self.training:
            raise RuntimeError("Call model.eval() before predict()")
        return self.forward(xyz, features, indices, counts, False)["logits"]
