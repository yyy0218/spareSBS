import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sbs.ops import directional_response


def sample_template_centers(channels, distances):
    if distances is None:
        raise ValueError("DE initialization requires calibrated distance samples")
    distances = torch.as_tensor(distances).float().cpu().flatten()
    stride = len(distances) // (channels + 2)
    if stride < 1:
        raise ValueError("Too few calibration samples for DE templates")
    radii = distances[stride::stride][:channels][torch.randperm(channels)]
    rnd = float(np.random.random() * channels)
    index = np.arange(channels)
    y = 2 * (index + 0.5) / channels - 1
    angle = ((index + rnd) % channels) * (math.pi * (3 - math.sqrt(5)))
    radial = np.sqrt(1 - y * y)
    sphere = torch.tensor(
        np.stack([np.cos(angle) * radial, y, np.sin(angle) * radial], -1), dtype=torch.float32
    )
    axis = F.normalize(torch.randn(1, 3), dim=1)
    rotation = axis_angle_rotation(axis, torch.rand(1) * torch.pi)[0]
    return (sphere @ rotation) * radii[:, None]


def axis_angle_rotation(axis, angle):
    axis = F.normalize(axis, dim=-1)
    a = axis.new_zeros((len(axis), 3, 3))
    a[:, 0, 1] = -axis[:, 2]
    a[:, 0, 2] = axis[:, 1]
    a[:, 1, 0] = axis[:, 2]
    a[:, 1, 2] = -axis[:, 0]
    a[:, 2, 0] = -axis[:, 1]
    a[:, 2, 1] = axis[:, 0]
    angle = angle.reshape(-1, 1, 1)
    return a * angle.sin() + (a @ a) * (1 - angle.cos()) + torch.eye(3, device=axis.device)


class DirectionalTemplateMatching(nn.Module):
    """One template bank; returns the per-point directional signature.

    `input_mode` is spatial, rgb, or normal. Indoor input observations include RGB/height and optional normals.
    """

    def __init__(self, channels, distances, input_mode="spatial"):
        super().__init__()
        self.channels = channels
        self.input_mode = input_mode
        center = sample_template_centers(channels, distances)
        scale = torch.empty(channels, 3).uniform_(0, 1)
        axis = F.normalize(torch.randn(channels, 3), dim=1)
        angle = torch.rand(channels, 1) * torch.pi
        # Keep geometry flattened: the paper optimizer excludes 1-D parameter
        # groups from weight decay. Changing this shape changes training.
        self.templates = nn.Parameter(torch.cat([center, scale, axis, angle], -1).flatten())
        if input_mode in {"rgb", "normal"}:
            h = torch.linspace(0, 1, channels)
            self.height = nn.Parameter(h[torch.randperm(channels)])
            self.color = nn.Parameter(torch.rand(3 * channels))
            if input_mode == "normal":
                self.normal = nn.Parameter(torch.randn(3 * channels))
        elif input_mode != "spatial":
            raise ValueError(f"Unsupported DE input mode: {input_mode}")

    def spatial_weights(self):
        raw = self.templates.reshape(-1, 10)
        transform = axis_angle_rotation(raw[:, 6:9], raw[:, 9]) @ torch.diag_embed(raw[:, 3:6])
        return torch.cat([transform.reshape(-1, 9), raw[:, :3]], 1).T.contiguous()

    def forward(self, xyz, neighbors, attributes=None):
        return self.encode(xyz, neighbors, self.spatial_weights(), attributes)

    def encode(self, xyz, neighbors, weights, attributes=None):
        coords = F.pad(xyz, (0, 1))
        mode = self.input_mode
        if mode in {"rgb", "normal"}:
            if attributes is None:
                raise ValueError("Input attributes required")
            anchors = [self.color.reshape(3, -1), self.height.reshape(1, -1)]
            if mode == "normal":
                anchors.append(F.normalize(self.normal.reshape(3, -1), dim=0))
            weights = torch.cat([weights, *anchors], 0)
            points = torch.cat(
                [coords, attributes if mode == "rgb" else F.pad(attributes, (0, 1))], 1
            )
        else:
            points = coords
        score = directional_response(
            points.unsqueeze(0), neighbors.unsqueeze(0).int(), weights, mode, self.training
        )
        return score.squeeze(0).sqrt()
