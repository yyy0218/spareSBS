"""Differentiable, chunked PyTorch reference operators (CPU or CUDA).

This backend is for tests and numerical debugging, not a throughput substitute
for the native kernels. Reference voxel selection is deliberately explicit;
its stochastic ordering is not claimed identical to native hash sampling.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def response(points, neighbors, weights, mode, chunk_size=256):
    # Input attributes/coordinates are observations, not trainable features.
    points = points.detach().float()
    weights = weights.float()
    groups = []
    for batch in range(points.shape[0]):
        x = points[batch]
        chunks = []
        transform = weights[:9].T.reshape(-1, 3, 3)
        centers = weights[9:12].T
        for start in range(0, len(x), chunk_size):
            ids = neighbors[batch, start : start + chunk_size].long()
            nbr = x[ids]
            delta = nbr[:, :, :3] - x[start : start + len(ids), None, :3]
            displacement = delta[:, :, None, :] - centers[None, None, :, :]
            projected = torch.einsum("nkci,cij->nkcj", displacement, transform)
            cost = projected.square().sum(-1) + (1.0001 if mode == "normal" else 1e-8)
            if mode in ("rgb", "normal"):
                dims = 4
                diff = nbr[:, :, None, 4 : 4 + dims] - weights[12 : 12 + dims].T[None, None]
                cost = cost + diff.square().sum(-1)
            if mode == "normal":
                cost = cost + torch.einsum("nkj,cj->nkc", nbr[:, :, 8:11], weights[16:19].T)
            # torch.min retains first-argmin semantics for gradient ties.
            chunks.append(cost.min(dim=1).values)
        groups.append(torch.cat(chunks, 0))
    return torch.stack(groups, 0)


def directional_pool(features, xyz, neighbors, weights, training=True, chunk_size=256):
    features = features.float()
    xyz = xyz.detach().float()
    weights = weights.float()
    groups = []
    centers = weights[:3].T.repeat_interleave(4, 0)
    scale = weights[3].repeat_interleave(4)
    for batch in range(features.shape[0]):
        f = features[batch]
        x = xyz[batch]
        chunks = []
        for start in range(0, len(f), chunk_size):
            ids = neighbors[batch, start : start + chunk_size].long()
            displacement = x[ids, :3] - x[start : start + len(ids), None, :3]
            distance = (displacement[:, :, None, :] - centers[None, None]).square().sum(-1) + 1e-8
            gate = torch.exp(-distance * scale)
            edge = (f[ids] - f[start : start + len(ids), None]) * gate
            chunks.append(edge.max(dim=1).values)
        groups.append(torch.cat(chunks, 0))
    return torch.stack(groups, 0)


class KDTree:
    def __init__(self, xyz, max_leaf_size=20):
        self.xyz = torch.as_tensor(xyz, dtype=torch.float32).cpu().contiguous()
        if len(self.xyz) == 0:
            raise ValueError("Cannot index an empty point cloud")
        self.tree = cKDTree(self.xyz.numpy(), leafsize=max_leaf_size)

    def knn(self, query, k=1, ordered=False):
        d, idx = self.tree.query(torch.as_tensor(query).cpu().numpy(), k=min(k, len(self.xyz)))
        idx = np.asarray(idx).reshape(len(query), -1)
        d = np.asarray(d).reshape(len(query), -1)
        if idx.shape[1] < k:
            padding = k - idx.shape[1]
            idx = np.concatenate([idx, np.repeat(idx[:, :1], padding, 1)], 1)
            d = np.concatenate([d, np.repeat(d[:, :1], padding, 1)], 1)
        return torch.from_numpy(idx.astype(np.int32)), torch.from_numpy((d * d).astype(np.float32))


def _voxel_groups(xyz, grid_size):
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    cells = np.trunc(
        torch.as_tensor(xyz).cpu().numpy().astype(np.float64) / np.float32(grid_size)
    ).astype(np.int64)
    _, first, inv = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    sorted_points = np.argsort(inv, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(inv))]
    return [sorted_points[offsets[g] : offsets[g + 1]] for g in order]


def grid_subsampling(xyz, grid_size, hash_size=1.0):
    groups = _voxel_groups(xyz, grid_size)
    return torch.tensor(
        [group[int(torch.randint(len(group), (1,)).item())] for group in groups], dtype=torch.long
    )


def grid_subsampling_test(xyz, grid_size, hash_size=1.0, pick=0):
    groups = _voxel_groups(xyz, grid_size)
    return torch.tensor(
        [
            group[pick] if pick < len(group) else group[int(torch.randint(len(group), (1,)).item())]
            for group in groups
        ],
        dtype=torch.long,
    )
