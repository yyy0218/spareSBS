"""Lazy bindings for the required native point operators; no build on import.

DeLA-v2 derived code: see NOTICE (DeLA-v2 section).
Modified Python symbol organization for Support Before Supervision.
"""

from pathlib import Path
from functools import lru_cache
import os
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "native_src"
    build = Path(
        os.environ.get("SBS_BUILD_DIR", str(Path.home() / ".cache" / "sbs" / "indoor_0p1_native"))
    )
    build.mkdir(parents=True, exist_ok=True)
    return load(
        "sbs_native",
        sources=[str(p) for p in sorted(source.iterdir()) if p.suffix in {".cpp", ".cu"}],
        extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
        extra_cuda_cflags=["-Xptxas", "-v"],
        build_directory=str(build),
        verbose=True,
    )


def next_prime(x) -> int:
    """
    Finds the next prime, x included.
    x should be >= 3 for a correct result.
    """
    x = int(x) | 1
    for i in range(x, 2 * x, 2):
        prime = True
        for j in range(3, int(i**0.5) + 1, 2):
            if i % j == 0:
                prime = False
                break
        if prime:
            return i


def grid_subsampling(xyz: torch.Tensor, grid_size: float, hash_size: float = 1.0) -> torch.Tensor:
    """
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and (xyz.dtype == torch.float)
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 3,), dtype=torch.int64)
    indices = _extension().grid_subsampling(xyz, grid_size, table, storage)
    return indices


def grid_subsampling_test(
    xyz: torch.Tensor, grid_size: float, hash_size: float = 1.0, pick=0
) -> torch.Tensor:
    """
    xyz: N x 3, float, non-negative coordinates
    grid_size: float, positive
    hash_size: How large the hash table should be relative to the original point cloud size.
                If estimated downsampling ratio is k, i.e., ori_size = k * subsampled_size,
                then recommended value is 2~3 / k.
                Must be greater than 1 / real_k
    pick:  the nth point in the same grid to pick, random picked if actual resident points < pick
    return value: M, int64, selected indices
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3 and (xyz.dtype == torch.float)
    if xyz.stride(0) != 3:
        xyz = xyz.contiguous()
    size = xyz.shape[0] * hash_size
    size = next_prime(size + 1)
    table = torch.zeros((size,), dtype=torch.int64)
    storage = torch.empty((size * 4,), dtype=torch.int64)
    indices = _extension().grid_subsampling_test(xyz, grid_size, table, storage, pick)
    return indices


class KDTree:
    """
    kdt = KDTree(xyz)
    indices, squared_dists = kdt.knn(query_xyz, k=16, ordered=True)
    indices: int32
    dists: float

    Setting ordered = False (default) can be 1.1-1.2x faster.
    If there are not enough neighbors, the nearest point is used for padding.
    Resources (reference to xyz, built tree) are freed when kdt goes out of life scope.
    """

    def __init__(self, xyz: torch.Tensor, max_leaf_size=20):
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and (xyz.dtype == torch.float)
        if xyz.stride(0) != 3:
            xyz = xyz.contiguous()
        self.xyz = xyz
        self.n = xyz.shape[0]
        self.tree, self.pca = _extension().kdtree_build(xyz, max_leaf_size)

    def __del__(self):
        _extension().kdtree_free(self.tree, self.pca)

    def knn(self, query: torch.Tensor, k=1, ordered=False):
        assert query.ndim == 2 and query.shape[1] == 3 and (query.dtype == torch.float)
        if query.stride(0) != 3:
            query = query.contiguous()
        queries = query.shape[0]
        nbrs = min(self.n, k)
        if self.n < k:
            ordered = True
        indices = torch.empty((queries, nbrs), dtype=torch.int32)
        dists = torch.empty((queries, nbrs), dtype=torch.float)
        _extension().kdtree_knn(self.tree, query, indices, dists, ordered)
        if self.n < k:
            indices = torch.cat([indices, indices[:, :1].expand(-1, k - self.n)], dim=1)
            dists = torch.cat([dists, dists[:, :1].expand(-1, k - self.n)], dim=1)
        return (indices, dists)


class RGBResponse(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 16
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        _extension().knn_ese_4_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 16, C, device=xyz.device)
        _extension().knn_ese_4_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return (None, None, wgrad, None)


rgb_response = RGBResponse.apply


class NormalResponse(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 19
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        _extension().knn_ese_4n_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 19, C, device=xyz.device)
        _extension().knn_ese_4n_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return (None, None, wgrad, None)


normal_response = NormalResponse.apply


class SpatialResponse(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, xyz: torch.Tensor, knn: torch.Tensor, weight: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        B, N, _ = xyz.shape
        C = weight.numel() // 12
        output = torch.empty(B, N, C, device=xyz.device)
        back_idx = torch.empty(B, N, C, dtype=torch.int32, device=xyz.device)
        _extension().knn_ese_forward(output, back_idx, xyz, knn, weight)
        if training or weight.requires_grad:
            ctx.save_for_backward(xyz, weight, back_idx)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        xyz, weight, back_idx = ctx.saved_tensors
        C = grad.shape[2]
        wgrad = torch.zeros(256, 12, C, device=xyz.device)
        _extension().knn_ese_backward(wgrad, grad, back_idx, xyz, weight)
        wgrad = wgrad.sum(dim=0).reshape(*weight.shape)
        return (None, None, wgrad, None)


spatial_response = SpatialResponse.apply


class DirectionalPoolFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feature: torch.Tensor,
        xyz: torch.Tensor,
        knn: torch.Tensor,
        weight: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        output = torch.empty_like(feature, dtype=torch.float)
        back_idx = torch.empty_like(feature, dtype=torch.int32)
        _extension().la_ese_a4_forward(
            output, back_idx, feature.float(), xyz.float(), knn, weight.float()
        )
        if training or weight.requires_grad or feature.requires_grad:
            ctx.save_for_backward(back_idx, feature, xyz, weight)
        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        grad = grad.float().contiguous()
        back_idx, feature, xyz, weight = ctx.saved_tensors
        weight = weight.view(4, -1, 1).repeat(1, 1, 4).view(4, -1)
        f_grad = torch.zeros_like(grad)
        w_grad = torch.zeros(256, *weight.shape, device=xyz.device)
        _extension().la_ese_backward(
            f_grad, w_grad, grad, back_idx, feature.float(), xyz.float(), weight.float()
        )
        w_grad = w_grad.sum(dim=0).view(4, -1, 4).sum(dim=2)
        return (f_grad, None, None, w_grad, None)


directional_pool = DirectionalPoolFunction.apply
