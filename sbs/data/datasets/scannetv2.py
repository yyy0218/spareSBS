"""scannetv2 data adapter; main-experiment geometry and sparse anchors.
See README.md for usage and NOTICE for attribution.
"""

import torch
import numpy as np
from torch.nn import functional as F
import random
import os
import math
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import warnings
from sbs.data.transforms import ElasticDistortion
from sbs.ops import grid_subsampling, KDTree, grid_subsampling_test
from sbs.data.retention import normalize_anchor_retention

mix_prob = torch.tensor([0.5])
mix_prob.share_memory_()


def change_mix_prob(p):
    mix_prob[0] = p


def get_mix_prob():
    return mix_prob[0].item()


def axis_to_matrix(axis, angle):
    axis = axis.view(1, 3)
    axis = F.normalize(axis, dim=-1)
    A = torch.zeros(axis.shape[0], 3, 3, device=axis.device)
    A[:, 0, 1] = -axis[:, 2]
    A[:, 0, 2] = axis[:, 1]
    A[:, 1, 0] = axis[:, 2]
    A[:, 1, 2] = -axis[:, 0]
    A[:, 2, 0] = -axis[:, 1]
    A[:, 2, 1] = axis[:, 0]
    M = A * angle.sin() + torch.bmm(A, A) * (1 - angle.cos()) + torch.eye(3, device=axis.device)
    return M.transpose(-2, -1).contiguous().view(3, 3)


class ScanNetDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root=None,
        ks=[24, 32, 32, 32, 32],
        grid_sizes=[0.02, 0.04, 0.08, 0.16, 0.32],
        max_pts=80000,
        loop=6,
        train=None,
        test=False,
        warmup=False,
        test_rotations=None,
        test_scales=None,
        anchor_file=None,
        sparse_ignore_index=-1,
        sparse_train_only=True,
        anchor_retention="preserve",
    ):
        if data_root is None:
            data_root = Path.home() / "data" / "scannetv2" / "processed"
            warnings.warn(f"data_root not specified, using default: {data_root}")
        self.data_root = Path(data_root)
        if split == "train":
            list_file = Path(__file__).parent / "scannetv2_train.txt"
        else:
            list_file = Path(__file__).parent / "scannetv2_val.txt"
        if not list_file.exists():
            raise FileNotFoundError(f"File list not found: {list_file}")
        with open(list_file, "r") as f:
            file_names = [line.strip() for line in f.readlines()]
        self.scene_names = []
        self.paths = []
        missing = []
        for name in file_names:
            candidates = [
                self.data_root / split / name,
                self.data_root / f"{name}.pt",
                self.data_root / name,
            ]
            resolved = None
            for c in candidates:
                if c.exists():
                    resolved = c
                    break
            if resolved is None:
                missing.append(name)
                continue
            self.scene_names.append(name)
            self.paths.append(resolved)
        if missing:
            print(f"⚠️  Warning: {len(missing)} scenes not found, will be skipped")
            for scene_name in missing[:5]:
                print(f"  - {scene_name}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
        if len(self.paths) == 0:
            raise RuntimeError(f"No valid ScanNet scenes found in {self.data_root}")
        print(f"Loaded {len(self.paths)} scenes ({split})")
        self.loop = loop
        self.train = train if train is not None else split == "train"
        self.test = test
        self.warmup = warmup
        self.test_rotations = [0, 0.5, 1, 1.5] if test_rotations is None else list(test_rotations)
        self.test_scales = [0.95, 1, 1.05] if test_scales is None else list(test_scales)
        self.sparse_ignore_index = int(sparse_ignore_index)
        self.sparse_train_only = bool(sparse_train_only)
        self.assert_sparse_keep = os.getenv("SBS_ASSERT_SPARSE_KEEP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.anchor_retention = normalize_anchor_retention(anchor_retention)
        self._return_dense_labels_with_sparse_geometry = False
        self.anchors = (
            torch.load(anchor_file, map_location="cpu", weights_only=False)
            if anchor_file is not None
            else None
        )
        if self.anchors is not None and (not isinstance(self.anchors, dict)):
            raise TypeError(f"la_file must be a dict-like mapping, got {type(self.anchors)}")
        self.k = list(ks)
        self.grid_size = list(grid_sizes)
        self.max_pts = 2**3**4
        self.max_pts = max_pts
        self.datas = [self._load_scene(path) for path in self.paths]
        if warmup:
            max_idx = int(np.argmax([x.shape[0] for x, _, _, _ in self.datas]))
            self.paths = [self.paths[max_idx]]
            self.scene_names = [self.scene_names[max_idx]]
            self.datas = [self.datas[max_idx]]
        if self.anchors is not None:
            matched = sum((1 for s in self.scene_names if s in self.anchors))
            print(
                f"Loaded sparse supervision: la_file={anchor_file}, matched={matched}/{len(self.scene_names)}, ignore={self.sparse_ignore_index}, train_only={self.sparse_train_only}"
            )
        self.els = ElasticDistortion()

    @staticmethod
    def _to_tensor(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype)

    @staticmethod
    def _normalize_dense_label(segment):
        segment = np.asarray(segment).reshape(-1).astype(np.int64, copy=False)
        invalid = (segment < 0) | (segment > 20)
        if invalid.any():
            segment = segment.copy()
            segment[invalid] = 20
        return segment

    def _load_scene(self, path: Path):
        if path.is_dir():
            coord = np.load(path / "coord.npy")
            color = np.load(path / "color.npy")
            normal = np.load(path / "normal.npy")
            seg20 = path / "segment20.npy"
            seg200 = path / "segment200.npy"
            if seg20.exists():
                segment = np.load(seg20)
            elif seg200.exists():
                segment = np.load(seg200)
            else:
                segment = np.full((coord.shape[0],), self.sparse_ignore_index, dtype=np.int64)
            segment = self._normalize_dense_label(segment)
            return (
                self._to_tensor(coord, torch.float32),
                self._to_tensor(color, torch.float32),
                self._to_tensor(normal, torch.float32),
                self._to_tensor(segment, torch.long),
            )
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            coord = data.get("coord")
            color = data.get("color")
            normal = data.get("normal")
            segment = (
                data.get("segment20")
                if "segment20" in data
                else data.get("segment", data.get("label", data.get("labels")))
            )
            if coord is None or color is None or normal is None:
                raise ValueError(f"Invalid scene dict in {path}: missing coord/color/normal")
            if segment is None:
                segment = np.full((len(coord),), self.sparse_ignore_index, dtype=np.int64)
            segment = self._normalize_dense_label(segment)
            return (
                self._to_tensor(coord, torch.float32),
                self._to_tensor(color, torch.float32),
                self._to_tensor(normal, torch.float32),
                self._to_tensor(segment, torch.long),
            )
        if isinstance(data, (list, tuple)) and len(data) >= 4:
            xyz, col, norm, lbl = data[:4]
            lbl = self._normalize_dense_label(lbl)
            return (
                self._to_tensor(xyz, torch.float32),
                self._to_tensor(col, torch.float32),
                self._to_tensor(norm, torch.float32),
                self._to_tensor(lbl, torch.long),
            )
        raise TypeError(f"Unsupported ScanNet scene format: {path}")

    def _get_sparse_index(self, scene_name: str, num_points: int):
        if self.anchors is None:
            return None
        if self.sparse_train_only and (not self.train):
            return None
        sampled_index = self.anchors.get(scene_name)
        if sampled_index is None:
            raise KeyError(f"Sparse indices missing for {scene_name}; dense fallback is disabled")
        idx = torch.as_tensor(sampled_index, dtype=torch.long)
        if (idx < 0).any() or (idx >= num_points).any():
            raise ValueError(f"Sparse indices out of bounds for {scene_name}")
        return idx.unique(sorted=True)

    def _make_sparse_mask(self, num_points: int, sampled_index: Optional[torch.Tensor]):
        if sampled_index is None:
            return None  # Intentional dense/geometry-only warmup; missing map keys fail earlier.
        mask = torch.zeros(num_points, dtype=torch.bool)
        if sampled_index.numel() > 0:
            mask[sampled_index] = True
        return mask

    def _preserve_sparse_indices(
        self, selected_index: torch.Tensor, sparse_mask: Optional[torch.Tensor]
    ):
        if sparse_mask is None:
            return selected_index
        sampled_index = torch.nonzero(sparse_mask, as_tuple=False).squeeze(1)
        if sampled_index.numel() == 0:
            return selected_index
        return torch.unique(torch.cat([selected_index, sampled_index], dim=0), sorted=True)

    def _crop_with_sparse_points(
        self,
        xyz: torch.Tensor,
        selected_index: torch.Tensor,
        sparse_mask: Optional[torch.Tensor],
        center: torch.Tensor,
    ):
        if xyz.shape[0] <= self.max_pts or not self.train:
            return selected_index
        candidate = (xyz - center).square().sum(dim=1).argsort()
        if sparse_mask is None:
            return candidate[: self.max_pts].sort()[0]
        sampled_index = torch.nonzero(sparse_mask, as_tuple=False).squeeze(1)
        if sampled_index.numel() == 0:
            return candidate[: self.max_pts].sort()[0]
        if sampled_index.numel() >= self.max_pts:
            return sampled_index[: self.max_pts].sort()[0]
        keep_mask = torch.zeros(xyz.shape[0], dtype=torch.bool)
        keep_mask[sampled_index] = True
        keep = [sampled_index]
        remain = self.max_pts - sampled_index.numel()
        extra = candidate[~keep_mask[candidate]][:remain]
        if extra.numel() > 0:
            keep.append(extra)
        return torch.cat(keep, dim=0).unique(sorted=True)

    def _crop_with_sparse_points_retention(
        self,
        xyz: torch.Tensor,
        selected_index: torch.Tensor,
        sparse_mask: Optional[torch.Tensor],
        center: torch.Tensor,
    ):
        if self.anchor_retention == "preserve":
            return self._crop_with_sparse_points(xyz, selected_index, sparse_mask, center)
        raise ValueError(f"Unsupported sparse_retention_mode={self.anchor_retention!r}")

    def _finalize_sparse_label(self, dense_lbl: torch.Tensor, sparse_mask: Optional[torch.Tensor]):
        if sparse_mask is None:
            return dense_lbl
        sparse_lbl = torch.full_like(dense_lbl, self.sparse_ignore_index)
        if sparse_mask.any():
            sparse_lbl[sparse_mask] = dense_lbl[sparse_mask]
        return sparse_lbl

    def _labels_for_output(self, dense_lbl: torch.Tensor, sparse_mask: Optional[torch.Tensor]):
        if self._return_dense_labels_with_sparse_geometry:
            return dense_lbl
        return self._finalize_sparse_label(dense_lbl, sparse_mask)

    def _assert_sparse_preserved(
        self,
        scene_tag: str,
        stage: str,
        expected_sparse_count: int,
        sparse_mask: Optional[torch.Tensor],
    ):
        if not self.assert_sparse_keep or sparse_mask is None:
            return
        actual_sparse_count = int(sparse_mask.sum().item())
        if actual_sparse_count != int(expected_sparse_count):
            raise RuntimeError(
                f"Sparse points must be fully preserved, but got a mismatch at {stage}: scene={scene_tag}, expected={expected_sparse_count}, actual={actual_sparse_count}"
            )

    def __len__(self):
        return len(self.paths) * self.loop

    def __getitem__(self, idx):
        if self.test:
            return self.get_test_item(idx)
        idx //= self.loop
        scene_name = self.scene_names[idx]
        xyz, col, norm, lbl = [x.clone() for x in self.datas[idx]]
        sparse_idx = self._get_sparse_index(scene_name, lbl.shape[0])
        sparse_mask = self._make_sparse_mask(lbl.shape[0], sparse_idx)
        scene_tag = scene_name
        mix = True if self.train and random.random() < get_mix_prob() else False
        if mix:
            mix_idx = random.randrange(0, len(self.datas))
            mix_name = self.scene_names[mix_idx]
            x2, c2, n2, l2 = [x.clone() for x in self.datas[mix_idx]]
            sparse_idx2 = self._get_sparse_index(mix_name, l2.shape[0])
            sparse_mask2 = self._make_sparse_mask(l2.shape[0], sparse_idx2)
        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = (math.cos(angle), math.sin(angle))
            rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
            norm = norm @ rotmat
            rotmat *= random.uniform(0.8, 1.2)
            xyz = xyz @ rotmat
            xyz = self.els(xyz)
            if mix:
                angle = random.random() * 2 * math.pi
                cos, sin = (math.cos(angle), math.sin(angle))
                rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
                n2 = n2 @ rotmat
                rotmat *= random.uniform(0.8, 1.2)
                x2 = x2 @ rotmat
                x2 = self.els(x2)
                xyz[:, 2] -= xyz[:, 2].min()
                x2[:, 2] -= x2[:, 2].min()
                pick1 = random.choice(xyz)
                pick2 = random.choice(x2)
                x2[:, :2] += pick1[:2] - pick2[:2]
                xyz = torch.cat([xyz, x2], dim=0)
                norm = torch.cat([norm, n2], dim=0)
                col = torch.cat([col, c2], dim=0)
                lbl = torch.cat([lbl, l2], dim=0)
                if sparse_mask is not None or sparse_mask2 is not None:
                    if sparse_mask is None:
                        sparse_mask = torch.zeros(xyz.shape[0] - x2.shape[0], dtype=torch.bool)
                    if sparse_mask2 is None:
                        sparse_mask2 = torch.zeros(x2.shape[0], dtype=torch.bool)
                    sparse_mask = torch.cat([sparse_mask, sparse_mask2], dim=0)
                scene_tag = f"{scene_name}+{mix_name}"
            xmin = xyz.min(dim=0)[0]
            xyz -= xmin
            if mix:
                pick1 -= xmin
            else:
                pick1 = random.choice(xyz)
        expected_sparse_count = 0 if sparse_mask is None else int(sparse_mask.sum().item())
        if self.train:
            indices = grid_subsampling(xyz, self.grid_size[0], 2.5 / 1.5)
        else:
            indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 1.5, pick=0)
        indices = self._preserve_sparse_indices(indices, sparse_mask)
        xyz = xyz[indices]
        if sparse_mask is not None:
            sparse_mask = sparse_mask[indices]
            self._assert_sparse_preserved(
                scene_tag, "post_subsample", expected_sparse_count, sparse_mask
            )
        if not self.train:
            xyz -= xyz.min(dim=0)[0]
        if xyz.shape[0] > self.max_pts and self.train:
            pt = pick1
            condition = self._crop_with_sparse_points_retention(xyz, indices, sparse_mask, pt)
            xyz = xyz[condition]
            indices = indices[condition]
            if sparse_mask is not None:
                sparse_mask = sparse_mask[condition]
                self._assert_sparse_preserved(
                    scene_tag, "post_crop", expected_sparse_count, sparse_mask
                )
        col = col[indices]
        lbl = lbl[indices]
        norm = norm[indices]
        col = col.float()
        lbl = self._labels_for_output(lbl, sparse_mask)
        if self.train and random.random() < 0.2:
            col.fill_(0.0)
        else:
            if self.train and random.random() < 0.2:
                colmin = col.min(dim=0, keepdim=True)[0]
                colmax = col.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                col = (1 - alpha + alpha * scale) * col - alpha * colmin * scale
            col.mul_(1 / 250.0)
        if self.train and random.random() < 0.2:
            norm.fill_(0.0)
        height = xyz[:, 2:] * (1 / 2.6)
        feature = torch.cat([col, height, norm], dim=1)
        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)
        xyz.mul_(60)
        return (xyz, feature, indices, lbl)

    def get_test_item(self, idx):
        rotations = self.test_rotations
        scales = self.test_scales
        augs = len(rotations) * len(scales)
        aug = idx % self.loop
        pick = aug // augs
        aug %= augs
        idx //= self.loop
        xyz, col, norm, lbl = [x.clone() for x in self.datas[idx]]
        angle = math.pi * rotations[aug // len(scales)]
        cos, sin = (math.cos(angle), math.sin(angle))
        rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        norm = norm @ rotmat
        rotmat *= scales[aug % len(scales)]
        xyz = xyz @ rotmat
        xyz -= xyz.min(dim=0)[0]
        full_xyz = xyz
        full_lbl = lbl
        indices = grid_subsampling_test(xyz, self.grid_size[0], 2.5 / 1.5, pick=pick)
        xyz = xyz[indices]
        col = col[indices].float()
        norm = norm[indices]
        full_nn = KDTree(xyz).knn(full_xyz, 1)[0].squeeze(-1)
        col.mul_(1 / 250.0)
        xyz -= xyz.min(dim=0)[0]
        feature = torch.cat([col, xyz[:, 2:] * (1 / 2.6), norm], dim=1)
        indices = []
        self.knn(xyz, self.grid_size[::-1], self.k[::-1], indices)
        xyz.mul_(60)
        return (xyz, feature, indices, full_nn, full_lbl)

    def knn(
        self,
        xyz: torch.Tensor,
        grid_size: list,
        k: list,
        indices: list,
        full_xyz: torch.Tensor = None,
    ):
        first = full_xyz is None
        last = len(k) == 1
        gs = grid_size.pop()
        if first:
            full_xyz = xyz
        else:
            if self.warmup:
                sub_indices = torch.randperm(xyz.shape[0])[: int(xyz.shape[0] / gs)].contiguous()
            else:
                sub_indices = grid_subsampling(xyz, gs)
            xyz = xyz[sub_indices]
            indices.append(sub_indices)
        kdt = KDTree(xyz)
        indices.append(kdt.knn(xyz, k.pop(), False)[0])
        if not last:
            self.knn(xyz, grid_size, k, indices, full_xyz)
        if not first:
            indices.append(kdt.knn(full_xyz, 1, False)[0].squeeze(-1))
        return


def fix_indices(indices, cnt1: list, cnt2: list):
    first = len(cnt2) == 0
    last = len(cnt1) == 1 or (len(cnt1) == 2 and len(cnt2) != 0)
    if first:
        c1 = cnt1[-1]
    else:
        indices.pop().add_(cnt1.pop())
        c1 = cnt1[-1]
    knn = indices.pop()
    knn.add_(c1)
    cnt2.append(c1 + knn.shape[0])
    if not last:
        fix_indices(indices, cnt1, cnt2)
    if not first:
        indices.pop().add_(c1)


def scan_collate_fn(batch):
    xyz, col, indices, lbl = list(zip(*batch))
    depth = (len(indices[0]) + 2) // 3
    cnt1 = [0] * depth
    pts = []
    for ids in indices:
        pts.extend((x.shape[0] for x in ids[: 2 * depth : 2]))
        cnt2 = []
        fix_indices(ids[::-1], cnt1[::-1], cnt2)
        cnt1 = cnt2
    xyz = torch.cat(xyz, dim=0)
    col = torch.cat(col, dim=0)
    lbl = torch.cat(lbl, dim=0)
    indices = [torch.cat(ids, dim=0) for ids in zip(*indices)]
    pts = torch.tensor(pts, dtype=torch.int64).view(-1, depth).transpose(0, 1).contiguous()
    return (xyz, col, indices, pts, lbl)


def scan_test_collate_fn(batch):
    return batch[0]


__all__ = [
    "ScanNetDataset",
    "scan_collate_fn",
    "scan_test_collate_fn",
    "fix_indices",
    "change_mix_prob",
]
