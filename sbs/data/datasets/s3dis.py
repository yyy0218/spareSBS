"""s3dis data adapter; main-experiment geometry and sparse anchors.
See README.md for usage and NOTICE for attribution.
"""

import torch
import random
import hashlib
import math
import os
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from dataclasses import dataclass
from sbs.ops import grid_subsampling, grid_subsampling_test, KDTree
from sbs.data.retention import normalize_anchor_retention


@dataclass(frozen=True)
class S3DISTestView:
    name: str
    rotation_degrees: float = 0.0
    scale: float = 1.0
    flip_xy: bool = False


def build_s3dis_test_views(profile: str) -> Tuple[S3DISTestView, ...]:
    if str(profile).strip().lower() != "none":
        raise ValueError("The main S3DIS protocol supports the identity view only")
    return (S3DISTestView(name="identity"),)


def apply_s3dis_test_view(xyz: torch.Tensor, view: S3DISTestView) -> torch.Tensor:
    return xyz.clone()


class S3DISDataset(Dataset):
    CLASSES = [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ]

    def __init__(
        self,
        split: str = "train",
        data_root: Optional[str] = None,
        area: str = "!5",
        num_points: int = 30000,
        grid_sizes: List[float] = [0.04, 0.08, 0.16, 0.32],
        ks: List[int] = [24, 32, 32, 32],
        loop: int = 30,
        warmup: bool = False,
        anchor_file: Optional[str] = None,
        sparse_ignore_index: int = -100,
        sparse_train_only: bool = True,
        anchor_retention: str = "preserve",
        point_order_mode: str = "original",
        point_order_seed: int = 0,
        point_order_voxel_size: float = 0.5,
        pre_subsample_shuffle: bool = False,
        pre_subsample_seed: int = 0,
        test_view_profile: str = "none",
        test_samples_per_view: int = 1,
        include_scene_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.split = split
        self.num_points = num_points
        self.grid_sizes = grid_sizes
        self.ks = ks
        self.loop = loop
        self.warmup = warmup
        self.train = split == "train"
        self.test = split == "test"
        self.sparse_ignore_index = int(sparse_ignore_index)
        self.sparse_train_only = bool(sparse_train_only)
        self._return_dense_labels_with_sparse_geometry = False
        self.assert_sparse_keep = os.getenv("SBS_ASSERT_SPARSE_KEEP", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        retention_mode = "preserve" if anchor_retention is None else str(anchor_retention).strip()
        self.anchor_retention = (
            "local_crop"
            if retention_mode == "local_crop"
            else normalize_anchor_retention(retention_mode)
        )
        self.point_order_mode = str(point_order_mode).strip().lower()
        if self.point_order_mode not in {"original", "scene_random", "spatial_voxel"}:
            raise ValueError(f"Unsupported point_order_mode={point_order_mode!r}")
        self.point_order_seed = int(point_order_seed)
        self.point_order_voxel_size = float(point_order_voxel_size)
        if self.point_order_voxel_size <= 0:
            raise ValueError("point_order_voxel_size must be positive")
        self.pre_subsample_shuffle = bool(pre_subsample_shuffle)
        self.pre_subsample_seed = int(pre_subsample_seed)
        self.test_view_profile = str(test_view_profile).strip().lower()
        self.test_views = build_s3dis_test_views(self.test_view_profile)
        self.test_samples_per_view = int(test_samples_per_view)
        if self.test_samples_per_view <= 0:
            raise ValueError("test_samples_per_view must be positive")
        if self.test_view_profile != "none":
            expected_loop = len(self.test_views) * self.test_samples_per_view
            if self.loop != expected_loop:
                raise ValueError(
                    f"loop={self.loop} does not match {self.test_view_profile} test protocol ({len(self.test_views)} views x {self.test_samples_per_view} samples)"
                )
        self.anchors = (
            torch.load(anchor_file, map_location="cpu", weights_only=False)
            if anchor_file is not None
            else None
        )
        if self.anchors is not None and (not isinstance(self.anchors, dict)):
            raise TypeError(f"la_file must be a dict-like mapping, got {type(self.anchors)}")
        if data_root is None:
            data_root = Path.home() / "data" / "s3dis" / "processed"
            warnings.warn(f"data_root not specified, using default: {data_root}")
        self.data_root = Path(data_root)
        self.paths = self._get_file_paths(area)
        if include_scene_names:
            requested_names = list(dict.fromkeys((str(name) for name in include_scene_names)))
            path_by_name = {path.stem: path for path in self.paths}
            missing = [name for name in requested_names if name not in path_by_name]
            if missing:
                raise ValueError(f"Requested S3DIS scenes are absent from area={area!r}: {missing}")
            self.paths = [path_by_name[name] for name in requested_names]
        self.scene_names = [p.stem for p in self.paths]
        if warmup and len(self.paths) > 0:
            maxpts = 0
            max_idx = -1
            for i, p in enumerate(self.paths):
                s = torch.load(p, weights_only=False)[0].shape[0]
                if s > maxpts:
                    maxpts = s
                    max_idx = i
            self.paths = [self.paths[max_idx]]
            self.scene_names = [self.scene_names[max_idx]]
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No data files found in {self.data_root} with area='{area}'. \nExpected format: {self.data_root}/[{area}]*.pt\nExample files: 1_conferenceRoom_1.pt, 2_office_1.pt, etc.\n\nPlease prepare S3DIS data first. Each .pt file should contain:\n  (xyz, rgb, labels) where xyz:(N,3), rgb:(N,3), labels:(N,)"
            )
        print(f"Loading S3DIS {split} dataset from {self.data_root}...")
        print(f"  Area: {area}")
        print(f"  Files: {len(self.paths)}")
        self.datas = []
        for i, path in enumerate(self.paths):
            try:
                data = torch.load(path, weights_only=False)
                if len(data) != 3:
                    raise ValueError(f"Expected (xyz, rgb, labels), got {len(data)} items")
                xyz, rgb, labels = data
                xyz = torch.as_tensor(xyz, dtype=torch.float32)
                rgb = torch.as_tensor(rgb)
                labels = torch.as_tensor(labels).reshape(-1).long()
                self.datas.append((xyz, rgb, labels))
                if i == 0:
                    print(f"  Example: {path.name}")
                    print(f"    Points: {xyz.shape[0]}")
                    print(f"    Classes: {labels.unique().tolist()}")
            except Exception as e:
                raise RuntimeError(f"Error loading {path}: {e}")
        print(f"✅ Loaded {len(self.datas)} scenes")
        if self.anchors is not None:
            matched = sum((1 for s in self.scene_names if s in self.anchors))
            print(
                f"Loaded sparse supervision: la_file={anchor_file}, matched={matched}/{len(self.scene_names)}, ignore={self.sparse_ignore_index}, train_only={self.sparse_train_only}"
            )

    def _get_file_paths(self, area: str) -> List[Path]:
        if not self.data_root.exists():
            return []

        def _is_scene_file(p: Path) -> bool:
            stem = p.stem
            if stem.startswith("s3dis_sparse_"):
                return False
            return len(stem) > 2 and stem[0] in "123456" and (stem[1] == "_")

        all_files = sorted([p for p in self.data_root.glob("*.pt") if _is_scene_file(p)])
        if area.startswith("!"):
            exclude_areas = list(area[1:])
            return [
                f for f in all_files if not any((f.stem.startswith(a + "_") for a in exclude_areas))
            ]
        else:
            include_areas = list(area)
            return [
                f for f in all_files if any((f.stem.startswith(a + "_") for a in include_areas))
            ]

    def __len__(self):
        return len(self.paths) * self.loop

    def __getitem__(self, idx):
        if self.test:
            return self._get_test_item(idx)
        idx //= self.loop
        scene_name = self.scene_names[idx]
        xyz, rgb, labels = self.datas[idx]
        xyz = xyz.clone()
        rgb = rgb.clone()
        labels = labels.clone()
        labels = self._sanitize_label_values(labels)
        sparse_idx = self._get_sparse_index(scene_name, labels.shape[0])
        sparse_mask = self._make_sparse_mask(labels.shape[0], sparse_idx)
        expected_sparse_count = 0 if sparse_mask is None else int(sparse_mask.sum().item())
        if self.train:
            angle = random.random() * 2 * math.pi
            cos, sin = (math.cos(angle), math.sin(angle))
            rotmat = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], dtype=xyz.dtype)
            rotmat *= random.uniform(0.8, 1.2)
            xyz = xyz @ rotmat
            xyz += torch.empty_like(xyz).normal_(std=0.005)
            xyz -= xyz.min(dim=0)[0]
        pre_order = self._pre_subsample_order(scene_name, xyz.shape[0])
        xyz = xyz[pre_order]
        rgb = rgb[pre_order]
        labels = labels[pre_order]
        if sparse_mask is not None:
            sparse_mask = sparse_mask[pre_order]
        if self.train:
            indices_sub = grid_subsampling(xyz, self.grid_sizes[0], 2.5 / 14)
        else:
            indices_sub = grid_subsampling_test(xyz, self.grid_sizes[0], 2.5 / 14, pick=0)
        indices_sub = self._preserve_sparse_indices(indices_sub, sparse_mask)
        xyz = xyz[indices_sub]
        rgb = rgb[indices_sub]
        labels = labels[indices_sub]
        if sparse_mask is not None:
            sparse_mask = sparse_mask[indices_sub]
            self._assert_sparse_preserved(
                scene_name, "post_subsample", expected_sparse_count, sparse_mask
            )
        if not self.train:
            xyz -= xyz.min(dim=0)[0]
        if xyz.shape[0] > self.num_points and self.train:
            pt = xyz[random.randint(0, xyz.shape[0] - 1)]
            condition = self._crop_with_sparse_points_retention(xyz, indices_sub, sparse_mask, pt)
            if sparse_mask is not None and self.anchor_retention == "local_crop":
                expected_sparse_count = int(sparse_mask[condition].sum().item())
            xyz = xyz[condition]
            rgb = rgb[condition]
            labels = labels[condition]
            if sparse_mask is not None:
                sparse_mask = sparse_mask[condition]
                self._assert_sparse_preserved(
                    scene_name, "post_crop", expected_sparse_count, sparse_mask
                )
        if self.train and (not self.warmup):
            mask = None
            up = 10
            up = max(1, round(xyz.shape[0] / self.num_points * up))
            for _ in range(random.randint(1, up)):
                pt = xyz[random.randint(0, xyz.shape[0] - 1)]
                cond = (xyz - pt).square().sum(dim=1) < 0.5**2
                mask = cond if mask is None else mask | cond
            mask = ~mask
            if sparse_mask is not None:
                mask = mask | sparse_mask
            xyz = xyz[mask]
            rgb = rgb[mask]
            labels = labels[mask]
            if sparse_mask is not None:
                sparse_mask = sparse_mask[mask]
                self._assert_sparse_preserved(
                    scene_name, "post_cutout", expected_sparse_count, sparse_mask
                )
        rgb = rgb.float()
        if self.train and random.random() < 0.2:
            rgb.fill_(0.0)
        else:
            if self.train and random.random() < 0.2:
                colmin = rgb.min(dim=0, keepdim=True)[0]
                colmax = rgb.max(dim=0, keepdim=True)[0]
                scale = 255 / (colmax - colmin)
                alpha = random.random()
                rgb = (1 - alpha + alpha * scale) * rgb - alpha * colmin * scale
            rgb.mul_(1 / 250.0)
        order = self._point_order(scene_name, xyz)
        xyz = xyz[order]
        rgb = rgb[order]
        labels = labels[order]
        if sparse_mask is not None:
            sparse_mask = sparse_mask[order]
        height = xyz[:, 2:] * (1 / 3.3)
        features = torch.cat([rgb, height], dim=1)
        indices = []
        self._compute_knn_indices(xyz, self.grid_sizes[::-1], self.ks[::-1], indices)
        xyz = xyz * 40
        if not self._return_dense_labels_with_sparse_geometry:
            labels = self._finalize_sparse_label(labels, sparse_mask)
        return (xyz, features, labels, indices)

    def _point_order(self, scene_name: str, xyz: torch.Tensor) -> torch.Tensor:
        num_points = xyz.shape[0]
        if self.point_order_mode == "original" or num_points <= 1:
            return torch.arange(num_points)
        digest = hashlib.blake2b(
            f"{self.point_order_seed}:{scene_name}".encode("utf-8"), digest_size=8
        ).digest()
        generator = torch.Generator()
        generator.manual_seed(int.from_bytes(digest, "little") & (1 << 63) - 1)
        random_order = torch.randperm(num_points, generator=generator)
        if self.point_order_mode == "scene_random":
            return random_order
        voxel = torch.floor(xyz / self.point_order_voxel_size).long()
        voxel -= voxel.min(dim=0).values
        extent = voxel.max(dim=0).values + 1
        voxel_key = (voxel[:, 0] * extent[1] + voxel[:, 1]) * extent[2] + voxel[:, 2]
        return random_order[torch.argsort(voxel_key[random_order], stable=True)]

    def _pre_subsample_order(self, scene_name: str, num_points: int, vote: int = 0) -> torch.Tensor:
        if not self.pre_subsample_shuffle or num_points <= 1:
            return torch.arange(num_points)
        if self.train:
            return torch.randperm(num_points)
        digest = hashlib.blake2b(
            f"{self.pre_subsample_seed}:{scene_name}:{vote}".encode("utf-8"), digest_size=8
        ).digest()
        generator = torch.Generator()
        generator.manual_seed(int.from_bytes(digest, "little") & (1 << 63) - 1)
        return torch.randperm(num_points, generator=generator)

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
        return torch.unique(
            torch.cat([selected_index, sampled_index.to(selected_index.device)], dim=0), sorted=True
        )

    def _crop_with_sparse_points(
        self,
        xyz: torch.Tensor,
        selected_index: torch.Tensor,
        sparse_mask: Optional[torch.Tensor],
        center: torch.Tensor,
    ):
        if xyz.shape[0] <= self.num_points or not self.train:
            return selected_index
        candidate = (xyz - center).square().sum(dim=1).argsort()
        if sparse_mask is None:
            return candidate[: self.num_points].sort()[0]
        sampled_index = torch.nonzero(sparse_mask, as_tuple=False).squeeze(1)
        if sampled_index.numel() == 0:
            return candidate[: self.num_points].sort()[0]
        if sampled_index.numel() >= self.num_points:
            return sampled_index[: self.num_points].sort()[0]
        keep_mask = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
        keep_mask[sampled_index.to(xyz.device)] = True
        remain = self.num_points - sampled_index.numel()
        extra = candidate[~keep_mask[candidate]][:remain]
        if extra.numel() == 0:
            return sampled_index.sort()[0]
        return torch.cat([sampled_index, extra.to(sampled_index.device)], dim=0).unique(sorted=True)

    def _crop_with_sparse_points_retention(
        self,
        xyz: torch.Tensor,
        selected_index: torch.Tensor,
        sparse_mask: Optional[torch.Tensor],
        center: torch.Tensor,
    ):
        if self.anchor_retention == "preserve":
            return self._crop_with_sparse_points(xyz, selected_index, sparse_mask, center)
        if self.anchor_retention == "local_crop":
            candidate = (xyz - center).square().sum(dim=1).argsort()
            return candidate[: self.num_points].sort()[0]
        raise ValueError(f"Unsupported sparse_retention_mode={self.anchor_retention!r}")

    def _finalize_sparse_label(self, dense_lbl: torch.Tensor, sparse_mask: Optional[torch.Tensor]):
        if sparse_mask is None:
            return dense_lbl
        sparse_lbl = torch.full_like(dense_lbl, self.sparse_ignore_index)
        if sparse_mask.any():
            sparse_lbl[sparse_mask] = dense_lbl[sparse_mask]
        return sparse_lbl

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

    def _apply_sparse_label(self, scene_name: str, lbl: torch.Tensor):
        lbl = self._sanitize_label_values(lbl)
        if self.anchors is None:
            return lbl
        if self.sparse_train_only and (not self.train):
            return lbl
        sampled_index = self.anchors.get(scene_name)
        if sampled_index is None:
            raise KeyError(f"Sparse indices missing for {scene_name}; dense fallback is disabled")
        idx = torch.as_tensor(sampled_index, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < lbl.shape[0])]
        sparse_lbl = torch.full(
            lbl.shape, self.sparse_ignore_index, dtype=torch.long, device=lbl.device
        )
        if idx.numel() > 0:
            sparse_lbl[idx] = lbl[idx]
        return sparse_lbl

    def _sanitize_label_values(self, lbl: torch.Tensor):
        lbl = lbl.long()
        num_classes = len(self.CLASSES)
        invalid = (lbl < 0) | (lbl >= num_classes)
        if invalid.any():
            lbl = lbl.clone()
            lbl[invalid] = self.sparse_ignore_index
        return lbl

    def _get_test_item(self, idx):
        sample = idx % self.loop
        if self.test_view_profile == "none":
            view = self.test_views[0]
            sample_in_view = sample
        else:
            view_index = sample // self.test_samples_per_view
            sample_in_view = sample % self.test_samples_per_view
            view = self.test_views[view_index]
        pick = sample_in_view * 5
        idx //= self.loop
        xyz, rgb, labels = self.datas[idx]
        xyz = apply_s3dis_test_view(xyz, view)
        rgb = rgb.clone()
        if self.test_view_profile != "none":
            xyz -= xyz.min(dim=0).values
        full_xyz = xyz
        full_labels = self._sanitize_label_values(labels)
        pre_order = self._pre_subsample_order(self.scene_names[idx], xyz.shape[0], vote=sample)
        xyz = xyz[pre_order]
        rgb = rgb[pre_order]
        indices_sub = grid_subsampling_test(xyz, self.grid_sizes[0], 2.5 / 14, pick=pick)
        xyz = xyz[indices_sub]
        rgb = rgb[indices_sub].float()
        order = self._point_order(self.scene_names[idx], xyz)
        xyz = xyz[order]
        rgb = rgb[order]
        full_nn = KDTree(xyz).knn(full_xyz, 1)[0].squeeze(-1)
        rgb.mul_(1 / 250.0)
        xyz -= xyz.min(dim=0)[0]
        height = xyz[:, 2:] * (1 / 3.3)
        features = torch.cat([rgb, height], dim=1)
        indices = []
        self._compute_knn_indices(xyz, self.grid_sizes[::-1], self.ks[::-1], indices)
        xyz = xyz * 40
        return (xyz, features, indices, full_nn, full_labels)

    def _compute_knn_indices(
        self,
        xyz: torch.Tensor,
        grid_sizes: List[float],
        ks: List[int],
        indices: List[torch.Tensor],
        full_xyz: Optional[torch.Tensor] = None,
    ):
        first = full_xyz is None
        last = len(ks) == 1
        gs = grid_sizes.pop()
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
        knn_indices = kdt.knn(xyz, ks.pop(), False)[0]
        indices.append(knn_indices)
        if not last:
            self._compute_knn_indices(xyz, grid_sizes, ks, indices, full_xyz)
        if not first:
            back_knn = kdt.knn(full_xyz, 1, False)[0].squeeze(-1)
            indices.append(back_knn)


def fix_indices(indices: List[torch.Tensor], cnt1: List[int], cnt2: List[int]):
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


def s3dis_collate_fn(batch):
    xyz_list, features_list, labels_list, indices_list = zip(*batch)
    depth = (len(indices_list[0]) + 2) // 3
    cnt1 = [0] * depth
    pts = []
    for ids in indices_list:
        pts.extend((x.shape[0] for x in ids[: 2 * depth : 2]))
        cnt2 = []
        fix_indices(ids[::-1], cnt1[::-1], cnt2)
        cnt1 = cnt2
    xyz = torch.cat(xyz_list, dim=0)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    indices = [torch.cat(ids, dim=0) for ids in zip(*indices_list)]
    pts = torch.tensor(pts, dtype=torch.int64).view(-1, depth).transpose(0, 1).contiguous()
    return (xyz, features, indices, pts, labels)


def s3dis_test_collate_fn(batch):
    return batch[0]


__all__ = ["S3DISDataset", "s3dis_collate_fn", "s3dis_test_collate_fn", "fix_indices"]
