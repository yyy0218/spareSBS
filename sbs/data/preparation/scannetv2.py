"""ScanNetV2 mesh conversion; keep vertex order, normals and 20-class labels."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import numpy as np
import torch
from .common import atomic_save

CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def label_mapping(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            try:
                label_id = int(row["nyu40id"])
            except (KeyError, ValueError, TypeError):
                continue
            for key in ("raw_category", "category", "nyuClass", "nyu40class"):
                word = (row.get(key) or "").strip().lower()
                if word:
                    result.setdefault(word, label_id)
    if not result:
        raise ValueError("Empty ScanNet label map; expected scannetv2-labels.combined.tsv")
    return result


def prepare(raw_root, processed_root, overwrite=False, names=None, workers=4, label_map=None):
    from plyfile import PlyData
    base, out = Path(raw_root), Path(processed_root)
    base = base / "scans" if (base / "scans").is_dir() else base
    if not base.is_dir():
        raise FileNotFoundError(f"ScanNet raw directory not found: {base}")
    names = sorted(x.name for x in base.glob("scene*_*") if x.is_dir()) if names is None else names
    if not names:
        raise FileNotFoundError("Expected sceneXXXX_YY directories under --raw-root or scans/")
    lookup = label_mapping(label_map) if label_map else None

    def convert(name):
        scene = base / name
        target = out / (name + ".pt")
        if target.exists() and not overwrite:
            return name
        mesh = PlyData.read(scene / (name + "_vh_clean_2.ply"),
                            known_list_len={"face": {"vertex_indices": 3}}, mmap="r")
        v = mesh["vertex"]
        xyz = np.stack([v[k] for k in ("x", "y", "z")], 1).astype(np.float32)
        rgb = np.stack([v[k] for k in ("red", "green", "blue")], 1).astype(np.uint8)
        raw_faces = mesh["face"]["vertex_indices"]
        faces = (np.stack(raw_faces) if raw_faces.dtype == object else np.asarray(raw_faces)).astype(np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{name}: expected triangular faces")
        labels_path = scene / (name + "_vh_clean_2.labels.ply")
        if labels_path.is_file():
            sv = PlyData.read(labels_path)["vertex"]
            sxyz = np.stack([sv[k] for k in ("x", "y", "z")], 1).astype(np.float32)
            if sxyz.shape != xyz.shape or not np.allclose(xyz, sxyz):
                raise ValueError(f"{name}: mesh/label point order differs")
            nyu = np.asarray(sv["label"], dtype=np.int64)
            labels = np.full(len(xyz), 20, dtype=np.int64)
            for cls, original in enumerate(CLASS_IDS):
                labels[nyu == original] = cls
        else:
            if lookup is None:
                raise FileNotFoundError(
                    f"{labels_path} missing. Download the official label PLY files, or provide "
                    "--label-map scannetv2-labels.combined.tsv with segment/aggregation JSONs."
                )
            seg = np.asarray(json.loads((scene / (name + "_vh_clean_2.0.010000.segs.json")).read_text())["segIndices"], dtype=np.int64)
            if len(seg) != len(xyz) or (seg < 0).any():
                raise ValueError(f"{name}: invalid segmentation indices")
            groups = json.loads((scene / (name + ".aggregation.json")).read_text())["segGroups"]
            seg_labels = np.full(int(seg.max()) + 1, 20, dtype=np.int64)
            ids = {v: i for i, v in enumerate(CLASS_IDS)}
            for group in groups:
                cls = ids.get(lookup.get(str(group["label"]).strip().lower()), 20)
                for index in group["segments"]:
                    if 0 <= int(index) < len(seg_labels):
                        seg_labels[int(index)] = cls
            labels = seg_labels[seg]
        xyz = xyz - xyz.min(axis=0, keepdims=True)
        fn = np.cross(xyz[faces[:, 1]] - xyz[faces[:, 0]], xyz[faces[:, 2]] - xyz[faces[:, 0]]).astype(np.float32)
        normals = np.zeros_like(xyz)
        # Preserve the accumulation order used by the evaluated preprocessing.
        for col in range(3):
            np.add.at(normals, faces[:, col], fn)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        atomic_save(tuple(torch.from_numpy(a) for a in (xyz, rgb, normals, labels)), target)
        return name

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, name in enumerate(pool.map(convert, names), 1):
            if i == 1 or i % 50 == 0 or i == len(names):
                print(f"ScanNetV2: {i}/{len(names)} scenes ({name})", flush=True)
    return len(names)
