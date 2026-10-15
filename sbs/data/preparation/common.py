"""Data checks and content fingerprints for the two released recipes."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[3]


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def fingerprint(dataset, value):
    """Hash ordered, canonical tensors, independent of torch.save container bytes."""
    expected = 3 if dataset == "s3dis" else 4
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"Expected {expected} scene tensors for {dataset}")
    dtypes = [torch.float32, torch.uint8]
    if dataset == "scannetv2":
        dtypes.append(torch.float32)
    dtypes.append(torch.int64)
    n = len(value[0])
    if n == 0:
        raise ValueError("Empty point cloud")
    h = hashlib.sha256()
    for i, (v, dtype) in enumerate(zip(value, dtypes)):
        t = torch.as_tensor(v).detach().cpu().to(dtype).contiguous()
        shape = (n,) if i == expected - 1 else (n, 3)
        if tuple(t.shape) != shape:
            raise ValueError(f"Tensor {i} has shape {tuple(t.shape)}, expected {shape}")
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise ValueError("Scene contains nonfinite observations")
        h.update(str(dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    labels = torch.as_tensor(value[-1]).long()
    max_label = 12 if dataset == "s3dis" else 20
    if (labels < 0).any() or (labels > max_label).any():
        raise ValueError("Unexpected prepared semantic label IDs")
    return {"points": n, "sha256": h.hexdigest()}


def expected_names(dataset, split):
    if split == "all":
        return expected_names(dataset, "train") + expected_names(dataset, "val")
    if dataset == "scannetv2":
        return (ROOT / "sbs/data/datasets" / f"scannetv2_{split}.txt").read_text().splitlines()
    manifest = json.loads((ROOT / "assets/s3dis/scene_fingerprints.json").read_text())
    return sorted(name for name in manifest if name.startswith("5_") == (split == "val"))


def prepared_names(dataset, root, split):
    root = Path(root)
    return [name for name in expected_names(dataset, split) if (root / (name + ".pt")).is_file()]


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_indices(indices, name, points, labels=None, classes=None):
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError(f"Invalid sparse index tensor: {name}")
    if not len(indices) or (indices < 0).any() or len(indices.unique()) != len(indices):
        raise ValueError(f"Empty, negative or duplicate sparse indices: {name}")
    if int(indices.max()) >= points:
        raise ValueError(f"Sparse indices out of bounds for {name}: points={points}")
    if labels is not None and classes is not None:
        selected = torch.as_tensor(labels).long()[indices]
        if (selected < 0).any() or (selected >= classes).any():
            raise ValueError(f"Sparse indices select ignored labels: {name}")


def validate_inputs(cfg, root, *, training=False, allow_partial=False,
                    anchor_file=None, verify_record=True):
    """Check complete splits, tensor content, point order and sparse supervision."""
    from sbs.config import load_statistics

    root = Path(root)
    splits = ("train", "val") if training else ("val",)
    counts, scenes = {}, {}
    for split in splits:
        expected = expected_names(cfg["dataset"], split)
        found = [n for n in expected if (root / (n + ".pt")).is_file()]
        missing = sorted(set(expected) - set(found))
        if not found or (missing and not allow_partial):
            raise FileNotFoundError(
                f"{cfg['dataset']} {split}: found {len(found)}/{len(expected)} scenes. "
                f"Missing: {missing[:5]}. Run scripts/prepare_data.py first."
            )
        if cfg["dataset"] == "s3dis":
            actual = {p.stem for p in root.glob("*.pt")
                      if len(p.stem) > 2 and p.stem[0] in "123456" and p.stem[1] == "_"
                      and (p.stem.startswith("5_") == (split == "val"))}
            if actual - set(expected):
                raise ValueError(f"Unexpected extra S3DIS scenes: {sorted(actual - set(expected))[:5]}")
        else:
            for name in found:
                if (root / split / name).exists() or (root / name).exists():
                    raise ValueError(f"Ambiguous prepared scene layout: {name}; use a flat .pt output directory")
        counts[split], scenes[split] = len(found), found
    stamp = root / "preparation.json"
    if verify_record and not allow_partial and not stamp.is_file():
        raise ValueError("Preparation record missing; run scripts/prepare_data.py --processed --dataset "
                         + cfg["dataset"] + " --output-root " + str(root))
    record = json.loads(stamp.read_text()) if verify_record and stamp.is_file() else {}
    if record and record.get("dataset") != cfg["dataset"]:
        raise ValueError("Preparation dataset does not match the configuration")
    anchors = None
    if training:
        path = Path(anchor_file) if anchor_file else root / cfg["data"]["anchors"]
        if not path.is_file():
            raise FileNotFoundError(f"Sparse indices missing: {path}; run scripts/prepare_data.py")
        anchors = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(anchors, dict):
            raise ValueError("Sparse annotation map must be a dict of index tensors")
        if anchor_file is None and record.get("anchor_sha256"):
            if file_digest(path) != record["anchor_sha256"]:
                raise ValueError("Sparse index file changed since preparation; restore it or rerun preparation explicitly")
    for split, names in scenes.items():
        for name in names:
            path = root / (name + ".pt")
            value = torch.load(path, map_location="cpu", weights_only=True)
            actual = fingerprint(cfg["dataset"], value)
            expected = record.get("scene_fingerprints", {}).get(name)
            if expected is not None and actual != expected:
                raise ValueError(f"Prepared point content/order changed for {name}; do not reuse stale sparse indices")
            if split == "train" and anchors is not None:
                if name not in anchors:
                    raise ValueError(f"Sparse indices missing for {name}; refusing dense fallback")
                validate_indices(anchors[name], name, actual["points"], value[-1], cfg["model"]["num_classes"])
    load_statistics(cfg, root)
    return counts
