#!/usr/bin/env python3
"""Prepare SparseSBS data, calibration and correctly aligned 0.1% annotations."""
from pathlib import Path
import sys
import argparse
import json
import shutil
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["s3dis", "scannetv2"], required=True)
    parser.add_argument("--raw-root", help="Raw aligned S3DIS directory or ScanNet root/scans directory")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--processed", action="store_true", help="Validate existing .pt scenes instead of converting raw data")
    parser.add_argument("--label-map", help="ScanNet TSV for meshes with aggregation/segment JSON instead of label PLY")
    parser.add_argument("--split", choices=["all", "val"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-scenes", type=int, default=0, help="Diagnostic subset; not a benchmark result")
    args = parser.parse_args()
    if not args.processed and not args.raw_root:
        parser.error("--raw-root is required unless --processed is used")
    if args.workers < 1 or args.limit_scenes < 0:
        parser.error("--workers must be positive and --limit-scenes nonnegative")
    import torch
    from sbs.config import load_config
    from sbs.data.preparation.common import (expected_names, fingerprint, atomic_save, validate_inputs,
                                             validate_indices, atomic_json, file_digest)
    from sbs.data.preparation.anchors import per_class_indices
    import numpy as np

    torch.set_num_threads(1)
    cfg = load_config(ROOT / f"configs/{args.dataset}/sparse_0p1.yaml")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    names = expected_names(args.dataset, args.split)
    if args.limit_scenes:
        # Include training and validation examples in a bounded diagnostic run.
        train = expected_names(args.dataset, "train")[:args.limit_scenes] if args.split == "all" else []
        val = expected_names(args.dataset, "val")[:args.limit_scenes]
        names = train + val
    if not args.processed:
        if args.dataset == "s3dis":
            from sbs.data.preparation.s3dis import prepare
            prepare(args.raw_root, root, args.overwrite, names=names, workers=args.workers)
        else:
            from sbs.data.preparation.scannetv2 import prepare
            prepare(args.raw_root, root, args.overwrite, names=names, workers=args.workers, label_map=args.label_map)
    assets = ROOT / "assets" / args.dataset
    reference = json.loads((assets / "scene_fingerprints.json").read_text())
    fingerprints = {}
    for i, name in enumerate(names, 1):
        path = root / (name + ".pt")
        fingerprints[name] = fingerprint(args.dataset, torch.load(path, map_location="cpu", weights_only=True))
        if i == 1 or i % 100 == 0 or i == len(names):
            print(f"Checked {i}/{len(names)} prepared scenes", flush=True)
    for i in (0, 1):
        src, dst = assets / f"dist{i}.pt", root / f"statistics/dist{i}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() != src.read_bytes() and not args.overwrite:
            raise FileExistsError(f"Different calibration already exists: {dst}; use a new output directory")
        shutil.copy2(src, dst)
    train_set = set(expected_names(args.dataset, "train"))
    train_names = [n for n in names if n in train_set]
    stamp = root / "preparation.json"
    previous = json.loads(stamp.read_text()) if stamp.is_file() else {}
    if previous and previous.get("dataset") != args.dataset:
        raise ValueError("Existing preparation belongs to another dataset; use a different output directory")
    path = root / cfg["data"]["anchors"]
    old_anchors = {}
    if path.exists() and not args.overwrite:
        if not previous:
            raise FileExistsError("Existing sparse indices have no preparation fingerprint. Use a new output directory.")
        if previous.get("anchor_sha256") and file_digest(path) != previous["anchor_sha256"]:
            raise ValueError("Sparse index file changed since preparation; use --overwrite only to regenerate it")
        old_anchors = torch.load(path, map_location="cpu", weights_only=True)
        for name in train_names:
            if name in old_anchors and previous.get("scene_fingerprints", {}).get(name) != fingerprints[name]:
                raise ValueError(f"Prepared points changed for {name}; use a new output directory")
    map_kind = previous.get("sparse_indices", "not_needed_for_evaluation")
    if train_names:
        published = torch.load(assets / "anchors.pt", map_location="cpu", weights_only=True)
        anchors = dict(old_anchors)
        all_fixed = True
        for name in train_names:
            matched = fingerprints[name] == reference.get(name)
            all_fixed = all_fixed and matched
            if name in old_anchors:
                indices = old_anchors[name]
            elif matched:
                indices = published[name]
            else:
                # Sampling for one scene is independent of preparation order.
                seed = int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{args.dataset}:{name}".encode()).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                item = torch.load(root / (name + ".pt"), map_location="cpu", weights_only=True)
                indices = torch.as_tensor(per_class_indices(item[-1], 0.001, cfg["model"]["num_classes"], rng), dtype=torch.int64)
            validate_indices(indices, name, fingerprints[name]["points"])
            anchors[name] = indices
        map_kind = "released_fixed_indices" if all_fixed else "deterministic_indices_for_this_point_order"
        if not all_fixed:
            print("Warning: point fingerprints differ from the released realization; indices are matched to the current points.", file=sys.stderr)
        if not old_anchors or set(anchors) != set(old_anchors) or args.overwrite:
            atomic_save(anchors, path)
    # A val-only pass must retain training provenance and its sparse-index record.
    merged = dict(previous.get("scene_fingerprints", {}))
    merged.update(fingerprints)
    counts = validate_inputs(cfg, root, training=bool(train_names), allow_partial=bool(args.limit_scenes), verify_record=False)
    metadata = {"dataset": args.dataset, "annotation_fraction": 0.001, "seed": cfg["seed"],
                "sparse_indices": map_kind, "scene_fingerprints": merged,
                "reference_point_content_match": all(merged[n] == reference.get(n) for n in merged),
                "partial": bool(args.limit_scenes), "counts": {**previous.get("counts", {}), **counts}}
    if path.is_file():
        metadata["anchor_sha256"] = file_digest(path)
    atomic_json(metadata, stamp)
    print(f"READY {args.dataset}: {counts}; sparse indices={map_kind}; partial={bool(args.limit_scenes)}")



if __name__ == "__main__":
    main()
