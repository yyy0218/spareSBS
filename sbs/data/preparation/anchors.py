"""Per-class point sampling for the two indoor 0.1% experiments.
Generated indices must refer to the same processed point ordering as training.
"""

from pathlib import Path
import json
import numpy as np
import torch


def per_class_indices(labels, fraction, classes, rng, min_per_class=1):
    labels = np.asarray(labels).reshape(-1)
    selected = []
    for label in np.unique(labels):
        if not 0 <= label < classes:
            continue
        indices = np.flatnonzero(labels == label)
        size = min(len(indices), max(min_per_class, round(len(indices) * fraction)))
        selected.append(
            indices if size == len(indices) else rng.choice(indices, size=size, replace=False)
        )
    return np.sort(np.unique(np.concatenate(selected))) if selected else np.empty(0, dtype=np.int64)


def sample(cfg, root, output):
    root = Path(root)
    ds = cfg["dataset"]
    if ds not in {"s3dis", "scannetv2"}:
        raise ValueError("Only the two indoor experiments are included")
    rng = np.random.default_rng(cfg["seed"])
    result = {}
    if ds == "s3dis":
        paths = sorted(x for x in root.glob("*.pt") if x.stem[:2] in {"1_", "2_", "3_", "4_", "6_"})
    else:
        names = (
            (Path(__file__).parents[1] / "datasets/scannetv2_train.txt").read_text().splitlines()
        )
        paths = []
        for name in names:
            path = next(
                (
                    x
                    for x in [root / "train" / name, root / (name + ".pt"), root / name]
                    if x.exists()
                ),
                None,
            )
            if path is None:
                raise FileNotFoundError(f"Missing training scene: {name}")
            paths.append(path)
    for path in paths:
        if path.is_dir():
            labels = np.load(path / "segment20.npy").reshape(-1)
        else:
            item = torch.load(path, map_location="cpu", weights_only=False)
            labels = (
                item.get("segment20", item.get("segment", item.get("labels")))
                if isinstance(item, dict)
                else item[2 if ds == "s3dis" else 3]
            )
        indices = per_class_indices(labels, 0.001, cfg["model"]["num_classes"], rng)
        result[path.stem] = torch.as_tensor(indices, dtype=torch.long)
    if not result:
        raise ValueError("No training scenes found")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing annotation map: {output}")
    torch.save(result, output)
    metadata = {
        "dataset": ds,
        "seed": cfg["seed"],
        "fraction": 0.001,
        "scenes": len(result),
        "selected_indices": sum(len(x) for x in result.values()),
        "historical_map_equivalence_verified": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
