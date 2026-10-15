"""CPU-only generation of new DE initialization statistics.

Records the actual hierarchy and seed. This collector uses stage-1
coordinates for stage-1 neighbors, and does not claim bitwise equivalence to
the separately archived calibration arrays. It does not train a network.
"""

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from .factory import build_dataset, collate_function
from sbs.engine.runtime import seed_everything, seed_worker
from sbs.ops import configure_backend


def collect(cfg, root, output, batches=600, workers=0):
    output = Path(output)
    if any((output / name).exists() for name in ("dist0.pt", "dist1.pt", "calibration.json")):
        raise FileExistsError(f"Calibration output already exists: {output}; choose a new --output directory")
    if workers < 0:
        raise ValueError("workers must be nonnegative")
    configure_backend("reference")
    seed_everything(cfg["seed"])
    if batches < 2:
        raise ValueError("At least two batches are required")
    dataset = build_dataset(cfg, "train", str(root), dense=True)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_function(cfg["dataset"]),
        generator=generator,
        worker_init_fn=seed_worker,
    )
    if not len(loader):
        raise ValueError("Calibration loader is empty")
    samples = [[], []]
    done = 0
    intervals = {
        "s3dis": [523, 131],
        "scannetv2": [1307, 523],
    }[cfg["dataset"]]
    while done < batches:
        for batch in loader:
            xyz, _, indices, *_ = batch
            # Adapters scale xyz by 40 indoors (S3DIS) or 60 otherwise.
            xyz = xyz / (40 if cfg["dataset"] == "s3dis" else 60)
            levels = [xyz, xyz[indices[1].long()]]
            for depth in (0, 1):
                knn = indices[2 * depth].long()
                coords = levels[depth]
                if len(coords) != len(knn):
                    raise ValueError("Calibration hierarchy/coordinate size mismatch")
                if knn.shape[1] <= 1:
                    continue
                j = torch.randint(knn.shape[1] - 1, (len(knn), 1))
                neighbor = torch.gather(knn[:, 1:], 1, j).flatten()
                relative = (coords[neighbor] - coords)[:: intervals[depth]] * cfg["model"][
                    "coordinate_scales"
                ][depth]
                samples[depth].append(relative.norm(dim=1))
            done += 1
            if done >= batches:
                break
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = []
    arrays = []
    for depth in (0, 1):
        values = torch.cat(samples[depth][1:]).sort().values.cpu()
        counts.append(len(values))
        widths = cfg["model"]["widths"]
        required = (widths[0] if depth == 0 else max(widths[1:] or widths)) + 2
        if len(values) < required:
            raise ValueError("Too few calibration samples; increase --batches")
        arrays.append(values)
    from .preparation.common import atomic_save
    for depth, values in enumerate(arrays):
        atomic_save(values, output / f"dist{depth}.pt")
    metadata = {
        "collector": "sbs_hierarchy_v1",
        "seed": cfg["seed"],
        "batches": done,
        "sample_counts": counts,
        "backend": "reference",
        "archived_array_equivalence_verified": False,
        "coordinate_scales": cfg["model"]["coordinate_scales"],
    }
    (output / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
