"""Prepare the licensed S3DIS aligned release without modifying raw files."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io
import numpy as np
import torch
from .common import atomic_save

CLASSES = ("ceiling", "floor", "wall", "beam", "column", "window", "door",
           "table", "chair", "sofa", "bookcase", "board", "clutter")
LABEL_IDS = {name: i for i, name in enumerate(CLASSES)}
LABEL_IDS["stairs"] = 12


def load_points(path):
    raw = Path(path).read_bytes().replace(b"\x00", b" ")
    if not raw.strip():
        return np.empty((0, 6), dtype=np.float32)
    # Known corruption in some official copies; preserve all rows and raw files.
    raw = raw.replace(b"103.0\xef\xbf\xbd0000", b"103.000000")
    try:
        points = np.loadtxt(io.BytesIO(raw), dtype=np.float32, usecols=range(6), ndmin=2)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot parse xyzrgb annotation {path}: {exc}") from exc
    if not np.isfinite(points).all():
        raise ValueError(f"Nonfinite xyzrgb values: {path}")
    return points


def prepare(raw_root, processed_root, overwrite=False, names=None, workers=4):
    base, out = Path(raw_root), Path(processed_root)
    if not base.is_dir():
        raise FileNotFoundError(f"S3DIS raw directory not found: {base}")
    jobs = []
    wanted = None if names is None else set(names)
    for area in range(1, 7):
        for room in sorted((base / f"Area_{area}").glob("*")):
            name = f"{area}_{room.name}"
            if room.is_dir() and (wanted is None or name in wanted):
                jobs.append((room, name))
    if not jobs:
        raise FileNotFoundError("Expected Area_1/.../Annotations/*.txt under --raw-root")
    found = {name for _, name in jobs}
    if wanted is not None and wanted - found:
        raise FileNotFoundError(f"Missing S3DIS rooms: {sorted(wanted - found)[:5]}")

    def convert(job):
        room, name = job
        target = out / (name + ".pt")
        if target.exists() and not overwrite:
            return name
        xyz, rgb, labels = [], [], []
        for annotation in sorted((room / "Annotations").glob("*.txt")):
            if annotation.name.startswith("."):
                continue
            category = annotation.stem.split("_", 1)[0].lower()
            if category not in LABEL_IDS:
                raise ValueError(f"Unknown S3DIS class: {annotation}")
            arr = load_points(annotation)
            if not len(arr):
                continue
            xyz.append(torch.from_numpy(arr[:, :3].copy()))
            rgb.append(torch.from_numpy(arr[:, 3:6].copy()).to(torch.uint8))
            labels.append(torch.full((len(arr),), LABEL_IDS[category], dtype=torch.uint8))
        if not xyz:
            raise ValueError(f"No annotation points: {room}")
        x = torch.cat(xyz)
        x = x - x.min(dim=0)[0]
        atomic_save((x, torch.cat(rgb), torch.cat(labels)), target)
        return name

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, name in enumerate(pool.map(convert, jobs), 1):
            if i == 1 or i % 20 == 0 or i == len(jobs):
                print(f"S3DIS: {i}/{len(jobs)} rooms ({name})", flush=True)
    return len(jobs)
