"""Pure-data YAML configuration. Loading never imports a model or reads tensors."""

from pathlib import Path
from copy import deepcopy
import yaml

DATASETS = {"s3dis", "scannetv2"}


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or cfg.get("dataset") not in DATASETS:
        raise ValueError("A supported dataset is required")
    required = {
        "name",
        "dataset",
        "seed",
        "ignore_index",
        "data",
        "model",
        "training",
        "arl",
        "evaluation",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    model = cfg["model"]
    if (
        model.get("representation", "directional") != "directional"
        or cfg["arl"].get("enabled", True) is not True
    ):
        raise ValueError("This release contains the complete DE + ARL method only")
    if model["input_mode"] not in {"rgb", "normal"}:
        raise ValueError("This release supports indoor RGB and RGB/normal input modes")
    stages = len(model["depths"])
    if any(
        len(model[k]) != stages
        for k in ["widths", "neighbors", "coordinate_scales", "drop_paths", "head_drops"]
    ):
        raise ValueError("Inconsistent hierarchy lengths")
    if any(w % 4 for w in model["widths"]):
        raise ValueError("Model widths must be divisible by 4")
    for k in ["local", "global_confirmation"]:
        if cfg["arl"][k]["start_epoch"] < 0:
            raise ValueError("ARL activation epoch must be non-negative")
    ramp = cfg["arl"]["local"].get("ramp_epochs", 0)
    if isinstance(ramp, bool) or not isinstance(ramp, int) or ramp < 0:
        raise ValueError("ARL local ramp_epochs must be a non-negative integer")
    for key in ("validation_batch_size", "validation_loop"):
        if key in cfg["training"]:
            value = cfg["training"][key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"training.{key} must be a positive integer")
    if cfg["arl"]["prototype_scope"] != "collated_batch":
        raise ValueError("Only the audited collated-batch prototype scope is implemented")
    if cfg["evaluation"]["votes"] != len(cfg["evaluation"]["rotations"]) * len(
        cfg["evaluation"]["scales"]
    ):
        raise ValueError("Votes must match explicit rotations and scales")
    return deepcopy(cfg)


def load_statistics(cfg, data_root=None):
    import torch

    root = Path(data_root or cfg["data"]["root"])
    result = []
    for rel in cfg["data"]["statistics"]:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"DE calibration missing: {path}. Run scripts/prepare_data.py first to install the bundled calibration."
            )
        samples = torch.load(path, map_location="cpu", weights_only=True)
        depth = len(result)
        widths = cfg["model"]["widths"]
        required = (widths[0] if depth == 0 else max(widths[1:] or widths)) + 2
        if (not isinstance(samples, torch.Tensor) or samples.ndim != 1
                or not samples.is_floating_point() or len(samples) < required
                or not torch.isfinite(samples).all() or (samples < 0).any()
                or (samples[1:] < samples[:-1]).any()):
            raise ValueError(f"Invalid distance samples: {path}; need at least {required} finite, nonnegative sorted values")
        result.append(samples)
    return result
