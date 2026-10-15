"""Construct the two indoor datasets from explicit, flattened public configurations."""

from copy import deepcopy
from pathlib import Path


def build_dataset(
    cfg,
    split="train",
    data_root=None,
    anchor_file=None,
    warmup=False,
    evaluation=False,
    dense=False,
):
    ds = cfg["dataset"]
    options = deepcopy(cfg["data"]["options"])
    root = Path(data_root or cfg["data"]["root"])
    options["data_root"] = str(root)
    options["split"] = split
    if split == "val" and not evaluation and "validation_loop" in cfg["training"]:
        options["loop"] = cfg["training"]["validation_loop"]
    if (split == "train" and (not warmup)) and (not dense):
        options["anchor_file"] = str(anchor_file or root / cfg["data"]["anchors"])
        if not Path(options["anchor_file"]).is_file():
            raise FileNotFoundError(f"Sparse anchor map missing: {options['anchor_file']}")
    else:
        options["anchor_file"] = None
    if ds == "s3dis":
        from .datasets.s3dis import S3DISDataset

        options["area"] = "!5" if split == "train" else "5"
        if evaluation:
            options.update(split="test", loop=cfg["evaluation"]["votes"])
        cls = S3DISDataset
    elif ds == "scannetv2":
        from .datasets.scannetv2 import ScanNetDataset

        cls = ScanNetDataset
    else:
        raise ValueError(f"Unsupported dataset: {ds}")
    if ds != "s3dis":
        options.update(train=split == "train", test=evaluation)
        if evaluation:
            options.update(
                loop=cfg["evaluation"]["votes"],
                test_rotations=[v / 180 for v in cfg["evaluation"]["rotations"]],
                test_scales=cfg["evaluation"]["scales"],
            )
    if warmup:
        options.pop("anchor_retention", None)
        options.update(
            warmup=True,
            loop=cfg["training"]["batch_size"],
            grid_sizes=cfg["training"]["bn_warmup_grid"],
        )
    return cls(**options)


def collate_function(dataset):
    if dataset == "s3dis":
        from .datasets.s3dis import s3dis_collate_fn

        return s3dis_collate_fn
    if dataset == "scannetv2":
        from .datasets.scannetv2 import scan_collate_fn

        return scan_collate_fn
    raise ValueError(f"Unsupported dataset: {dataset}")
