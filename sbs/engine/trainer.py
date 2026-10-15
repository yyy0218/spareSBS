"""From-scratch training with DE, sparse CE and training-only ARL.

No experiment dispatcher, teacher, memory bank or inherited config module.
"""

from pathlib import Path
import json
import os
import random
from copy import deepcopy
import warnings
import numpy as np
import torch
import torch.distributed as distributed
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from sbs.config import load_statistics
from sbs.data.factory import build_dataset, collate_function
from sbs.models import SparseSBS
from sbs.supervision import AnchorRelayLearning
from sbs.evaluation.metrics import update_confusion, summarize
from .runtime import seed_everything, seed_worker, precision_context, unpack_batch, checked_device
from .optimization import build_optimizer, learning_rate
from .checkpoints import load_checkpoint, save_checkpoint


def _loader(dataset, cfg, shuffle, generator, distributed_run=False):
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, seed=cfg["seed"]) if distributed_run else None
    )
    return DataLoader(
        dataset,
        batch_size=(
            cfg["training"]["batch_size"]
            if shuffle
            else cfg["training"].get("validation_batch_size", cfg["training"]["batch_size"])
        ),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=cfg["training"]["workers"],
        pin_memory=True,
        collate_fn=collate_function(cfg["dataset"]),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=shuffle,
    )


@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    classes = cfg["model"]["num_classes"]
    hist = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    for batch in loader:
        xyz, features, indices, counts, labels = unpack_batch(batch, device)
        with precision_context(device, cfg["training"]["precision"]):
            output = model(xyz, features, indices, counts)
        update_confusion(hist, output["logits"].argmax(-1), labels, cfg["ignore_index"])
    if distributed.is_initialized():
        distributed.all_reduce(hist)
    return summarize(hist.cpu())


def _check_run_directory(output_dir, resume=None):
    """Never silently append a fresh run to an existing training history."""
    root = Path(output_dir)
    artifacts = [root / "metrics.jsonl", root / "resolved_config.json", root / "best.pt"]
    artifacts.extend(root.glob("epoch*.pt"))
    if resume is None and any(path.exists() for path in artifacts):
        raise FileExistsError(f"Training output already exists: {root}; use a new --output or --resume")
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("format") != "sbs.v2" or "optimizer_state" not in saved:
            raise ValueError("--resume needs a training checkpoint with optimizer state; the released pretrained files are evaluation-only")
        if saved.get("partial_run"):
            raise ValueError("A bounded partial-run checkpoint is not an epoch-boundary resume checkpoint")
        history = root / "metrics.jsonl"
        if history.is_file():
            rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            if rows and rows[-1].get("epoch") != saved["epoch"]:
                raise ValueError("Resume checkpoint does not match the last recorded epoch; use a new --output")
        return saved
    return None


def _input_signature(cfg, data_root, anchor_file):
    from sbs.data.preparation.common import file_digest
    root = Path(data_root or cfg["data"]["root"])
    anchors = Path(anchor_file) if anchor_file else root / cfg["data"]["anchors"]
    files = {"anchors": anchors}
    files.update({name: root / name for name in cfg["data"]["statistics"]})
    return {name: file_digest(path) for name, path in files.items()}


def _validate_resume_recipe(saved, cfg, signature):
    old, current = deepcopy(saved["config"]), deepcopy(cfg)
    # Moving an identical dataset is allowed; changing its recipe is not.
    old.get("data", {}).pop("root", None)
    current.get("data", {}).pop("root", None)
    if old != current:
        raise ValueError("Resume configuration differs from the saved training recipe")
    if saved.get("input_signature") is not None and saved["input_signature"] != signature:
        raise ValueError("Resume annotation/calibration inputs changed")
    if saved["epoch"] >= cfg["training"]["epochs"]:
        raise ValueError("This checkpoint has already completed the configured training")


def train(
    cfg,
    output_dir,
    device_request="cuda:0",
    data_root=None,
    anchor_file=None,
    resume=None,
    allow_busy=False,
    max_steps=None,
):
    saved_header = _check_run_directory(output_dir, resume)
    signature = _input_signature(cfg, data_root, anchor_file)
    if saved_header is not None:
        _validate_resume_recipe(saved_header, cfg, signature)
        if max_steps is not None and max_steps <= saved_header["step"]:
            raise ValueError("--max-steps must exceed the resume checkpoint's global step")
        del saved_header
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        device_request = f"cuda:{local_rank}"
    device = checked_device(device_request, allow_busy)
    if world > 1:
        torch.cuda.set_device(device)
        distributed.init_process_group("nccl")
    rank = distributed.get_rank() if distributed.is_initialized() else 0
    seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(cfg["seed"])
    training = cfg["training"]
    root = Path(output_dir)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    network = SparseSBS(cfg["model"], load_statistics(cfg, data_root)).to(device)
    dataset = build_dataset(cfg, "train", data_root, anchor_file)
    loader = _loader(dataset, cfg, True, generator, world > 1)
    if not len(loader):
        raise ValueError("Training loader has no complete batches")
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "training_ready",
                    "name": cfg["name"],
                    "representation": cfg["model"].get("representation", "directional"),
                    "arl_enabled": cfg["arl"].get("enabled", True),
                    "seed": cfg["seed"],
                    "epochs": training["epochs"],
                    "training_batches_per_epoch": len(loader),
                    "training_samples": len(dataset),
                    "parameters": sum(p.numel() for p in network.parameters()),
                }
            ),
            flush=True,
        )
    validation = None
    if training["validation_interval"]:
        valset = build_dataset(cfg, "val", data_root)
        validation = _loader(valset, cfg, False, generator, world > 1)
    if training["bn_warmup_grid"] is not None and resume is None:
        warmset = build_dataset(cfg, "train", data_root, warmup=True)
        warmloader = DataLoader(
            warmset,
            batch_size=len(warmset),
            num_workers=min(training["workers"], 6),
            collate_fn=collate_function(cfg["dataset"]),
            generator=generator,
            worker_init_fn=seed_worker,
        )
        network.train()
        for batch in warmloader:
            xyz, features, indices, counts, _ = unpack_batch(batch, device)
            with precision_context(device, training["precision"]):
                _, coordinate, _ = network.encode(xyz, features, indices, counts)
            coordinate.backward()
        network.zero_grad(set_to_none=True)
    if world > 1:
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
        network = DistributedDataParallel(
            network, device_ids=[local_rank], find_unused_parameters=True
        )
    optimizer = build_optimizer(network, training)
    start = step = 0
    best = -1.0
    if resume:
        saved = load_checkpoint(resume, network, optimizer, expected_config=cfg)
        if saved.get("partial_run"):
            raise ValueError(
                "A bounded partial-run checkpoint is not an epoch-boundary resume checkpoint"
            )
        if saved["config"]["model"] != cfg["model"]:
            raise ValueError("Resume model configuration mismatch")
        start = saved["epoch"]
        step = saved["step"]
        best = saved.get("best_miou", -1.0) if (root / "best.pt").is_file() else -1.0
        if device.type == "cuda":
            if "cuda_rng_state" in saved:
                torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
            else:
                warnings.warn("Legacy checkpoint has no CUDA RNG state; resume is not bitwise replay")
        if "loader_rng" in saved:
            generator.set_state(saved["loader_rng"])
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
        if "python_rng" in saved:
            random.setstate(saved["python_rng"])
        if "numpy_rng" in saved:
            name, keys, position, gaussian, cached = saved["numpy_rng"]
            np.random.set_state((name, keys.numpy().astype(np.uint32), position, gaussian, cached))
    if resume and cfg["dataset"] == "scannetv2":
        from sbs.data.datasets.scannetv2 import change_mix_prob
        previous_probability = (1 - max(0, step - 1) / (training["epochs"] * len(loader))) * 0.5
        change_mix_prob(saved.get("data_mix_probability", previous_probability))
    relay = AnchorRelayLearning(cfg["arl"], cfg["model"]["num_classes"], cfg["ignore_index"])
    weights = training["class_weights"]
    weights = None if weights is None else torch.tensor(weights, device=device)
    horizon = training["epochs"] * len(loader)
    warmup = training["warmup_epochs"] * len(loader)
    try:
        for epoch in range(start, training["epochs"]):
            if world > 1:
                loader.sampler.set_epoch(epoch)
            network.train()
            totals = {
                "loss": 0.0,
                "batches": 0,
                "local_points": 0,
                "global_points": 0,
                "skipped_empty": 0,
            }
            for batch in loader:
                if max_steps is not None and step >= max_steps:
                    break
                progress = step / horizon
                if cfg["dataset"] == "scannetv2":
                    from sbs.data.datasets.scannetv2 import change_mix_prob

                    change_mix_prob((1 - progress) * 0.5)
                lr = learning_rate(step, horizon, warmup, training["learning_rate"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                step += 1
                xyz, features, indices, counts, labels = unpack_batch(batch, device)
                labels = labels.clone()
                labels[(labels < 0) | (labels >= cfg["model"]["num_classes"])] = cfg["ignore_index"]
                valid = bool((labels != cfg["ignore_index"]).any())
                if world > 1:
                    gate = torch.tensor(int(valid), device=device)
                    distributed.all_reduce(gate, op=distributed.ReduceOp.MIN)
                    valid = bool(gate)
                if not valid:
                    totals["skipped_empty"] += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, training["precision"]):
                    output = network(xyz, features, indices, counts, return_support=True)
                    main = F.cross_entropy(
                        output["logits"],
                        labels,
                        weight=weights,
                        ignore_index=cfg["ignore_index"],
                        label_smoothing=training["label_smoothing"],
                    )
                    extra, metrics = relay(output, labels, epoch)
                    loss = (
                        main
                        + extra
                        + (3e-3**progress)
                        * training["coordinate_loss_scale"]
                        * output["coordinate_loss"]
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite training loss at epoch={epoch} step={step}"
                    )
                loss.backward()
                optimizer.step()
                totals["loss"] += float(loss.detach())
                totals["batches"] += 1
                totals["local_points"] += metrics["local_points"]
                totals["global_points"] += metrics["global_points"]
                log_interval = training.get("log_every_steps", 0)
                if rank == 0 and log_interval and step % log_interval == 0:
                    print(
                        json.dumps(
                            {
                                "event": "train_progress",
                                "epoch_index": epoch,
                                "epoch": epoch + 1,
                                "step": step,
                                "loss": float(loss.detach()),
                                "learning_rate": lr,
                                "local_points": metrics["local_points"],
                                "global_points": metrics["global_points"],
                            }
                        ),
                        flush=True,
                    )
            is_partial = max_steps is not None and step >= max_steps
            stats = {
                "epoch": epoch + 1,
                "epoch_index": epoch,
                "step": step,
                **totals,
                "local_loss_weight": relay.local_weight(epoch),
                "partial_run": is_partial,
            }
            if (
                validation is not None
                and not is_partial
                and (epoch + 1) % training["validation_interval"] == 0
            ):
                stats["validation"] = validate(network, validation, cfg, device)
            if rank == 0:
                current = stats.get("validation", {}).get("miou", -1.0)
                improved = current > best
                if improved:
                    best = current
                state = {
                    "best_miou": best,
                    "loader_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": (
                        np.random.get_state()[0],
                        torch.from_numpy(np.random.get_state()[1].astype(np.int64)),
                        *np.random.get_state()[2:],
                    ),
                    "partial_run": is_partial,
                    "input_signature": signature,
                }
                if device.type == "cuda":
                    state["cuda_rng_state"] = torch.cuda.get_rng_state(device)
                if cfg["dataset"] == "scannetv2":
                    from sbs.data.datasets.scannetv2 import get_mix_prob
                    state["data_mix_probability"] = get_mix_prob()
                if improved:
                    save_checkpoint(
                        root / "best.pt", network, cfg, epoch + 1, step, optimizer, **state
                    )
                if (
                    (epoch + 1) % training["save_every"] == 0
                    or epoch + 1 == training["epochs"]
                    or is_partial
                ):
                    save_checkpoint(
                        root / f"epoch{epoch + 1}.pt",
                        network,
                        cfg,
                        epoch + 1,
                        step,
                        optimizer,
                        **state,
                    )
                with (root / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(stats) + "\n")
                print(json.dumps(stats), flush=True)
            if is_partial:
                break
    finally:
        if world > 1:
            distributed.destroy_process_group()
