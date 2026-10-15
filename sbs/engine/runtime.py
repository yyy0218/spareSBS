"""Runtime utilities. Imports and CLI inspection never allocate a GPU."""

from contextlib import nullcontext
import os
import random
import subprocess
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def precision_context(device, precision="bf16"):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("Only bf16 and fp32 are supported")
    return torch.amp.autocast("cuda", dtype=torch.bfloat16)


def checked_device(request, allow_busy=False):
    device = torch.device(request)
    if device.type != "cuda":
        return device
    index = 0 if device.index is None else device.index
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[index] if visible and visible.strip() else str(index)
    try:
        uuid = subprocess.check_output(
            ["nvidia-smi", "--id=" + physical, "--query-gpu=uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        lines = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("GPU occupancy could not be checked; refusing to allocate CUDA") from exc
    busy = []
    for line in lines:
        parts = [v.strip() for v in line.split(",")]
        if len(parts) >= 2 and parts[0] == uuid and parts[1] != str(os.getpid()):
            busy.append(line)
    if busy and not allow_busy:
        raise RuntimeError(
            "Requested GPU is already in use: "
            + "; ".join(busy)
            + ". Choose another device, or explicitly pass --allow-busy-gpu."
        )
    print("GPU occupancy checked:", physical, "existing processes:", busy, flush=True)
    return device


def unpack_batch(batch, device):
    if len(batch) not in (5, 6):
        raise ValueError(f"Expected 5 or 6 collated fields, got {len(batch)}")
    xyz, features, indices, counts, labels = batch[:5]
    indices = [index.to(device, non_blocking=True).long() for index in indices[::-1]]
    if counts is not None:
        counts = counts.tolist()[::-1] if isinstance(counts, torch.Tensor) else list(counts)[::-1]
    return xyz.to(device), features.to(device), indices, counts, labels.to(device).long()
