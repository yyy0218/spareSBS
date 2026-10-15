#!/usr/bin/env python3
"""Train SparseSBS (Support Before Supervision) from a main configuration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--data-root")
    parser.add_argument("--anchors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=["native", "reference"], default="native")
    parser.add_argument("--resume", help="An epoch-boundary checkpoint produced by this codebase")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop early and mark outputs as partial; skips full validation",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Explicitly permit other processes after the GPU occupancy check",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration only; no model/data/CUDA initialization",
    )
    args = parser.parse_args()
    from sbs.config import load_config

    cfg = load_config(args.config)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "configuration_only",
                    "name": cfg["name"],
                    "dataset": cfg["dataset"],
                    "epochs": cfg["training"]["epochs"],
                    "gpu_used": False,
                }
            )
        )
        return
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    from sbs.ops import configure_backend

    configure_backend(args.backend)
    if args.device == "cpu" and args.backend != "reference":
        parser.error("CPU training requires --backend reference")
    from sbs.engine.trainer import train
    from sbs.data.preparation.common import validate_inputs

    validate_inputs(cfg, args.data_root or cfg["data"]["root"], training=True,
                    allow_partial=bool(args.max_steps), anchor_file=args.anchors)

    if args.backend == "native":
        # Preload in the parent before DataLoader forks: worker-side JIT discovery
        # must not try to initialize CUDA in an already forked process.
        from sbs.engine.runtime import checked_device
        from sbs.ops.native import _extension

        checked_device(args.device, args.allow_busy_gpu)
        _extension()

    train(
        cfg,
        args.output or str(Path("runs") / cfg["name"]),
        args.device,
        args.data_root,
        args.anchors,
        args.resume,
        args.allow_busy_gpu,
        args.max_steps,
    )


if __name__ == "__main__":
    main()
