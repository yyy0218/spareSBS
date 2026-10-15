#!/usr/bin/env python3
"""Evaluate SparseSBS at full resolution using a new-format checkpoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--data-root",
        help="Benchmark root containing prepared scenes and statistics/",
    )
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=["native", "reference"], default="native")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.max_scenes < 0:
        parser.error("--max-scenes must be nonnegative")
    from sbs.config import load_config, load_statistics

    cfg = load_config(args.config)
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "configuration_only",
                    "name": cfg["name"],
                    "evaluation": cfg["evaluation"],
                    "gpu_used": False,
                }
            )
        )
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for evaluation")
    from sbs.ops import configure_backend
    from sbs.engine.runtime import checked_device, seed_everything
    from sbs.models import SparseSBS
    from sbs.engine.checkpoints import load_checkpoint
    from sbs.data.factory import build_dataset

    configure_backend(args.backend)
    if args.device == "cpu" and args.backend != "reference":
        parser.error("CPU evaluation requires --backend reference")
    seed_everything(cfg["seed"])
    device = checked_device(args.device, args.allow_busy_gpu)
    root = Path(args.data_root or cfg["data"]["root"])
    from sbs.data.preparation.common import validate_inputs

    validate_inputs(cfg, root, allow_partial=bool(args.max_scenes))
    model = SparseSBS(cfg["model"], load_statistics(cfg, root)).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, expected_config=cfg)
    if checkpoint.get("partial_run"):
        print("Warning: evaluating a partial-training checkpoint", file=sys.stderr)
    from sbs.evaluation.full_resolution import evaluate

    dataset = build_dataset(cfg, "val", str(root), evaluation=True)
    result = evaluate(model, dataset, cfg, device, args.max_scenes)
    result.update(
        config_name=cfg["name"],
        checkpoint_epoch=checkpoint["epoch"],
        backend=args.backend,
        partial_evaluation=bool(args.max_scenes),
        evaluation_settings=cfg["evaluation"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
