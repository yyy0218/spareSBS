#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Generate SparseSBS DE distance statistics on CPU; does not train a model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--batches", type=int, default=600)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    from sbs.config import load_config
    from sbs.data.calibration import collect

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg["data"]["root"])
    print(
        json.dumps(
            collect(cfg, root, args.output or root / "statistics", args.batches, args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
