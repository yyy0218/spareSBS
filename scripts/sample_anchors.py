#!/usr/bin/env python3
"""Generate new SparseSBS 0.1% annotation maps; not needed for bundled indices."""

from pathlib import Path
import sys
import argparse
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    from sbs.config import load_config
    from sbs.data.preparation.anchors import sample

    cfg = load_config(args.config)
    root = Path(args.data_root or cfg["data"]["root"])
    print(json.dumps(sample(cfg, root, args.output or root / cfg["data"]["anchors"]), indent=2))


if __name__ == "__main__":
    main()
