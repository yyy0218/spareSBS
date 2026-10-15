#!/usr/bin/env python3
"""Check SparseSBS runtime dependencies and optional native CUDA operators."""
from pathlib import Path
import argparse
import importlib.metadata
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Build native operators and run CUDA forward/backward")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch
    from sbs.config import load_config
    for ds in ("s3dis", "scannetv2"):
        load_config(ROOT / f"configs/{ds}/sparse_0p1.yaml")
    print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
    for pkg in ("numpy", "scipy", "PyYAML", "plyfile", "ninja", "tqdm"):
        print(pkg, importlib.metadata.version(pkg))
    if args.native:
        from torch.utils.cpp_extension import CUDA_HOME
        from sbs.engine.runtime import checked_device
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Native benchmark kernels require Linux x86-64")
        if not CUDA_HOME or not (Path(CUDA_HOME) / "bin/nvcc").is_file():
            raise RuntimeError("Install the CUDA 12.8 toolkit and export CUDA_HOME=/usr/local/cuda-12.8")
        if not shutil.which("g++") or not shutil.which("ninja"):
            raise RuntimeError("Install g++ and ninja and expose them on PATH")
        device = checked_device(args.device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable: check NVIDIA driver and CUDA PyTorch build")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("The released recipes require a GPU supporting BF16")
        from sbs.ops import configure_backend, directional_response
        configure_backend("native")
        for mode, pdim, wdim in [("spatial", 4, 12), ("rgb", 8, 16), ("normal", 12, 19)]:
            points = torch.rand(1, 32, pdim, device=device)
            nbr = torch.arange(32, device=device, dtype=torch.int32).reshape(1, 32, 1).repeat(1, 1, 4)
            weights = torch.rand(wdim, 8, device=device, requires_grad=True)
            out = directional_response(points, nbr, weights, mode)
            out.square().mean().backward()
            if not torch.isfinite(out).all() or weights.grad is None or not torch.isfinite(weights.grad).all():
                raise RuntimeError(f"Native {mode} forward/backward failed")
        print("Native CUDA forward/backward: PASS")
    print("SparseSBS checks passed")


if __name__ == "__main__":
    main()
