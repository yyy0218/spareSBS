#!/usr/bin/env python3
"""Run the two SparseSBS 0.1% experiments sequentially on one checked GPU.

Each model is trained from scratch for 100 epochs and then evaluated on the
complete validation split. A failed stage stops the queue; nothing is published.
"""

from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENTS = ("s3dis", "scannetv2")
EXPECTED_SCENES = {"s3dis": 68, "scannetv2": 312}
REFERENCE_MIOU = {"s3dis": 0.6989614963531494, "scannetv2": 0.7299975752830505}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_manifest():
    return {
        str(f.relative_to(ROOT)): digest(f)
        for group in ("sbs", "scripts", "configs")
        for f in sorted((ROOT / group).rglob("*"))
        if f.is_file() and f.suffix in {".py", ".yaml", ".txt", ".cpp", ".cu", ".h", ".hpp"}
    }


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_result(value, dataset):
    if value.get("num_scenes") != EXPECTED_SCENES[dataset]:
        raise ValueError("Evaluation did not cover the full split")
    if value.get("partial_evaluation") or value.get("votes") != 1:
        raise ValueError("Expected complete one-vote evaluation")
    miou = value.get("miou")
    if not isinstance(miou, (int, float)) or not math.isfinite(miou) or not 0 <= miou <= 1:
        raise ValueError("Invalid mIoU")
    return miou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3dis-root", required=True)
    parser.add_argument("--scannet-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from sbs.config import load_config
    from sbs.engine.runtime import checked_device

    if not args.device.startswith("cuda"):
        parser.error("This formal two-experiment runner requires one CUDA device")
    checked_device(args.device)
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "queue_status.json").exists():
        parser.error("An existing queue state is present; use a new output directory")
    roots = {
        "s3dis": Path(args.s3dis_root).resolve(),
        "scannetv2": Path(args.scannet_root).resolve(),
    }
    source = source_manifest()
    assets = {}
    configs = {}
    for ds in EXPERIMENTS:
        cfg = load_config(ROOT / f"configs/{ds}/sparse_0p1.yaml")
        configs[ds] = cfg
        if cfg["training"]["epochs"] != 100:
            raise ValueError("Formal recipes must run 100 epochs")
        required = [
            roots[ds] / cfg["data"]["anchors"],
            *(roots[ds] / r for r in cfg["data"]["statistics"]),
        ]
        for f in required:
            if not f.is_file():
                raise FileNotFoundError(f)
        assets[ds] = {str(f): digest(f) for f in required}
        if (out / ds).exists():
            raise FileExistsError(out / ds)
        from sbs.data.preparation.common import validate_inputs
        validate_inputs(cfg, roots[ds], training=True)
    # Lock this output and GPU selection for cooperating launchers.
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "all")
    lockid = hashlib.sha256((visible + args.device).encode()).hexdigest()[:16]
    lock = open(Path(tempfile.gettempdir()) / ("sparsesbs_indoor_" + lockid + ".lock"), "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {
        "created": now(),
        "queue_pid": os.getpid(),
        "cuda_visible_devices": visible,
        "device": args.device,
        "serial": True,
        "order": list(EXPERIMENTS),
        "phase": "ready",
        "runs": {},
        "source_sha256": source,
        "input_sha256": assets,
    }
    atomic_json(out / "source_manifest.json", source)
    child = None

    def save():
        state["updated"] = now()
        atomic_json(out / "queue_status.json", state)

    def interrupted(signum, frame):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        state.update(phase="interrupted", signal=signum)
        save()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def immutable():
        if source_manifest() != source:
            raise RuntimeError("Source changed during the reproduction")
        for entries in assets.values():
            for path, value in entries.items():
                if digest(path) != value:
                    raise RuntimeError("Input asset changed: " + path)

    def execute(command, log_path, phase, dataset):
        nonlocal child
        immutable()
        checked_device(args.device)
        state.update(phase=phase, current=dataset)
        with Path(log_path).open("w") as log:
            log.write("START " + now() + "\n")
            log.flush()
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state["child_pid"] = child.pid
            state["runs"][dataset].update(phase=phase, child_pid=child.pid)
            save()
            print(
                json.dumps({"event": phase, "dataset": dataset, "pid": child.pid, "time": now()}),
                flush=True,
            )
            rc = child.wait()
        state["runs"][dataset][phase + "_exit_code"] = rc
        if rc != 0:
            raise RuntimeError(f"{dataset} {phase} failed with exit code {rc}")
        child = None

    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", SBS_BACKEND="native", PYTHONUNBUFFERED="1")
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        env.pop(key, None)
    try:
        save()
        for ds in EXPERIMENTS:
            run = out / ds
            run.mkdir()
            cfg_path = ROOT / f"configs/{ds}/sparse_0p1.yaml"
            state["runs"][ds] = {
                "started": now(),
                "configuration": str(cfg_path),
                "training_seed": configs[ds]["seed"],
            }
            execute(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(cfg_path),
                    "--data-root",
                    str(roots[ds]),
                    "--device",
                    args.device,
                    "--backend",
                    "native",
                    "--output",
                    str(run),
                ],
                run / "train.log",
                "training",
                ds,
            )
            import torch

            best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            final = torch.load(run / "epoch100.pt", map_location="cpu", weights_only=True)
            if final.get("epoch") != 100 or final.get("partial_run") or best.get("partial_run"):
                raise RuntimeError(
                    "Expected a completed 100-epoch run and non-partial best checkpoint"
                )
            for payload in [best, final]:
                if payload.get("format") != "sbs.v2":
                    raise RuntimeError("Wrong checkpoint format")
                if not all(
                    not t.is_floating_point() or bool(torch.isfinite(t).all())
                    for t in payload["model_state"].values()
                ):
                    raise RuntimeError("Nonfinite checkpoint tensor")
            state["runs"][ds]["selected_epoch"] = best["epoch"]
            state["runs"][ds]["best_checkpoint_sha256"] = digest(run / "best.pt")
            del best, final
            execute(
                [
                    sys.executable,
                    "scripts/evaluate.py",
                    "--config",
                    str(cfg_path),
                    "--checkpoint",
                    str(run / "best.pt"),
                    "--data-root",
                    str(roots[ds]),
                    "--device",
                    args.device,
                    "--backend",
                    "native",
                    "--output",
                    str(run / "full_resolution.json"),
                ],
                run / "evaluate.log",
                "evaluation",
                ds,
            )
            result = json.loads((run / "full_resolution.json").read_text())
            miou = validate_result(result, ds)
            state["runs"][ds].update(
                phase="completed",
                completed=now(),
                miou=miou,
                num_scenes=result["num_scenes"],
                reference_miou=REFERENCE_MIOU[ds],
                delta_percentage_points=100 * (miou - REFERENCE_MIOU[ds]),
            )
            immutable()
            save()
        state.update(phase="completed_review_required", current=None, child_pid=None)
        save()
        atomic_json(
            out / "summary.json",
            {
                "runs": state["runs"],
                "source_sha256": source,
                "status": "completed_review_required",
                "publication_performed": False,
            },
        )
    except Exception as exc:
        state.update(phase="failed", error=str(exc))
        save()
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
