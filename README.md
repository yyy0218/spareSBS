# SparseSBS: Support Before Supervision

Preliminary code and pretrained checkpoints for **S3DIS Area 5 (0.1%)** and **ScanNetV2 (0.1%)**. Both experiments use the complete Directional Encoding (DE) and training-only Anchor Relay Learning (ARL).

**The complete codebase and remaining experiments will be released after the paper is accepted.**

## 1. Installation

Use Linux x86-64 (AVX2), Python 3.11, a BF16-capable NVIDIA GPU, the CUDA 12.8 toolkit (`nvcc`), and `g++`. The released runs used an H100 80 GB. Allow sufficient CPU RAM: the loaders cache scenes in memory (64 GB or more is recommended).

```bash
cd SparseSBS
conda create -n sparsesbs python=3.11 -y
conda activate sparsesbs
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip check

export CUDA_HOME=/usr/local/cuda-12.8   # adjust to your installed toolkit
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4
export CUDA_VISIBLE_DEVICES=0          # select one free GPU
python scripts/check.py --native
```

The last command checks dependencies and builds/tests the native operators. It does not train a model. If the selected GPU is occupied, choose a free GPU instead.

## 2. Prepare data

Download the datasets through the official access process and accept their terms. No raw dataset is redistributed here.

**S3DIS:** obtain `Stanford3dDataset_v1.2_Aligned_Version` from the [official dataset request form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform). Extract it so that `Area_1` through `Area_6` are directly under the supplied raw root.

```bash
python scripts/prepare_data.py --dataset s3dis \
  --raw-root /path/to/Stanford3dDataset_v1.2_Aligned_Version \
  --output-root data/s3dis
```

**ScanNetV2:** follow the [official ScanNet download instructions](https://github.com/ScanNet/ScanNet#scannet-data). Download the training/validation scenes, including `sceneXXXX_YY_vh_clean_2.ply` and `sceneXXXX_YY_vh_clean_2.labels.ply` in each scene directory. Images and `.sens` files are not needed.

```bash
python scripts/prepare_data.py --dataset scannetv2 \
  --raw-root /path/to/scannet \
  --output-root data/scannetv2
```

`--raw-root` may contain `scans/` or directly contain the scene directories. For a download containing meshes, `.aggregation.json` and `.segs.json` instead of label PLY files, append `--label-map /path/to/scannetv2-labels.combined.tsv`.

Each preparation command converts the data, installs the bundled DE calibration, prepares the 0.1% sparse indices, and checks the complete split. Expected counts are **204 training / 68 evaluation rooms** for S3DIS and **1,201 training / 312 validation scans** for ScanNetV2. To prepare only evaluation data, append `--split val`.

Run all commands from `SparseSBS/`, the directory containing this README. The YAML `data.root` (for example, `data/s3dis`) is a **generated output directory**, not a folder included in the ZIP and not relative to the YAML file. Use the same directory for preparation `--output-root` and training/evaluation `--data-root`; `anchors` and `statistics` are resolved inside that directory.

| Included in the ZIP | Created under `data/s3dis/` by preparation |
|---|---|
| `assets/s3dis/anchors.pt` | `anchors.pt` (after point-order checks) |
| `assets/s3dis/dist0.pt` | `statistics/dist0.pt` |
| `assets/s3dis/dist1.pt` | `statistics/dist1.pt` |

ScanNetV2 follows the same layout using `assets/scannetv2/` and `data/scannetv2/`; its prepared index file is `anchors.pt`. The following tree shows **preparation outputs**, not the archive contents.

```text
data/
├── s3dis/
│   ├── 1_room_name.pt ... 6_room_name.pt
│   ├── anchors.pt
│   └── statistics/dist0.pt, dist1.pt
└── scannetv2/
    ├── sceneXXXX_YY.pt
    ├── anchors.pt
    └── statistics/dist0.pt, dist1.pt
```

The public annotation filename is `anchors.pt` for both datasets; the seed recorded in each YAML is the training RNG seed, not a filename or guarantee of resampling the same annotations. Fixed indices are reused only when the prepared point content and ordering match the included fingerprints. Otherwise, the script creates a deterministic per-scene 0.1% sample for the new ordering; numerical results can differ. Existing valid preparation outputs are reused, including when switching between validation-only and full preparation. Training and evaluation check the recorded point content and required input files before running. Do not mix sparse indices from another preprocessing pipeline.

## 3. Evaluate the included checkpoints

Pretrained checkpoint can be downloaded from: [https://github.com/yyy0218/spareSBS/releases/tag/checkpoint](https://github.com/yyy0218/spareSBS/releases/tag/checkpoint), please scroll down to the Assets section to get the weight file.

| Dataset | Labels | Full-resolution mIoU | Checkpoint |
|---|---:|---:|---|
| S3DIS Area 5 | 0.1% | 69.90% | `checkpoints/s3dis_0p1.pt` |
| ScanNetV2 validation | 0.1% | 73.00% | `checkpoints/scannetv2_0p1.pt` |

```bash
python scripts/evaluate.py --config configs/s3dis/sparse_0p1.yaml \
  --checkpoint checkpoints/s3dis_0p1.pt --data-root data/s3dis \
  --device cuda:0 --output runs/s3dis_pretrained.json

python scripts/evaluate.py --config configs/scannetv2/sparse_0p1.yaml \
  --checkpoint checkpoints/scannetv2_0p1.pt --data-root data/scannetv2 \
  --device cuda:0 --output runs/scannetv2_pretrained.json
```

Both commands use one-vote full-resolution evaluation. JSON metrics are fractions: multiply `miou` by 100. For a quick check, append `--max-scenes 1`; that result is partial and is not a benchmark score.

## 4. Train from scratch

Run after preparing both training and validation data. The following commands train for **100 epochs** and save `best.pt` selected by training-time validation. The pretrained files do not include optimizer state and are not resume checkpoints.

```bash
# S3DIS: 
python scripts/train.py --config configs/s3dis/sparse_0p1.yaml \
  --data-root data/s3dis --device cuda:0 --output runs/s3dis_0p1

# ScanNetV2:
python scripts/train.py --config configs/scannetv2/sparse_0p1.yaml \
  --data-root data/scannetv2 --device cuda:0 --output runs/scannetv2_0p1
```

Evaluate the trained models with the commands in Section 3, replacing the checkpoint arguments with `runs/s3dis_0p1/best.pt` and `runs/scannetv2_0p1/best.pt`. Use different output JSON paths. From-scratch runs may vary with the environment and sparse sampling realization.

To run both training/evaluation pipelines sequentially:

```bash
python scripts/run_indoor_experiments.py --s3dis-root data/s3dis \
  --scannet-root data/scannetv2 --device cuda:0 --output-root runs/indoor_0p1
```

Use a new output directory for each fresh training run. To continue an interrupted run, append `--resume runs/s3dis_0p1/epochN.pt` (or the ScanNetV2 equivalent) with the same configuration and inputs. The checkpoint epoch must match the last recorded epoch; otherwise use a new output directory.

Check the extracted files using `sha256sum -c SHA256SUMS`. If host RAM is limited, lower `training.workers` in the YAML. Smaller GPUs may require lower batch size, which changes the training recipe.


