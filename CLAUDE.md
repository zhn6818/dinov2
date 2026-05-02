# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DINOv2 is a self-supervised vision transformer framework from Meta AI. This fork extends it with custom datasets (JinXiang metallographic images), Cell-DINO/Channel-Adaptive DINO for biology, and analysis scripts for attention visualization and feature exploration.

## Build & Run Commands

**Environment setup:**
```shell
conda env create -f conda.yaml && conda activate dinov2
# or: pip install -r requirements.txt
```

**Training (requires `PYTHONPATH=.`):**
```shell
# Custom metallographic dataset (single GPU)
PYTHONPATH=. python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/jinxiang_vitb14.yaml \
    --output-dir output/jinxiang_run/

# Standard ImageNet training (multi-node via SLURM)
python dinov2/run/train/train.py --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml --output-dir <OUTDIR>
```

**Evaluation:**
```shell
PYTHONPATH=. python dinov2/run/eval/knn.py --config-file <config.yaml> ...
PYTHONPATH=. python dinov2/run/eval/linear.py --config-file <config.yaml> ...
PYTHONPATH=. python dinov2/run/eval/log_regression.py --config-file <config.yaml> ...
```

**Analysis scripts:**
```shell
python scripts/analyze_attention.py --image <path> [--weights <path>] [--detailed] [--per-head]
python scripts/extract_feature.py --image <path> [--layers 0 3 6 11]
python scripts/inspect_dinov2_layers.py
```

**Linting:**
```shell
flake8                    # config in setup.cfg: max-line-length=120, ignore E203,E501,W503
black --check dinov2      # line-length=120
pylint --exit-zero dinov2 # only checks FIXME/XXX/TODO and similarity
```

## Architecture

### Core Training Pipeline
- `dinov2/train/ssl_meta_arch.py` — `SSLMetaArch`: the central model combining student/teacher backbones with DINO + iBOT heads. Student is trained with EMA-updated teacher.
- `dinov2/train/train.py` — Training loop: data loading, FSDP wrapping, optimization, periodic checkpointing.
- `dinov2/run/train/train.py` — SLURM submitit entry point; instantiates `Trainer` and submits jobs.
- `dinov2/run/submit.py` — Shared argument parser for distributed job submission.

### Model
- `dinov2/models/vision_transformer.py` — `DinoVisionTransformer` (ViT) with factory functions `vit_small`, `vit_base`, `vit_large`, `vit_giant2`. Supports registers, block chunks, SwiGLU FFN, channel-adaptive input.
- `dinov2/layers/` — Building blocks: `MemEffAttention` (xFormers-backed), `NestedTensorBlock`, `PatchEmbed`, `DINOHead`, `SwiGLUFFNFused`, `MLP`.
- `dinov2/loss/` — `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`.

### Data
- `dinov2/data/datasets/` — Dataset classes inheriting from `ExtendedVisionDataset`. Each uses `path:format` syntax in config (e.g., `ImageNet:split=TRAIN:root=...`, `JinXiang:root=...`).
- `dinov2/data/augmentations.py` — `DataAugmentationDINO` (multi-crop), `CellAugmentationDINO`.
- `dinov2/data/transforms.py`, `dinov2/data/collate.py` — Transform pipelines and collation with masking.

### Configs
- `dinov2/configs/ssl_default_config.yaml` — Base config defining all defaults (model, optimization, crops, precision).
- Training configs in `dinov2/configs/train/` override defaults. Key params: `student.arch`, `student.patch_size`, `train.dataset_path`, `optim.epochs`, `crops.*`.
- Eval configs in `dinov2/configs/eval/` for each model variant.

### Distributed Training
- `dinov2/fsdp/` — FSDP wrappers and `FSDPCheckpointer` for model sharding.
- `dinov2/distributed/` — Distributed utilities (GPU count, rank, etc.).

### Evaluation Downstream Tasks
- `dinov2/eval/` — k-NN, linear probing, logistic regression, depth estimation, semantic segmentation.
- `dinov2/eval/segmentation_m2f/` — Mask2Former-based segmentation pipeline.

### Hub & Inference
- `hubconf.py` — PyTorch Hub entry points for loading pretrained backbones and task heads.
- `dinov2/hub/` — Hub loader functions for backbones, classifiers, depthers, dino.txt, Cell-DINO, XRay-DINO.

## Key Conventions

- All commands must be prefixed with `PYTHONPATH=.` unless the package is installed.
- Config system uses OmegaConf; `--config-file` loads YAML, CLI args override fields via dot notation (`train.dataset_path=...`).
- Vision transformer variants: ViT-S/14 (21M), ViT-B/14 (86M), ViT-L/14 (300M), ViT-g/14 (1.1B). Patch sizes: 14 or 16.
- xFormers is required for training (memory-efficient attention). Analysis scripts set `XFORMERS_DISABLED=1` for CPU fallback.
- Pretrained weights stored in `pretrain/` directory.
- Code style: Black formatting with 120-char line length.
