# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

DINOv2 是 Meta AI 的自监督视觉 Transformer 框架。本 fork 扩展了自定义数据集（JinXiang 金相图像）、用于生物领域的 Cell-DINO/Channel-Adaptive DINO，以及注意力可视化和特征探索的分析脚本。

## 构建与运行命令

**运行环境：** 所有 Python 脚本在 Docker 容器 `JHCVTrain` 的 `ai` 虚拟环境中运行，不要在本地安装新环境。

**训练（需设置 `PYTHONPATH=.`）：**
```shell
# 自定义金相数据集（单 GPU）
PYTHONPATH=. python dinov2/run/train/train.py \
    --nodes 1 \
    --config-file dinov2/configs/train/jinxiang_vitb14.yaml \
    --output-dir output/jinxiang_run/

# 标准 ImageNet 训练（多节点，通过 SLURM）
python dinov2/run/train/train.py --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml --output-dir <OUTDIR>
```

**评估：**
```shell
PYTHONPATH=. python dinov2/run/eval/knn.py --config-file <config.yaml> ...
PYTHONPATH=. python dinov2/run/eval/linear.py --config-file <config.yaml> ...
PYTHONPATH=. python dinov2/run/eval/log_regression.py --config-file <config.yaml> ...
```

**分析脚本：**
```shell
python scripts/analyze_attention.py --image <path> [--weights <path>] [--detailed] [--per-head]
python scripts/extract_feature.py --image <path> [--layers 0 3 6 11]
python scripts/inspect_dinov2_layers.py
```

**代码检查：**
```shell
flake8                    # 配置在 setup.cfg：max-line-length=120，忽略 E203,E501,W503
black --check dinov2      # line-length=120
pylint --exit-zero dinov2 # 仅检查 FIXME/XXX/TODO 和相似度
```

## 架构

### 核心训练流水线
- `dinov2/train/ssl_meta_arch.py` — `SSLMetaArch`：核心模型，结合学生/教师骨干网络与 DINO + iBOT 头。学生模型通过 EMA 更新的教师模型进行训练。
- `dinov2/train/train.py` — 训练循环：数据加载、FSDP 封装、优化、定期保存检查点。
- `dinov2/run/train/train.py` — SLURM submitit 入口；实例化 `Trainer` 并提交作业。
- `dinov2/run/submit.py` — 分布式作业提交的共享参数解析器。

### 模型
- `dinov2/models/vision_transformer.py` — `DinoVisionTransformer` (ViT)，提供工厂函数 `vit_small`、`vit_base`、`vit_large`、`vit_giant2`。支持 registers、block chunks、SwiGLU FFN、通道自适应输入。
- `dinov2/layers/` — 构建模块：`MemEffAttention`（基于 xFormers）、`NestedTensorBlock`、`PatchEmbed`、`DINOHead`、`SwiGLUFFNFused`、`MLP`。
- `dinov2/loss/` — `DINOLoss`、`iBOTPatchLoss`、`KoLeoLoss`。

### 数据
- `dinov2/data/datasets/` — 继承自 `ExtendedVisionDataset` 的数据集类。配置中使用 `path:format` 语法（如 `ImageNet:split=TRAIN:root=...`、`JinXiang:root=...`）。
- `dinov2/data/augmentations.py` — `DataAugmentationDINO`（多裁剪）、`CellAugmentationDINO`。
- `dinov2/data/transforms.py`、`dinov2/data/collate.py` — 变换流水线和带掩码的合并。

### 配置
- `dinov2/configs/ssl_default_config.yaml` — 基础配置，定义所有默认值（模型、优化、裁剪、精度）。
- `dinov2/configs/train/` 中的训练配置覆盖默认值。关键参数：`student.arch`、`student.patch_size`、`train.dataset_path`、`optim.epochs`、`crops.*`。
- `dinov2/configs/eval/` 中为各模型变体的评估配置。

### 分布式训练
- `dinov2/fsdp/` — FSDP 封装器和 `FSDPCheckpointer`，用于模型分片。
- `dinov2/distributed/` — 分布式工具（GPU 数量、rank 等）。

### 评估下游任务
- `dinov2/eval/` — k-NN、线性探测、逻辑回归、深度估计、语义分割。
- `dinov2/eval/segmentation_m2f/` — 基于 Mask2Former 的分割流水线。

### Hub 与推理
- `hubconf.py` — PyTorch Hub 入口，用于加载预训练骨干网络和任务头。
- `dinov2/hub/` — Hub 加载函数：骨干网络、分类器、深度估计器、dino.txt、Cell-DINO、XRay-DINO。

## 关键约定

- 所有命令必须加 `PYTHONPATH=.` 前缀，除非已安装该包。
- 配置系统使用 OmegaConf；`--config-file` 加载 YAML，CLI 参数通过点号覆盖字段（`train.dataset_path=...`）。
- 视觉 Transformer 变体：ViT-S/14 (21M)、ViT-B/14 (86M)、ViT-L/14 (300M)、ViT-g/14 (1.1B)。Patch 大小：14 或 16。
- 训练需要 xFormers（内存高效注意力）。分析脚本设置 `XFORMERS_DISABLED=1` 以回退到 CPU。
- 预训练权重存放在 `pretrain/` 目录。
- 代码风格：Black 格式化，120 字符行宽。
