# DINOv2 线性探测分割训练指南

本文档说明如何在 DINOv2 上进行**语义分割的线性探测（Linear Probing）**训练，以及如何整理自定义分割数据集格式。

---

## 一、前置依赖

分割训练依赖 **mmcv** 和 **mmsegmentation**，需使用 extras 环境：

```bash
# 使用 conda-extras 环境（包含 mmcv、mmsegmentation）
conda env create -f conda-extras.yaml
conda activate dinov2-extras

# 或 pip 安装
pip install -r requirements.txt -r requirements-extras.txt
```

依赖版本：`mmcv-full==1.5.0`，`mmsegmentation==0.27.0`

---

## 二、分割数据集格式

DINOv2 分割使用 **mmsegmentation** 的数据格式，支持 ADE20K、VOC、以及自定义 CustomDataset。

### 2.1 ADE20K 格式（推荐参考）

```
data_root/
├── images/
│   ├── training/          # 训练图像
│   │   ├── xxx.jpg
│   │   └── yyy.jpg
│   └── validation/        # 验证图像
│       ├── aaa.jpg
│       └── bbb.jpg
└── annotations/
    ├── training/          # 训练标注（与图像同名，扩展名 .png）
    │   ├── xxx.png
    │   └── yyy.png
    └── validation/
        ├── aaa.png
        └── bbb.png
```

**标注图要求**：
- 单通道 PNG，像素值 = 类别 ID（0, 1, 2, ...）
- 背景通常用 255 或 0 表示（可配置 `reduce_zero_label`）
- 图像与标注**文件名必须一致**（仅扩展名不同）

### 2.2 VOC 格式

```
VOCdevkit/VOC2012/
├── JPEGImages/            # 原图
│   ├── 2007_000032.jpg
│   └── ...
├── SegmentationClass/     # 标注（.png）
│   ├── 2007_000032.png
│   └── ...
└── ImageSets/Segmentation/
    ├── train.txt          # 训练集文件名列表（无扩展名）
    ├── val.txt
    └── trainval.txt
```

### 2.3 自定义数据集（CustomDataset）

适用于任意自定义分割数据，目录结构：

```
my_dataset/
├── img_dir/
│   ├── train/             # 或 images/training
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── img3.jpg
└── ann_dir/
    ├── train/             # 或 annotations/training
    │   ├── img1.png       # 与图像同名
    │   └── img2.png
    └── val/
        └── img3.png
```

**可选**：使用 split 文件指定 train/val：

```
my_dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── annotations/
│   ├── img1.png
│   ├── img2.png
│   └── img3.png
└── splits/
    ├── train.txt          # 每行一个文件名（无扩展名）
    └── val.txt
```

**标注图像素值**：
- 0 = 背景（或第一个类别，取决于 `reduce_zero_label`）
- 1, 2, 3, ... = 各类别 ID
- 255 = 忽略区域（不参与 loss）

---

## 三、线性探测分割训练流程

### 3.1 原理

- **冻结** DINOv2 backbone，只训练一个**线性分割头**（BN + 1×1 Conv）
- 输入：DINOv2 patch 特征
- 输出：每个像素的类别 logits

### 3.2 快速开始：使用项目自带脚本（推荐）

项目提供 `scripts/train_segmentation_linear.py`，**无需 mmsegmentation**，仅需 PyTorch 和 DINOv2：

```bash
cd /data1/code/dinov2
export PYTHONPATH="${PWD}:${PYTHONPATH}"

python scripts/train_segmentation_linear.py \
    --data-root /path/to/your/segmentation_dataset \
    --img-dir images/training \
    --ann-dir annotations/training \
    --num-classes 3 \
    --backbone dinov2_vitb14 \
    --backbone-weights pretrain/dinov2_vitb14_pretrain.pth \
    --output-dir ./output/segmentation_linear \
    --epochs 50 \
    --batch-size 4
```

**金相晶界等细线结构分割**（前景极少、类别不平衡）建议使用：

```bash
python scripts/train_segmentation_linear.py \
    --train-list /path/to/train.txt \
    --num-classes 3 \
    --loss focal \
    --use-multiscale \
    --crop-size 512 \
    --epochs 80 \
    --batch-size 4

# 或组合损失 + 类别权重
python scripts/train_segmentation_linear.py \
    --train-list /path/to/train.txt \
    --num-classes 3 \
    --loss ce+dice \
    --class-weights "0.2,2.0,2.0" \
    --use-multiscale
```

| 参数 | 说明 |
|------|------|
| `--loss focal` | Focal Loss，缓解前景极少的类别不平衡 |
| `--loss ce+dice` | CE + Dice 组合，对细线结构更友好 |
| `--class-weights "0.2,2.0,2.0"` | 提高前景类权重（背景, 晶界1, 晶界2） |
| `--use-multiscale` | 融合最后 4 层特征，提升细节 |
| `--no-augment` | 禁用数据增强（默认开启翻转、旋转、颜色扰动） |

**使用微调后的 backbone**：
```bash
python scripts/train_segmentation_linear.py \
    --data-root /path/to/seg_dataset \
    --num-classes 3 \
    --backbone-weights output/vitb14_finetune_jinxiang/eval/training_10000/teacher_checkpoint.pth \
    --output-dir ./output/segmentation_linear
```

**测试训练好的模型**（`scripts/eval_segmentation_linear.py`）：
```bash
# 使用 train.txt 中的图像，保存预测图并计算 mIoU
python scripts/eval_segmentation_linear.py \
    --checkpoint output/seg_u2netgrain/seg_head_epoch10.pth \
    --train-list /data1/code/u2netGrain/datasetv2/train.txt \
    --output-dir ./output/seg_eval \
    --num-classes 3 --crop-size 512

# 单张图像
python scripts/eval_segmentation_linear.py --checkpoint ... --image path/to/img.jpg

# 图像目录
python scripts/eval_segmentation_linear.py --checkpoint ... --image-dir path/to/imgs/
```

或使用快捷脚本：`bash scripts/eval_seg_jld.sh [checkpoint路径]`

### 3.3 使用 mmsegmentation 的完整配置（可选）

基于官方 [dinov2_vitb14_ade20k_linear_config.py](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_config.py)，创建自定义配置。

在 `dinov2/configs/eval/` 下新建 `segmentation_custom.py`：

```python
# dinov2/configs/eval/segmentation_custom.py
# 自定义分割线性探测配置（以 CustomDataset 为例）

# 导入 DINOv2 分割模块，以便 mmseg 能找到 DinoVisionTransformer、BNHead
custom_imports = dict(
    imports=[
        'dinov2.eval.segmentation.models.backbones.vision_transformer',
        'dinov2.eval.segmentation.models.decode_heads.linear_head',
    ],
    allow_failed_imports=False
)

dataset_type = 'CustomDataset'  # 或 'ADE20KDataset', 'PascalVOCDataset'
data_root = '/path/to/your/segmentation_dataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)
crop_size = (512, 512)

# 类别数：根据你的数据集修改
num_classes = 3  # 例如：背景 + 2 个前景类

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),  # 若背景为 0 则减 1
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=test_pipeline
    )
)

# 若使用 split 文件
# data['train']['split'] = 'splits/train.txt'
# data['val']['split'] = 'splits/val.txt'

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False
)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='mIoU', pre_eval=True)
fp16 = None
find_unused_parameters = True

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DinoVisionTransformer',
        out_indices=[8, 9, 10, 11],  # ViT-B 最后 4 层
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        )
    ),
    decode_head=dict(
        type='BNHead',
        in_channels=[768],
        in_index=[3],
        input_transform='resize_concat',
        channels=768,
        dropout_ratio=0,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

work_dir = './output/segmentation_linear'
gpu_ids = range(0, 1)  # 单卡可设为 range(0, 1)
```

若需使用 mmseg 的完整 pipeline（数据增强、多尺度等），可参考下方配置并配合 mmseg 的 `tools/train.py`。注意：DINOv2 的 backbone 在 mmseg 中为占位符，需通过自定义脚本注入真实 backbone。

### 3.4 使用微调后的 DINOv2 backbone

若你已有金相图微调后的 teacher 权重，可将 backbone 的 `checkpoint` 改为本地路径：

```python
# 在 model.backbone.init_cfg 中
checkpoint='output/vitb14_finetune_jinxiang/eval/training_XXXXX/teacher_checkpoint.pth'
```

---

## 四、推理与可视化

训练完成后，使用 `notebooks/semantic_segmentation.ipynb` 中的方式加载模型并推理：

```python
from mmseg.apis import init_segmentor, inference_segmentor

config_file = 'dinov2/configs/eval/segmentation_custom.py'
checkpoint_file = 'output/segmentation_linear/iter_40000.pth'

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
result = inference_segmentor(model, 'path/to/image.jpg')
# result[0] 为 (H, W) 的预测类别 ID
```

---

## 五、常见问题

### Q0：金相晶界分割效果差怎么办？

晶界是细线结构，前景占比通常 <5%，存在严重类别不平衡。建议：

1. **使用 Focal Loss 或 ce+dice**：`--loss focal` 或 `--loss ce+dice`
2. **提高前景权重**：`--class-weights "0.2,2.0,2.0"`（按你实际类别数调整）
3. **开启多尺度**：`--use-multiscale` 融合多层特征
4. **保持数据增强**：默认开启，不要加 `--no-augment`
5. **增大 crop-size**：若晶界很细，可尝试 1024
6. **增加 epoch**：50→80 或更多

若仍不佳，可考虑微调 backbone（解冻最后几层）或使用 U-Net 等专门结构。

### Q1：CustomDataset 找不到图像

检查 `img_dir`、`ann_dir` 是否相对于 `data_root` 正确，且图像与标注**文件名一致**。

### Q2：reduce_zero_label 含义

- `True`：标注中 0 视为背景，实际类别从 1 开始，会减 1 后送入网络
- `False`：0 即第 0 类，不做转换

### Q3：类别数 num_classes

包含背景在内的**总类别数**。例如 3 类（背景 + 2 类前景）则 `num_classes=3`。

### Q4：DinoVisionTransformer 未注册

确保在配置中通过 `custom_imports` 导入了 `dinov2.eval.segmentation.models`，且 `PYTHONPATH` 包含项目根目录。

---

## 六、参考链接

- [mmsegmentation 自定义数据集](https://mmsegmentation.readthedocs.io/en/latest/tutorials/customize_datasets.html)
- [DINOv2 官方分割 config](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_config.py)
- [notebooks/semantic_segmentation.ipynb](../notebooks/semantic_segmentation.ipynb)
