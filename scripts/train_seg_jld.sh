#!/bin/bash
# DINOv2 分割线性探测训练 - u2netGrain/datasetv2
# 使用前请先激活 conda 环境：conda activate ai
#
# 数据：/data1/code/u2netGrain/datasetv2/
#   - img/*.jpg, label/*.png
#   - train.txt: 每行 "img_path label_path"
# 标注：0, 1, 2 -> 三分类

set -e
cd /data1/code/dinov2
export PYTHONPATH="${PWD}:${PYTHONPATH}"

DATA_ROOT="/data1/code/u2netGrain/datasetv2"
BACKBONE_WEIGHTS="pretrain/dinov2_vitb14_pretrain.pth"
OUTPUT_DIR="./output/seg_u2netgrain"

python scripts/train_segmentation_linear.py \
    --train-list "$DATA_ROOT/train.txt" \
    --num-classes 3 \
    --backbone-weights "$BACKBONE_WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-3 \
    --crop-size 512 \
    --num-workers 4

echo "训练完成，权重保存在 $OUTPUT_DIR"
