#!/bin/bash
# 测试分割线性探测模型（测试文件夹内图像）
# 用法: bash scripts/eval_seg_jld.sh [checkpoint路径] [图像目录]
# 使用前：conda activate ai

set -e
cd /data1/code/dinov2
export PYTHONPATH="${PWD}:${PYTHONPATH}"

CHECKPOINT="${1:-output/seg_u2netgrain/seg_head_epoch50.pth}"
IMAGE_DIR="${2:-/data1/code/u2netGrain/datasetv2/img}"
OUTPUT_DIR="./output/seg_eval"

python scripts/eval_segmentation_linear.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-classes 3 \
    --crop-size 512

echo "测试目录: $IMAGE_DIR, 预测图保存在 $OUTPUT_DIR"
