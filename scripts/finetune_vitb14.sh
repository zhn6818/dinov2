#!/bin/bash
# DINOv2 ViT-B/14 垂直领域微调脚本
# 适用于 1-2 GPU、1-3 万无标签图像

# ==================== 配置区域 ====================

# 修改这里：设置你的图像数据路径
DATA_PATH="/Volumes/data1/JH/projects/JLD_imgprocess/datasetv2/img"

# 修改这里：设置输出目录
OUTPUT_DIR="./output/vitb14_finetune"

# 预训练权重（可选，默认使用官方 URL）
# PRETRAINED_WEIGHTS="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
# 或者使用本地路径：
# PRETRAINED_WEIGHTS="/path/to/pretrained_weights/dinov2_vitb14_pretrain.pth"

# ==================== 训练配置 ====================

# GPU 数量（1 或 2）
NUM_GPUS=1

# 配置文件路径
CONFIG_FILE="dinov2/configs/train/vitb14_finetune.yaml"

# ==================== 训练命令 ====================

echo "=================================="
echo "DINOv2 ViT-B/14 垂直领域微调"
echo "=================================="
echo "数据路径: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "GPU 数量: $NUM_GPUS"
echo "=================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查数据路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    echo "请修改脚本中的 DATA_PATH 变量"
    exit 1
fi

# 计算数据集中的图像数量
IMG_COUNT=$(find "$DATA_PATH" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.webp" \) | wc -l)
echo "找到 $IMG_COUNT 张图像"

if [ "$IMG_COUNT" -eq 0 ]; then
    echo "错误: 在 $DATA_PATH 中未找到任何图像文件"
    exit 1
fi

echo "开始训练..."
echo ""

# 根据GPU数量选择训练方式
if [ "$NUM_GPUS" -eq 1 ]; then
    # 单 GPU 训练
    python dinov2/run/train/train.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        train.dataset_path=FlatFolder:root="$DATA_PATH" \
        student.pretrained_weights=$PRETRAINED_WEIGHTS

elif [ "$NUM_GPUS" -ge 2 ]; then
    # 多 GPU 训练（使用 torchrun）
    torchrun --nproc_per_node=$NUM_GPUS dinov2/run/train/train.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        train.dataset_path=FlatFolder:root="$DATA_PATH" \
        student.pretrained_weights=$PRETRAINED_WEIGHTS
fi

echo ""
echo "=================================="
echo "训练完成!"
echo "模型保存在: $OUTPUT_DIR/eval/"
echo "=================================="
