#!/usr/bin/env python3
"""
对单张或目录下图像提取 DINOv2 微调模型特征。
用于验证微调效果：检索、聚类、可视化等。

用法:
    # 单张图像
    python scripts/extract_features.py \
        --checkpoint output/vitb14_finetune2/teacher_checkpoint_0749.pth \
        --config dinov2/configs/eval/vitb14_finetune.yaml \
        --image path/to/image.jpg \
        --output features.npy

    # 目录下所有图像（扁平目录）
    python scripts/extract_features.py \
        --checkpoint output/vitb14_finetune2/teacher_checkpoint_0749.pth \
        --config dinov2/configs/eval/vitb14_finetune.yaml \
        --image-dir path/to/images/ \
        --output features.npy

    # ImageFolder 结构 (root/类别名/图片.jpg)，同时输出 labels.txt 供可视化
    python scripts/extract_features.py \
        --checkpoint ... --config ... \
        --image-dir path/to/images/ \
        --image-folder \
        --output features.npy
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import load_pretrained_weights


def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path, config_path):
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.load(config_path))
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    load_pretrained_weights(model, checkpoint_path, "teacher")
    model.eval()
    return model, cfg.crops.global_crops_size


def extract_features(model, image_paths, img_size, device="cuda", batch_size=32):
    transform = get_transform(img_size)
    model = model.to(device)
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))
        x = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model(x)
            # cls token: [B, 1, C] -> [B, C]
            if feats.dim() == 3:
                feats = feats[:, 0]
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="teacher checkpoint 路径")
    parser.add_argument("--config", type=str, default="dinov2/configs/eval/vitb14_finetune.yaml")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--image-dir", type=str, help="图像目录路径")
    parser.add_argument(
        "--image-folder",
        action="store_true",
        help="image-dir 为 ImageFolder 结构 (root/类别名/图片.jpg)，同时输出 labels.txt",
    )
    parser.add_argument("--output", type=str, default="features.npy", help="输出 .npy 路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    labels_out = None
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        if args.image_folder:
            image_paths, labels_out = [], []
            for class_name in sorted(os.listdir(args.image_dir)):
                class_path = os.path.join(args.image_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                for f in sorted(os.listdir(class_path)):
                    if f.lower().endswith(exts):
                        image_paths.append(os.path.join(class_path, f))
                        labels_out.append(class_name)
        else:
            image_paths = [
                os.path.join(args.image_dir, f)
                for f in sorted(os.listdir(args.image_dir))
                if f.lower().endswith(exts)
            ]
        if not image_paths:
            print(f"错误: 在 {args.image_dir} 中未找到图像")
            sys.exit(1)
    else:
        print("请指定 --image 或 --image-dir")
        sys.exit(1)

    print(f"加载模型: {args.checkpoint}")
    model, img_size = load_model(args.checkpoint, args.config)
    print(f"提取 {len(image_paths)} 张图像的特征 (img_size={img_size})...")
    features = extract_features(
        model, image_paths, img_size, device=args.device, batch_size=args.batch_size
    )
    np.save(args.output, features)
    print(f"特征已保存到 {args.output}, shape={features.shape}")
    if labels_out is not None:
        labels_path = args.output.replace(".npy", "_labels.txt")
        with open(labels_path, "w") as f:
            f.write("\n".join(labels_out))
        print(f"标签已保存到 {labels_path}，可用于 t-SNE/UMAP 可视化")


if __name__ == "__main__":
    main()
