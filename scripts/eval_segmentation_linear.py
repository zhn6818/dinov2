#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 线性探测分割模型测试脚本

加载训练好的 seg_head，对单张/多张图像推理，保存预测图，可选计算 mIoU。
"""

import argparse
import math
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 复用训练脚本的模型构建（scripts 需在 path 中）
_scripts_dir = os.path.join(_project_root, "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from train_segmentation_linear import build_model


def get_args():
    parser = argparse.ArgumentParser(description="DINOv2 分割线性探测测试")
    parser.add_argument("--checkpoint", type=str, default="output/seg_u2netgrain/seg_head_epoch10.pth", help="分割头权重路径")
    parser.add_argument("--backbone-weights", type=str, default="pretrain/dinov2_vitb14_pretrain.pth", help="backbone 权重")
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--use-multiscale", action="store_true", help="若训练时用了 --use-multiscale 则需加此参数")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--image", type=str, default="", help="单张图像路径")
    parser.add_argument("--image-dir", type=str, default="", help="图像目录")
    parser.add_argument("--train-list", type=str, default="", help="train.txt，用于有 GT 时计算 mIoU")
    parser.add_argument("--output-dir", type=str, default="./output/seg_eval", help="预测图保存目录")
    parser.add_argument("--save-pred", action="store_true", help="保存预测图为彩色 PNG")
    parser.add_argument("--no-save-pred", dest="save_pred", action="store_false")
    parser.set_defaults(save_pred=True)
    return parser.parse_args()


def load_model(args):
    """构建模型并加载分割头权重"""
    # 构造 build_model 所需的参数对象，避免调用 get_train_args() 解析 sys.argv
    class TrainArgs:
        pass
    train_args = TrainArgs()
    train_args.backbone = args.backbone
    train_args.backbone_weights = args.backbone_weights
    train_args.num_classes = args.num_classes
    train_args.crop_size = args.crop_size
    train_args.use_multiscale = getattr(args, "use_multiscale", False)
    model = build_model(train_args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "decode_head" in ckpt:
        model.decode_head.load_state_dict(ckpt["decode_head"], strict=True)
        if "epoch" in ckpt:
            print(f"加载 checkpoint epoch {ckpt['epoch']}")
    else:
        model.decode_head.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def preprocess(img_path, crop_size=512):
    """与训练时一致的预处理"""
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    h, w = img.shape[1], img.shape[2]
    img = F.interpolate(img.unsqueeze(0), size=(crop_size, crop_size), mode="bilinear", align_corners=False).squeeze(0)
    return img, (h, w)


def pred_to_color(pred, num_classes=3):
    """预测索引 -> 彩色图 (RGB)"""
    palette = [
        [0, 0, 0],      # 0: 黑
        [255, 0, 0],    # 1: 红
        [0, 255, 0],    # 2: 绿
        [0, 0, 255],    # 3: 蓝
    ][:num_classes]
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(palette))):
        out[pred == c] = palette[c]
    return out


def compute_miou(pred, gt, num_classes, ignore_index=255):
    """单张图 mIoU"""
    pred = pred.flatten()
    gt = gt.flatten()
    mask = gt != ignore_index
    pred, gt = pred[mask], gt[mask]
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum().item()
        union = ((pred == c) | (gt == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
        else:
            ious.append(float("nan"))
    return np.nanmean(ious) if any(not np.isnan(i) for i in ious) else 0.0


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args).to(device)

    # 收集待测图像
    samples = []
    if args.image and os.path.isfile(args.image):
        samples.append((args.image, None))
    elif args.image_dir and os.path.isdir(args.image_dir):
        for f in sorted(os.listdir(args.image_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                samples.append((os.path.join(args.image_dir, f), None))
    elif args.train_list and os.path.isfile(args.train_list):
        with open(args.train_list) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    samples.append((parts[0], parts[1]))
    else:
        print("请指定 --image / --image-dir / --train-list")
        return

    if not samples:
        print("未找到有效图像")
        return

    print(f"共 {len(samples)} 张图像")
    ious = []

    with torch.no_grad():
        for img_path, ann_path in tqdm(samples, desc="推理"):
            img, orig_size = preprocess(img_path, args.crop_size)
            img = img.unsqueeze(0).to(device)
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = F.interpolate(logits, size=(args.crop_size, args.crop_size), mode="bilinear", align_corners=False)
            pred = logits.argmax(1).squeeze(0).cpu().numpy()

            if args.save_pred:
                color = pred_to_color(pred, args.num_classes)
                out_path = os.path.join(args.output_dir, os.path.basename(img_path) + "_pred.png")
                Image.fromarray(color).save(out_path)

            if ann_path and os.path.isfile(ann_path):
                ann = np.array(Image.open(ann_path))
                if len(ann.shape) == 3:
                    ann = ann[:, :, 0]
                ann = np.array(Image.fromarray(ann.astype(np.uint8)).resize(
                    (args.crop_size, args.crop_size), Image.NEAREST))
                ann[ann >= args.num_classes] = 255
                miou = compute_miou(pred, ann, args.num_classes)
                ious.append(miou)

    if ious:
        print(f"mIoU: {np.mean(ious):.4f}")
    print(f"预测图已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
