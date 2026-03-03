#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2 线性探测分割训练脚本

冻结 DINOv2 backbone，仅训练线性分割头。
支持两种数据格式：
  1) train.txt：每行 "img_path label_path"（如 u2netGrain/datasetv2）
  2) img_dir + ann_dir：目录下图像与标注同名

针对金相晶界等细线结构分割的改进：
  - Focal Loss / Dice Loss：缓解前景极少的类别不平衡
  - 数据增强：随机翻转、旋转、颜色扰动
  - 多尺度特征融合：使用最后 4 层特征 concat
"""

import argparse
import math
import os
import random
import sys

# 将项目根目录加入 sys.path，无需安装 dinov2
# 脚本位于 scripts/，向上 2 层得到项目根 dinov2/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
from tqdm import tqdm


# ============== 针对类别不平衡的损失函数 ==============

class FocalLoss(nn.Module):
    """Focal Loss：对易分类样本降权，缓解前景极少的类别不平衡（如晶界）"""

    def __init__(self, gamma=2.0, alpha=None, ignore_index=255, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 各类别权重，如 [0.1, 2.0, 2.0] 提高前景权重
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W)
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)
        mask = target != self.ignore_index
        logits = logits[mask]
        target = target[mask]
        if logits.numel() == 0:
            return logits.sum() * 0.0
        pt = F.softmax(logits, dim=1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - pt) ** self.gamma) * torch.log(pt + 1e-8)
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)[target]
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class DiceLoss(nn.Module):
    """Dice Loss：直接优化 IoU，对前景少的细线结构（晶界）更友好"""

    def __init__(self, num_classes, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (B, C, H, W), target: (B, H, W)
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        target = target.view(-1)
        mask = target != self.ignore_index
        probs = probs[mask]
        target = target[mask]
        if probs.numel() == 0:
            return probs.sum() * 0.0
        target_onehot = F.one_hot(target.long(), C).float()
        intersection = (probs * target_onehot).sum(dim=0)
        union = probs.sum(dim=0) + target_onehot.sum(dim=0)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def get_args():
    parser = argparse.ArgumentParser(description="DINOv2 线性探测分割训练")
    parser.add_argument("--data-root", type=str, default="", help="数据集根目录（与 train-list 二选一）")
    parser.add_argument("--train-list", type=str, default="/data1/code/u2netGrain/datasetv2/train.txt", help="train.txt 路径，每行 img_path label_path")
    parser.add_argument("--img-dir", type=str, default="images/training", help="图像子目录（仅 data-root 模式）")
    parser.add_argument("--ann-dir", type=str, default="annotations/training", help="标注子目录（仅 data-root 模式）")
    parser.add_argument("--val-img-dir", type=str, default="", help="验证集图像目录，空则不做验证")
    parser.add_argument("--val-ann-dir", type=str, default="", help="验证集标注目录")
    parser.add_argument("--num-classes", type=int, default=3, help="类别数（含背景）")
    parser.add_argument("--backbone", type=str, default="dinov2_vitb14", help="backbone 名称: dinov2_vits14/vitb14/vitl14")
    parser.add_argument("--backbone-weights", type=str, default="pretrain/dinov2_vitb14_pretrain.pth", help="backbone 权重路径")
    parser.add_argument("--output-dir", type=str, default="./output/seg_u2netgrain", help="输出目录")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=512, help="输入/输出尺寸，支持 224/512/1024 等（会 pad 到 14 的倍数）")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=10, help="每 N epoch 保存一次")
    # 针对金相晶界的改进
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal", "dice", "ce+dice"],
                        help="损失函数: ce=CrossEntropy, focal=FocalLoss, dice=DiceLoss, ce+dice=组合")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal Loss 的 gamma")
    parser.add_argument("--class-weights", type=str, default="", help="类别权重，逗号分隔，如 '0.2,2.0,2.0' 提高前景")
    parser.add_argument("--use-multiscale", action="store_true", help="使用最后 4 层特征融合（更强但更慢）")
    parser.add_argument("--no-augment", action="store_true", help="禁用数据增强")
    return parser.parse_args()


class CenterPadding(nn.Module):
    """将输入 pad 到 patch_size 的倍数"""

    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        return pad_size // 2, pad_size - pad_size // 2

    def forward(self, x):
        *dims, h, w = x.shape
        pad_h = self._get_pad(h)
        pad_w = self._get_pad(w)
        return F.pad(x, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]])


def _augment_pil_and_mask(pil_img, ann_np):
    """金相晶界数据增强：翻转、旋转、颜色扰动（在归一化前对 PIL 操作）"""
    # 随机水平/垂直翻转
    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        ann_np = np.fliplr(ann_np)
    if random.random() > 0.5:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        ann_np = np.flipud(ann_np)
    # 随机 90° 旋转
    k = random.randint(0, 3)
    if k > 0:
        pil_img = pil_img.rotate(-90 * k, expand=False)
        ann_np = np.rot90(ann_np, k)
    # 颜色扰动（亮度、对比度）
    if random.random() > 0.5:
        enh = ImageEnhance.Brightness(pil_img)
        pil_img = enh.enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        enh = ImageEnhance.Contrast(pil_img)
        pil_img = enh.enhance(random.uniform(0.8, 1.2))
    return pil_img, ann_np


class SegmentationDataset(Dataset):
    """分割数据集：支持 train.txt 或 img_dir+ann_dir"""

    def __init__(self, train_list=None, img_dir=None, ann_dir=None, crop_size=224, training=True, num_classes=150, augment=True):
        self.crop_size = crop_size
        self.training = training
        self.num_classes = num_classes
        self.augment = augment and training
        self.samples = []

        if train_list and os.path.isfile(train_list):
            with open(train_list) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and os.path.isfile(parts[0]) and os.path.isfile(parts[1]):
                        self.samples.append((parts[0], parts[1]))
        elif img_dir and ann_dir and os.path.isdir(img_dir) and os.path.isdir(ann_dir):
            for f in os.listdir(img_dir):
                base, ext = os.path.splitext(f)
                if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    ann_path = os.path.join(ann_dir, base + ".png")
                    if os.path.isfile(ann_path):
                        self.samples.append((os.path.join(img_dir, f), ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path)
        ann = np.array(ann)
        if len(ann.shape) == 3:
            ann = ann[:, :, 0]  # 若为 RGB 标注，取单通道

        # 数据增强（在归一化前）
        if self.augment:
            img, ann = _augment_pil_and_mask(img, ann)

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        ann = torch.from_numpy(ann).long()

        # ImageNet 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        h, w = img.shape[1], img.shape[2]
        if self.training and (h > self.crop_size or w > self.crop_size):
            top = torch.randint(0, max(1, h - self.crop_size), (1,)).item()
            left = torch.randint(0, max(1, w - self.crop_size), (1,)).item()
            img = img[:, top : top + self.crop_size, left : left + self.crop_size]
            ann = ann[top : top + self.crop_size, left : left + self.crop_size]
        else:
            img = F.interpolate(img.unsqueeze(0), size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False).squeeze(0)
            ann = F.interpolate(ann.unsqueeze(0).unsqueeze(0).float(), size=(self.crop_size, self.crop_size), mode="nearest").squeeze().long()

        # 255 为 ignore
        ann[ann >= self.num_classes] = 255
        return img, ann


class SimpleSegHead(nn.Module):
    """简单线性分割头：BN + 1x1 Conv，仅用最后一层"""

    def __init__(self, in_channels=768, num_classes=3):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        # x: list of (B,C,H,W)，取最后一层
        x = x[-1]
        return self.conv(self.bn(x))


class MultiScaleSegHead(nn.Module):
    """多尺度分割头：融合最后 4 层特征，对细线结构（晶界）更友好"""

    def __init__(self, embed_dim=768, num_layers=4, num_classes=3):
        super().__init__()
        in_channels = embed_dim * num_layers
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        # x: list of (B,C,H,W)，concat 所有层
        x = torch.cat(x, dim=1)
        return self.conv(self.bn(x))


def _interpolate_pos_embed(ckpt_pos_embed, target_shape):
    """将 checkpoint 的 pos_embed 插值到目标尺寸（如 224 对应 16x16）"""
    cls_pos = ckpt_pos_embed[:, :1]
    patch_pos = ckpt_pos_embed[:, 1:]
    N = patch_pos.shape[1]
    M = int(math.sqrt(N))
    assert N == M * M
    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, M, M, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=target_shape, mode="bicubic", antialias=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((cls_pos, patch_pos), dim=1)


def build_model(args):
    """构建分割模型：DINOv2 backbone + 线性分割头"""

    # 加载 backbone
    if args.backbone_weights:
        from urllib.parse import urlparse
        from dinov2.models import vision_transformer as vits

        arch_map = {"dinov2_vits14": "vit_small", "dinov2_vitb14": "vit_base", "dinov2_vitl14": "vit_large"}
        arch_name = arch_map.get(args.backbone, "vit_base")
        backbone = vits.__dict__[arch_name](patch_size=14, num_register_tokens=0)

        # 加载权重，处理 pos_embed 尺寸不匹配（预训练 518x518 -> 224x224）
        if urlparse(args.backbone_weights).scheme:
            state_dict = torch.hub.load_state_dict_from_url(args.backbone_weights, map_location="cpu")
        else:
            state_dict = torch.load(args.backbone_weights, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        if "pos_embed" in state_dict and state_dict["pos_embed"].shape != backbone.pos_embed.shape:
            n_patches = backbone.pos_embed.shape[1] - 1
            w = int(math.sqrt(n_patches))
            state_dict["pos_embed"] = _interpolate_pos_embed(state_dict["pos_embed"], (w, w))
        backbone.load_state_dict(state_dict, strict=False)
        embed_dim = 384 if "small" in arch_name else 768 if "base" in arch_name else 1024
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", args.backbone)
        embed_dim = backbone.embed_dim

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    use_multiscale = getattr(args, "use_multiscale", False)
    if use_multiscale:
        decode_head = MultiScaleSegHead(embed_dim=embed_dim, num_layers=4, num_classes=args.num_classes)
    else:
        decode_head = SimpleSegHead(in_channels=embed_dim, num_classes=args.num_classes)

    # 不同 backbone 的最后一层索引
    out_indices_map = {
        "dinov2_vits14": [8, 9, 10, 11],
        "dinov2_vitb14": [8, 9, 10, 11],
        "dinov2_vitl14": [20, 21, 22, 23],
    }
    out_indices = out_indices_map.get(args.backbone, [8, 9, 10, 11])

    class Segmentor(nn.Module):
        def __init__(self, backbone, decode_head, patch_size=14, out_indices=None):
            super().__init__()
            self.backbone = backbone
            self.decode_head = decode_head
            self.patch_size = getattr(backbone, "patch_size", 14)
            self.out_indices = out_indices or [8, 9, 10, 11]
            self.pad = CenterPadding(self.patch_size)

        def forward(self, x):
            x = self.pad(x)
            feats = self.backbone.get_intermediate_layers(x, n=self.out_indices, reshape=True)
            # feats 为 tuple of tensors，转为 list 供 decode_head 取最后一层
            return self.decode_head(list(feats))

    model = Segmentor(backbone, decode_head, out_indices=out_indices)
    return model


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印训练尺寸与输出维度
    patch_size = 14
    padded_h = math.ceil(args.crop_size / patch_size) * patch_size
    feat_hw = padded_h // patch_size
    embed_dim = 384 if "small" in args.backbone else 768 if "base" in args.backbone else 1024
    use_augment = not getattr(args, "no_augment", False)
    print(f"[尺寸] 输入: {args.crop_size}x{args.crop_size} -> pad 后: {padded_h}x{padded_h} -> 特征图: {feat_hw}x{feat_hw}")
    print(f"[输出] 分割头: 特征 {embed_dim} -> {args.num_classes} 类, 多尺度={getattr(args, 'use_multiscale', False)}, 增强={use_augment}")
    print(f"[损失] {args.loss} (金相晶界建议: focal 或 ce+dice)")

    if args.train_list and os.path.isfile(args.train_list):
        dataset = SegmentationDataset(
            train_list=args.train_list,
            crop_size=args.crop_size,
            training=True,
            num_classes=args.num_classes,
            augment=use_augment,
        )
        if len(dataset) == 0:
            raise ValueError(f"train.txt 中无有效样本: {args.train_list}")
    elif args.data_root:
        img_dir = os.path.join(args.data_root, args.img_dir)
        ann_dir = os.path.join(args.data_root, args.ann_dir)
        if not os.path.isdir(img_dir) or not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"请确保 {img_dir} 和 {ann_dir} 存在")
        dataset = SegmentationDataset(
            img_dir=img_dir,
            ann_dir=ann_dir,
            crop_size=args.crop_size,
            training=True,
            num_classes=args.num_classes,
            augment=use_augment,
        )
        if len(dataset) == 0:
            raise ValueError(f"未找到有效样本: {img_dir} / {ann_dir}")
    else:
        raise ValueError("请指定 --train-list 或 --data-root")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.decode_head.parameters(), lr=args.lr, weight_decay=0.0001)

    # 构建损失函数
    class_weights = None
    if args.class_weights:
        class_weights = [float(x) for x in args.class_weights.split(",")]
        assert len(class_weights) == args.num_classes, f"class_weights 需 {args.num_classes} 个值"
    if args.loss == "ce":
        w = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=w)
        criterion_ce, criterion_dice = None, None
    elif args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights, ignore_index=255)
        criterion_ce, criterion_dice = None, None
    elif args.loss == "dice":
        criterion = DiceLoss(num_classes=args.num_classes, ignore_index=255)
        criterion_ce, criterion_dice = None, None
    elif args.loss == "ce+dice":
        w = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
        criterion_ce = nn.CrossEntropyLoss(ignore_index=255, weight=w)
        criterion_dice = DiceLoss(num_classes=args.num_classes, ignore_index=255)
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        criterion_ce, criterion_dice = None, None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, gts in pbar:
            imgs, gts = imgs.to(device), gts.to(device)
            logits = model(imgs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            # logits 为 patch 分辨率，上采样到与 gts 一致
            if logits.shape[-2:] != gts.shape[-2:]:
                logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
            if criterion is not None:
                loss = criterion(logits, gts)
            else:
                loss = 0.5 * criterion_ce(logits, gts) + 0.5 * criterion_dice(logits, gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg_loss={avg_loss:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            ckpt = {"decode_head": model.decode_head.state_dict(), "epoch": epoch + 1}
            torch.save(ckpt, os.path.join(args.output_dir, f"seg_head_epoch{epoch+1}.pth"))

    torch.save({"decode_head": model.decode_head.state_dict()}, os.path.join(args.output_dir, "seg_head_final.pth"))
    print(f"训练完成，权重已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
