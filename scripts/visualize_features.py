#!/usr/bin/env python3
"""
DINOv2 不同层特征图可视化
核心方法来自 facebookresearch/dino 的 visualize_attention.py：
  1. CLS-to-Patch 注意力热力图 + 阈值分割 + 轮廓叠加（无监督分割效果最好）
  2. PCA 降维特征可视化（语义分割效果）
"""

import os
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import colorsys
import random
from pathlib import Path

import cv2
import skimage.io
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from PIL import Image
from torchvision import transforms


WEIGHTS_PATH = "pretrain/dinov2_vitb14_pretrain.pth"
IMG_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── 模型加载 ────────────────────────────────────────────


def load_model(weights_path: str):
    repo = str(Path(__file__).parent.parent)
    model = torch.hub.load(repo, "dinov2_vitb14", source="local", pretrained=False)

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("teacher", ckpt)
    cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    if "pos_embed" in cleaned and hasattr(model, "pos_embed"):
        ckpt_pe = cleaned["pos_embed"]
        model_pe = model.pos_embed
        if ckpt_pe.shape != model_pe.shape:
            cls_pe = ckpt_pe[:, :1, :]
            patch_pe = ckpt_pe[:, 1:, :]
            n_old, n_new = patch_pe.shape[1], model_pe.shape[1] - 1
            h_old = w_old = int(n_old**0.5)
            h_new = w_new = int(n_new**0.5)
            patch_pe = patch_pe.reshape(1, h_old, w_old, -1).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, (h_new, w_new), mode="bicubic", align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, h_new * w_new, -1)
            cleaned["pos_embed"] = torch.cat([cls_pe, patch_pe], dim=1)

    print(f"权重加载: {model.load_state_dict(cleaned, strict=False)}")
    return model


def preprocess(img_path: str):
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0), img


# ── 注意力提取（hook 方式，适配 DINOv2） ──────────────────


def _get_blocks(model):
    if getattr(model, "chunked_blocks", False):
        return [b for chunk in model.blocks for b in chunk if not isinstance(b, nn.Identity)]
    return list(model.blocks)


def extract_cls_attention(model, x, layer_indices=None):
    """
    通过 hook 提取各层 CLS token 对 patch token 的注意力权重。
    返回 dict: {layer_idx: attention (1, num_heads, H, W)}
    """
    blocks = _get_blocks(model)
    if layer_indices is None:
        layer_indices = list(range(len(blocks)))

    attention_maps = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inp, _out):
            B, N, C = inp[0].shape
            qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            # CLS token(index=0) 对 patch tokens(index=1:) 的注意力
            attention_maps[idx] = attn[:, :, 0, 1:].detach()
        return hook

    for idx in layer_indices:
        hooks.append(blocks[idx].attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return attention_maps


# ── 可视化方法（来自 facebookresearch/dino/visualize_attention.py） ──


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, output_path, figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    """完全复现 facebookresearch/dino 的 display_instances"""
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    mask = mask[None, :, :]
    colors = random_colors(1)

    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")

    masked_image = image.astype(np.uint32).copy()
    color = colors[0]
    _mask = mask[0]
    if blur:
        _mask = cv2.blur(_mask, (10, 10))
    masked_image = apply_mask(masked_image, _mask, color, alpha)
    if contour:
        padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
        padded_mask[1:-1, 1:-1] = _mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(str(output_path))
    plt.close()


def threshold_attention(attn, threshold, patch_size):
    """对注意力做阈值过滤，保留 top threshold 的质量，然后上采样到原图尺寸"""
    # attn: (num_heads, N_patches) → 排序并保留 top-threshold 的 mass
    nh = attn.shape[0]
    val, idx = torch.sort(attn, dim=1)
    val = val / val.sum(dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx, dim=1)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    return th_attn.float()


def pca_to_rgb(features: np.ndarray, H: int, W: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(features)
    for j in range(3):
        if rgb[:, j].mean() < 0:
            rgb[:, j] = -rgb[:, j]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb.reshape(H, W, 3)


# ── 绘图 ──────────────────────────────────────────────


def save_per_head_attention(orig_img_np, attn, patch_size, layer_idx, threshold, output_dir, name):
    """
    严格复现 facebookresearch/dino/visualize_attention.py 的做法：
    - 每个 head 独立热力图，用 nearest 上采样 + plt.imsave 直接保存
    - 每个 head 独立阈值分割 + 轮廓叠加
    """
    nh = attn.shape[0]
    n_patches = attn.shape[1]
    h_feat = w_feat = int(n_patches**0.5)

    # 关键：用 scale_factor + nearest 上采样，和原始 DINO 一致
    attn_2d = attn.reshape(nh, h_feat, w_feat)
    attn_up = F.interpolate(
        attn_2d.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0].cpu().numpy()

    head_dir = Path(output_dir) / f"layer{layer_idx:02d}_heads"
    head_dir.mkdir(parents=True, exist_ok=True)

    # 每个 head 单独保存热力图（和原始 DINO 一致，不指定 cmap）
    for j in range(nh):
        plt.imsave(head_dir / f"{name}_head{j:02d}.png", attn_up[j], format="png")

    # 每个 head 单独做阈值分割 + 轮廓叠加（和原始 DINO 完全一致）
    th_attn = threshold_attention(attn, threshold, patch_size)
    th_attn = th_attn.reshape(nh, h_feat, w_feat).float()
    th_up = F.interpolate(
        th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0].cpu().numpy()

    for j in range(nh):
        display_instances(
            orig_img_np, th_up[j],
            head_dir / f"{name}_head{j:02d}_mask_th{threshold}.png",
            alpha=0.5,
        )


def save_single_layer(
    orig_img_np, orig_img, attn, pca_rgb, cls_sim, layer_idx, patch_size, threshold, output_dir, name,
):
    """单层完整可视化：注意力热力图 + PCA + 阈值分割"""
    nh = attn.shape[0]
    n_patches = attn.shape[1]
    h_feat = w_feat = int(n_patches**0.5)

    # 注意力上采样：和原始 DINO 一致，用 nearest + scale_factor
    attn_up = F.interpolate(
        attn.reshape(1, nh, h_feat, w_feat),
        scale_factor=patch_size, mode="nearest",
    )[0].cpu().numpy()
    mean_attn = attn_up.mean(axis=0)

    # 每个head独立阈值后取平均做分割叠加
    th_attn = threshold_attention(attn, threshold, patch_size)
    th_attn = th_attn.reshape(nh, h_feat, w_feat).float()
    th_up = F.interpolate(
        th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0].cpu().numpy()
    # 取所有 head 的 mask 并集
    combined_mask = th_up.max(axis=0)

    masked = orig_img_np.astype(np.uint32).copy()
    colors = random_colors(1)
    masked = apply_mask(masked, combined_mask, colors[0], alpha=0.5)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mean_attn)
    axes[1].set_title(f"Layer {layer_idx} Attention (mean)")
    axes[1].axis("off")

    im2 = axes[2].imshow(cls_sim, cmap="inferno")
    axes[2].set_title(f"Layer {layer_idx} CLS Similarity")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pca_rgb)
    axes[3].set_title(f"Layer {layer_idx} PCA")
    axes[3].axis("off")

    axes[4].imshow(masked.astype(np.uint8))
    axes[4].set_title(f"Layer {layer_idx} Segmentation")
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_layer{layer_idx:02d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_overview(orig_img_np, orig_img, all_attn, all_pca, all_cls_sim, layer_indices, patch_size, threshold, output_dir, name):
    """所有层总览：注意力 + PCA + 分割"""
    n_layers = len(layer_indices)

    fig, axes = plt.subplots(n_layers, 4, figsize=(20, 5 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    for i, layer_idx in enumerate(layer_indices):
        nh = all_attn[i].shape[0]
        n_patches = all_attn[i].shape[1]
        h_feat = w_feat = int(n_patches**0.5)

        attn_up = F.interpolate(
            all_attn[i].reshape(1, nh, h_feat, w_feat),
            scale_factor=patch_size, mode="nearest",
        )[0].cpu().numpy()
        mean_attn = attn_up.mean(axis=0)

        th_attn = threshold_attention(all_attn[i], threshold, patch_size)
        th_attn = th_attn.reshape(nh, h_feat, w_feat).float()
        th_up = F.interpolate(
            th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
        )[0].cpu().numpy()
        combined_mask = th_up.max(axis=0)

        masked = orig_img_np.astype(np.uint32).copy()
        colors = random_colors(1)
        masked = apply_mask(masked, combined_mask, colors[0], alpha=0.5)

        axes[i, 0].imshow(mean_attn)
        axes[i, 0].set_title(f"Layer {layer_idx} Attention")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(all_cls_sim[i], cmap="inferno")
        axes[i, 1].set_title(f"Layer {layer_idx} CLS Sim")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(all_pca[i])
        axes[i, 2].set_title(f"Layer {layer_idx} PCA")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(masked.astype(np.uint8))
        axes[i, 3].set_title(f"Layer {layer_idx} Segmentation")
        axes[i, 3].axis("off")

    plt.suptitle("DINOv2 Feature Visualization Across Layers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_overview.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── 主逻辑 ──────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="DINOv2 不同层特征图可视化")
    parser.add_argument("--image", type=str, default="output/test.jpg")
    parser.add_argument("--weights", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--output-dir", type=str, default="output/feature_vis")
    parser.add_argument("--threshold", type=float, default=0.6, help="注意力阈值，保留 top X%% 的 mass")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="层索引，如 0 5 11；不传则全部")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"图像不存在: {img_path}")

    output_dir = Path(args.output_dir) / img_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    name = img_path.stem

    # 加载模型
    print("加载模型...")
    model = load_model(args.weights)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"设备: {device}")

    # 预处理
    x, orig_img = preprocess(str(img_path))
    x = x.to(device)
    patch_size = model.patch_size
    h_feat = IMG_SIZE // patch_size
    w_feat = IMG_SIZE // patch_size

    # 和原始 DINO 一致：保存归一化后的输入图，再读回作为叠加底图
    output_dir.mkdir(parents=True, exist_ok=True)
    img_save_path = str(output_dir / f"{name}_input.png")
    torchvision.utils.save_image(
        torchvision.utils.make_grid(x.cpu(), normalize=True, scale_each=True),
        img_save_path,
    )
    orig_img_np = skimage.io.imread(img_save_path)

    # 确定层索引
    n_blocks = getattr(model, "n_blocks", len(model.blocks))
    layer_indices = args.layers if args.layers else list(range(n_blocks))
    print(f"可视化层: {layer_indices}")

    # 1) 提取注意力（hook）
    print("提取注意力...")
    attn_maps = extract_cls_attention(model, x, layer_indices)

    # 2) 提取特征（用于 PCA + CLS 相似度）
    print("提取特征...")
    with torch.no_grad():
        features = model.get_intermediate_layers(
            x, n=layer_indices, reshape=True, return_class_token=True, norm=True
        )

    # 3) 生成可视化
    print("生成可视化...")
    all_attn, all_pca, all_cls_sim = [], [], []

    for layer_idx, (patch_tokens, cls_token) in zip(layer_indices, features):
        attn = attn_maps[layer_idx]  # (1, nh, N)
        attn = attn[0].cpu()         # (nh, N)
        all_attn.append(attn)

        # PCA
        pt = patch_tokens[0].permute(1, 2, 0).reshape(-1, patch_tokens.shape[1])
        pca_rgb = pca_to_rgb(pt.cpu().numpy(), h_feat, w_feat)
        all_pca.append(pca_rgb)

        # CLS 相似度
        pt_norm = F.normalize(pt, dim=-1)
        ct_norm = F.normalize(cls_token, dim=-1)
        sim = (pt_norm * ct_norm).sum(dim=-1)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        all_cls_sim.append(sim.cpu().numpy().reshape(h_feat, w_feat))

        # 保存单层完整图
        save_single_layer(
            orig_img_np, orig_img, attn, pca_rgb,
            all_cls_sim[-1], layer_idx, patch_size, args.threshold, output_dir, name,
        )
        # 保存每个 head 的独立热力图 + 阈值分割
        save_per_head_attention(
            orig_img_np, attn, patch_size, layer_idx, args.threshold, output_dir, name,
        )
        print(f"  Layer {layer_idx} 完成")

    # 总览图
    save_overview(
        orig_img_np, orig_img, all_attn, all_pca, all_cls_sim,
        layer_indices, patch_size, args.threshold, output_dir, name,
    )
    print(f"\n完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
