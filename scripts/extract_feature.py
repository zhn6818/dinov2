#!/usr/bin/env python3
"""
Patch 特征 PCA 可视化：使用 DINOv2 提取各层 patch token 特征，PCA 降维到 RGB 并保存。
支持自定义可视化层数：单层、多层或全部（ViT-B 共 12 层，索引 0~11）。
"""
# 在 CPU 上运行时必须禁用 xFormers（xFormers 的 memory_efficient_attention 仅支持 CUDA）
from typing import Any


import os
import argparse
os.environ["XFORMERS_DISABLED"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ===== 配置 =====
REPO_DIR = "/data1/code/dinov2"
# WEIGHTS_PATH = "pretrain/dinov2_vitb14_pretrain.pth"
# 若使用自定义训练权重（如 teacher_checkpoint.pth），修改为对应路径
WEIGHTS_PATH = "output/jinxiang/eval/training_87499/teacher_checkpoint.pth"

IMG_PATH = "/data1/code/dinov2/output/test5.jpg"  # 单张图片的完整路径
VIS_OUTPUT_DIR = "/data1/code/dinov2/output/vis_result"

# 可视化的层索引，None=全部层；[11]=仅最后一层；[0,5,11]=第 0/5/11 层
LAYER_INDICES = None

IMG_SIZE = 518
PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_weights(model: torch.nn.Module, weights_path: str) -> None:
    """
    加载权重，兼容：
    - 官方 dinov2_vitb14_pretrain.pth
    - 自定义 teacher_checkpoint.pth（224 输入），自动将 pos_embed 从 16×16 插值到 37×37
    """
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "teacher" in ckpt:
        state_dict = ckpt["teacher"]
    else:
        state_dict = ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        new_state_dict[k] = v

    if "pos_embed" in new_state_dict and hasattr(model, "pos_embed"):
        pos_embed_ckpt = new_state_dict["pos_embed"]
        pos_embed_model = model.pos_embed

        if pos_embed_ckpt.shape != pos_embed_model.shape:
            print(
                f"检测到 pos_embed 形状不一致，执行插值："
                f"ckpt {pos_embed_ckpt.shape} -> model {pos_embed_model.shape}"
            )
            cls_pos_ckpt = pos_embed_ckpt[:, :1, :]
            patch_pos_ckpt = pos_embed_ckpt[:, 1:, :]

            num_patches_ckpt = patch_pos_ckpt.shape[1]
            num_patches_model = pos_embed_model.shape[1] - 1

            H_ckpt = W_ckpt = int(num_patches_ckpt ** 0.5)
            H_model = W_model = int(num_patches_model ** 0.5)

            patch_pos_ckpt = patch_pos_ckpt.reshape(1, H_ckpt, W_ckpt, -1).permute(0, 3, 1, 2)
            patch_pos_resized = F.interpolate(
                patch_pos_ckpt,
                size=(H_model, W_model),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos_resized = patch_pos_resized.permute(0, 2, 3, 1).reshape(1, H_model * W_model, -1)
            new_pos_embed = torch.cat([cls_pos_ckpt, patch_pos_resized], dim=1)
            new_state_dict["pos_embed"] = new_pos_embed

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"load_state_dict 消息：{msg}")


def _pca_to_rgb(patch_flat: np.ndarray, H: int, W: int):
    """将 [N, C] 的 patch 特征 PCA 降维到 RGB [H, W, 3]，并对主成分做符号标准化"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    rgb = pca.fit_transform(patch_flat)
    # 消除 PCA 主成分的符号不确定性：使每列均值非负
    for j in range(3):
        if rgb[:, j].mean() < 0:
            rgb[:, j] = -rgb[:, j]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = np.clip(rgb, 0, 1).reshape(H, W, 3)
    return rgb


def visualize_layers_pca(model, x, img_path, output_dir, name, layer_indices=None):
    """
    对指定层 Transformer Block 的 patch token 做 PCA 可视化并保存。
    layer_indices: None=全部层；[11]=仅最后一层；[0,5,11]=第 0/5/11 层
    """
    try:
        from sklearn.decomposition import PCA  # noqa: F401
    except ImportError:
        print("  跳过 PCA 可视化: 需要安装 sklearn (pip install scikit-learn)")
        return

    # 使用 n_blocks 而非 len(blocks)：chunked 模型下 len(blocks)=chunk 数，非总 block 数
    n_blocks = getattr(model, "n_blocks", len(model.blocks))
    if layer_indices is None:
        layer_indices = list(range(n_blocks))
    else:
        for i in layer_indices:
            if i < 0 or i >= n_blocks:
                raise ValueError(f"层索引 {i} 超出范围 [0, {n_blocks-1}]")

    with torch.no_grad():
        features_list = model.get_intermediate_layers(
            x, n=layer_indices, reshape=True, return_class_token=False
        )

    orig_img = Image.open(img_path).convert("RGB")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for layer_idx, patch_tokens in zip(layer_indices, features_list):
        # patch_tokens: [1, 768, H, W]
        _, C, H, W = patch_tokens.shape
        patch_flat = patch_tokens[0].permute(1, 2, 0).reshape(-1, C).numpy()

        rgb = _pca_to_rgb(patch_flat, H, W)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(rgb)
        axes[1].set_title(f"Layer {layer_idx} Patch PCA (top3 components -> RGB)")
        axes[1].axis("off")
        plt.tight_layout()
        out_path = Path(output_dir) / f"{name}_patch_pca_layer{layer_idx:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  保存: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 Patch 特征 PCA 可视化")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="可视化的层索引，如 --layers 0 5 11 表示第 0/5/11 层；不传则全部层",
    )
    parser.add_argument(
        "--img",
        type=str,
        default=None,
        help="图片路径，覆盖配置中的 IMG_PATH",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出目录，覆盖配置中的 VIS_OUTPUT_DIR",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    img_path = args.img if args.img is not None else IMG_PATH
    out_dir = args.out if args.out is not None else VIS_OUTPUT_DIR
    layer_indices = args.layers if args.layers is not None else LAYER_INDICES

    print("加载模型...")
    model = torch.hub.load(REPO_DIR, "dinov2_vitb14", source="local", pretrained=False)
    load_weights(model, WEIGHTS_PATH)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    path = Path(img_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"未找到图片: {path}")
    name = path.stem
    print(f"\n处理: {name} ({path})")
    n_blocks = getattr(model, "n_blocks", len(model.blocks))
    if layer_indices is not None:
        print(f"可视化层: {layer_indices}")
    else:
        print(f"可视化层: 全部 ({n_blocks} 层)")
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)
    visualize_layers_pca(model, x, str(path), out_dir, name, layer_indices)

    print(f"\nPCA 可视化完成，结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
