#!/usr/bin/env python3
"""
检查 DINOv2 ViT-B/14 各层输出特征维度，用于学习网络结构。
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dinov2.layers import NestedTensorBlock


# 在 CPU 上运行时必须禁用 xFormers（xFormers 的 memory_efficient_attention 仅支持 CUDA）
os.environ["XFORMERS_DISABLED"] = "1"


# ===== 基本配置 =====
REPO_DIR = "/data1/code/dinov2"

# 如果使用官方 ImageNet 预训练权重：
#   下载后的路径例如：pretrain/dinov2_vitb14_pretrain.pth
# 如果使用你在金相数据集上训练得到的 teacher_checkpoint.pth，
#   把这个路径改成对应的文件即可。
WEIGHTS_PATH = "pretrain/dinov2_vitb14_pretrain.pth"

# 用于构造输入的图片（可选）
IMAGE_DIR = "/data1/code/dinov2/output"
IMG_NAME = "test"  # 不含扩展名，会自动尝试 .jpg/.png/.jpeg 等

# 输入尺寸（与你在 test_similar.py 中保持一致）
IMG_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_weights_with_pos_interp(model: torch.nn.Module, weights_path: str) -> None:
    """
    加载权重，兼容：
    - 官方 dinov2_vitb14_pretrain.pth
    - 你训练得到的 teacher_checkpoint.pth（224 输入），
      自动把 16×16 的 pos_embed 插值到 37×37。
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
                f" ckpt {pos_embed_ckpt.shape} -> model {pos_embed_model.shape}"
            )
            cls_pos_ckpt = pos_embed_ckpt[:, :1, :]
            patch_pos_ckpt = pos_embed_ckpt[:, 1:, :]

            num_patches_ckpt = patch_pos_ckpt.shape[1]
            num_patches_model = pos_embed_model.shape[1] - 1

            H_ckpt = W_ckpt = int(num_patches_ckpt ** 0.5)
            H_model = W_model = int(num_patches_model ** 0.5)

            patch_pos_ckpt = (
                patch_pos_ckpt.reshape(1, H_ckpt, W_ckpt, -1)
                .permute(0, 3, 1, 2)
            )

            patch_pos_resized = F.interpolate(
                patch_pos_ckpt,
                size=(H_model, W_model),
                mode="bicubic",
                align_corners=False,
            )

            patch_pos_resized = (
                patch_pos_resized.permute(0, 2, 3, 1)
                .reshape(1, H_model * W_model, -1)
            )

            new_pos_embed = torch.cat([cls_pos_ckpt, patch_pos_resized], dim=1)
            new_state_dict["pos_embed"] = new_pos_embed

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"load_state_dict 消息：{msg}")


def find_image_path(base_name: str, directory: str) -> str:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = Path(directory) / f"{base_name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"未找到图片: {base_name} 在 {directory}")


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def register_layer_hooks(model: torch.nn.Module):
    """
    在 patch_embed、每个 NestedTensorBlock、norm、head 上注册 forward hook，
    打印并记录输出特征的维度。
    """
    layer_shapes = {}

    def make_hook(name):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(out, torch.Tensor):
                return
            shape = tuple(out.shape)
            if name not in layer_shapes:
                layer_shapes[name] = shape
            print(f"{name}: {shape}")

        return hook

    if hasattr(model, "patch_embed"):
        model.patch_embed.register_forward_hook(make_hook("patch_embed"))

    for name, module in model.named_modules():
        if isinstance(module, NestedTensorBlock):
            module.register_forward_hook(make_hook(f"block::{name}"))

    if hasattr(model, "norm"):
        model.norm.register_forward_hook(make_hook("norm"))

    if hasattr(model, "head"):
        model.head.register_forward_hook(make_hook("head"))

    return layer_shapes


def inspect_model_layers():
    print("=== 加载 DINOv2 ViT-B/14 模型 ===")
    model = torch.hub.load(REPO_DIR, "dinov2_vitb14", source="local", pretrained=False)
    load_weights_with_pos_interp(model, WEIGHTS_PATH)
    model.eval()

    print("\n=== 注册 forward hook，准备前向 ===")
    layer_shapes = register_layer_hooks(model)

    transform = build_transform()
    try:
        img_path = find_image_path(IMG_NAME, IMAGE_DIR)
        print(f"\n使用图片作为输入: {img_path}")
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0)
    except FileNotFoundError as e:
        print(f"\n{e}，改用随机张量作为输入")
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"\ninput: {tuple(x.shape)}")

    with torch.no_grad():
        _ = model.forward_features(x)

    print("\n=== 各层输出维度汇总（按首次触发顺序） ===")
    for name, shape in layer_shapes.items():
        print(f"{name}: {shape}")


def main():
    inspect_model_layers()


if __name__ == "__main__":
    main()

