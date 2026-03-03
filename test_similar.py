"""
使用 DINOv2 提取图像特征并进行可视化
"""
# 在 CPU 上运行时必须禁用 xFormers（xFormers 的 memory_efficient_attention 仅支持 CUDA）
import os
os.environ["XFORMERS_DISABLED"] = "1"

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 配置
REPO_DIR = "/data1/code/dinov2"
# 如果使用官方 ImageNet 预训练权重，保持如下路径；
# 如果使用你在金相数据集上从头训练得到的权重，
# 将该路径改为对应的 teacher_checkpoint.pth，例如：
# WEIGHTS_PATH = "/data1/zhn/jinxiang_runs/run1/eval/training_24999/teacher_checkpoint.pth"
WEIGHTS_PATH = "pretrain/dinov2_vitb14_pretrain.pth"
IMAGE_DIR = "/data1/code/dinov2/output"  # 图片所在目录
IMG_NAMES = ["test", "test2"]  # 不含扩展名，脚本会自动尝试 .jpg .png .jpeg
VIS_OUTPUT_DIR = "/data1/code/dinov2/output/vis"  # 可视化结果保存目录

# DINOv2 使用的图像尺寸
IMG_SIZE = 518
PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_image_path(base_name, directory):
    """根据文件名查找图片（支持多种扩展名）"""
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = Path(directory) / f"{base_name}{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"未找到图片: {base_name} 在 {directory}")


def visualize_patch_pca(model, x, img_path, output_dir, name):
    """Patch 特征 PCA 可视化：将前 3 个主分量映射到 RGB"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  跳过 PCA 可视化: 需要安装 sklearn (pip install scikit-learn)")
        return

    with torch.no_grad():
        features = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
    # features[0]: [1, 768, H, W]
    patch_tokens = features[0]
    B, C, H, W = patch_tokens.shape
    patch_flat = patch_tokens[0].permute(1, 2, 0).reshape(-1, C).numpy()

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(patch_flat)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = np.clip(rgb, 0, 1).reshape(H, W, 3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orig_img = Image.open(img_path).convert("RGB")
    axes[0].imshow(orig_img)
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].set_title("Patch 特征 PCA (前3主分量→RGB)")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = Path(output_dir) / f"{name}_patch_pca.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out_path}")


def visualize_feature_heatmap(model, x, img_path, output_dir, name):
    """特征通道均值热力图"""
    with torch.no_grad():
        features = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
    patch_tokens = features[0]  # [1, 768, H, W]
    feat_map = patch_tokens[0].mean(dim=0).numpy()  # [37, 37]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orig_img = Image.open(img_path).convert("RGB")
    axes[0].imshow(orig_img)
    axes[0].set_title("原图")
    axes[0].axis("off")
    im = axes[1].imshow(feat_map, cmap="viridis")
    axes[1].set_title("特征通道均值热力图")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    out_path = Path(output_dir) / f"{name}_feature_heatmap.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out_path}")


def visualize_cls_patch_similarity(model, x, img_path, output_dir, name):
    """CLS token 与各 patch 的相似度图（注意力式可视化）"""
    with torch.no_grad():
        out = model.forward_features(x)
    cls_token = out["x_norm_clstoken"]  # [1, 768]
    patch_tokens = out["x_norm_patchtokens"]  # [1, N, 768]
    # 计算每个 patch 与 CLS 的余弦相似度
    cls_norm = cls_token / (cls_token.norm(dim=1, keepdim=True) + 1e-8)
    patch_norm = patch_tokens / (patch_tokens.norm(dim=2, keepdim=True) + 1e-8)
    sim = (patch_norm * cls_norm).sum(dim=-1)  # [1, N]
    n_patches = int((IMG_SIZE / PATCH_SIZE) ** 2)
    sim_map = sim[0].reshape(IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    orig_img = Image.open(img_path).convert("RGB")
    axes[0].imshow(orig_img)
    axes[0].set_title("原图")
    axes[0].axis("off")
    im = axes[1].imshow(sim_map, cmap="hot")
    axes[1].set_title("CLS 与 Patch 相似度（越亮越相关）")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    out_path = Path(output_dir) / f"{name}_cls_patch_sim.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out_path}")


def load_weights(model, weights_path):
    """加载权重，支持官方格式和自训练格式（含 teacher 字段）"""
    import dinov2.utils.utils as dinov2_utils
    dinov2_utils.load_pretrained_weights(model, weights_path, checkpoint_key="teacher")


def main():
    # 1. 加载模型
    print("加载模型...")
    model = torch.hub.load(REPO_DIR, "dinov2_vitb14", source="local", pretrained=False)
    load_weights(model, WEIGHTS_PATH)
    model.eval()

    # 2. 预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 3. 对每张图片进行特征可视化
    for name in IMG_NAMES:
        path = find_image_path(name, IMAGE_DIR)
        print(f"\n处理: {name} ({path})")
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0)

        visualize_patch_pca(model, x, path, VIS_OUTPUT_DIR, name)
        visualize_feature_heatmap(model, x, path, VIS_OUTPUT_DIR, name)
        visualize_cls_patch_similarity(model, x, path, VIS_OUTPUT_DIR, name)

    print(f"\n可视化完成，结果保存在: {VIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
