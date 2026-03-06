#!/usr/bin/env python3
"""
分析 DINOv2 不同层的 Attention Map
计算每层的平均 attention distance，判断局部/全局特征倾向
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from dinov2.models.vision_transformer import DinoVisionTransformer, vit_small, vit_base, vit_large, vit_giant2

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

WEIGHTS_PATH = "output/jinxiang/eval/training_87499/teacher_checkpoint.pth"
ARCH_MAP = {"vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large, "vit_giant2": vit_giant2}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_blocks(model):
    """获取 transformer blocks（兼容 chunked/普通结构）"""
    if getattr(model, "chunked_blocks", False):
        return [b for chunk in model.blocks for b in chunk if not isinstance(b, torch.nn.Identity)]
    return list(model.blocks)


def load_weights(model: torch.nn.Module, weights_path: str) -> None:
    """加载权重，兼容官方/自定义 checkpoint，自动插值 pos_embed"""
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = ckpt.get("teacher", ckpt)
    new_state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    if "pos_embed" in new_state_dict and hasattr(model, "pos_embed"):
        pos_ckpt, pos_model = new_state_dict["pos_embed"], model.pos_embed
        if pos_ckpt.shape != pos_model.shape:
            print(f"pos_embed 插值: {pos_ckpt.shape} -> {pos_model.shape}")
            cls_pos = pos_ckpt[:, :1, :]
            patch = pos_ckpt[:, 1:, :]
            h_ckpt = w_ckpt = int(patch.shape[1] ** 0.5)
            h_model = w_model = int((pos_model.shape[1] - 1) ** 0.5)
            patch = patch.reshape(1, h_ckpt, w_ckpt, -1).permute(0, 3, 1, 2)
            patch = F.interpolate(patch, (h_model, w_model), mode="bicubic", align_corners=False)
            patch = patch.permute(0, 2, 3, 1).reshape(1, h_model * w_model, -1)
            new_state_dict["pos_embed"] = torch.cat([cls_pos, patch], dim=1)

    print(f"load_state_dict: {model.load_state_dict(new_state_dict, strict=False)}")


def preprocess_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """加载并预处理图像"""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    img = torch.from_numpy(np.array(img)).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = (img - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return img.to(device)


def compute_attention_distance(attn: torch.Tensor, patch_size: int) -> float:
    """计算平均 attention distance（像素）"""
    n = attn.shape[2]
    h = w = int(n ** 0.5)
    assert h * w == n

    y, x = torch.meshgrid(torch.arange(h, device=attn.device).float(), torch.arange(w, device=attn.device).float(), indexing="ij")
    y, x = y.reshape(-1), x.reshape(-1)
    dist = torch.sqrt((y.unsqueeze(0) - y.unsqueeze(1)).pow(2) + (x.unsqueeze(0) - x.unsqueeze(1)).pow(2))
    return (attn * dist.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean().item() * patch_size


class AttentionExtractor:
    """提取各层 attention weights"""

    def __init__(self, model: DinoVisionTransformer):
        self.model = model
        self.attention_weights = []
        self.hooks = []

    def register_hooks(self):
        blocks = _get_blocks(self.model)
        self.attention_weights = [None] * len(blocks)

        def make_hook(i):
            def hook(module, inp, _):
                B, N, C = inp[0].shape
                qkv = module.qkv(inp[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
                q, k, v = qkv.unbind(2)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                attn = (q @ k.transpose(-2, -1)) * module.scale
                self.attention_weights[i] = attn.softmax(dim=-1)[:, :, 1:, 1:].detach()
            return hook

        for i, blk in enumerate(blocks):
            self.hooks.append(blk.attn.register_forward_hook(make_hook(i)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract(self, image: torch.Tensor) -> list:
        with torch.no_grad():
            _ = self.model(image)
        return self.attention_weights


def run_analysis(results: dict, output_path: Path) -> None:
    """打印分析并保存可视化"""
    dists = results["layer_distances"]
    threshold = (max(dists) + min(dists)) / 2

    print("\n" + "=" * 50)
    print("Attention Distance 分析结果")
    print("=" * 50)
    print(f"最小/最大/平均: {min(dists):.2f} / {max(dists):.2f} / {np.mean(dists):.2f} pixels")
    print(f"阈值: {threshold:.2f} pixels")
    print(f"\n{'Layer':<8} {'Distance':<10} {'Type'}")
    print("-" * 30)
    for idx, d in zip(results["layer_indices"], dists):
        print(f"{idx:<8} {d:<10.2f} {'Local' if d < threshold else 'Global'}")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(results["layer_indices"], dists, "o-", lw=2, color="#2E86AB")
    for i, d in enumerate(dists):
        ax1.scatter(i, d, color="#FF6B6B" if d < threshold else "#4ECDC4", s=100, zorder=5)
    ax1.axhline(threshold, color="gray", ls="--", alpha=0.5)
    ax1.set(xlabel="Layer", ylabel="Attention Distance (px)", title="Attention Distance Across Layers")
    ax1.grid(True, alpha=0.3)

    local_count = sum(1 for d in dists if d < threshold)
    bars = ax2.bar(["Local", "Global"], [local_count, len(dists) - local_count], color=["#FF6B6B", "#4ECDC4"], alpha=0.8)
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}", ha="center", va="bottom")
    ax2.set(ylabel="Layers", title="Local vs Global")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n可视化已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="分析 DINOv2 各层 Attention Map")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="模型权重路径")
    parser.add_argument("--arch", default="vit_base", choices=list(ARCH_MAP))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--output", default=None, help="输出图片路径")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    output_path = Path(args.output) if args.output else image_path.parent / f"{image_path.stem}_attention_analysis.png"

    model = ARCH_MAP[args.arch](patch_size=args.patch_size, img_size=args.img_size)
    load_weights(model, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    image = preprocess_image(image_path, args.img_size, device)
    attention_weights = extractor.extract(image)
    extractor.remove_hooks()

    results = {
        "layer_indices": list(range(len(attention_weights))),
        "layer_distances": [compute_attention_distance(attn, args.patch_size) for attn in attention_weights],
    }
    run_analysis(results, output_path)


if __name__ == "__main__":
    main()

# python  scripts/analyze_attention.py  --image output/test2.jpg --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth --output output/attention.png