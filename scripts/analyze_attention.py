#!/usr/bin/env python3
"""
分析 DINOv2 不同层的 Attention Map
计算每层的平均 attention distance，判断局部/全局特征倾向
支持按 head 分析，提供详细的统计信息
"""

import argparse
import sys
from pathlib import Path
import csv

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


def compute_attention_distance(attn: torch.Tensor, patch_size: int, return_stats: bool = False):
    """
    计算平均 attention distance（像素）
    
    Args:
        attn: attention weights, shape (B, num_heads, N, N) 或 (B, num_heads, N, N) 的平均
        patch_size: patch 大小（像素）
        return_stats: 是否返回详细统计信息
    
    Returns:
        如果 return_stats=False: 返回平均距离（float）
        如果 return_stats=True: 返回字典，包含平均距离、标准差、分位数等
    """
    # 处理输入：如果是 4D (B, H, N, N)，取 batch 和 head 的平均
    if attn.dim() == 4:
        attn = attn.mean(dim=0).mean(dim=0)  # (N, N)
    elif attn.dim() == 3:
        attn = attn.mean(dim=0)  # (N, N)
    
    n = attn.shape[0]
    h = w = int(n ** 0.5)
    assert h * w == n

    y, x = torch.meshgrid(torch.arange(h, device=attn.device).float(), torch.arange(w, device=attn.device).float(), indexing="ij")
    y, x = y.reshape(-1), x.reshape(-1)
    dist = torch.sqrt((y.unsqueeze(0) - y.unsqueeze(1)).pow(2) + (x.unsqueeze(0) - x.unsqueeze(1)).pow(2))
    
    # 加权平均距离（patch 单位）
    weighted_dist = (attn * dist).sum(dim=-1)  # (N,) 每个 query patch 的平均距离
    
    mean_dist = weighted_dist.mean().item() * patch_size
    std_dist = weighted_dist.std().item() * patch_size
    
    if not return_stats:
        return mean_dist
    
    # 计算分位数和局部/全局比例
    dist_pixels = weighted_dist.cpu().numpy() * patch_size
    median_dist = np.median(dist_pixels)
    p25 = np.percentile(dist_pixels, 25)
    p75 = np.percentile(dist_pixels, 75)
    
    # 局部/全局阈值：使用中位数或平均值的某个比例
    threshold = median_dist
    local_ratio = (dist_pixels < threshold).mean()
    
    return {
        "mean": mean_dist,
        "std": std_dist,
        "median": median_dist,
        "p25": p25,
        "p75": p75,
        "local_ratio": local_ratio,
        "global_ratio": 1 - local_ratio,
    }


def compute_attention_distance_per_head(attn: torch.Tensor, patch_size: int):
    """
    按 head 计算 attention distance
    
    Args:
        attn: attention weights, shape (B, num_heads, N, N)
        patch_size: patch 大小
    
    Returns:
        list of dict: 每个 head 的统计信息
    """
    B, num_heads, N, _ = attn.shape
    h = w = int(N ** 0.5)
    
    y, x = torch.meshgrid(torch.arange(h, device=attn.device).float(), torch.arange(w, device=attn.device).float(), indexing="ij")
    y, x = y.reshape(-1), x.reshape(-1)
    dist = torch.sqrt((y.unsqueeze(0) - y.unsqueeze(1)).pow(2) + (x.unsqueeze(0) - x.unsqueeze(1)).pow(2))
    
    head_stats = []
    for h_idx in range(num_heads):
        attn_head = attn[:, h_idx, :, :].mean(dim=0)  # (N, N) 平均 batch
        weighted_dist = (attn_head * dist).sum(dim=-1)  # (N,)
        mean_dist = weighted_dist.mean().item() * patch_size
        
        dist_pixels = weighted_dist.cpu().numpy() * patch_size
        median_dist = np.median(dist_pixels)
        threshold = median_dist
        local_ratio = (dist_pixels < threshold).mean()
        
        head_stats.append({
            "head": h_idx,
            "mean": mean_dist,
            "median": median_dist,
            "local_ratio": local_ratio,
        })
    
    return head_stats


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
    
    def get_num_heads(self) -> int:
        """获取模型的 head 数量"""
        blocks = _get_blocks(self.model)
        if len(blocks) > 0:
            return blocks[0].attn.num_heads
        return 0


def run_analysis(results: dict, output_path: Path, csv_path: Path = None, per_head_stats: list = None) -> None:
    """
    打印分析并保存可视化
    
    Args:
        results: 包含 layer_indices 和 layer_distances 的字典
        output_path: 可视化图片保存路径
        csv_path: CSV 结果保存路径（可选）
        per_head_stats: 每层每个 head 的统计信息（可选）
    """
    dists = results["layer_distances"]
    layer_stats = results.get("layer_stats", [])
    
    # 使用中位数作为阈值（更稳健）
    threshold = np.median(dists)

    print("\n" + "=" * 70)
    print("Attention Distance 分析结果 - 每层局部/全局特征分析")
    print("=" * 70)
    print(f"统计信息:")
    print(f"  最小/最大/平均/中位数: {min(dists):.2f} / {max(dists):.2f} / {np.mean(dists):.2f} / {np.median(dists):.2f} pixels")
    print(f"  标准差: {np.std(dists):.2f} pixels")
    print(f"  阈值（中位数）: {threshold:.2f} pixels")
    
    print(f"\n{'Layer':<8} {'Mean Dist':<12} {'Std':<10} {'Median':<10} {'Local%':<10} {'Type':<10}")
    print("-" * 70)
    
    csv_rows = []
    for idx, d in zip(results["layer_indices"], dists):
        layer_type = 'Local' if d < threshold else 'Global'
        if layer_stats and idx < len(layer_stats):
            stats = layer_stats[idx]
            local_pct = stats.get("local_ratio", 0) * 100
            std_val = stats.get("std", 0)
            median_val = stats.get("median", d)
            print(f"{idx:<8} {d:<12.2f} {std_val:<10.2f} {median_val:<10.2f} {local_pct:<10.1f} {layer_type:<10}")
            csv_rows.append({
                "layer": idx,
                "mean_distance": f"{d:.4f}",
                "std_distance": f"{std_val:.4f}",
                "median_distance": f"{median_val:.4f}",
                "p25": f"{stats.get('p25', 0):.4f}",
                "p75": f"{stats.get('p75', 0):.4f}",
                "local_ratio": f"{stats.get('local_ratio', 0):.4f}",
                "global_ratio": f"{stats.get('global_ratio', 0):.4f}",
                "type": layer_type,
            })
        else:
            print(f"{idx:<8} {d:<12.2f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {layer_type:<10}")
            csv_rows.append({
                "layer": idx,
                "mean_distance": f"{d:.4f}",
                "std_distance": "N/A",
                "median_distance": "N/A",
                "p25": "N/A",
                "p75": "N/A",
                "local_ratio": "N/A",
                "global_ratio": "N/A",
                "type": layer_type,
            })
    
    print("=" * 70)
    
    # 保存 CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n详细结果已保存到 CSV: {csv_path}")
    
    # 可视化
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 子图1: 每层平均距离
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results["layer_indices"], dists, "o-", lw=2, color="#2E86AB", markersize=8)
    for i, d in enumerate(dists):
        ax1.scatter(i, d, color="#FF6B6B" if d < threshold else "#4ECDC4", s=120, zorder=5, edgecolors='black', linewidth=1)
    ax1.axhline(threshold, color="gray", ls="--", alpha=0.7, linewidth=2, label=f'Threshold ({threshold:.1f}px)')
    ax1.set_xlabel("Layer Index", fontsize=11)
    ax1.set_ylabel("Mean Attention Distance (pixels)", fontsize=11)
    ax1.set_title("Attention Distance Across Layers", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(labelsize=10)
    
    # 子图2: 局部/全局层数统计
    ax2 = fig.add_subplot(gs[0, 1])
    local_count = sum(1 for d in dists if d < threshold)
    bars = ax2.bar(["Local\nLayers", "Global\nLayers"], 
                   [local_count, len(dists) - local_count], 
                   color=["#FF6B6B", "#4ECDC4"], alpha=0.8, edgecolor='black', linewidth=1.5)
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, 
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Number of Layers", fontsize=11)
    ax2.set_title("Local vs Global Layer Count", fontsize=11)
    ax2.set_ylim(0, max(local_count, len(dists) - local_count) * 1.2)
    ax2.tick_params(labelsize=10)
    
    # 子图3: 距离分布箱线图（如果有详细统计）
    ax3 = fig.add_subplot(gs[1, 0])
    if layer_stats:
        medians = [s.get("median", d) for s, d in zip(layer_stats, dists)]
        p25s = [s.get("p25", d) for s, d in zip(layer_stats, dists)]
        p75s = [s.get("p75", d) for s, d in zip(layer_stats, dists)]
        
        positions = results["layer_indices"]
        ax3.vlines(positions, p25s, p75s, colors="#2E86AB", linewidth=2, alpha=0.7)
        ax3.scatter(positions, medians, color="#FF6B6B", s=100, zorder=5, label='Median', edgecolors='black', linewidth=1)
        ax3.scatter(positions, dists, color="#4ECDC4", s=60, zorder=4, marker='x', label='Mean', linewidth=2)
        ax3.axhline(threshold, color="gray", ls="--", alpha=0.5)
        ax3.set_xlabel("Layer Index", fontsize=11)
        ax3.set_ylabel("Attention Distance (pixels)", fontsize=11)
        ax3.set_title("Distance Distribution (Median, 25%/75% Percentiles)", fontsize=11)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=10)
    else:
        ax3.text(0.5, 0.5, "No detailed statistics", ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Distance Distribution", fontsize=11)
    
    # 子图4: 每层局部比例（如果有详细统计）
    ax4 = fig.add_subplot(gs[1, 1])
    if layer_stats:
        local_ratios = [s.get("local_ratio", 0.5) * 100 for s in layer_stats]
        ax4.bar(results["layer_indices"], local_ratios, color="#FF6B6B", alpha=0.7, edgecolor='black', linewidth=1)
        ax4.axhline(50, color="gray", ls="--", alpha=0.5, label='50% Threshold')
        ax4.set_xlabel("Layer Index", fontsize=11)
        ax4.set_ylabel("Local Attention Ratio (%)", fontsize=11)
        ax4.set_title("Local Attention Ratio per Layer", fontsize=11)
        ax4.set_ylim(0, 100)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(labelsize=10)
    else:
        ax4.text(0.5, 0.5, "No detailed statistics", ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title("Local Attention Ratio per Layer", fontsize=11)
    
    plt.suptitle("DINOv2 Attention Local/Global Feature Analysis", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n可视化已保存: {output_path}")
    plt.close()
    
    # 如果有 per-head 统计，打印摘要
    if per_head_stats:
        print("\n" + "=" * 70)
        print("按 Head 分析摘要（前3层示例）")
        print("=" * 70)
        for layer_idx in range(min(3, len(per_head_stats))):
            if per_head_stats[layer_idx]:
                print(f"\n层 {layer_idx}:")
                print(f"  {'Head':<8} {'Mean Dist':<12} {'Median':<10} {'Local%':<10}")
                print("  " + "-" * 40)
                for h_stat in per_head_stats[layer_idx][:min(8, len(per_head_stats[layer_idx]))]:  # 最多显示8个head
                    print(f"  {h_stat['head']:<8} {h_stat['mean']:<12.2f} {h_stat['median']:<10.2f} {h_stat['local_ratio']*100:<10.1f}")
                if len(per_head_stats[layer_idx]) > 8:
                    print(f"  ... (共 {len(per_head_stats[layer_idx])} 个 heads)")


def main():
    parser = argparse.ArgumentParser(description="分析 DINOv2 各层 Attention Map - 判断局部/全局特征倾向")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--weights", default=WEIGHTS_PATH, help="模型权重路径")
    parser.add_argument("--arch", default="vit_base", choices=list(ARCH_MAP))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--output", default=None, help="输出图片路径")
    parser.add_argument("--csv", default=None, help="输出 CSV 结果路径（可选）")
    parser.add_argument("--detailed", action="store_true", help="启用详细统计（包括标准差、分位数等）")
    parser.add_argument("--per-head", action="store_true", help="按 head 分析（会显示每层每个 head 的统计）")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_attention_analysis.png"
    
    csv_path = Path(args.csv) if args.csv else (output_path.parent / f"{output_path.stem}.csv" if args.detailed else None)

    print("=" * 70)
    print("DINOv2 Attention 局部/全局特征分析")
    print("=" * 70)
    print(f"模型架构: {args.arch}")
    print(f"图像路径: {image_path}")
    print(f"权重路径: {args.weights}")
    print(f"图像尺寸: {args.img_size}x{args.img_size}")
    print(f"Patch 大小: {args.patch_size}")
    print(f"详细统计: {'启用' if args.detailed else '禁用'}")
    print(f"按 Head 分析: {'启用' if args.per_head else '禁用'}")
    print("=" * 70)

    # 加载模型
    print("\n正在加载模型...")
    model = ARCH_MAP[args.arch](patch_size=args.patch_size, img_size=args.img_size)
    load_weights(model, args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device).eval()

    # 提取 attention
    print("\n正在提取各层 attention weights...")
    extractor = AttentionExtractor(model)
    extractor.register_hooks()
    image = preprocess_image(image_path, args.img_size, device)
    attention_weights = extractor.extract(image)
    extractor.remove_hooks()
    print(f"成功提取 {len(attention_weights)} 层的 attention weights")

    # 计算距离
    print("\n正在计算 attention distance...")
    layer_distances = []
    layer_stats = []
    per_head_stats = []
    
    for layer_idx, attn in enumerate(attention_weights):
        if attn is None:
            print(f"警告: 层 {layer_idx} 的 attention 为空，跳过")
            continue
        
        # 基本距离计算
        mean_dist = compute_attention_distance(attn, args.patch_size, return_stats=False)
        layer_distances.append(mean_dist)
        
        # 详细统计
        if args.detailed:
            stats = compute_attention_distance(attn, args.patch_size, return_stats=True)
            layer_stats.append(stats)
        
        # 按 head 分析
        if args.per_head:
            head_stats = compute_attention_distance_per_head(attn, args.patch_size)
            per_head_stats.append(head_stats)
        else:
            per_head_stats.append(None)
        
        if (layer_idx + 1) % 5 == 0:
            print(f"  已处理 {layer_idx + 1}/{len(attention_weights)} 层")

    results = {
        "layer_indices": list(range(len(layer_distances))),
        "layer_distances": layer_distances,
        "layer_stats": layer_stats if args.detailed else None,
    }
    
    # 运行分析并保存结果
    run_analysis(results, output_path, csv_path, per_head_stats if args.per_head else None)
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

# python  scripts/analyze_attention.py  --image output/test2.jpg --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth --output output/attention.png