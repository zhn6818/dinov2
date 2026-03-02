#!/usr/bin/env python3
"""
对 features.npy 做 t-SNE/UMAP 降维可视化，检查同类是否聚在一起。

用法:
    # 无标签（所有点同色，看自然聚类）
    python scripts/visualize_tsne_umap.py --features features.npy --output viz.png

    # 有标签（按类别着色）
    python scripts/visualize_tsne_umap.py --features features.npy --labels labels.txt --output viz.png

    # 从 ImageFolder 目录推断标签（目录结构: root/类别名/图片.jpg）
    python scripts/visualize_tsne_umap.py --features features.npy --image-dir /path/to/images/ --output viz.png

labels.txt 格式：每行一个标签，顺序与特征一一对应。
"""
import argparse
import os
import sys

import numpy as np

# 项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_labels_from_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_labels_from_image_folder(root, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    """从 ImageFolder 结构 (root/类别名/图片.jpg) 按字母序收集路径和标签"""
    paths, labels = [], []
    for class_name in sorted(os.listdir(root)):
        class_path = os.path.join(root, class_name)
        if not os.path.isdir(class_path):
            continue
        for f in sorted(os.listdir(class_path)):
            if f.lower().endswith(exts):
                paths.append(os.path.join(class_path, f))
                labels.append(class_name)
    return paths, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, required=True, help="features.npy 路径")
    parser.add_argument("--labels", type=str, help="标签文件路径，每行一个标签")
    parser.add_argument(
        "--image-dir",
        type=str,
        help="ImageFolder 目录 (root/类别/图片.jpg)，用于推断标签。注意：特征需按相同顺序提取",
    )
    parser.add_argument("--output", type=str, default="tsne_umap.png", help="输出图像路径")
    parser.add_argument("--method", choices=["tsne", "umap", "both"], default="both")
    parser.add_argument("--sample", type=int, default=0, help="抽样数量，0 表示不抽样（大数据集可设 2000 加速）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = np.load(args.features)
    if features.ndim != 2:
        print(f"错误: 特征应为 2 维 (N, D)，当前 shape={features.shape}")
        sys.exit(1)
    n, dim = features.shape
    print(f"加载特征: {n} 样本, {dim} 维")

    labels = None
    if args.labels:
        labels = load_labels_from_file(args.labels)
        if len(labels) != n:
            print(f"错误: 标签数 {len(labels)} 与特征数 {n} 不一致")
            sys.exit(1)
        print(f"加载标签: {len(set(labels))} 类")
    elif args.image_dir:
        paths, labels = load_labels_from_image_folder(args.image_dir)
        if len(paths) != n:
            print(
                f"警告: 目录下图像数 {len(paths)} 与特征数 {n} 不一致，"
                "请确保特征是用 extract_features --image-folder 按该目录顺序提取的"
            )
        else:
            print(f"从目录推断标签: {len(set(labels))} 类")
    else:
        # 自动查找同目录的 _labels.txt（由 extract_features --image-folder 生成）
        auto_labels = args.features.replace(".npy", "_labels.txt")
        if os.path.exists(auto_labels):
            labels = load_labels_from_file(auto_labels)
            if len(labels) == n:
                print(f"自动加载标签: {auto_labels}, {len(set(labels))} 类")
            else:
                labels = None

    if args.sample > 0 and n > args.sample:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n, args.sample, replace=False)
        features = features[idx]
        if labels is not None:
            labels = [labels[i] for i in idx]
        print(f"抽样 {args.sample} 个点进行可视化")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("请安装 matplotlib: pip install matplotlib")
        sys.exit(1)

    if args.method in ("tsne", "both"):
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("请安装 scikit-learn: pip install scikit-learn")
            sys.exit(1)
        print("运行 t-SNE...")
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, len(features) - 1))
        coords_tsne = tsne.fit_transform(features)

    if args.method in ("umap", "both"):
        try:
            import umap
        except ImportError:
            print("请安装 umap-learn: pip install umap-learn")
            sys.exit(1)
        print("运行 UMAP...")
        reducer = umap.UMAP(random_state=args.seed, n_neighbors=min(15, len(features) - 1))
        coords_umap = reducer.fit_transform(features)

    def plot(coords, title, ax):
        if labels is not None:
            unique_labels = sorted(set(labels))
            n_classes = len(unique_labels)
            cmap = plt.cm.get_cmap("tab20" if n_classes <= 20 else "tab20b", max(n_classes, 1))
            for i, lab in enumerate(unique_labels):
                mask = np.array([l == lab for l in labels])
                color = cmap(i % 20) if n_classes > 20 else cmap(i)
                n_pts = mask.sum()
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=[color] * n_pts,
                    label=lab,
                    alpha=0.6,
                    s=10,
                )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=6)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=10, c="steelblue")
        ax.set_title(title)

    if args.method == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot(coords_tsne, "t-SNE", axes[0])
        plot(coords_umap, "UMAP", axes[1])
    elif args.method == "tsne":
        fig, ax = plt.subplots(figsize=(8, 6))
        plot(coords_tsne, "t-SNE", ax)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot(coords_umap, "UMAP", ax)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"已保存到 {args.output}")


if __name__ == "__main__":
    main()
