# DINOv2 垂直领域微调模型使用与验证指南

本文档说明：训练产出的微调模型能做什么、如何提取权重、以及如何用**你自己的数据**验证微调是否成功。

---

## 一、微调模型能做什么

基于 `vitb14_finetune.yaml` 训练得到的 DINOv2 ViT-B/14 微调模型，可用于：

| 用途 | 说明 |
|------|------|
| **图像特征提取** | 作为 backbone 提取视觉特征，用于下游任务 |
| **图像分类** | 在特征上接线性层或 k-NN 做分类 |
| **检索 / 聚类** | 用特征做相似度检索、聚类 |
| **语义分割** | 作为分割 backbone（需额外 head） |
| **深度估计** | 作为深度估计 backbone（需额外 head） |

---

## 二、从训练 checkpoint 提取 teacher 权重

训练产出的 `model_XXXXX.rank_0.pth` 是完整训练状态（含 student、teacher、optimizer），eval 和推理需要单独的 teacher 权重。

### 命令

```bash
conda activate ai  # 或你的 conda 环境名
cd /data1/code/dinov2

python scripts/extract_teacher_from_checkpoint.py \
    --input output/vitb14_finetune2/model_0000749.rank_0.pth \
    --output output/vitb14_finetune2/teacher_checkpoint_0749.pth
```

### 输出

- `teacher_checkpoint_0749.pth`：eval 和推理可直接使用的 teacher 权重

---

## 三、用你自己的数据验证微调是否成功

你的数据是垂直领域，不需要在 ImageNet 上测通用能力，**只要在你自己的数据上表现正常即可**。

### 方式 1：特征提取 + 定性检查（无标注也可用）

加载模型，对训练/验证图像提取特征，做以下检查：

- **检索一致性**：同一类/同一场景的图像，特征相似度应较高
- **聚类效果**：对特征做聚类，看是否与业务语义一致
- **可视化**：用 t-SNE/UMAP 降维，看同类是否聚在一起

### 方式 2：有标注数据时做 k-NN / Linear Probe

若你有**带类别标签**的数据，可按 ImageFolder 组织：

```
你的数据根目录/
├── 类别A/
│   ├── img1.jpg
│   └── img2.jpg
├── 类别B/
│   └── ...
└── ...
```

然后使用 k-NN 或 linear probe 评估准确率，对比**微调前**和**微调后**在同一数据上的表现，判断微调是否有效。

---

## 四、提取特征验证微调效果

使用 `scripts/extract_features.py` 对你的图像提取特征，用于检索、聚类或可视化。

### 单张图像

```bash
conda activate ai
cd /data1/code/dinov2

python scripts/extract_features.py \
    --checkpoint output/vitb14_finetune2/teacher_checkpoint_11249.pth \
    --config dinov2/configs/eval/vitb14_finetune.yaml \
    --image 你的图像路径.jpg \
    --output features.npy
```

### 目录下所有图像

```bash
python scripts/extract_features.py \
    --checkpoint output/vitb14_finetune2/teacher_checkpoint_11249.pth \
    --config dinov2/configs/eval/vitb14_finetune.yaml \
    --image-dir /data1/zhn/tt/ \
    --output features.npy
```

### 目录下各个文件夹
```bash
python scripts/extract_features.py \
    --checkpoint output/vitb14_finetune2/teacher_checkpoint_11249.pth \
    --image-dir /data1/zhn/tt/ \
    --image-folder \
    --output features.npy
```

输出 `features.npy` 为 `(N, 768)` 的 numpy 数组，可用于余弦相似度检索、k-means 聚类、t-SNE/UMAP 可视化等。

### t-SNE/UMAP 可视化（看同类是否聚在一起）

```bash
# 依赖: pip install matplotlib scikit-learn umap-learn

# 无标签（所有点同色，看自然聚类）
python scripts/visualize_tsne_umap.py --features features.npy --output viz.png

# 有标签（按类别着色）：需 labels.txt，每行一个标签，顺序与特征对应
python scripts/visualize_tsne_umap.py --features features.npy --labels labels.txt --output viz.png

# 若用 extract_features --image-folder 提取，会生成 features_labels.txt，可自动识别
python scripts/visualize_tsne_umap.py --features features.npy --output viz.png

# 大数据集可抽样加速（如 500 点）
python scripts/visualize_tsne_umap.py --features features.npy --output viz.png --sample 500
```

---

## 五、相关文件

| 文件 | 说明 |
|------|------|
| `dinov2/configs/train/vitb14_finetune.yaml` | 微调训练配置 |
| `dinov2/configs/eval/vitb14_finetune.yaml` | 评估配置（crop_size=224） |
| `scripts/finetune_vitb14.sh` | 微调启动脚本 |
| `scripts/extract_teacher_from_checkpoint.py` | 从训练 checkpoint 提取 teacher |
| `scripts/extract_features.py` | 对图像提取特征（用于检索、聚类、可视化） |
| `scripts/visualize_tsne_umap.py` | t-SNE/UMAP 降维可视化，检查同类是否聚在一起 |

---

## 六、微调成功的判断标准（针对你的数据）

- **无标注**：检索、聚类、可视化结果符合业务预期
- **有标注**：k-NN / linear probe 准确率高于微调前，或达到预期水平

无需在 ImageNet 上评估，只要在你自己的垂直领域数据上表现满足需求即可。
