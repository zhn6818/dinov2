# Attention 局部/全局特征分析脚本使用说明

## 功能概述

`analyze_attention.py` 脚本用于分析 DINOv2 模型每一层的 attention 机制，判断各层是更偏向**局部特征提取**还是**全局特征整合**。

### 核心指标：Attention Distance（Attention 距离）

脚本通过计算**加权平均 attention 距离**来衡量每一层的局部/全局倾向：

- **距离小** → 该层更关注**邻近的 patch** → **局部特征提取**
- **距离大** → 该层更关注**远处的 patch** → **全局特征整合**

### 计算原理

1. **提取 Attention Weights**：
   - 在每层 transformer block 的 attention 模块上注册 forward hook
   - 手动计算 Q、K 的 scaled dot-product attention
   - 只保留 **patch-patch** 之间的 attention（去掉 CLS token）

2. **计算加权平均距离**：
   - 在 patch 网格上建立 2D 坐标
   - 计算任意两个 patch 之间的欧氏距离矩阵
   - 用 attention 权重对距离做加权平均
   - 乘以 `patch_size` 得到**像素单位的平均距离**

3. **判断局部/全局**：
   - 使用中位数作为阈值（比平均值更稳健）
   - 距离 < 阈值 → Local（局部层）
   - 距离 ≥ 阈值 → Global（全局层）

---

## 使用方法

### 基本用法

```bash
python scripts/analyze_attention.py \
    --image output/test2.jpg \
    --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth \
    --arch vit_base \
    --img-size 224 \
    --patch-size 14
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|---------|------|
| `--image` | ✅ | - | 输入图像路径 |
| `--weights` | ❌ | `output/jinxiang/eval/training_87499/teacher_checkpoint.pth` | 模型权重路径 |
| `--arch` | ❌ | `vit_base` | 模型架构：`vit_small`, `vit_base`, `vit_large`, `vit_giant2` |
| `--img-size` | ❌ | `224` | 输入图像尺寸（像素） |
| `--patch-size` | ❌ | `14` | Patch 大小（像素） |
| `--output` | ❌ | `{图像名}_attention_analysis.png` | 输出可视化图片路径 |
| `--csv` | ❌ | `{图像名}_attention_analysis.csv` | 输出 CSV 结果路径（需配合 `--detailed`） |
| `--detailed` | ❌ | `False` | 启用详细统计（标准差、分位数、局部/全局比例） |
| `--per-head` | ❌ | `False` | 按 head 分析（显示每层每个 attention head 的统计） |

### 高级用法示例

#### 1. 启用详细统计并保存 CSV

```bash
python scripts/analyze_attention.py \
    --image output/test2.jpg \
    --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth \
    --detailed \
    --csv output/attention_results.csv \
    --output output/attention_analysis.png
```

**输出内容**：
- 控制台：每层的平均距离、标准差、中位数、25%/75% 分位数、局部/全局比例
- CSV 文件：包含所有层的详细统计信息
- 可视化图片：4 个子图（距离趋势、层数统计、距离分布、局部比例）

#### 2. 按 Head 分析

```bash
python scripts/analyze_attention.py \
    --image output/test2.jpg \
    --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth \
    --per-head
```

**输出内容**：
- 控制台：显示前 3 层每个 head 的平均距离、中位数、局部比例
- 可视化图片：标准分析图
- 可用于发现不同 head 的注意力模式差异

#### 3. 完整分析（详细统计 + 按 Head）

```bash
python scripts/analyze_attention.py \
    --image output/test2.jpg \
    --weights output/jinxiang/eval/training_87499/teacher_checkpoint.pth \
    --detailed \
    --per-head \
    --csv output/full_analysis.csv \
    --output output/full_analysis.png
```

---

## 输出说明

### 控制台输出

#### 基本模式

```
======================================================================
Attention Distance 分析结果 - 每层局部/全局特征分析
======================================================================
统计信息:
  最小/最大/平均/中位数: 45.23 / 156.78 / 98.45 / 95.12 pixels
  标准差: 28.34 pixels
  阈值（中位数）: 95.12 pixels

Layer    Mean Dist    Std        Median      Local%     Type
----------------------------------------------------------------------
0        45.23      12.34      42.10      85.2      Local
1        52.11      15.67      48.90      78.5      Local
...
12       145.67     28.90      142.30     25.3      Global
======================================================================
```

#### 详细模式（`--detailed`）

- 显示每层的**标准差**、**中位数**、**25%/75% 分位数**
- 显示每层的**局部/全局比例**（基于 patch 级别的统计）

#### 按 Head 模式（`--per-head`）

```
======================================================================
按 Head 分析摘要（前3层示例）
======================================================================

层 0:
  Head     Mean Dist     Median      Local%
  ----------------------------------------
  0        42.10      38.50      88.5
  1        48.20      45.10      82.3
层 1:
  Head     Mean Dist     Median      Local%
  ----------------------------------------
  0        55.30      52.10      75.2
  ...
```

### 可视化图片

脚本会生成包含 **4 个子图** 的综合分析图：

1. **各层 Attention 距离变化趋势**（左上）
   - 折线图显示每层平均距离
   - 散点颜色：红色=局部层，绿色=全局层
   - 灰色虚线：阈值（中位数）

2. **局部 vs 全局层数统计**（右上）
   - 柱状图显示局部层和全局层的数量

3. **各层距离分布**（左下，需 `--detailed`）
   - 箱线图显示中位数、25%/75% 分位数
   - 平均值用 X 标记

4. **各层局部 Attention 占比**（右下，需 `--detailed`）
   - 柱状图显示每层局部 attention 的百分比

### CSV 文件格式（需 `--detailed`）

| 列名 | 说明 |
|--------|------|
| `layer` | 层索引（从 0 开始） |
| `mean_distance` | 平均 attention 距离（像素） |
| `std_distance` | 距离标准差（像素） |
| `median_distance` | 距离中位数（像素） |
| `p25` | 25% 分位数（像素） |
| `p75` | 75% 分位数（像素） |
| `local_ratio` | 局部 attention 比例（0-1） |
| `global_ratio` | 全局 attention 比例（0-1） |
| `type` | 分类：`Local` 或 `Global` |

---

## 分析结果解读

### 典型模式

#### 1. 标准 ViT 模式（浅层局部 → 深层全局）

```
层 0-3:   Local  (距离: 40-60 px)
层 4-8:   Transition (距离: 60-100 px)
层 9-12:  Global (距离: 100-150 px)
```

**解读**：
- 浅层关注局部细节（边缘、纹理）
- 深层整合全局信息（物体形状、空间关系）
- 符合 ViT 的层级设计理念

#### 2. 全局部模式（所有层都局部）

```
所有层: Local (距离: 30-50 px)
```

**可能原因**：
- 模型训练不充分
- 图像内容简单（不需要全局整合）
- 模型架构特殊（如使用了局部 attention 机制）

#### 3. 全全局模式（所有层都全局）

```
所有层: Global (距离: 120-180 px)
```

**可能原因**：
- 图像内容复杂（需要大量全局信息）
- 模型设计偏向全局特征（如使用了全局 attention）

### 使用建议

1. **对比不同模型**：
   - 用相同图像分析不同 checkpoint
   - 对比训练前后的变化
   - 对比不同架构（vit_small vs vit_large）

2. **分析训练过程**：
   - 定期保存 checkpoint 并分析
   - 观察浅层/深层的变化趋势
   - 判断模型是否收敛

3. **诊断问题**：
   - 如果所有层都是局部 → 可能欠拟合
   - 如果浅层就是全局 → 可能过拟合或架构问题
   - 如果距离波动很大 → 可能训练不稳定

4. **按 Head 分析**：
   - 发现不同 head 的专门化（有的关注局部、有的关注全局）
   - 判断 head 数量是否合适
   - 分析 head 的多样性

---

## 技术细节

### Attention 提取机制

- 使用 **forward hook** 在 attention 模块的 forward 之前拦截输入
- 手动计算 attention：`attn = (Q @ K^T) * scale`
- 只保留 patch-patch attention：`attn[:, :, 1:, 1:]`（去掉 CLS token）

### 距离计算

- Patch 网格坐标：`(y, x) = meshgrid(0..h-1, 0..w-1)`
- 欧氏距离：`dist = sqrt((y_i - y_j)^2 + (x_i - x_j)^2)`
- 加权平均：`mean_dist = sum(attn * dist) / sum(attn) * patch_size`

### 阈值选择

- **默认阈值**：中位数（`median(dists)`）
- **优势**：对异常值更稳健，不受极值影响
- **替代方案**：平均值、固定阈值（如 80 px）

---

## 常见问题

### Q: 为什么去掉 CLS token？

**A**: CLS token 通常对所有 patch 都有很强的 attention，会主导距离计算。去掉 CLS 后，距离指标更能反映**空间上的局部/全局倾向**。

### Q: 为什么用加权平均而不是简单平均？

**A**: Attention 权重反映了模型**实际关注的程度**。加权平均能更准确地反映模型"平均关注多远"，而不是"所有 patch 对之间的平均距离"。

### Q: 如何判断阈值是否合理？

**A**: 
- 如果阈值接近平均值 → 合理
- 如果阈值与平均值差异很大 → 检查是否有异常层
- 可以手动设置阈值：修改代码中的 `threshold` 计算

### Q: 按 Head 分析有什么用？

**A**: 
- 不同 head 可能专门化：有的关注局部细节，有的关注全局结构
- 可以发现 head 的多样性
- 如果所有 head 都相似 → 可能 head 数量过多

---

## 参考文献

- DINOv2 论文：使用 attention distance 分析各层特征
- ViT 论文：浅层局部、深层全局的层级设计
- Vision Transformer 相关研究：attention 机制分析

---

## 更新日志

- **2026-03-06**: 增强版本
  - ✅ 添加详细统计（标准差、分位数、局部/全局比例）
  - ✅ 添加按 head 分析功能
  - ✅ 添加 CSV 导出功能
  - ✅ 改进可视化（4 个子图）
  - ✅ 使用中位数作为阈值（更稳健）
