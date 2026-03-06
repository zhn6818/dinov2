# DINOv2 架构与特征可视化说明

本文档总结 DINOv2 Vision Transformer 中的关键概念，包括 **Patch 特征 PCA 可视化** 和 **Register Tokens**。

---

## 一、Patch 特征 PCA 可视化

### 1.1 可视化的是什么？

PCA 可视化展示的是 **最后一层 Transformer Block 的 patch token 特征**，将每个 patch 的 768 维特征向量通过 PCA 降维到 3 维（RGB），用于直观展示不同空间位置 patch 在特征空间中的相似性。

### 1.2 模型输出的层与特征

| 项目 | 说明 |
|------|------|
| **层** | 第 12 层（最后一个 Transformer Block） |
| **特征类型** | Patch tokens（不含 CLS token、不含 register tokens） |
| **是否 norm** | 是，经过 layer norm |
| **特征维度** | 768 维（ViT-B） |
| **空间分辨率** | 37×37（输入 518×518，patch_size=14） |
| **PCA 作用** | 768 维 → 3 维 RGB，用于可视化 patch 间的特征相似性 |

### 1.3 代码逻辑

`get_intermediate_layers(x, n=1, ...)` 中，当 `n` 为整数时，取的是 **最后 n 个 block** 的输出：

```python
# vision_transformer.py
blocks_to_take = range(total_block_len - n, total_block_len)  # n=1 → 最后一层
```

提取的特征经过 layer norm，并排除 CLS token 和 register tokens，只保留 patch tokens。每个 patch 对应一个 768 维向量，PCA 将其投影到 3 个主成分并映射为 RGB 颜色。

### 1.4 可视化含义

- **颜色相近**：patch 在 768 维特征空间中相似
- **颜色差异大**：patch 特征差异大

---

## 二、Register Tokens（寄存器 Token）

### 2.1 是什么？

**Register Tokens** 是 DINOv2 中引入的一类 **可学习的额外 token**，与 CLS token 类似，但数量通常为 4 个，插入在 CLS token 与 patch tokens 之间。

来源论文：[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)

### 2.2 在序列中的位置

Token 顺序为：

```
[CLS] [Reg1] [Reg2] [Reg3] [Reg4] [Patch1] [Patch2] ... [PatchN]
```

对应代码（`vision_transformer.py`）：

```python
if self.register_tokens is not None:
    x = torch.cat(
        (
            x[:, :1],                              # CLS
            self.register_tokens.expand(x.shape[0], -1, -1),  # Reg1~4
            x[:, 1:],                              # Patch tokens
        ),
        dim=1,
    )
```

### 2.3 实现方式

```python
# 定义
self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
# 初始化
nn.init.normal_(self.register_tokens, std=1e-6)
```

- 可学习参数：`nn.Parameter`，形状 `[1, num_register_tokens, embed_dim]`
- 常见配置：`num_register_tokens=4`
- 初始化方式与 CLS token 类似

### 2.4 作用（论文动机）

论文指出：深层 ViT 的 patch tokens 容易积累 **冗余、低秩** 信息，影响表达能力。加入少量 register tokens 后，它们充当“寄存器”，吸收这些冗余，让 patch tokens 更专注于 **discriminative 特征**，从而提升下游任务表现。

### 2.5 模型变体

| 模型 | `num_register_tokens` |
|------|------------------------|
| `dinov2_vitb14`（标准） | 0 |
| `dinov2_vitb14_reg`（带 registers） | 4 |

标准预训练模型（如 `dinov2_vitb14_pretrain.pth`）通常 `num_register_tokens=0`，即没有 register tokens。此时 `outputs = [out[:, 1:]]` 等价于只去掉 CLS token。

---

## 三、相关代码位置

| 功能 | 文件 | 说明 |
|------|------|------|
| `get_intermediate_layers` | `dinov2/models/vision_transformer.py` | 提取中间层特征 |
| Patch PCA 可视化 | `test_similar.py` | `visualize_patch_pca()` |
| Register tokens 定义 | `dinov2/models/vision_transformer.py` | `prepare_tokens_with_masks()` |

---

## 四、参考

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- [MODEL_CARD.md](../MODEL_CARD.md)
