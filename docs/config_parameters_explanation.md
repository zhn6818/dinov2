# DINOv2 训练配置参数详细说明

本文档详细解释 DINOv2 训练配置文件中的每个参数及其作用。

---

## 1. MODEL 模块

### `WEIGHTS: ''`
- **含义**: 预训练模型权重路径
- **作用**: 指定加载的预训练模型文件路径,空字符串表示从头开始训练
- **默认值**: 空字符串

---

## 2. compute_precision 计算精度模块

控制模型训练过程中的数值精度和分布式训练策略。

### `grad_scaler: true`
- **含义**: 梯度缩放器开关
- **作用**: 在混合精度训练中防止梯度下溢,自动调整损失缩放因子
- **建议**: 使用 FP16 训练时建议开启

### teacher / student 配置

#### backbone / dino_head / ibot_head 共同参数

##### `sharding_strategy: SHARD_GRAD_OP`
- **含义**: 分布式分片策略
- **作用**: 指定如何在不同 GPU 之间分片模型参数
  - `SHARD_GRAD_OP`: 在反向传播时分片梯度,平衡内存和通信效率
- **适用场景**: 大规模分布式训练

##### `mixed_precision` 混合精度设置

###### `param_dtype: fp16`
- **含义**: 参数数据类型
- **作用**: 模型参数存储为 16 位浮点数
- **优势**: 减少显存占用,加速计算

###### `reduce_dtype: fp16 / fp32`
- **含义**: 归约操作数据类型
- **作用**: 
  - Teacher 使用 `fp16`: 梯度归约时使用 16 位
  - Student 使用 `fp32`: 梯度归约时使用 32 位,保持更高精度
- **区别原因**: Student 网络需要更稳定的梯度更新

###### `buffer_dtype: fp32`
- **含义**: 缓冲区数据类型
- **作用**: BatchNorm 等层的统计缓冲区使用 32 位浮点数
- **建议**: 保持 fp32 以确保数值稳定性

---

## 3. dino 模块 - DINO 自蒸馏损失

DINO (Distillation with No Labels) 的核心损失函数配置。

### `loss_weight: 1.0`
- **含义**: DINO 损失权重
- **作用**: 控制自蒸馏损失在总损失中的占比
- **范围**: 0.0 ~ 多个损失并存时可调整

### `head_n_prototypes: 65536`
- **含义**: 原型数量(聚类中心数)
- **作用**: 
  - DINO 使用 SwAV 风格的对比学习
  - 65536 个原型相当于一个大的代码本(codebook)
  - 学生网络的输出会被映射到这些原型上
- **原理**: 更多的原型可以学习更细粒度的特征表示

### `head_bottleneck_dim: 256`
- **含义**: DINO head 的瓶颈层维度
- **作用**: 
  - 特征在进入原型分类前先降维到 256 维
  - 减少参数量和计算量
- **建议**: 256 是平衡性能和效率的常用值

### `head_nlayers: 3`
- **含义**: DINO head 的层数
- **作用**: 投影头的 MLP 层数
- **典型结构**: Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear

### `head_hidden_dim: 2048`
- **含义**: DINO head 隐藏层维度
- **作用**: 投影头中间层的维度
- **建议**: 通常为 bottleneck_dim 的 4-8 倍

### `koleo_loss_weight: 0.1`
- **含义**: KoLeo 损失权重
- **作用**: 
  - KoLeo (Kozachenko-Leonenko) 损失鼓励特征均匀分布
  - 防止特征坍塌到某个子空间
  - 提高特征的多样性和表示能力
- **原理**: 基于特征点之间的距离,鼓励它们在空间中均匀分布

---

## 4. ibot 模块 - 掩码图像建模损失

iBOT (Image BERT Pre-training with Online Tokenizer) 的配置。

### `loss_weight: 1.0`
- **含义**: iBOT 损失权重
- **作用**: 控制掩码建模损失在总损失中的占比

### `mask_sample_probability: 0.5`
- **含义**: 掩码采样概率
- **作用**: 每个图像应用掩码增强的概率
- **解释**: 50% 的图像会被掩码,其余保持完整

### `mask_ratio_min_max: [0.1, 0.5]`
- **含义**: 掩码比例范围
- **作用**: 
  - 随机掩码 10%~50% 的图像 patch
  - 掩码太少:任务太简单,学不到有用特征
  - 掩码太多:信息不足,难以重建
- **建议**: 0.1-0.5 是经验最佳范围

### `separate_head: false`
- **含义**: 是否使用独立的 iBOT head
- **作用**: 
  - `false`: iBOT 和 DINO 共享投影头
  - `true`: 使用两个独立的投影头
- **建议**: 共享头可以减少参数,通常效果相当

### `head_n_prototypes: 65536`
- **含义**: iBOT head 的原型数量
- **作用**: 同 DINO,用于掩码 patch 的预测

### `head_bottleneck_dim: 256`
- **含义**: iBOT head 瓶颈维度
- **作用**: 掩码 patch 特征的降维维度

### `head_nlayers: 3`
- **含义**: iBOT head 层数
- **作用**: 同 DINO head

### `head_hidden_dim: 2048`
- **含义**: iBOT head 隐藏层维度
- **作用**: 同 DINO head

---

## 5. train 模块 - 训练基础配置

### `batch_size_per_gpu: 2`
- **含义**: 每个 GPU 的批大小
- **作用**: 
  - 有效批大小 = batch_size_per_gpu × GPU 数量
  - 2 是较小的值,适合显存有限的情况
- **建议**: 根据显存大小调整,越大越好(需保持总批大小)

### `dataset_path: FlatFolder:root=/data1/zhn/jinxiang`
- **含义**: 数据集路径
- **作用**: 
  - `FlatFolder`: 数据集格式,所有图像在一个文件夹下
  - `root=`: 指定数据根目录
- **格式**: 支持多种数据集格式(ImageNet, FlatFolder, etc.)

### `output_dir: /data1/code/dinov2/output/vitb14_finetune_jinxiang`
- **含义**: 输出目录
- **作用**: 保存训练日志、检查点、配置文件等

### `saveckp_freq: 5000`
- **含义**: 保存检查点频率
- **作用**: 每训练 5000 次迭代保存一次模型
- **建议**: 根据训练时长调整,避免丢失太多进度

### `seed: 0`
- **含义**: 随机种子
- **作用**: 确保实验可复现
- **建议**: 固定种子便于调试和对比

### `num_workers: 0`
- **含义**: 数据加载工作进程数
- **作用**: 
  - 0: 主进程加载数据(单进程)
  - >0: 使用多进程并行加载
- **建议**: 通常设置为 4-8,但共享内存受限时可设为 0

### `OFFICIAL_EPOCH_LENGTH: 250`
- **含义**: 一个 epoch 的迭代次数
- **作用**: 
  - 定义"epoch"的概念,与数据集实际大小无关
  - 250 次迭代算一个 epoch
- **原因**: 大规模数据集可能太大,人为定义 epoch 更方便

### `cache_dataset: false`
- **含义**: 是否缓存数据集到内存
- **作用**: 
  - `true`: 将整个数据集加载到内存(需要足够 RAM)
  - `false`: 每次从磁盘读取
- **建议**: 小数据集可缓存,大数据集不建议

### `centering: centering`
- **含义**: 特征中心化方法
- **作用**: 对特征进行中心化处理,提高训练稳定性
- **选项**: `centering` / `sinkhorn_knopp` 等

### `cell_augmentation: false`
- **含义**: 细胞级增强(医学图像专用)
- **作用**: 针对细胞图像的特殊数据增强
- **适用**: 医学图像数据集

---

## 6. student 模块 - 学生网络配置

学生网络是实际要训练的模型,通过向教师网络学习来提升性能。

### `arch: vit_base`
- **含义**: 模型架构
- **作用**: 使用 Vision Transformer Base 版本
- **可选**: `vit_small`, `vit_base`, `vit_large`, `vit_giant`

### `patch_size: 14`
- **含义**: Patch 大小
- **作用**: 
  - 将图像切分成 14×14 像素的 patch
  - 224×224 图像 → 16×16 = 256 个 patch
- **影响**: patch 越小,计算量越大,特征越精细

### `drop_path_rate: 0.3`
- **含义**: DropPath 概率(随机深度)
- **作用**: 
  - 训练时随机丢弃某些 transformer 层
  - 正则化手段,防止过拟合
  - 0.3 表示 30% 概率丢弃某层
- **原理**: 类似 Dropout,但作用于整个层

### `layerscale: 1.0e-05`
- **含义**: LayerScale 初始值
- **作用**: 
  - 每个 transformer 层的残差连接上的可学习缩放因子
  - 小的初始值使训练更稳定
- **建议**: 深层网络建议使用(1e-5 ~ 1e-6)

### `drop_path_uniform: true`
- **含义**: 是否均匀应用 DropPath
- **作用**: 
  - `true`: 所有层使用相同的 drop_path_rate
  - `false`: 深层使用更高的 drop_path_rate
- **建议**: 通常 uniform 效果不错

### `pretrained_weights: pretrain/dinov2_vitb14_pretrain.pth`
- **含义**: 预训练权重路径
- **作用**: 加载预训练的 DINOv2 权重进行微调
- **场景**: 从已有模型开始训练,而非从头训练

### `ffn_layer: mlp`
- **含义**: 前馈网络类型
- **作用**: Transformer 中 FFN 层的实现方式
- **可选**: `mlp` / `swiglu` 等
- **MLP**: 标准 2 层 MLP (Linear -> GELU -> Linear)

### `block_chunks: 0`
- **含义**: 块分块数
- **作用**: 
  - 0: 不分块,所有层一起处理
  - >0: 将 transformer 分成多个块,节省显存
- **场景**: 显存受限时可使用

### `qkv_bias: true`
- **含义**: Query/Key/Value 是否使用偏置
- **作用**: 注意力层的 QKV 投影是否加 bias
- **建议**: true 通常效果更好

### `proj_bias: true`
- **含义**: 输出投影是否使用偏置
- **作用**: 注意力输出投影层的 bias

### `ffn_bias: true`
- **含义**: FFN 是否使用偏置
- **作用**: 前馈网络层的 bias

### `num_register_tokens: 0`
- **含义**: Register token 数量
- **作用**: 
  - 注册令牌,用于存储全局信息
  - 0: 不使用(DINOv2 原版)
  - >0: 添加额外的 token 来缓解伪影
- **论文**: "Vision Transformers with Registers"

### `interpolate_antialias: false`
- **含义**: 插值时是否使用抗锯齿
- **作用**: 位置编码插值时的抗锯齿处理
- **场景**: 改变输入分辨率时需要插值位置编码

### `interpolate_offset: 0.1`
- **含义**: 插值偏移量
- **作用**: 位置编码插值的偏移参数
- **影响**: 微调时改变分辨率会用到

### `in_chans: 3`
- **含义**: 输入通道数
- **作用**: RGB 图像为 3 通道
- **可改**: 灰度图为 1,多光谱图像可为更多

### `channel_adaptive: false`
- **含义**: 是否启用通道自适应
- **作用**: 适应不同通道数的输入(如医学多通道图像)

---

## 7. teacher 模块 - 教师网络配置

教师网络是学生网络的指数移动平均(EMA),不通过梯度下降更新。

### `momentum_teacher: 0.992`
- **含义**: 教师 EMA 动量(初始值)
- **作用**: 
  - 教师 = 0.992 × 教师 + 0.008 × 学生
  - 教师网络缓慢跟踪学生网络
- **原理**: 提供稳定的训练目标,避免震荡

### `final_momentum_teacher: 1`
- **含义**: 教师 EMA 动量(最终值)
- **作用**: 训练结束时动量达到 1.0
- **调度**: 从 0.992 线性增加到 1.0

### `warmup_teacher_temp: 0.04`
- **含义**: 教师温度(预热期)
- **作用**: 
  - 控制教师输出的软化程度
  - 0.04 较低,输出分布较尖锐
- **原理**: 温度越低,分布越接近 one-hot

### `teacher_temp: 0.07`
- **含义**: 教师温度(稳定期)
- **作用**: 预热后的教师温度
- **建议**: 0.04-0.07 是常用范围

### `warmup_teacher_temp_epochs: 20`
- **含义**: 教师温度预热 epoch 数
- **作用**: 前 20 个 epoch 温度从 0.04 逐渐升到 0.07

### `in_chans: 3`
- **含义**: 输入通道数(同 student)

### `channel_adaptive: false`
- **含义**: 通道自适应(同 student)

---

## 8. optim 模块 - 优化器配置

### `epochs: 250`
- **含义**: 训练总 epoch 数
- **作用**: 完整训练 250 个 epoch
- **建议**: DINOv2 通常需要较长时间训练

### `weight_decay: 0.04`
- **含义**: 权重衰减(初始值)
- **作用**: L2 正则化,防止过拟合
- **原理**: 在损失函数中添加 λ||θ||²

### `weight_decay_end: 0.4`
- **含义**: 权重衰减(最终值)
- **作用**: 训练过程中从 0.04 增加到 0.4
- **原理**: 训练后期需要更强的正则化

### `base_lr: 0.004`
- **含义**: 基础学习率
- **作用**: 参考学习率,实际学习率会根据批大小缩放

### `lr: 0.0001767766952966369`
- **含义**: 实际学习率
- **计算**: base_lr × (batch_size / 1024)^0.5
- **作用**: 根据你的批大小自动计算得到

### `warmup_epochs: 10`
- **含义**: 学习率预热 epoch 数
- **作用**: 
  - 前 10 个 epoch 学习率从 0 逐渐升到 lr
  - 防止训练初期不稳定
- **原理**: 模型参数还是随机时,大学习率可能导致发散

### `min_lr: 1.0e-06`
- **含义**: 最小学习率
- **作用**: 学习率衰减的下限
- **调度**: 余弦退火后不低于此值

### `clip_grad: 3.0`
- **含义**: 梯度裁剪阈值
- **作用**: 
  - 梯度范数超过 3.0 时进行裁剪
  - 防止梯度爆炸
- **原理**: grad = grad × (3.0 / ||grad||)

### `freeze_last_layer_epochs: 1`
- **含义**: 冻结最后一层的 epoch 数
- **作用**: 
  - 第一个 epoch 冻结投影头的最后一层
  - 让 backbone 先适应,再微调 head
- **建议**: 通常 1-3 个 epoch 即可

### `scaling_rule: sqrt_wrt_1024`
- **含义**: 学习率缩放规则
- **作用**: 
  - 根据批大小缩放学习率
  - sqrt_wrt_1024: lr ∝ sqrt(batch_size / 1024)
- **原理**: 线性缩放规则在大批大小时可能过于激进

### `patch_embed_lr_mult: 0.2`
- **含义**: Patch Embedding 层学习率倍率
- **作用**: 
  - Patch Embedding 使用 0.2×lr
  - 底层特征变化应更缓慢
- **建议**: 微调时底层用较小学习率

### `layerwise_decay: 0.9`
- **含义**: 层级学习率衰减
- **作用**: 
  - 从底层到顶层,学习率逐层衰减
  - 第 i 层 lr = base_lr × (0.9)^i
- **原理**: 底层学习通用特征,顶层学习任务特定特征

### `adamw_beta1: 0.9`
- **含义**: AdamW 的 β1 参数
- **作用**: 一阶矩估计的指数衰减率
- **含义**: 控制动量项

### `adamw_beta2: 0.999`
- **含义**: AdamW 的 β2 参数
- **作用**: 二阶矩估计的指数衰减率
- **含义**: 控制 RMSProp 项

---

## 9. crops 模块 - 多尺度裁剪增强

DINOv2 使用多尺度裁剪进行数据增强。

### `global_crops_scale: [0.9, 1.0]`
- **含义**: 全局裁剪缩放范围
- **作用**: 
  - 裁剪图像的 90%~100% 区域
  - 生成 2 个全局裁剪(大尺度)
- **解释**: 保留大部分图像内容,学习全局特征

### `local_crops_number: 4`
- **含义**: 局部裁剪数量
- **作用**: 生成 4 个小尺度裁剪
- **原理**: Multi-crop 策略,增加训练数据多样性

### `local_crops_scale: [0.05, 0.32]`
- **含义**: 局部裁剪缩放范围
- **作用**: 
  - 裁剪图像的 5%~32% 区域
  - 生成小尺度裁剪
- **原理**: 学习局部细节特征

### `global_crops_size: 224`
- **含义**: 全局裁剪尺寸
- **作用**: 全局裁剪 resize 到 224×224
- **建议**: 根据模型输入尺寸调整

### `local_crops_size: 98`
- **含义**: 局部裁剪尺寸
- **作用**: 局部裁剪 resize 到 98×98
- **原理**: 小裁剪用小分辨率,减少计算量

**Multi-crop 训练流程**:
1. 对每张图像生成 2 个全局裁剪(224×224) + 4 个局部裁剪(98×98)
2. 全局裁剪之间做自蒸馏
3. 局部裁剪向全局裁剪学习
4. 鼓励模型学习多尺度特征

---

## 10. evaluation 模块 - 评估配置

### `eval_period_iterations: 5000`
- **含义**: 评估频率
- **作用**: 每训练 5000 次迭代进行一次验证
- **建议**: 根据训练时长调整,评估会耗时

---

## 核心训练机制总结

本节结合前述配置参数,深入介绍 DINOv2 的训练框架与设计思路。

---

### 1. 自蒸馏框架 (DINO - Distillation with No Labels)

DINO 是 DINOv2 的核心,实现**无标签自监督蒸馏**。

#### 1.1 双网络结构

| 角色 | 更新方式 | 对应参数 | 说明 |
|------|----------|----------|------|
| **Student** | 梯度下降 | `student.*` | 实际训练的模型,接收梯度 |
| **Teacher** | EMA 更新 | `teacher.momentum_teacher` | 不接收梯度,仅作为目标 |

- **EMA 公式**: `θ_teacher = m × θ_teacher + (1-m) × θ_student`
- **动量**: `momentum_teacher: 0.992` → 教师缓慢跟踪学生,提供稳定目标
- **调度**: `final_momentum_teacher: 1` → 训练末期动量线性增至 1.0,教师逐渐"冻结"

#### 1.2 损失计算

- **形式**: 交叉熵(等价于 KL 散度),让学生 softmax 输出接近教师
- **公式**: `L = -Σ t_i × log(s_i)`,其中 t 为教师 softmax,temp 软化;s 为学生 log_softmax
- **温度**:
  - `warmup_teacher_temp: 0.04` → 预热期较低,分布更尖锐
  - `teacher_temp: 0.07` → 稳定期温度
  - `warmup_teacher_temp_epochs: 20` → 温度从 0.04 线性升至 0.07 的 epoch 数

#### 1.3 中心化 (Centering)

- **作用**: 对教师输出做中心化,防止所有样本塌缩到同一原型
- **方式**: `centering` 或 `sinkhorn_knopp`
- **Centering**: 维护全局中心 `center`,教师输出减去 center 后再 softmax
- **Sinkhorn-Knopp**: 用迭代归一化替代简单中心化,使原型分配更均匀

#### 1.4 投影头 (DINO Head)

- **结构**: MLP,将 backbone 输出映射到 65536 维原型空间
- **参数**: `head_n_prototypes: 65536`, `head_bottleneck_dim: 256`, `head_nlayers: 3`, `head_hidden_dim: 2048`
- **作用**: 相当于大 codebook,学生输出被软分配到这些原型上

---

### 2. 掩码图像建模 (iBOT - Image BERT Pre-training with Online Tokenizer)

iBOT 在 DINO 基础上增加**局部 patch 级**的自监督信号。

#### 2.1 掩码策略

- **掩码概率**: `mask_sample_probability: 0.5` → 50% 图像被掩码
- **掩码比例**: `mask_ratio_min_max: [0.1, 0.5]` → 随机掩码 10%~50% 的 patch
- **实现**: 在 collate 时对每张图随机采样比例,用 `MaskingGenerator` 生成二值掩码

#### 2.2 训练目标

- **输入**: Student 接收**带掩码**的全局裁剪;Teacher 接收**完整**图像(无掩码)
- **目标**: Student 对被掩码 patch 的特征预测,要接近 Teacher 对**同一位置** patch 的特征
- **损失**: 与 DINO 相同形式的交叉熵,但只对掩码位置计算

#### 2.3 与 DINO 的关系

- **共享头**: `separate_head: false` 时,iBOT 与 DINO 共用 `dino_head`
- **效果**: 同时学习全局语义(CLS token)与局部语义(patch token),增强表示能力

---

### 3. 多尺度训练 (Multi-crop)

通过不同尺度的裁剪,让模型同时学习**全局结构**和**局部细节**。

#### 3.1 裁剪配置

| 类型 | 参数 | 典型值 | 作用 |
|------|------|--------|------|
| 全局裁剪 | `global_crops_scale`, `global_crops_size` | [0.9,1.0], 224 | 覆盖 90%~100% 图像,学习整体 |
| 局部裁剪 | `local_crops_scale`, `local_crops_size`, `local_crops_number` | [0.05,0.32], 98, 4 | 覆盖 5%~32%,学习细节 |

#### 3.2 损失配对

- **全局-全局**: 2 个全局裁剪之间做 DINO 自蒸馏(A↔B)
- **局部-全局**: 4 个局部裁剪的 CLS token 向 2 个全局裁剪的教师输出学习
- **iBOT**: 仅对全局裁剪的 patch 做掩码预测

#### 3.3 设计动机

- 局部裁剪与全局裁剪来自同一图像的不同区域,迫使模型建立**多尺度一致性**
- 提高对尺度变化、裁剪、遮挡的鲁棒性

---

### 4. 特征均匀性 (KoLeo Loss)

#### 4.1 原理

- **来源**: Kozachenko-Leonenko 熵估计,来自 Sablayrolles 等人的相似性搜索工作
- **目标**: 鼓励特征在单位超球面上**均匀分布**,避免塌缩到少数方向

#### 4.2 计算方式

- 对 L2 归一化后的 CLS token 计算**最近邻距离**
- 损失: `L = -mean(log(distance_to_NN + ε))`
- 距离越大 → 分布越均匀 → 损失越小

#### 4.3 配置

- `koleo_loss_weight: 0.1` → 作为正则项,与 DINO/iBOT 损失加权求和
- 仅作用于**全局裁剪的 CLS token**,且不在同一图像的 2 个 crop 之间计算(避免 trivial solution)

---

### 5. 优化策略

#### 5.1 学习率

- **基础**: `base_lr: 0.004`,按 `sqrt_wrt_1024` 规则根据 batch size 缩放
- **层级衰减**: `layerwise_decay: 0.9` → 第 i 层 lr = base_lr × 0.9^i
- **Patch Embedding**: `patch_embed_lr_mult: 0.2` → 底层用更小学习率
- **最后一层冻结**: `freeze_last_layer_epochs: 1` → 前 1 个 epoch 投影头最后一层 lr=0

#### 5.2 学习率调度

- **Warmup**: `warmup_epochs: 10` → 前 10 epoch 从 0 线性升至 lr
- **主阶段**: Cosine Annealing → 从 lr 余弦衰减至 `min_lr: 1e-6`

#### 5.3 权重衰减

- **调度**: 从 `weight_decay: 0.04` 余弦增至 `weight_decay_end: 0.4`
- **作用**: 训练后期加强正则化,防止过拟合

#### 5.4 梯度裁剪

- `clip_grad: 3.0` → 梯度范数超过 3.0 时按比例缩放,防止梯度爆炸

---

### 6. 单次迭代的完整流程

1. **数据**: 每张图 → 2 个全局裁剪(224×224) + 4 个局部裁剪(98×98);50% 概率对全局裁剪做 patch 掩码
2. **Teacher 前向**: 2 个全局裁剪 → Teacher backbone → DINO head → 中心化 + softmax
3. **Student 前向**: 全局(带掩码) + 局部 → Student backbone → DINO head
4. **损失**:
   - DINO 全局: Student 全局 CLS vs Teacher 全局(交叉熵)
   - DINO 局部: Student 局部 CLS vs Teacher 全局(交叉熵)
   - iBOT: Student 掩码 patch vs Teacher 对应 patch(交叉熵)
   - KoLeo: Student 全局 CLS 的均匀性正则
5. **反向传播**: 仅对 Student 计算梯度
6. **更新**: Student 用 AdamW 更新;Teacher 用 EMA 更新

---

## 调参建议

### 显存不足时
1. 减小 `batch_size_per_gpu`
2. 减小 `global_crops_size` 和 `local_crops_size`
3. 减小 `local_crops_number`
4. 使用 `block_chunks` 分块处理
5. 使用梯度检查点(gradient checkpointing)

### 训练速度慢时
1. 增加 `num_workers`(如果 I/O 是瓶颈)
2. 使用 `cache_dataset`(如果内存够大)
3. 减少 `saveckp_freq` 和 `eval_period_iterations`

### 提升性能时
1. 增加训练时长(epochs)
2. 增加批大小(需相应调整学习率)
3. 调整 `mask_ratio_min_max` 范围
4. 尝试不同的 `drop_path_rate`
5. 调整损失权重 `loss_weight`

### 微调时
1. 使用较小的学习率
2. 减小 `weight_decay`
3. 可冻结部分底层(`layerwise_decay` 调小)
4. 减少 `warmup_epochs`

---

## 参考资料

- [DINOv2 论文](https://arxiv.org/abs/2304.07193)
- [DINO 论文](https://arxiv.org/abs/2104.14294)
- [iBOT 论文](https://arxiv.org/abs/2111.07832)
- [官方代码库](https://github.com/facebookresearch/dinov2)
