# DINOv2 在 macOS (Apple Silicon) 上的安装说明

## xformers 编译错误说明

在 macOS Apple Silicon (M1/M2/M3) 上安装 xformers 时，可能遇到：

```
clang++: error: unsupported option '-fopenmp'
```

**原因**：Apple 自带的 clang 不支持 OpenMP，而 xformers 从源码编译时需要该选项。

---

## 解决方案：使用 Homebrew LLVM 编译

### 步骤 1：安装 LLVM 和 libomp

```bash
brew install llvm libomp
```

### 步骤 2：设置编译环境变量

在安装 xformers 之前，临时设置以下环境变量，让 pip 使用支持 OpenMP 的编译器：

```bash
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"
```

### 步骤 3：安装依赖（跳过 CUDA 相关）

macOS 没有 NVIDIA GPU，请使用 **CPU 版本** 的 PyTorch，并**不要安装** cuml-cu11：

```bash
# 创建 conda 环境
conda create -n dinov2 python=3.9 -y
conda activate dinov2

# 安装 PyTorch（CPU 版本，适用于 Mac）
conda install pytorch==2.0.0 torchvision==0.15.0 -c pytorch

# 安装其他依赖（不含 cuml）
pip install omegaconf torchmetrics==0.10.3 fvcore iopath

# 在设置好 CC/CXX 后安装 xformers
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"

pip install xformers==0.0.18

# 安装 submitit
pip install git+https://github.com/facebookincubator/submitit
```

### 步骤 4：一键安装脚本（可选）

将上述命令合并为一段脚本，在项目根目录执行：

```bash
#!/bin/bash
brew install llvm libomp 2>/dev/null || true

export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib -Wl,-rpath,$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"

pip install xformers==0.0.18
```

---

## 重要说明

1. **训练需要 NVIDIA GPU**：DINOv2 官方训练代码面向 Linux + CUDA，在 Mac 上即使安装成功，**训练也会非常慢**（CPU 或 MPS 均非官方支持）。建议使用云 GPU（如 Colab、AWS、AutoDL 等）进行训练。

2. **cuml-cu11**：这是 NVIDIA 的 CUDA 库，Mac 上**无法安装**，可跳过。若代码中有 cuml 调用，需要做兼容处理。

3. **submitit**：用于 SLURM 集群提交任务，单机训练时可能用不到，但安装无妨。

---

## 若仍无法安装 xformers

可考虑：

- 使用 **Linux 机器** 或 **带 NVIDIA GPU 的云服务器** 进行训练（推荐）
- 使用 **Google Colab** 等在线环境，通常已预装 xformers
