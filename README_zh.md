🌐 [English](README.md) | [Français](README_fr.md) | **[中文](README_zh.md)**

# MLP & 自动微分引擎 (MNIST)

使用 C++ 从零实现的 MNIST 手写数字识别神经网络。  
**项目：** PPN（数值编程项目），M1 CHPS，UVSQ / 巴黎萨克雷大学

## ✨ 功能特性

- **MLP 实现**：全可配置的多层感知机，支持前向和反向传播
- **自动微分引擎**：使用动态计算图（DAG）的反向模式自动微分
- **多种优化**：朴素、缓存优化、OpenMP 和 BLAS 矩阵乘法
- **训练流程**：SGD 优化器、交叉熵损失、小批量训练
- **验证集准确率约 98.2%**

## 🛠️ 环境要求

- CMake 3.16+
- 支持 C++17 的 GCC/Clang
- OpenBLAS（可选，用于优化矩阵运算）

```bash
# Fedora/RHEL
sudo dnf install cmake gcc-c++ openblas-devel

# Ubuntu/Debian
sudo apt install cmake g++ libopenblas-dev
```

## 🚀 快速开始

### 编译

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 下载 MNIST 数据集

```bash
./scripts/get_mnist.sh
```

### 运行训练

```bash
./build/mnist_mlp --epochs 20 --lr 0.01 --batch_size 64 --hidden_sizes 128
```

### 命令行选项

| 选项 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 0.01 | 学习率 |
| `--batch_size` | 64 | 小批量大小 |
| `--hidden_sizes` | 128 | 隐藏层大小（逗号分隔） |
| `--activation` | relu | 激活函数（relu/sigmoid/tanh） |
| `--init` | he | 权重初始化方式（he/xavier/manual） |
| `--seed` | 42 | 随机种子（用于可复现性） |

## 📊 网络架构

```text
输入 (784) → 线性层 → ReLU → 线性层 → Softmax → 输出 (10)
```

![架构图](Docs/Images/phase3.png)

## 📖 文档

- [需求说明 (中文)](Docs/demande_zh.md) / [Spécification des besoins (法语)](Docs/demande_fr.md)
- [详细设计](Docs/conception_detaillee_fr.md)
- [理论：自动微分与反向传播](Docs/PPN_NN_zh.md)

## 📈 实验结果

| 指标 | 值 |
| ---- | -- |
| 验证集准确率 | ~98.2% |
| 最佳配置 | LR=0.01, Batch=64, Hidden=128, ReLU |
| 加速比（BLAS vs 朴素） | ~200× |

## 👥 作者

- 史建业 (Jianye Shi)
- 钱浩 (Hao Qian)
- 卞翔 (Xiang Bian)
- Abdennour Boulmis

**指导老师：** Aurélien Delval

## 📄 许可证

本项目为 UVSQ / 巴黎萨克雷大学 M1 CHPS 课程项目。
