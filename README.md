🌐 **[English](README.md)** | [Français](README_fr.md) | [中文](README_zh.md)

# MLP & Autodiff Engine for MNIST

Implementation of a neural network from scratch in C++ for the MNIST dataset.  
**Project:** PPN (Projet de Programmation Numérique), M1 CHPS, UVSQ / Université Paris-Saclay

## ✨ Features

- **MLP Implementation**: Fully configurable multi-layer perceptron with forward and backward propagation
- **Autodifferentiation Engine**: Reverse-mode autodiff using a dynamic computation graph (DAG)
- **Multiple Optimizations**: Naive, cache-optimized, OpenMP, and BLAS matrix multiplication
- **Training Pipeline**: SGD optimizer, CrossEntropy loss, mini-batch training
- **~98.2% Accuracy** on MNIST validation set

## 🛠️ Prerequisites

- CMake 3.16+
- GCC/Clang with C++17 support
- OpenBLAS (optional, for optimized matrix operations)

```bash
# Fedora/RHEL
sudo dnf install cmake gcc-c++ openblas-devel

# Ubuntu/Debian
sudo apt install cmake g++ libopenblas-dev
```

## 🚀 Quick Start

### Build

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Download MNIST Dataset

```bash
./scripts/get_mnist.sh
```

### Run Training

```bash
./build/mnist_mlp --epochs 20 --lr 0.01 --batch_size 64 --hidden_sizes 128
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 0.01 | Learning rate |
| `--batch_size` | 64 | Mini-batch size |
| `--hidden_sizes` | 128 | Hidden layer sizes (comma-separated) |
| `--activation` | relu | Activation function (relu/sigmoid/tanh) |
| `--init` | he | Weight initialization (he/xavier/manual) |
| `--seed` | 42 | Random seed for reproducibility |

## 📊 Architecture

```
Input (784) → Linear → ReLU → Linear → Softmax → Output (10)
```

![Architecture](Docs/Images/phase3.png)

## 📖 Documentation

- [Spécification des besoins (FR)](Docs/demande_fr.md) / [需求说明 (ZH)](Docs/demande_zh.md)
- [Conception détaillée](Docs/conception_detaillee_fr.md)
- [Théorie: Autodiff & Backpropagation](Docs/PPN_NN.md)

## 📈 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~98.2% |
| Best Configuration | LR=0.01, Batch=64, Hidden=128, ReLU |
| Speedup (BLAS vs Naive) | ~200× |

## 👥 Authors

- Jianye Shi
- Hao Qian
- Xiang Bian
- Abdennour Boulmis

**Supervised by:** Aurélien Delval

## 📄 License

This project was developed as part of the M1 CHPS curriculum at UVSQ / Université Paris-Saclay.
