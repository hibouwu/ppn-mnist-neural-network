# MLP & Autodifferentiation Engine for MNIST

**[English](README.md)** | [Français](README_fr.md) | [中文](README_zh.md)

This repository contains a complete implementation of a multi-layer perceptron (MLP) neural network written from scratch in C++.

It was developed as part of the **Numerical Programming Project (PPN)** for the Master 1 CHPS (High Performance Computing & Simulation) curriculum at the **University of Paris-Saclay (UVSQ)**.

The primary objective of this project is to understand the internal mechanisms of deep learning frameworks by implementing a custom reverse-mode autodifferentiation engine and optimized matrix operations without relying on external machine learning libraries like PyTorch or TensorFlow.

## Project Features

* **Custom Autodifferentiation Engine**: A dynamic computation graph (DAG) implementation supporting reverse-mode automatic differentiation.
* **Optimized Tensor Operations**: Matrix multiplication kernels optimized using cache blocking, OpenMP multithreading, and optional BLAS integration.
* **Configurable Neural Network**: Support for arbitrary layer configurations, activation functions (ReLU, Sigmoid, Tanh), and weight initialization schemes (He, Xavier).
* **Training Pipeline**: Complete training loop with Stochastic Gradient Descent (SGD), CrossEntropy loss, and mini-batch processing.

## Prerequisites

The project requires a C++17 compliant compiler and CMake. OpenBLAS is recommended for optimal performance.

* CMake 3.16 or higher
* GCC or Clang with C++17 support
* OpenBLAS (Optional, but highly recommended)

### Installation

* Fedora / RHEL

```bash
sudo dnf install cmake gcc-c++ openblas-devel
```

* Ubuntu / Debian

```bash
sudo apt install cmake g++ libopenblas-dev
```

## Build and Usage

### 1. Compilation

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 2. Dataset Preparation

A script is provided to download the MNIST dataset:

```bash
./scripts/get_mnist.sh
```

### 3. Execution

To train the model with the default configuration:

```bash
./build/ppn_train --epochs 20 --learning_rate 0.01 --batch_size 64 --hidden_size 128
```

### Command Line Options

The application supports the following command-line arguments:

| Option | Default | Description |
| -------- | --------- | ------------- |
| `--epochs` | 1 | Number of training epochs |
| `--learning_rate` | 0.01 | Learning rate |
| `--batch_size` | 64 | Mini-batch size |
| `--hidden_size` | 128 | Size of a single hidden layer |
| `--hidden_sizes` | "" | Sizes for multiple hidden layers (comma-separated, e.g., "128,64"). Overrides `--hidden_size`. |
| `--data_dir` | "mnist" | Directory containing MNIST dataset files |
| `--activation` | relu | Activation function (relu/sigmoid/tanh) |
| `--init` | he | Weight initialization strategy (he/xavier/manual) |
| `--seed` | 0 | Random seed (0 = random) |

## Helper Scripts

The `scripts/` directory contains various utilities for benchmarking and running experiments:

* **Benchmarks**:
  * `benchmark_matmul.sh`: Compare naive vs. optimized matrix multiplication.
  * `benchmark_e2e.sh`: End-to-end training performance test.
* **Experiments**:
  * `exp_learning_rate.sh`, `exp_batch_size.sh`, `exp_hidden_size.sh`: Run hyperparameter sweeps.
  * `exp_init_comparison.sh`: Compare weight initialization strategies.
* **Visualization**:
  * Python scripts (e.g., `plot_metrics.py`) are used by the shell scripts to generate performance plots.

## Architecture

The network architecture consists of a dynamic graph of operations.

```text
Input (784) -> Linear -> ReLU -> Linear -> Softmax -> Output (10)
```

[View Detailed Architecture Diagram (UML)](Docs/Images/phase4-6.png)

## Performance

We benchmarked the implementation on an **AMD Ryzen** processor. The optimized BLAS version shows significant speedup compared to the naive implementation.

| Implementation | Training Time (per epoch) | Speedup |
| ---------------- | --------------------------- | --------- |
| Naive C++ | ~60s | 1x |
| **Optimized (BLAS)** | **~0.3s** | **~200x** |

## Documentation

* [Technical Report (French)](Docs/PPN_NN.md)
* Detailed Design: [Docs/conception_detaillee_fr.md](Docs/conception_detaillee_fr.md)

## Authors

* **Jianye Shi**
* **Hao Qian**
* **Xiang Bian**
* **Abdennour Boulmis**

**Supervisor**: Aurélien Delval

## License

No license file is provided. This project is intended for academic and educational use only.
