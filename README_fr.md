🌐 [English](README.md) | **[Français](README_fr.md)** | [中文](README_zh.md)

# MLP & Moteur d'Autodifférentiation pour MNIST

Implémentation d'un réseau de neurones from scratch en C++ pour le dataset MNIST.  
**Projet :** PPN (Projet de Programmation Numérique), M1 CHPS, UVSQ / Université Paris-Saclay

## ✨ Fonctionnalités

- **Implémentation MLP** : Perceptron multicouche entièrement configurable avec propagation avant et arrière
- **Moteur d'Autodifférentiation** : Autodiff en mode inverse utilisant un graphe de calcul dynamique (DAG)
- **Optimisations Multiples** : Multiplication matricielle naïve, cache-optimisée, OpenMP et BLAS
- **Pipeline d'Entraînement** : Optimiseur SGD, perte CrossEntropy, entraînement par mini-batch
- **~98.2% de Précision** sur l'ensemble de validation MNIST

## 🛠️ Prérequis

- CMake 3.16+
- GCC/Clang avec support C++17
- OpenBLAS (optionnel, pour les opérations matricielles optimisées)

```bash
# Fedora/RHEL
sudo dnf install cmake gcc-c++ openblas-devel

# Ubuntu/Debian
sudo apt install cmake g++ libopenblas-dev
```

## 🚀 Démarrage Rapide

### Compilation

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Télécharger le Dataset MNIST

```bash
./scripts/get_mnist.sh
```

### Lancer l'Entraînement

```bash
./build/mnist_mlp --epochs 20 --lr 0.01 --batch_size 64 --hidden_sizes 128
```

### Options de Ligne de Commande

| Option | Défaut | Description |
| ------ | ------ | ----------- |
| `--epochs` | 10 | Nombre d'époques d'entraînement |
| `--lr` | 0.01 | Taux d'apprentissage |
| `--batch_size` | 64 | Taille du mini-batch |
| `--hidden_sizes` | 128 | Tailles des couches cachées (séparées par virgule) |
| `--activation` | relu | Fonction d'activation (relu/sigmoid/tanh) |
| `--init` | he | Initialisation des poids (he/xavier/manual) |
| `--seed` | 42 | Graine aléatoire pour la reproductibilité |

## 📊 Architecture

```text
Entrée (784) → Linéaire → ReLU → Linéaire → Softmax → Sortie (10)
```

![Architecture](Docs/Images/phase3.png)

## 📖 Documentation

- [Spécification des besoins (FR)](Docs/demande_fr.md) / [需求说明 (ZH)](Docs/demande_zh.md)
- [Conception détaillée](Docs/conception_detaillee_fr.md)
- [Théorie : Autodiff & Backpropagation](Docs/PPN_NN.md)

## 📈 Résultats

| Métrique | Valeur |
| -------- | ------ |
| Précision Validation | ~98.2% |
| Meilleure Configuration | LR=0.01, Batch=64, Hidden=128, ReLU |
| Speedup (BLAS vs Naïf) | ~200× |

## 👥 Auteurs

- Jianye Shi
- Hao Qian
- Xiang Bian
- Abdennour Boulmis

**Encadrant :** Aurélien Delval

## 📄 Licence

Ce projet a été développé dans le cadre du cursus M1 CHPS à l'UVSQ / Université Paris-Saclay.
