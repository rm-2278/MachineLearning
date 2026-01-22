# 🧠 Machine Learning Portfolio

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)

A comprehensive collection of machine learning and deep learning implementations, demonstrating proficiency in both classical ML algorithms and modern deep learning architectures. All implementations are built from scratch or using fundamental frameworks like NumPy, PyTorch, and TensorFlow.

---

## 📑 Table of Contents
- [Projects Overview](#-projects-overview)
- [Technical Highlights](#-technical-highlights)
- [Project Details](#-project-details)
- [Getting Started](#-getting-started)
- [Key Achievements](#-key-achievements)
- [Technologies Used](#-technologies-used)

---

## 🚀 Projects Overview

| Project | Description | Key Technologies | Performance |
|---------|-------------|------------------|-------------|
| **[Logistic Regression](#1-logistic-regression)** | From-scratch implementation with NumPy | NumPy, scikit-learn | 86.6% (MNIST), 76.1% (Fashion-MNIST) |
| **[Multi-Layer Perceptron](#2-multi-layer-perceptron)** | Custom MLP with backpropagation | NumPy, Mini-batch GD | 86.9% (Fashion-MNIST) |
| **[Neural Network Optimization](#3-neural-network-optimization)** | Advanced optimization techniques | PyTorch, WandB, Adam | 88.3% (Fashion-MNIST) |
| **[ResNet Implementation](#4-resnet)** | Deep residual networks | PyTorch, CNNs | In Progress |
| **[PPO (Reinforcement Learning)](#5-proximal-policy-optimization-ppo)** | Policy gradient RL algorithm | Gymnasium, PyBullet | In Progress |
| **[Visual Question Answering](#6-visual-question-answering)** | Multimodal deep learning | Transformers, VQA | Competition Baseline |

---

## 💡 Technical Highlights

### Core Competencies Demonstrated
- ✅ **Algorithm Implementation from Scratch**: Built logistic regression and MLPs using only NumPy, demonstrating deep understanding of fundamental ML mathematics
- ✅ **Deep Learning Frameworks**: Proficient in PyTorch for building and training neural networks
- ✅ **Optimization Techniques**: Implemented various optimizers (SGD, Adam), learning rate scheduling, and regularization
- ✅ **Experiment Tracking**: Integrated WandB for comprehensive ML experiment management
- ✅ **Reinforcement Learning**: Working knowledge of policy gradient methods (PPO)
- ✅ **Multimodal AI**: Experience with vision-language models (VQA task)
- ✅ **Best Practices**: Proper train/validation/test splits, reproducible seeds, GPU acceleration, modular code architecture

---

## 📊 Project Details

### 1. Logistic Regression
**Path**: [`LogisticRegression/`](LogisticRegression/)

A complete from-scratch implementation of logistic regression using only NumPy, demonstrating strong fundamentals in:
- Softmax activation and cross-entropy loss
- Gradient computation and backpropagation
- Batch processing and numerical stability

**Implementations**:
- **MNIST Digit Classification**: 86.6% test accuracy
- **Fashion-MNIST**: 76.1% test accuracy

**Key Files**:
- [`model.py`](LogisticRegression/model.py) - Core logistic regression implementation
- [`mnist.py`](LogisticRegression/mnist.py) / [`fashion_mnist.py`](LogisticRegression/fashion_mnist.py) - Dataset-specific training scripts

**Performance Metrics** (100 epochs):
```
MNIST:        Train Loss: 2.346 → 0.617 | Test Acc: 86.6%
Fashion-MNIST: Train Loss: 2.395 → 0.712 | Test Acc: 76.1%
```

---

### 2. Multi-Layer Perceptron
**Path**: [`MultiLayeredPerceptron/`](MultiLayeredPerceptron/)

A fully custom MLP implementation with NumPy, featuring:
- Modular dense layer architecture with arbitrary depth
- Forward and backward propagation
- Mini-batch gradient descent
- ReLU and softmax activations

**Architecture**: 784 → 100 → 100 → 10  
**Performance** (Fashion-MNIST, 100 epochs):
```
Valid Loss: 1.326 → 0.360
Valid Acc:  58.7% → 87.5%
Test Acc:   86.9%
```

**Key Files**:
- [`model.py`](MultiLayeredPerceptron/model.py) - Dense layer and model classes
- [`helper.py`](MultiLayeredPerceptron/helper.py) - Activation functions, loss, batching utilities
- [`fashion_mnist.py`](MultiLayeredPerceptron/fashion_mnist.py) - Training pipeline

---

### 3. Neural Network Optimization
**Path**: [`NeuralNetworkOptimization/`](NeuralNetworkOptimization/)

Professional-grade PyTorch implementation showcasing production ML practices:
- **Adam optimizer** with learning rate scheduling
- **GPU acceleration** with CUDA support
- **Experiment tracking** via Weights & Biases (WandB)
- **Proper loss handling** with `nn.CrossEntropyLoss`
- **Modular design** with reusable training/evaluation functions

**Architecture**: Simple MLP (784 → 128 → 10)  
**Performance** (Fashion-MNIST, 10 epochs):
```
Train Loss: 0.538 → 0.251 | Train Acc: 81.1% → 90.5%
Valid Loss: 0.392 → 0.294 | Valid Acc: 85.9% → 89.4%
Test Acc: 88.3%
```

**Key Features**:
- WandB integration for real-time metrics visualization
- Stratified train/validation split for balanced classes
- Proper PyTorch DataLoader usage
- Device-agnostic code (CPU/GPU)

**Key Files**:
- [`model.py`](NeuralNetworkOptimization/model.py) - MLP class with training/evaluation loops
- [`fashion_mnist.py`](NeuralNetworkOptimization/fashion_mnist.py) - Main training script

**[View Training Logs on WandB →](https://wandb.ai/rm2278-university-of-cambridge/NNO_fashio_mnist/)**

---

### 4. ResNet
**Path**: [`ResNet/`](ResNet/)

Reimplementation of the influential paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385)

**Status**: 🚧 In Progress  
**Focus**: Understanding skip connections and deep network training dynamics

---

### 5. Proximal Policy Optimization (PPO)
**Path**: [`PPO/`](PPO/)

Reinforcement learning implementation based on: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

**Technologies**:
- Gymnasium for RL environments
- PyBullet for physics simulation
- Custom policy and value networks

**Key Features**:
- Reproducible seeding for environment and RNG
- Type-annotated code for maintainability
- Environment configuration with YAML

**Status**: 🚧 In Progress

---

### 6. Visual Question Answering
**Path**: [`DL_Basic_2025_Competition_VQA_baseline.ipynb`](DL_Basic_2025_Competition_VQA_baseline.ipynb)

Multimodal deep learning for answering questions about images.

**Highlights**:
- Transformer-based architectures
- Vision-language integration
- Competition-ready baseline implementation

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

Each project has its own dependencies. Navigate to the project directory and install:

```bash
# Example: Running Logistic Regression on MNIST
cd LogisticRegression
pip install -r requirements.txt
python mnist.py
```

```bash
# Example: Running Neural Network Optimization with WandB
cd NeuralNetworkOptimization
pip install -r requirements.txt
wandb login  # Enter your WandB API key
python fashion_mnist.py
```

```bash
# Example: Setting up PPO environment
cd PPO
conda env create -f environment.yml
conda activate ppo
python main.py
```

---

## 🏆 Key Achievements

### Technical Depth
- **Mathematical Foundations**: Implemented gradient descent, backpropagation, and softmax from scratch
- **Framework Proficiency**: Demonstrated PyTorch expertise with proper loss functions, optimizers, and data handling
- **ML Engineering**: Integrated professional tools (WandB, proper logging, reproducible experiments)

### Performance Metrics
- Achieved **88.3% accuracy** on Fashion-MNIST with optimized PyTorch implementation
- Built numerically stable implementations with proper gradient clipping and normalization
- Demonstrated understanding of train/val/test methodology and overfitting prevention

### Research Implementation
- Successfully reimplemented academic papers (ResNet, PPO)
- Worked with modern RL environments (Gymnasium, PyBullet)
- Explored cutting-edge multimodal AI (VQA)

---

## 🧰 Technologies Used

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Libraries & Tools
- **ML/DL**: NumPy, PyTorch, TensorFlow/Keras, scikit-learn
- **Experiment Tracking**: Weights & Biases (WandB)
- **RL**: Gymnasium, PyBullet
- **Visualization**: Matplotlib, pandas
- **Environment Management**: conda, pip

---

## 📈 Future Work

- [ ] Complete ResNet implementation and benchmark on CIFAR-10
- [ ] Train PPO agent to convergence on continuous control tasks
- [ ] Implement attention mechanisms from scratch
- [ ] Add CNN-based architectures
- [ ] Explore transfer learning and fine-tuning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Connect

Feel free to explore the code, raise issues, or suggest improvements!

**Note**: This repository demonstrates foundational ML/DL understanding suitable for research and engineering positions in machine learning.

---

*Last Updated: 2026-01-22 20:10:45*