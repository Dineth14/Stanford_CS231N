# CNN Classifier — CIFAR-10

## Overview

A VGG-style convolutional neural network with three double-convolution blocks and a fully connected head. Trained end-to-end on CIFAR-10 with batch normalisation, dropout regularisation, data augmentation, and a cosine annealing learning-rate schedule.

---

## Architecture

```
Input (3 × 32 × 32)
│
├─ Block 1
│   Conv2d(3→32, 3×3, pad=1) → BN → ReLU
│   Conv2d(32→32, 3×3, pad=1) → BN → ReLU
│   MaxPool2d(2×2)  →  [32 × 16 × 16]
│   Dropout2d(p=0.2)
│
├─ Block 2
│   Conv2d(32→64, 3×3, pad=1) → BN → ReLU
│   Conv2d(64→64, 3×3, pad=1) → BN → ReLU
│   MaxPool2d(2×2)  →  [64 × 8 × 8]
│   Dropout2d(p=0.3)
│
├─ Block 3
│   Conv2d(64→128, 3×3, pad=1) → BN → ReLU
│   Conv2d(128→128, 3×3, pad=1) → BN → ReLU
│   MaxPool2d(2×2)  →  [128 × 4 × 4]  =  2048-d feature
│   Dropout2d(p=0.4)
│
└─ Classifier Head
    Flatten
    Linear(2048 → 256) → ReLU → Dropout(p=0.5)
    Linear(256 → 10)  →  logits
```

### Key Design Choices

| Choice | Rationale |
|---|---|
| Double convolution per block | Increases receptive field depth before pooling (VGG-style) |
| Batch Normalisation after every conv | Stabilises training and allows a higher learning rate |
| Increasing dropout (0.2 → 0.3 → 0.4 → 0.5) | More regularisation as the spatial size shrinks and over-fitting risk grows |
| Cosine annealing LR | Smooth decay avoids oscillating around optima near the end of training |

---

## Loss Function

**Cross-Entropy Loss**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{10} y_{i,c}\,\log\hat{p}_{i,c}$$

where $y_{i,c}$ is the one-hot label and $\hat{p}_{i,c}$ is the softmax probability for class $c$.

---

## Learnable Parameters

| Component | Parameters |
|---|---:|
| Block 1 — Conv + BN layers | 10,272 |
| Block 2 — Conv + BN layers | 55,680 |
| Block 3 — Conv + BN layers | 221,952 |
| FC  2048 → 256 | 524,544 |
| FC  256 → 10 | 2,570 |
| **Total** | **815,018** |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| LR schedule | Cosine Annealing (T_max = 30) |
| Data augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalisation mean | (0.4914, 0.4822, 0.4465) |
| Normalisation std | (0.2470, 0.2435, 0.2616) |

---

## Results (CIFAR-10 test set — 10,000 samples)

| Metric | Value |
|---|---|
| Accuracy | **82.61%** |
| Macro Precision | 82.55% |
| Macro Recall | 82.61% |
| Macro F1 | 82.41% |
| Parameters | 815,018 |

Checkpoint saved to `best_cnn_cifar10.pth`.

---

## How to Run

```bash
cd CNN-Classifier
python CNN_classifier.py
```

CIFAR-10 is downloaded automatically to `data/` on first run. A CUDA-capable GPU is recommended but not required.

---

## File Structure

```
CNN-Classifier/
├── CNN_classifier.py        # Model definition + training loop
├── best_cnn_cifar10.pth     # Best checkpoint (saved during training)
└── data/
    └── cifar-10-batches-py/ # CIFAR-10 dataset
```

---

## References

- Simonyan, K., Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/abs/1409.1556
- Ioffe, S., Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167
- Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/cifar.html

