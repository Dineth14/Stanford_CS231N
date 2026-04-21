# Mamba (SSM) Classifier — CIFAR-10

## Overview

A pure-PyTorch implementation of the Mamba selective state-space model (S6) applied to image classification. The image is split into 4×4 patches, projected to 128-d tokens, and processed through 4 Mamba blocks whose state-transition parameters are *dynamically computed from the input* at each time step — giving the model the ability to selectively remember or forget information along the patch sequence.

---

## Architecture

```
Input (3 × 32 × 32)
│
├─ Patch Embedding
│   Conv2d(3→128, kernel=4, stride=4)  →  [64 × 128]  (64 patch tokens)
├─ Add learnable positional embedding  →  [64 × 128]
├─ Dropout(p=0.1)
│
├─ × 4  Mamba Blocks
│   ├─ LayerNorm
│   └─ SelectiveSSM
│       ├─ in_proj: Linear(128 → 512)  →  split into x_inner (256) and gate z (256)
│       ├─ Causal depthwise Conv1d(256, kernel=4, padding=3) → slice [:L] → SiLU
│       ├─ x_proj: Linear(256 → 17)  →  B (8-d), C (8-d), dt_raw (1-d)
│       ├─ dt_proj: Linear(1 → 256) + softplus  →  Δt  (input-dependent time step)
│       ├─ A: learnable log-parameterised [256 × 8] state matrix
│       ├─ Selective scan (sequential over L=64 tokens):
│       │     dA_t = exp(A · Δt_t)                   ← discretised transition
│       │     dB_t = Δt_t · B_t                       ← discretised input matrix
│       │     h_t  = dA_t · h_{t-1} + dB_t · x_t     ← state update
│       │     y_t  = (h_t · C_t).sum(-1)              ← output
│       ├─ Residual skip: y = y + x · D
│       ├─ Gating: y = y · SiLU(z)
│       └─ out_proj: Linear(256 → 128)
│       Residual add around block
│
├─ LayerNorm
└─ Mean pool over 64 tokens  →  Linear(128 → 10)  →  logits
```

### Selective Scan — How It Works

Unlike standard RNNs with fixed transition matrices, Mamba's **A**, **B**, **C**, and **Δt** are all functions of the current token:

$$h_t = e^{A \cdot \Delta t_t} \cdot h_{t-1} + \Delta t_t \cdot B_t \cdot x_t$$
$$y_t = C_t \cdot h_t$$

This allows the model to *selectively* compress information: for uninformative tokens it learns a large Δt (fast decay); for important tokens it learns a small Δt (slow decay).

### Key Design Choices

| Choice | Rationale |
|---|---|
| d_model=128, expand=2 (d_inner=256) | Compact model — competitive accuracy with far fewer parameters than the ViT |
| d_state=8 per channel | Controls SSM memory capacity; higher values increase expressiveness but cost |
| Causal Conv1d (kernel=4) before SSM | Short local context before the global sequential scan; analogue of QK local attention |
| Mean pooling for CLS | No CLS token — averaging all patch states works well for classification |
| Gradient clipping (max_norm=1.0) | SSM hidden states can produce large gradients; clipping stabilises early training |

---

## Loss Function

**Cross-Entropy Loss**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{10} y_{i,c}\,\log\hat{p}_{i,c}$$

---

## Learnable Parameters

| Component | Parameters |
|---|---:|
| Patch embedding (Conv2d) | 6,272 |
| Positional embedding (64 × 128) | 8,192 |
| 4 × Mamba blocks | 429,568 |
| Classification head (128 → 10) | 1,290 |
| **Total** | **444,042** |

The Mamba classifier achieves the best accuracy of the three trained models while using **~11× fewer parameters than the Transformer** and only about half the parameters of the CNN.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.05 |
| LR schedule | Cosine Annealing (T_max = 30) |
| Gradient clipping | max_norm = 1.0 |
| Data augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalisation mean | (0.4914, 0.4822, 0.4465) |
| Normalisation std | (0.2470, 0.2435, 0.2616) |

---

## Results (CIFAR-10 test set — 10,000 samples)

| Metric | Value |
|---|---|
| Accuracy | **83.09%** |
| Macro Precision | 83.16% |
| Macro Recall | 83.09% |
| Macro F1 | 83.11% |
| Parameters | 444,042 |

Checkpoint saved to `best_mamba_cifar10.pth`.

> Mamba achieves the highest accuracy among all models in this project, using the fewest parameters  
> of any deep model. The strong parameter efficiency is attributed to the inductive sequential bias  
> of the selective scan, which is well-suited to processing ordered patch sequences.

---

## How to Run

```bash
cd Mamba-Classifier
python Mamba_classifier.py
```

This implementation is **pure PyTorch** — no Triton or custom CUDA kernels are required. CIFAR-10 is downloaded automatically to `data/` on first run. A CUDA-capable GPU is recommended as the sequential scan is slow on CPU.

---

## File Structure

```
Mamba-Classifier/
├── Mamba_classifier.py        # Model definition + training loop
├── best_mamba_cifar10.pth     # Best checkpoint (saved during training)
└── data/
    └── cifar-10-batches-py/   # CIFAR-10 dataset
```

---

## References

- Gu, A., Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023. https://arxiv.org/abs/2312.00752  
- Gu, A. et al. *Efficiently Modeling Long Sequences with Structured State Spaces (S4).* ICLR 2022. https://arxiv.org/abs/2111.00396
