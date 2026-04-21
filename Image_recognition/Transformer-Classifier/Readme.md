# Vision Transformer (ViT) Classifier — CIFAR-10

## Overview

A from-scratch Vision Transformer adapted for CIFAR-10 (32×32 images). The image is divided into 4×4 patches, linearly projected to a 256-dimensional embedding space, and processed through 6 Transformer encoder blocks. A prepended CLS token aggregates global context; its final representation is fed into a linear classification head.

---

## Architecture

```
Input (3 × 32 × 32)
│
├─ Patch Embedding
│   Conv2d(3→256, kernel=4, stride=4)  →  64 patch tokens  [64 × 256]
│
├─ Prepend CLS token                   →  [65 × 256]
├─ Add learnable positional embedding  →  [65 × 256]
├─ Dropout(p=0.1)
│
├─ × 6  Transformer Encoder Blocks
│   ├─ LayerNorm
│   ├─ Multi-Head Self-Attention  (8 heads, head_dim=32)
│   │   Q, K, V = Linear(256 → 256×3)  split per head
│   │   Attention = softmax( QKᵀ / √32 ) · V
│   │   Dropout(p=0.1) on attention weights and projection
│   ├─ Residual add
│   ├─ LayerNorm
│   ├─ MLP: Linear(256→1024) → GELU → Dropout → Linear(1024→256) → Dropout
│   └─ Residual add
│
├─ LayerNorm
└─ CLS token  →  Linear(256 → 10)  →  logits
```

### Attention Formula

For each head with dimension $d_k = 32$:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The 8 heads are computed in parallel and their outputs concatenated, then projected back to 256-d.

### Key Design Choices

| Choice | Rationale |
|---|---|
| Patch size 4 × 4 | Produces 64 tokens — enough sequence length for self-attention to be useful on 32×32 images |
| CLS token | Single vector that attends globally to all patches; avoids spatial bias of average pooling |
| Depth 6, embed_dim 256, heads 8 | Scaled-down ViT-S configuration appropriate for CIFAR-10 without pre-training |
| Learnable positional embedding | Allows the model to learn spatial relationships rather than using fixed sinusoids |
| AdamW + weight_decay 0.05 | Standard for Transformers; decoupled weight decay avoids decay on biases/norms |

---

## Loss Function

**Cross-Entropy Loss**

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{10} y_{i,c}\,\log\hat{p}_{i,c}$$

---

## Learnable Parameters

| Component | Parameters |
|---|---:|
| Patch embedding (Conv2d) | 12,544 |
| CLS token | 256 |
| Positional embedding (65 × 256) | 16,640 |
| 6 × MHSA (QKV proj + out proj) | 1,572,864 |
| 6 × MLP (256→1024→256) | 3,149,824 |
| 6 × LayerNorms (×2 each) | 6,144 |
| Classification head (256→10) | 2,570 |
| **Total** | **4,771,082** |

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 0.05 |
| LR schedule | Cosine Annealing (T_max = 30) |
| Data augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |
| Normalisation mean | (0.4914, 0.4822, 0.4465) |
| Normalisation std | (0.2470, 0.2435, 0.2616) |

---

## Results (CIFAR-10 test set — 10,000 samples)

| Metric | Value |
|---|---|
| Accuracy | **79.70%** |
| Macro Precision | 79.61% |
| Macro Recall | 79.70% |
| Macro F1 | 79.56% |
| Parameters | 4,771,082 |

Checkpoint saved to `best_transformer_cifar10.pth`.

> ViTs typically underperform CNNs on small datasets without large-scale pre-training, because  
> self-attention has a weaker inductive bias than convolutions. The ~3% gap vs. the CNN here  
> is consistent with this known behaviour.

---

## How to Run

```bash
cd Transformer-Classifier
python Transformer_classifier.py
```

CIFAR-10 is downloaded automatically to `data/` on first run. A CUDA-capable GPU is strongly recommended.

---

## File Structure

```
Transformer-Classifier/
├── Transformer_classifier.py        # Model definition + training loop
├── best_transformer_cifar10.pth     # Best checkpoint (saved during training)
└── data/
    └── cifar-10-batches-py/         # CIFAR-10 dataset
```

---

## References

- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. https://arxiv.org/abs/2010.11929  
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. https://arxiv.org/abs/1706.03762
