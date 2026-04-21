# K-Nearest Neighbours (KNN) Classifier — CIFAR-10

## Overview

A from-scratch KNN classifier that uses raw pixel values as features. There is no training phase; the model memorises the training set and classifies test samples via majority vote among the *k* closest training examples in Euclidean space.

---

## Architecture

```
Input image (32×32×3)
    │
    ▼
Flatten  →  3072-dimensional vector
    │
    ▼
Compute L2 distance to every training vector
    │
    ▼
Select k=5 nearest neighbours
    │
    ▼
Majority vote  →  Predicted class (0–9)
```

| Property | Value |
|---|---|
| k (neighbours) | 5 |
| Distance metric | Euclidean (L2) |
| Feature space | Raw flattened pixel values (float32) |
| Training subset used | 3,000 samples |
| Test subset used | 300 samples |

> **Note:** KNN is evaluated on a small subset to keep runtime manageable.  
> Full-dataset evaluation with L2 distance on 50,000 train × 10,000 test would take several hours on CPU.

---

## Loss Function

KNN has **no loss function**. There is no gradient-based optimisation step. Classification is entirely determined by the distance metric and the value of *k* at inference time.

---

## Learnable Parameters

**0** — the model carries no learnable weights. The training set acts as the model.

---

## Results (CIFAR-10 subset — 300 test samples)

| Metric | Value |
|---|---|
| Accuracy | 27.00% |
| Macro Precision | 30.00% |
| Macro Recall | 25.00% |
| Macro F1 | 23.00% |

These numbers are **not directly comparable** to the CNN / Transformer / Mamba results, which are evaluated on the full 10,000-sample test set.

---

## How to Run

```bash
cd KNN-Classifier
python KNN_classifier.py
```

No dependencies beyond `numpy`, `scikit-learn`, and `matplotlib`. CIFAR-10 is downloaded automatically on first run.

---

## Design Notes

- Pixel-level L2 distance is a weak feature for images; background clutter and lighting shifts hurt performance significantly.
- Accuracy can be improved by using HOG or PCA-reduced features, or by tuning *k* via cross-validation.
- The primary value of this implementation is as a **baseline** and for understanding the distance-based classification paradigm before moving to learned representations.
