## CIFAR-10 Image Classification

This folder contains multiple classifiers trained on CIFAR-10:

- KNN classifier
- CNN classifier
- Transformer classifier (ViT-style)
- Mamba classifier (SSM-based)

## Results (CIFAR-10 test set — 10,000 samples, measured 2026-04-14)

| Model | Accuracy | Precision | Recall | F1 | Parameters |
|---|---:|---:|---:|---:|---:|
| [KNN](KNN-Classifier/Readme.md) | 27.00%† | 30.00% | 25.00% | 23.00% | 0 |
| [CNN](CNN-Classifier/Readme.md) | 82.61% | 82.55% | 82.61% | 82.41% | 815,018 |
| [Transformer](Transformer-Classifier/Readme.md) | 80.54% | 79.61% | 79.70% | 79.56% | 4,771,082 |
| [Mamba](Mamba-Classifier/Readme.md) | **83.09%** | 83.16% | 83.09% | 83.11% | 444,042 |

† KNN was evaluated on a 300-sample subset; not directly comparable to the full-set results above.

## Notes

- The KNN number uses a limited subset (3,000 train / 300 test) for runtime reasons; pixel L2 distance is a weak feature for images.
- CNN, Transformer, and Mamba results are from their respective best checkpoints evaluated on the full 10,000-sample CIFAR-10 test split.
- Mamba achieves the highest accuracy with the fewest parameters of any trained model (~11× smaller than the Transformer).

## How Accuracy Was Checked

- **KNN:** ran `KNN-Classifier/KNN_classifier.py` directly (script default subset).
- **CNN / Transformer / Mamba:** loaded model definitions, restored checkpoints, and evaluated on the full CIFAR-10 test split with the same normalisation used during training.

## References

1. Krizhevsky, A., Hinton, G. Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/cifar.html
2. Simonyan, K., Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/abs/1409.1556
3. Ioffe, S., Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. https://arxiv.org/abs/1502.03167
4. Vaswani, A. et al. Attention Is All You Need. https://arxiv.org/abs/1706.03762
5. Gu, A., Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. https://arxiv.org/abs/2312.00752
