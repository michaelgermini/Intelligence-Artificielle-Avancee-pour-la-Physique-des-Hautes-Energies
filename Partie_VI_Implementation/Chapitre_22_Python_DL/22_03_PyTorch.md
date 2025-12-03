# 22.3 PyTorch - Fondamentaux

---

## Introduction

**PyTorch** est un framework de deep learning développé par Facebook/Meta, particulièrement apprécié pour sa flexibilité, son mode de calcul dynamique, et son interface Pythonique. Il est largement utilisé dans la recherche et l'industrie pour développer des modèles de deep learning complexes.

Cette section présente les fondamentaux de PyTorch, organisés en trois sous-sections qui couvrent les tenseurs et l'automatique differentiation, les modules et optimiseurs, et la gestion des données.

---

## Plan de la Section

1. [Tenseurs et Autograd](./22_03_01_Tenseurs_Autograd.md)
2. [Modules et Optimizers](./22_03_02_Modules_Optimizers.md)
3. [DataLoaders et Datasets](./22_03_03_DataLoaders.md)

---

## Vue d'Ensemble PyTorch

### Caractéristiques Principales

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# PyTorch offre:
# - Tenseurs similaires à NumPy mais avec support GPU
# - Automatic differentiation (autograd)
# - Modules de deep learning pré-construits
# - Interface Pythonique et dynamique
```

---

## Pourquoi PyTorch ?

### Avantages

- **Flexibilité**: Mode de calcul dynamique permet debugging facile
- **Pythonic**: Interface naturelle pour programmeurs Python
- **GPU Support**: Calcul sur GPU transparent
- **Ecosystème riche**: Nombreuses bibliothèques (Torchvision, Torchaudio)
- **Research-friendly**: Facile à prototyper et expérimenter

---

## Architecture PyTorch

```
PyTorch
├── torch: Core library (tenseurs, autograd)
├── torch.nn: Modules de neural networks
├── torch.optim: Optimiseurs
├── torch.utils.data: Datasets et DataLoaders
├── torchvision: Vision (datasets, transforms)
└── torchaudio: Audio
```

---

*Section suivante : [22.3.1 Tenseurs et Autograd](./22_03_01_Tenseurs_Autograd.md)*

