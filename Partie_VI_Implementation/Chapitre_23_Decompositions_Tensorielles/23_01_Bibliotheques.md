# 23.1 Bibliothèques Python (tensorly, tntorch)

---

## Introduction

Les bibliothèques **tensorly** et **tntorch** fournissent des implémentations efficaces des décompositions tensorielles avec support pour différents backends (NumPy, PyTorch, TensorFlow). Cette section présente comment utiliser ces bibliothèques pour effectuer des décompositions tensorielles.

---

## TensorLy

### Installation et Configuration

```python
"""
Installation:
pip install tensorly

Backends supportés:
- NumPy (défaut)
- PyTorch
- TensorFlow
- JAX
- CuPy (GPU)
"""

import tensorly as tl
import numpy as np
import torch

# Changer backend
tl.set_backend('numpy')  # NumPy (défaut)
tl.set_backend('pytorch')  # PyTorch
tl.set_backend('tensorflow')  # TensorFlow

print(f"Backend actuel: {tl.get_backend()}")
```

---

## Décompositions avec TensorLy

### CP Decomposition

```python
import tensorly as tl
from tensorly.decomposition import parafac
import numpy as np

# Créer tenseur 3D
tensor = np.random.randn(10, 20, 30)

# Décomposition CP avec rank R
rank = 5
factors = parafac(tensor, rank=rank)

# factors est liste de matrices
print(f"Nombre de facteurs: {len(factors)}")
for i, factor in enumerate(factors):
    print(f"Facteur {i} shape: {factor.shape}")

# Reconstruire tenseur
reconstructed = tl.cp_to_tensor(factors)
print(f"Erreur reconstruction: {np.linalg.norm(tensor - reconstructed):.6f}")
```

### Tucker Decomposition

```python
from tensorly.decomposition import tucker

# Décomposition Tucker
tucker_rank = [5, 10, 8]  # Ranks pour chaque mode
core, factors = tucker(tensor, rank=tucker_rank)

print(f"Core shape: {core.shape}")
print(f"Nombre de facteurs: {len(factors)}")

# Reconstruire
reconstructed = tl.tucker_to_tensor((core, factors))
print(f"Erreur reconstruction: {np.linalg.norm(tensor - reconstructed):.6f}")
```

### Tensor Train

```python
from tensorly.decomposition import matrix_product_state

# Décomposition Tensor Train
tt_rank = [1, 5, 5, 1]  # Bond dimensions
factors = matrix_product_state(tensor, rank=tt_rank)

# Reconstruire
reconstructed = tl.tt_to_tensor(factors)
print(f"Erreur reconstruction: {np.linalg.norm(tensor - reconstructed):.6f}")
```

---

## TensorLy avec PyTorch

### Backend PyTorch

```python
import tensorly as tl
import torch

# Changer backend PyTorch
tl.set_backend('pytorch')

# Créer tenseur PyTorch
tensor = torch.randn(10, 20, 30, requires_grad=True)

# Décomposition CP
rank = 5
factors = parafac(tensor, rank=rank)

# Les facteurs sont aussi des tenseurs PyTorch avec gradients
print(f"Facteur 0 requires_grad: {factors[0].requires_grad}")

# Reconstruire
reconstructed = tl.cp_to_tensor(factors)

# Utiliser dans loss
loss = torch.nn.functional.mse_loss(reconstructed, tensor)
loss.backward()
print(f"Gradients calculés pour facteurs")
```

---

## tntorch (Tensor Train)

### Installation et Utilisation

```python
"""
Installation:
pip install tntorch

Focus sur Tensor Train et formats compressés
"""

import tntorch as tn
import torch

# Créer tenseur PyTorch
tensor = torch.randn(10, 10, 10, 10)

# Compression en Tensor Train
tt = tn.TensorTrain(tensor, ranks_tt=5)  # Bond dimension 5

print(f"Tenseur original: {tensor.numel()} paramètres")
print(f"Tensor Train: {tt.numel()} paramètres")
print(f"Compression: {tensor.numel() / tt.numel():.2f}×")

# Opérations sur Tensor Train
tt_sum = tt + tt  # Addition
tt_prod = tt * 2  # Multiplication scalaire
tt_dot = tn.dot(tt, tt)  # Produit scalaire

# Reconstruire tenseur complet
reconstructed = tt.torch()
print(f"Erreur reconstruction: {torch.norm(tensor - reconstructed):.6f}")
```

---

## Opérations avec tntorch

### Manipulation de Tensor Trains

```python
import tntorch as tn
import torch

# Créer Tensor Train
tt = tn.TensorTrain(tensor, ranks_tt=5)

# Accéder aux cores
cores = tt.cores
print(f"Nombre de cores: {len(cores)}")
for i, core in enumerate(cores):
    print(f"Core {i} shape: {core.shape}")

# Modifier bond dimension
tt_compressed = tt.round(eps=1e-4)  # Compression additionnelle
print(f"Nouvelle compression: {tt_compressed.numel()} paramètres")

# Slicing et indexing
tt_slice = tt[0:5, :, :, :]  # Slice première dimension

# Fonctions mathématiques
tt_exp = tn.exp(tt)
tt_log = tn.log(tt + 1e-8)
```

---

## Comparaison TensorLy vs tntorch

### Caractéristiques

```python
class LibraryComparison:
    """
    Comparaison des bibliothèques
    """
    
    def __init__(self):
        self.comparison = {
            'tensorly': {
                'strengths': [
                    'Multiple décompositions (CP, Tucker, TT)',
                    'Plusieurs backends (NumPy, PyTorch, TensorFlow)',
                    'Interface unifiée',
                    'Bien documenté'
                ],
                'weaknesses': [
                    'Performance parfois limitée',
                    'Moins optimisé que bibliothèques spécialisées'
                ],
                'best_for': 'Prototypage, expérimentation, comparaison méthodes'
            },
            'tntorch': {
                'strengths': [
                    'Optimisé pour Tensor Train',
                    'Interface PyTorch native',
                    'Opérations sur TT compressés',
                    'Meilleures performances'
                ],
                'weaknesses': [
                    'Focus sur TT uniquement',
                    'Backend PyTorch seulement'
                ],
                'best_for': 'Production, modèles TT compressés, PyTorch'
            }
        }
    
    def display_comparison(self):
        """Affiche comparaison"""
        print("\n" + "="*70)
        print("Comparaison TensorLy vs tntorch")
        print("="*70)
        
        for lib, info in self.comparison.items():
            print(f"\n{lib.upper()}:")
            print("  Forces:")
            for strength in info['strengths']:
                print(f"    + {strength}")
            print("  Faiblesses:")
            for weakness in info['weaknesses']:
                print(f"    - {weakness}")
            print(f"  Idéal pour: {info['best_for']}")

comparison = LibraryComparison()
comparison.display_comparison()
```

---

## Exemple Complet: Compression de Poids

### Application Réelle

```python
import tensorly as tl
import torch
from tensorly.decomposition import parafac

# Simuler poids d'une couche dense
original_weights = torch.randn(100, 50)  # Couche 100 → 50

# Tensoriser en 3D: (10, 10) × (10, 5)
weights_tensorized = original_weights.reshape(10, 10, 10, 5)

# Décomposition CP
rank = 8
factors = parafac(weights_tensorized, rank=rank)

# Reconstruire
reconstructed = tl.cp_to_tensor(factors)
reconstructed_weights = reconstructed.reshape(100, 50)

# Comparaison
original_params = original_weights.numel()  # 5000
compressed_params = sum(f.numel() for f in factors)  # ~800 (dépend rank)

compression_ratio = original_params / compressed_params
error = torch.norm(original_weights - reconstructed_weights) / torch.norm(original_weights)

print(f"\nCompression de Poids:")
print(f"  Paramètres originaux: {original_params}")
print(f"  Paramètres compressés: {compressed_params}")
print(f"  Ratio compression: {compression_ratio:.2f}×")
print(f"  Erreur relative: {error:.4f}")
```

---

## Exercices

### Exercice 23.1.1
Installez tensorly et testez décomposition CP, Tucker, et TT sur un tenseur 4D.

### Exercice 23.1.2
Comparez performance entre backend NumPy et PyTorch dans tensorly pour même décomposition.

### Exercice 23.1.3
Utilisez tntorch pour compresser un tenseur 5D et comparez erreur de reconstruction vs compression ratio.

### Exercice 23.1.4
Implémentez compression de couche dense avec décomposition CP en utilisant tensorly avec backend PyTorch.

---

## Points Clés à Retenir

> 📌 **TensorLy supporte multiple décompositions et backends**

> 📌 **tntorch est optimisé pour Tensor Train avec PyTorch**

> 📌 **Le choix de backend impact performance et fonctionnalités**

> 📌 **Les bibliothèques simplifient utilisation mais comprendre implémentation reste important**

> 📌 **La compression peut réduire paramètres significativement avec faible erreur**

---

*Section précédente : [23.0 Introduction](./23_introduction.md) | Section suivante : [23.2 Décomposition CP](./23_02_Decomposition_CP.md)*

