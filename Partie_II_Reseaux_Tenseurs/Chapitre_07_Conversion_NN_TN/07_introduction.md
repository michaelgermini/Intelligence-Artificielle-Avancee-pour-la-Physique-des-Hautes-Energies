# Chapitre 7 : Conversion de Réseaux de Neurones en Réseaux de Tenseurs

---

## Introduction

Ce chapitre présente les techniques pour convertir des réseaux de neurones classiques en réseaux de tenseurs, permettant ainsi d'exploiter la structure tensorielle pour la compression.

---

## Plan du Chapitre

1. [Tensorisation des Couches Denses](./07_01_Tensorisation_Dense.md)
2. [Tensorisation des Couches Convolutionnelles](./07_02_Tensorisation_Conv.md)
3. [Formats TT pour les Poids de Réseaux](./07_03_Formats_TT.md)
4. [Entraînement Bout-en-Bout avec Contraintes Tensorielles](./07_04_Entrainement.md)
5. [Analyse de la Perte d'Expressivité](./07_05_Expressivite.md)

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│              Conversion NN → Réseau de Tenseurs                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Réseau Standard                Réseau Tensoriel                │
│  ──────────────                ────────────────                │
│                                                                 │
│  Linear(1024, 512)      →      TT-Layer(rank=32)              │
│     512K params                    32K params                   │
│                                                                 │
│  Conv2d(64, 128, 3)     →      Tensorized Conv                 │
│     184K params                    40K params                   │
│                                                                 │
│  Avantage: Compression significative avec perte minimale       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tensorisation des Couches Denses

```python
import torch
import torch.nn as nn
import numpy as np

class TensorizedLinear(nn.Module):
    """
    Couche linéaire tensorisée en format Tensor Train
    """
    
    def __init__(self, input_dims, output_dims, rank):
        """
        Args:
            input_dims: tuple (d₁, d₂, ..., dₙ) pour reshape de l'input
            output_dims: tuple (d'₁, d'₂, ..., d'ₘ) pour reshape de l'output
            rank: rang TT
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Crée les cores TT
        self.cores = nn.ModuleList()
        
        n_cores = len(input_dims) + len(output_dims)
        prev_rank = 1
        
        # Cores pour les dimensions d'entrée
        for i, dim in enumerate(input_dims):
            next_rank = rank if i < len(input_dims) - 1 else rank
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            self.cores.append(core)
            prev_rank = next_rank
        
        # Cores pour les dimensions de sortie
        for i, dim in enumerate(output_dims):
            next_rank = rank if i < len(output_dims) - 1 else 1
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            self.cores.append(core)
            prev_rank = next_rank
        
    def forward(self, x):
        """
        Forward pass avec contraction TT
        """
        batch_size = x.shape[0]
        
        # Reshape l'input
        x = x.view(batch_size, *self.input_dims)
        
        # Contracte avec les cores d'entrée
        result = x
        for i in range(len(self.input_dims)):
            # result: (batch, d₁, ..., dᵢ, ...)
            # core: (r_{i-1}, d_i, r_i)
            
            # Contracte sur la dimension i
            result = torch.tensordot(result, self.cores[i], dims=([i+1], [1]))
            # Nouvelle shape: (batch, d₁, ..., d_{i-1}, r_i, d_{i+1}, ...)
            
            # Réarrange pour mettre r_i au bon endroit
            perm = list(range(result.dim()))
            perm.insert(-len(self.input_dims)+i, perm.pop(i+1))
            result = result.permute(perm)
        
        # Contracte avec les cores de sortie
        for i in range(len(self.output_dims)):
            core = self.cores[len(self.input_dims) + i]
            # Contracte et ajoute dimension de sortie
            result = torch.tensordot(result, core, dims=([1], [0]))
            # Réarrange
            result = result.permute(0, 2, 1)
        
        # Reshape finale
        result = result.view(batch_size, self.output_size)
        
        return result

# Exemple: tensorisation d'une couche 1024 → 512
# Reshape: 1024 = 16×16×4, 512 = 16×16×2
tensorized = TensorizedLinear(
    input_dims=(16, 16, 4),
    output_dims=(16, 16, 2),
    rank=8
)

print(f"Tensorized Layer:")
print(f"  Input: {tensorized.input_size}, Output: {tensorized.output_size}")
print(f"  Cores: {len(tensorized.cores)}")
total_params = sum(c.numel() for c in tensorized.cores)
print(f"  Paramètres: {total_params:,}")
print(f"  vs couche dense: {tensorized.input_size * tensorized.output_size:,}")
print(f"  Compression: {tensorized.input_size * tensorized.output_size / total_params:.2f}x")
```

---

## Conversion Automatique

```python
def convert_dense_to_tensorized(linear_layer, input_dims, output_dims, rank):
    """
    Convertit une couche dense en couche tensorisée
    
    Utilise la SVD pour initialiser les cores
    """
    # Matrice de poids originale
    W = linear_layer.weight.data  # (out_features, in_features)
    
    # Reshape en tenseur
    W_tensor = W.view(*output_dims, *input_dims)
    
    # Convertit en Tensor Train
    from tensor_compression import tensor_train_decomposition
    
    cores = tensor_train_decomposition(W_tensor, max_rank=rank)
    
    # Crée la couche tensorisée
    tensorized = TensorizedLinear(input_dims, output_dims, rank)
    
    # Initialise avec les cores
    for i, core in enumerate(cores):
        tensorized.cores[i].data = core
    
    return tensorized
```

---

## Points Clés à Retenir

> 📌 **La tensorisation réduit exponentiellement le nombre de paramètres**

> 📌 **Le choix des facteurs de reshape est crucial pour la compression**

> 📌 **L'entraînement doit respecter la structure tensorielle**

> 📌 **La perte d'expressivité peut être minimisée avec un rang suffisant**

---

*Section suivante : [7.1 Tensorisation des Couches Denses](./07_01_Tensorisation_Dense.md)*

