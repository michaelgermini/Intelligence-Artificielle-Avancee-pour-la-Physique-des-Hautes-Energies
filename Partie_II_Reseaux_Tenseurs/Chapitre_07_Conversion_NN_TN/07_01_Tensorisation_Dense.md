# 7.1 Tensorisation des Couches Denses

---

## Introduction

La **tensorisation** des couches denses consiste à représenter les matrices de poids comme des réseaux de tenseurs (typiquement Tensor Train). Cela permet de réduire drastiquement le nombre de paramètres tout en préservant l'expressivité pour de nombreuses applications.

---

## Motivation

### Problème des Couches Denses

Une couche dense `Linear(m, n)` a :
- **m × n paramètres**
- Pour m=1024, n=512 → **524,288 paramètres**

### Solution : Tensorisation

En factorisant les dimensions et utilisant Tensor Train :
- **O(rank × (m + n)) paramètres**
- Pour rank=32 → **~50,000 paramètres**
- **Compression ~10x** avec perte minimale

---

## Principe de Tensorisation

### Factorisation des Dimensions

Une matrice $W \in \mathbb{R}^{m \times n}$ est reshapée en tenseur :
- $m = m_1 \times m_2 \times \cdots \times m_k$
- $n = n_1 \times n_2 \times \cdots \times n_l$
- $W \in \mathbb{R}^{m_1 \times m_2 \times \cdots \times m_k \times n_1 \times n_2 \times \cdots \times n_l}$

Puis décomposée en Tensor Train (TT).

---

## Implémentation

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import svd

class TensorizedLinear(nn.Module):
    """
    Couche linéaire tensorisée en format Tensor Train
    """
    
    def __init__(self, input_dims, output_dims, tt_rank, bias=True):
        """
        Args:
            input_dims: tuple (d₁, d₂, ..., dₙ) pour factoriser input
            output_dims: tuple (d'₁, d'₂, ..., d'ₘ) pour factoriser output
            tt_rank: rang Tensor Train (peut être liste ou scalaire)
            bias: utiliser un biais ou non
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Normalise tt_rank
        if isinstance(tt_rank, (int, float)):
            total_cores = len(input_dims) + len(output_dims)
            self.tt_ranks = [int(tt_rank)] * (total_cores - 1)
        else:
            self.tt_ranks = list(tt_rank)
        
        # Crée les cores TT
        self.cores = nn.ModuleList()
        
        prev_rank = 1
        
        # Cores pour les dimensions d'entrée
        for i, dim in enumerate(input_dims):
            if i < len(input_dims) - 1:
                next_rank = self.tt_ranks[i]
            else:
                # Dernier core d'entrée connecte à la sortie
                next_rank = self.tt_ranks[len(input_dims) - 1]
            
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            nn.init.xavier_uniform_(core)
            self.cores.append(core)
            prev_rank = next_rank
        
        # Cores pour les dimensions de sortie
        for i, dim in enumerate(output_dims):
            if i < len(output_dims) - 1:
                next_rank = self.tt_ranks[len(input_dims) + i]
            else:
                next_rank = 1
            
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            nn.init.xavier_uniform_(core)
            self.cores.append(core)
            prev_rank = next_rank
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.output_size))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """
        Forward pass: y = TT(x) + b
        
        Args:
            x: (batch, input_size)
        
        Returns:
            y: (batch, output_size)
        """
        batch_size = x.shape[0]
        
        # Reshape input
        x = x.view(batch_size, *self.input_dims)
        
        # Contracte avec les cores d'entrée
        result = x
        
        for i, core in enumerate(self.cores[:len(self.input_dims)]):
            # Contracte sur la dimension i+1 (après batch)
            # result: (batch, d₁, ..., dᵢ, d_{i+1}, ..., dₙ)
            # core: (r_{i-1}, d_i, r_i)
            
            result = torch.tensordot(result, core, dims=([i+1], [1]))
            # Résultat: (batch, d₁, ..., dᵢ, r_i, d_{i+2}, ..., dₙ)
            
            # Réarrange: déplace r_i après batch
            n_dims = result.dim()
            perm = list(range(n_dims))
            perm.remove(i+1)  # Enlève la position de r_i
            perm.insert(1, i+1)  # Remet r_i après batch
            result = result.permute(*perm)
        
        # Contracte avec les cores de sortie
        for i, core in enumerate(self.cores[len(self.input_dims):]):
            # Contracte avec le core de sortie
            result = torch.tensordot(result, core, dims=([1], [0]))
            # Réarrange si nécessaire
            if i < len(self.output_dims) - 1:
                result = result.permute(0, 2, 1)
        
        # Reshape final
        result = result.view(batch_size, self.output_size)
        
        # Ajoute bias
        if self.bias is not None:
            result = result + self.bias
        
        return result
    
    def count_parameters(self):
        """Compte le nombre de paramètres"""
        params = sum(c.numel() for c in self.cores)
        if self.bias is not None:
            params += self.bias.numel()
        return params
    
    def compression_ratio(self):
        """Ratio de compression"""
        dense_params = self.input_size * self.output_size
        return dense_params / self.count_parameters()

# Exemple
tensorized = TensorizedLinear(
    input_dims=(16, 16, 4),  # 1024 = 16×16×4
    output_dims=(16, 16, 2),  # 512 = 16×16×2
    tt_rank=8
)

print("Couche Linéaire Tensorisée:")
print(f"  Input: {tensorized.input_size}, Output: {tensorized.output_size}")
print(f"  Paramètres: {tensorized.count_parameters():,}")
print(f"  Dense équivalent: {tensorized.input_size * tensorized.output_size:,}")
print(f"  Compression: {tensorized.compression_ratio():.2f}x")
```

---

## Conversion depuis une Couche Dense

### Méthode 1 : TT-SVD

```python
def dense_to_tensorized_svd(linear_layer, input_dims, output_dims, max_rank):
    """
    Convertit une couche dense en tensorisée via TT-SVD
    
    Args:
        linear_layer: nn.Linear
        input_dims: tuple factorisant in_features
        output_dims: tuple factorisant out_features
        max_rank: rang TT maximal
    """
    W = linear_layer.weight.data.numpy()  # (out_features, in_features)
    
    # Vérifie les dimensions
    assert np.prod(input_dims) == W.shape[1], \
        f"∏input_dims ({np.prod(input_dims)}) doit égaler in_features ({W.shape[1]})"
    assert np.prod(output_dims) == W.shape[0], \
        f"∏output_dims ({np.prod(output_dims)}) doit égaler out_features ({W.shape[0]})"
    
    # Reshape en tenseur: (d'₁, ..., d'ₘ, d₁, ..., dₙ)
    W_tensor = W.reshape(*output_dims, *input_dims)
    
    # Convertit en Tensor Train
    cores = tt_svd_decomposition(W_tensor, max_rank=max_rank)
    
    # Crée la couche tensorisée
    tensorized = TensorizedLinear(input_dims, output_dims, tt_rank=max_rank)
    
    # Initialise les cores
    for i, core in enumerate(cores):
        tensorized.cores[i].data = torch.from_numpy(core).float()
    
    # Copie le bias
    if linear_layer.bias is not None:
        tensorized.bias.data = linear_layer.bias.data.clone()
    
    return tensorized

def tt_svd_decomposition(tensor, max_rank=None, tolerance=1e-6):
    """
    Décomposition TT-SVD d'un tenseur
    
    Args:
        tensor: tenseur NumPy à décomposer
        max_rank: rang TT maximal
        tolerance: tolérance pour la troncature
    
    Returns:
        Liste de cores TT
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    cores = []
    remainder = tensor.copy()
    rank_left = 1
    
    for k in range(n_modes - 1):
        # Reshape pour SVD
        remainder = remainder.reshape(rank_left * shape[k], -1)
        
        # SVD
        U, S, Vt = svd(remainder, full_matrices=False)
        
        # Troncature
        if max_rank is not None:
            rank = min(max_rank, len(S))
        else:
            # Troncature par tolérance
            rank = np.sum(S > tolerance * S[0])
        
        rank = max(1, rank)
        
        # Core k
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]
        
        # Forme le core: (rank_left, shape[k], rank)
        core = U_trunc.reshape(rank_left, shape[k], rank)
        cores.append(core)
        
        # Prépare pour l'itération suivante
        remainder = np.diag(S_trunc) @ Vt_trunc
        rank_left = rank
    
    # Dernier core
    cores.append(remainder.reshape(rank_left, shape[-1], 1))
    
    return cores

# Test de conversion
original = nn.Linear(1024, 512)
input_tensor = torch.randn(32, 1024)

# Convertit en tensorisé
tensorized = dense_to_tensorized_svd(
    original,
    input_dims=(16, 16, 4),
    output_dims=(16, 16, 2),
    max_rank=8
)

# Test forward
output_original = original(input_tensor)
output_tensorized = tensorized(input_tensor)

print(f"\nConversion Dense → Tensorisé:")
print(f"  Erreur relative: {torch.norm(output_original - output_tensorized) / torch.norm(output_original):.4f}")
```

### Méthode 2 : Initialisation Aléatoire puis Fine-tuning

```python
def create_tensorized_with_same_output_shape(linear_layer, input_dims, output_dims, tt_rank):
    """
    Crée une couche tensorisée avec la même forme de sortie
    
    Initialise aléatoirement (pour entraînement de zéro)
    """
    tensorized = TensorizedLinear(input_dims, output_dims, tt_rank)
    
    # Copie le bias si présent
    if linear_layer.bias is not None and tensorized.bias is not None:
        tensorized.bias.data = linear_layer.bias.data.clone()
    
    return tensorized
```

---

## Factorisation des Dimensions

### Stratégies de Factorisation

```python
def factorize_dimension(n, preferred_factors=None):
    """
    Factorise une dimension en facteurs
    
    Args:
        n: dimension à factoriser
        preferred_factors: facteurs préférés (ex: puissances de 2)
    
    Returns:
        Liste de facteurs dont le produit = n
    """
    if preferred_factors is None:
        preferred_factors = [2, 3, 4, 5, 7, 8, 9, 16, 32]
    
    # Trouve la factorisation avec les facteurs préférés
    factors = []
    remainder = n
    
    # Essaie les facteurs dans l'ordre décroissant
    for factor in sorted(preferred_factors, reverse=True):
        while remainder % factor == 0:
            factors.append(factor)
            remainder = remainder // factor
    
    # Si reste > 1, ajoute le reste comme facteur
    if remainder > 1:
        factors.append(remainder)
    
    return factors

def find_optimal_factorization(n, max_factors=5):
    """
    Trouve une factorisation optimale (facteurs proches de √n)
    """
    import math
    
    # Cible: facteurs proches de √n
    target = math.sqrt(n)
    
    # Essaie différentes factorisations
    best_factors = None
    best_score = float('inf')
    
    # Diviseurs simples
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors = [i, n // i]
            # Continue à factoriser si nécessaire
            while len(factors) < max_factors:
                max_factor = max(factors)
                # Essaie de factoriser le plus grand facteur
                # (Simplifié)
                break
            
            # Score: variance des facteurs (plus proche de target = mieux)
            score = sum((f - target)**2 for f in factors)
            if score < best_score:
                best_score = score
                best_factors = factors
    
    return best_factors if best_factors else [n]

# Exemples
print("Factorisations:")
print(f"  1024: {factorize_dimension(1024)}")
print(f"  784: {factorize_dimension(784)}")
print(f"  512: {factorize_dimension(512)}")
print(f"  256: {factorize_dimension(256)}")
```

---

## Optimisation de la Compression

### Choix du Rang TT

```python
def find_optimal_tt_rank(linear_layer, input_dims, output_dims, target_compression):
    """
    Trouve le rang TT optimal pour une compression cible
    """
    target_params = (linear_layer.weight.numel()) / target_compression
    
    # Estime le rang nécessaire
    # Approximation: params ≈ rank × (Σ input_dims + Σ output_dims)
    total_dims = sum(input_dims) + sum(output_dims)
    estimated_rank = int(target_params / total_dims)
    
    # Essaie différents rangs autour de l'estimation
    best_rank = None
    best_error = float('inf')
    
    for rank in [max(1, estimated_rank - 5), estimated_rank, estimated_rank + 5]:
        tensorized = dense_to_tensorized_svd(
            linear_layer,
            input_dims,
            output_dims,
            max_rank=rank
        )
        
        # Teste la reconstruction
        test_input = torch.randn(100, linear_layer.in_features)
        output_original = linear_layer(test_input)
        output_tensorized = tensorized(test_input)
        
        error = torch.norm(output_original - output_tensorized) / torch.norm(output_original)
        
        compression = linear_layer.weight.numel() / tensorized.count_parameters()
        
        if compression >= target_compression and error < best_error:
            best_error = error
            best_rank = rank
    
    return best_rank, best_error

# Exemple
linear_test = nn.Linear(1024, 512)
optimal_rank, error = find_optimal_tt_rank(
    linear_test,
    input_dims=(16, 16, 4),
    output_dims=(16, 16, 2),
    target_compression=8
)

print(f"Rang TT optimal: {optimal_rank}, Erreur: {error:.4f}")
```

---

## Entraînement Direct

### Initialisation pour Entraînement

```python
def initialize_tensorized_for_training(input_dims, output_dims, tt_rank, init_scale=0.1):
    """
    Initialise une couche tensorisée pour l'entraînement
    
    Utilise une initialisation adaptée aux réseaux de tenseurs
    """
    tensorized = TensorizedLinear(input_dims, output_dims, tt_rank)
    
    # Initialise chaque core avec variance adaptée
    for i, core in enumerate(tensorized.cores):
        # Variance adaptée pour maintenir la variance du produit
        scale = init_scale / np.sqrt(core.shape[1])
        core.data = torch.randn_like(core) * scale
    
    return tensorized
```

---

## Exercices

### Exercice 7.1.1
Implémentez une fonction qui trouve automatiquement la meilleure factorisation pour tensoriser une couche dense.

### Exercice 7.1.2
Comparez différentes stratégies de conversion (TT-SVD vs initialisation aléatoire) sur un petit réseau.

### Exercice 7.1.3
Mesurez la perte d'expressivité en fonction du rang TT pour différentes architectures.

---

## Points Clés à Retenir

> 📌 **La tensorisation réduit exponentiellement le nombre de paramètres**

> 📌 **La factorisation des dimensions est cruciale pour la compression**

> 📌 **TT-SVD permet de convertir une couche existante avec perte minimale**

> 📌 **Le choix du rang TT est un compromis compression/performance**

> 📌 **L'entraînement direct est possible mais nécessite une bonne initialisation**

---

*Section suivante : [7.2 Tensorisation des Couches Convolutionnelles](./07_02_Tensorisation_Conv.md)*

