# 7.3 Formats TT pour les Poids de Réseaux

---

## Introduction

Le **Tensor Train (TT)** est un format particulièrement efficace pour compresser les grandes matrices de poids. Cette section détaille les différents formats TT et leur utilisation pour les réseaux de neurones.

---

## Structure Tensor Train

### Rappel : Définition TT

Un Tensor Train représente une matrice $W \in \mathbb{R}^{m \times n}$ comme :

$$W[i, j] = \sum_{r_1, \ldots, r_{d-1}} G_1[i_1, r_1] \cdot G_2[r_1, i_2, r_2] \cdots G_d[r_{d-1}, j_d]$$

où $i = (i_1, \ldots, i_k)$, $j = (j_1, \ldots, j_l)$ sont des indices factorisés.

---

## Formats TT pour Matrices

### TT-Matrix

```python
import torch
import torch.nn as nn
import numpy as np

class TTMatrix:
    """
    Représente une matrice en format Tensor Train
    """
    
    def __init__(self, input_dims, output_dims, tt_ranks):
        """
        Args:
            input_dims: tuple (d₁, ..., dₖ) factorisant les lignes
            output_dims: tuple (d'₁, ..., d'ₗ) factorisant les colonnes
            tt_ranks: liste des rangs TT [r₁, ..., r_{k+l-1}]
        """
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.tt_ranks = tt_ranks
        
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Crée les cores TT
        self.cores = []
        
        total_dims = len(input_dims) + len(output_dims)
        
        # Cores pour les dimensions d'entrée
        prev_rank = 1
        for i, dim in enumerate(input_dims):
            next_rank = tt_ranks[i]
            # Core: (prev_rank, dim, next_rank)
            core = np.random.randn(prev_rank, dim, next_rank)
            self.cores.append(core)
            prev_rank = next_rank
        
        # Cores pour les dimensions de sortie
        for i, dim in enumerate(output_dims):
            if i < len(output_dims) - 1:
                next_rank = tt_ranks[len(input_dims) + i]
            else:
                next_rank = 1
            
            # Core: (prev_rank, dim, next_rank)
            core = np.random.randn(prev_rank, dim, next_rank)
            self.cores.append(core)
            prev_rank = next_rank
    
    def to_matrix(self):
        """
        Reconstruit la matrice complète (coûteux!)
        """
        # Contracte tous les cores
        result = self.cores[0]
        for core in self.cores[1:]:
            result = np.tensordot(result, core, axes=([-1], [0]))
        
        # Reshape en matrice
        result = result.reshape(self.input_size, self.output_size)
        return result
    
    def matvec(self, x):
        """
        Produit matrice-vecteur: W @ x
        
        Efficace sans reconstruire la matrice complète
        """
        # Reshape input
        x = x.reshape(*self.input_dims)
        
        # Contracte avec les cores d'entrée
        result = x
        for i, core in enumerate(self.cores[:len(self.input_dims)]):
            # Contracte sur la dimension i
            result = np.tensordot(result, core, axes=([i], [1]))
            # Réarrange
            # (Simplifié - nécessite reshape approprié)
            pass
        
        # Contracte avec les cores de sortie
        for core in self.cores[len(self.input_dims):]:
            result = np.tensordot(result, core, axes=([0], [0]))
        
        return result.flatten()
    
    def count_parameters(self):
        """Nombre de paramètres"""
        return sum(c.size for c in self.cores)

# Exemple
tt_matrix = TTMatrix(
    input_dims=(16, 16, 4),   # 1024 = 16×16×4
    output_dims=(16, 16, 2),  # 512 = 16×16×2
    tt_ranks=[4, 4, 4, 4, 4]
)

print("TT-Matrix:")
print(f"  Input: {tt_matrix.input_size}, Output: {tt_matrix.output_size}")
print(f"  Paramètres: {tt_matrix.count_parameters():,}")
print(f"  vs Matrice dense: {tt_matrix.input_size * tt_matrix.output_size:,}")
print(f"  Compression: {tt_matrix.input_size * tt_matrix.output_size / tt_matrix.count_parameters():.2f}x")
```

---

## Conversion Matrice → TT

### TT-SVD pour Matrices

```python
def matrix_to_tt_svd(W, input_dims, output_dims, max_rank=None, tolerance=1e-6):
    """
    Convertit une matrice en format TT via SVD
    
    Args:
        W: matrice (m, n)
        input_dims: tuple factorisant m
        output_dims: tuple factorisant n
        max_rank: rang TT maximal
        tolerance: tolérance pour troncature
    """
    assert np.prod(input_dims) == W.shape[0], \
        f"∏input_dims doit égaler {W.shape[0]}"
    assert np.prod(output_dims) == W.shape[1], \
        f"∏output_dims doit égaler {W.shape[1]}"
    
    # Reshape en tenseur 4D: (d₁, ..., dₖ, d'₁, ..., d'ₗ)
    W_tensor = W.reshape(*input_dims, *output_dims)
    
    # TT-SVD
    cores = tt_svd_decomposition(W_tensor, max_rank, tolerance)
    
    # Crée TTMatrix
    if max_rank is None:
        # Infère les rangs depuis les cores
        tt_ranks = [c.shape[2] for c in cores[:-1]]
    else:
        tt_ranks = [max_rank] * (len(input_dims) + len(output_dims) - 1)
    
    tt_matrix = TTMatrix(input_dims, output_dims, tt_ranks)
    tt_matrix.cores = cores
    
    return tt_matrix

def tt_svd_decomposition(tensor, max_rank=None, tolerance=1e-6):
    """
    Décomposition TT-SVD d'un tenseur
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
        U, S, Vt = np.linalg.svd(remainder, full_matrices=False)
        
        # Troncature
        if max_rank is not None:
            rank = min(max_rank, len(S))
        else:
            rank = np.sum(S > tolerance * S[0])
        
        rank = max(1, rank)
        
        # Core k
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]
        
        core = U_trunc.reshape(rank_left, shape[k], rank)
        cores.append(core)
        
        # Prépare pour l'itération suivante
        remainder = np.diag(S_trunc) @ Vt_trunc
        rank_left = rank
    
    # Dernier core
    cores.append(remainder.reshape(rank_left, shape[-1], 1))
    
    return cores

# Test
W = np.random.randn(1024, 512)
tt_W = matrix_to_tt_svd(W, 
                        input_dims=(16, 16, 4),
                        output_dims=(16, 16, 2),
                        max_rank=8)

print(f"Conversion Matrice → TT:")
print(f"  Erreur reconstruction: {np.linalg.norm(W - tt_W.to_matrix()) / np.linalg.norm(W):.6f}")
```

---

## Formats TT pour PyTorch

### Couche Linéaire TT

```python
class TTLinear(nn.Module):
    """
    Couche linéaire avec poids en format TT
    """
    
    def __init__(self, input_dims, output_dims, tt_ranks, bias=True):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Normalise tt_ranks
        if isinstance(tt_ranks, int):
            total_cores = len(input_dims) + len(output_dims)
            tt_ranks = [tt_ranks] * (total_cores - 1)
        
        self.tt_ranks = tt_ranks
        
        # Crée les cores comme Paramètres
        self.cores = nn.ModuleList()
        
        prev_rank = 1
        
        # Cores d'entrée
        for i, dim in enumerate(input_dims):
            if i < len(input_dims) - 1:
                next_rank = tt_ranks[i]
            else:
                next_rank = tt_ranks[len(input_dims) - 1]
            
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            nn.init.xavier_uniform_(core)
            self.cores.append(core)
            prev_rank = next_rank
        
        # Cores de sortie
        for i, dim in enumerate(output_dims):
            if i < len(output_dims) - 1:
                next_rank = tt_ranks[len(input_dims) + i]
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
        """Forward pass avec contraction TT"""
        batch_size = x.shape[0]
        x = x.view(batch_size, *self.input_dims)
        
        # Contracte avec les cores
        result = x
        
        for i, core in enumerate(self.cores[:len(self.input_dims)]):
            result = torch.tensordot(result, core, dims=([i+1], [1]))
            # Réarrange les dimensions
            perm = list(range(result.dim()))
            perm.remove(i+1)
            perm.insert(1, i+1)
            result = result.permute(*perm)
        
        for core in self.cores[len(self.input_dims):]:
            result = torch.tensordot(result, core, dims=([1], [0]))
        
        result = result.view(batch_size, self.output_size)
        
        if self.bias is not None:
            result = result + self.bias
        
        return result
```

---

## Optimisation du Format TT

### Rounding TT

```python
def tt_rounding(tt_matrix, max_rank, tolerance=1e-10):
    """
    Arrondit un TT pour réduire les rangs
    
    Utilise SVD pour compresser chaque core
    """
    cores = tt_matrix.cores.copy()
    n_cores = len(cores)
    
    # Passe gauche-droite
    for k in range(n_cores - 1):
        core = cores[k]
        
        # Reshape: (r_{k-1} × d_k, r_k)
        matrix = core.reshape(-1, core.shape[2])
        
        # SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Troncature
        rank = min(max_rank, np.sum(S > tolerance * S[0]))
        rank = max(1, rank)
        
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Mise à jour
        cores[k] = U.reshape(core.shape[0], core.shape[1], rank)
        
        # Absorbe S × Vt dans le core suivant
        if k < n_cores - 1:
            cores[k+1] = np.tensordot(
                np.diag(S) @ Vt,
                cores[k+1],
                axes=([1], [0])
            )
    
    # Crée nouveau TT avec cores arrondis
    rounded = TTMatrix(tt_matrix.input_dims, tt_matrix.output_dims, 
                      [c.shape[2] for c in cores[:-1]])
    rounded.cores = cores
    
    return rounded
```

### Initialisation Adaptative

```python
def initialize_tt_from_distribution(input_dims, output_dims, tt_ranks, 
                                   distribution='xavier'):
    """
    Initialise un TT avec une distribution adaptée
    """
    tt_matrix = TTMatrix(input_dims, output_dims, tt_ranks)
    
    if distribution == 'xavier':
        for core in tt_matrix.cores:
            scale = np.sqrt(2.0 / (core.shape[0] * core.shape[1]))
            core[:] = np.random.randn(*core.shape) * scale
    elif distribution == 'he':
        for core in tt_matrix.cores:
            scale = np.sqrt(2.0 / core.shape[1])
            core[:] = np.random.randn(*core.shape) * scale
    
    return tt_matrix
```

---

## Formats Spécialisés

### TT avec Rangs Variables

```python
class AdaptiveTTMatrix(TTMatrix):
    """
    TT avec rangs adaptatifs selon l'importance
    """
    
    def __init__(self, input_dims, output_dims, base_rank, importance_weights=None):
        """
        Args:
            importance_weights: poids d'importance pour ajuster les rangs
        """
        # Calcule les rangs adaptatifs
        if importance_weights is None:
            tt_ranks = [base_rank] * (len(input_dims) + len(output_dims) - 1)
        else:
            # Ajuste les rangs selon l'importance
            tt_ranks = [int(base_rank * w) for w in importance_weights]
        
        super().__init__(input_dims, output_dims, tt_ranks)
```

---

## Comparaison avec Autres Formats

```python
def compare_compression_formats(W, input_dims, output_dims):
    """
    Compare TT, SVD, et matrice dense
    """
    m, n = W.shape
    
    # SVD (low-rank)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank_svd = min(m, n) // 4
    W_svd = U[:, :rank_svd] @ np.diag(S[:rank_svd]) @ Vt[:rank_svd, :]
    params_svd = m * rank_svd + rank_svd * n
    
    # TT
    tt_W = matrix_to_tt_svd(W, input_dims, output_dims, max_rank=8)
    W_tt = tt_W.to_matrix()
    params_tt = tt_W.count_parameters()
    
    print("Comparaison formats compression:")
    print(f"  Original: {m * n:,} paramètres")
    print(f"  SVD (rank={rank_svd}): {params_svd:,} params, "
          f"erreur: {np.linalg.norm(W - W_svd) / np.linalg.norm(W):.6f}")
    print(f"  TT: {params_tt:,} params, "
          f"erreur: {np.linalg.norm(W - W_tt) / np.linalg.norm(W):.6f}")
    
    return {
        'svd': (params_svd, np.linalg.norm(W - W_svd)),
        'tt': (params_tt, np.linalg.norm(W - W_tt))
    }

W_test = np.random.randn(1024, 512)
compare_compression_formats(W_test, 
                           input_dims=(16, 16, 4),
                           output_dims=(16, 16, 2))
```

---

## Exercices

### Exercice 7.3.1
Implémentez une fonction qui trouve automatiquement la meilleure factorisation pour minimiser le nombre de paramètres TT.

### Exercice 7.3.2
Comparez les performances (vitesse, mémoire) d'une multiplication matrice-vecteur avec TT vs SVD.

### Exercice 7.3.3
Implémentez un algorithme de rounding TT adaptatif qui ajuste les rangs selon l'erreur locale.

---

## Points Clés à Retenir

> 📌 **TT est particulièrement efficace pour les grandes matrices**

> 📌 **La factorisation des dimensions est cruciale pour la compression**

> 📌 **TT-SVD permet une conversion avec contrôle de l'erreur**

> 📌 **Le rounding TT permet de réduire encore les rangs**

> 📌 **Les formats TT peuvent être intégrés directement dans PyTorch**

---

*Section suivante : [7.4 Entraînement Bout-en-Bout avec Contraintes Tensorielles](./07_04_Entrainement.md)*

