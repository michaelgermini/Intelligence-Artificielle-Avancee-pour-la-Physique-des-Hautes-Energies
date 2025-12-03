# Chapitre 5 : Décompositions Tensorielles Fondamentales

---

## Introduction

Les **décompositions tensorielles** sont au cœur de la compression de modèles par réseaux de tenseurs. Ce chapitre présente les principales décompositions : CP, Tucker, Tensor Train, et leurs variantes.

---

## Plan du Chapitre

1. [Décomposition CP (CANDECOMP/PARAFAC)](./05_01_CP.md)
2. [Décomposition Tucker](./05_02_Tucker.md)
3. [Tensor Train / Matrix Product States](./05_03_TensorTrain.md)
4. [Hierarchical Tucker](./05_04_HierarchicalTucker.md)
5. [Tensor Ring Decomposition](./05_05_TensorRing.md)
6. [Comparaison et Choix de Décomposition](./05_06_Comparaison.md)

---

## Vue d'Ensemble des Décompositions

```
┌─────────────────────────────────────────────────────────────────┐
│              Décompositions Tensorielles                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CP (Canonical Polyadic):                                       │
│    T = Σᵣ aᵣ ⊗ bᵣ ⊗ cᵣ                                         │
│    Paramètres: R(n₁ + n₂ + n₃)                                 │
│                                                                 │
│  Tucker:                                                        │
│    T = G ×₁ A ×₂ B ×₃ C                                        │
│    Paramètres: r₁r₂r₃ + n₁r₁ + n₂r₂ + n₃r₃                    │
│                                                                 │
│  Tensor Train (TT):                                             │
│    T[i₁,...,iₙ] = G₁[i₁] G₂[i₂] ... Gₙ[iₙ]                    │
│    Paramètres: Σₖ rₖ₋₁ nₖ rₖ                                   │
│                                                                 │
│  Tensor Ring (TR):                                              │
│    Comme TT mais avec trace finale                             │
│    Plus flexible, moins de contraintes aux bords               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Décomposition CP

```python
import numpy as np
from scipy.linalg import khatri_rao

def cp_decomposition_als(tensor, rank, max_iter=100, tol=1e-6):
    """
    Décomposition CP par Alternating Least Squares (ALS)
    
    T ≈ Σᵣ λᵣ a⁽ʳ⁾ ⊗ b⁽ʳ⁾ ⊗ c⁽ʳ⁾
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    # Initialisation aléatoire des facteurs
    factors = [np.random.randn(shape[i], rank) for i in range(n_modes)]
    
    for iteration in range(max_iter):
        for mode in range(n_modes):
            # Matricisation selon le mode
            unfolding = unfold(tensor, mode)
            
            # Produit Khatri-Rao des autres facteurs
            kr_product = khatri_rao_except(factors, mode)
            
            # Mise à jour du facteur (moindres carrés)
            factors[mode] = unfolding @ kr_product @ np.linalg.pinv(
                kr_product.T @ kr_product
            )
        
        # Vérification de convergence
        reconstruction = cp_reconstruct(factors)
        error = np.linalg.norm(tensor - reconstruction) / np.linalg.norm(tensor)
        
        if error < tol:
            break
    
    return factors, error

def unfold(tensor, mode):
    """Matricisation selon un mode"""
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)

def khatri_rao_except(factors, skip_mode):
    """Produit Khatri-Rao de tous les facteurs sauf un"""
    result = None
    for i, factor in enumerate(factors):
        if i != skip_mode:
            if result is None:
                result = factor
            else:
                result = khatri_rao(result, factor)
    return result

def cp_reconstruct(factors):
    """Reconstruit le tenseur à partir des facteurs CP"""
    result = factors[0]
    for factor in factors[1:]:
        result = np.einsum('ir,jr->ijr', result, factor)
    return result.sum(axis=-1)
```

---

## Décomposition Tucker

```python
def tucker_decomposition(tensor, ranks, max_iter=100):
    """
    Décomposition Tucker par Higher-Order SVD (HOSVD)
    
    T ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃
    
    G: tenseur noyau de shape (r₁, r₂, r₃)
    Uₖ: matrices facteurs de shape (nₖ, rₖ)
    """
    n_modes = tensor.ndim
    
    # HOSVD: SVD de chaque matricisation
    factors = []
    for mode in range(n_modes):
        unfolding = unfold(tensor, mode)
        U, _, _ = np.linalg.svd(unfolding, full_matrices=False)
        factors.append(U[:, :ranks[mode]])
    
    # Calcul du noyau
    core = tensor.copy()
    for mode in range(n_modes):
        core = mode_n_product(core, factors[mode].T, mode)
    
    return core, factors

def mode_n_product(tensor, matrix, mode):
    """Produit mode-n : T ×ₙ M"""
    # Déplace le mode en première position
    tensor = np.moveaxis(tensor, mode, 0)
    shape = tensor.shape
    
    # Reshape pour multiplication matricielle
    tensor = tensor.reshape(shape[0], -1)
    result = matrix @ tensor
    
    # Reshape et remet le mode à sa place
    new_shape = (matrix.shape[0],) + shape[1:]
    result = result.reshape(new_shape)
    result = np.moveaxis(result, 0, mode)
    
    return result

def tucker_reconstruct(core, factors):
    """Reconstruit le tenseur à partir de la décomposition Tucker"""
    result = core.copy()
    for mode, factor in enumerate(factors):
        result = mode_n_product(result, factor, mode)
    return result
```

---

## Tensor Train (TT)

```python
def tt_decomposition(tensor, max_rank=None, tol=1e-10):
    """
    Décomposition Tensor Train par TT-SVD
    
    T[i₁,...,iₙ] = G₁[i₁] × G₂[i₂] × ... × Gₙ[iₙ]
    
    Gₖ: tenseur 3D de shape (rₖ₋₁, nₖ, rₖ)
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    if max_rank is None:
        max_rank = [None] * (n_modes - 1)
    
    cores = []
    remainder = tensor.copy()
    rank_left = 1
    
    for k in range(n_modes - 1):
        # Reshape en matrice
        remainder = remainder.reshape(rank_left * shape[k], -1)
        
        # SVD tronquée
        U, S, Vt = np.linalg.svd(remainder, full_matrices=False)
        
        # Détermine le rang (par seuil ou max_rank)
        if max_rank[k] is not None:
            rank = min(max_rank[k], len(S))
        else:
            rank = np.sum(S > tol * S[0])
        rank = max(1, rank)
        
        # Tronque
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Core k
        core = U.reshape(rank_left, shape[k], rank)
        cores.append(core)
        
        # Prépare pour l'itération suivante
        remainder = np.diag(S) @ Vt
        rank_left = rank
    
    # Dernier core
    cores.append(remainder.reshape(rank_left, shape[-1], 1))
    
    return cores

def tt_reconstruct(cores):
    """Reconstruit le tenseur à partir des cores TT"""
    result = cores[0]
    for core in cores[1:]:
        # Contraction sur le dernier indice de result et le premier de core
        result = np.tensordot(result, core, axes=([-1], [0]))
    return result.squeeze()

def tt_ranks(cores):
    """Retourne les rangs TT"""
    return [core.shape[2] for core in cores[:-1]]

# Exemple
tensor = np.random.randn(10, 12, 8, 6)
cores = tt_decomposition(tensor, max_rank=[5, 5, 5])

print("Tensor Train Decomposition:")
print(f"  Original shape: {tensor.shape}")
print(f"  TT-ranks: {tt_ranks(cores)}")
print(f"  Core shapes: {[c.shape for c in cores]}")

# Compression
original_params = tensor.size
tt_params = sum(c.size for c in cores)
print(f"  Paramètres: {original_params:,} → {tt_params:,} ({tt_params/original_params:.1%})")

# Erreur
reconstruction = tt_reconstruct(cores)
error = np.linalg.norm(tensor - reconstruction) / np.linalg.norm(tensor)
print(f"  Erreur relative: {error:.2e}")
```

---

## Comparaison des Décompositions

```python
def compare_decompositions(tensor, ranks):
    """
    Compare les différentes décompositions sur un même tenseur
    """
    results = {}
    
    # CP
    cp_rank = ranks.get('cp', 10)
    factors, cp_error = cp_decomposition_als(tensor, cp_rank)
    cp_params = sum(f.size for f in factors)
    results['CP'] = {'error': cp_error, 'params': cp_params}
    
    # Tucker
    tucker_ranks = ranks.get('tucker', [5, 5, 5])
    core, factors = tucker_decomposition(tensor, tucker_ranks)
    tucker_recon = tucker_reconstruct(core, factors)
    tucker_error = np.linalg.norm(tensor - tucker_recon) / np.linalg.norm(tensor)
    tucker_params = core.size + sum(f.size for f in factors)
    results['Tucker'] = {'error': tucker_error, 'params': tucker_params}
    
    # TT
    tt_max_ranks = ranks.get('tt', [5, 5])
    cores = tt_decomposition(tensor, max_rank=tt_max_ranks)
    tt_recon = tt_reconstruct(cores)
    tt_error = np.linalg.norm(tensor - tt_recon) / np.linalg.norm(tensor)
    tt_params = sum(c.size for c in cores)
    results['TT'] = {'error': tt_error, 'params': tt_params}
    
    # Affichage
    print("Comparaison des décompositions:")
    print(f"Tenseur original: {tensor.shape}, {tensor.size:,} éléments")
    print("\n{:10} | {:>12} | {:>12} | {:>10}".format(
        "Méthode", "Paramètres", "Compression", "Erreur"))
    print("-" * 50)
    
    for name, res in results.items():
        compression = tensor.size / res['params']
        print(f"{name:10} | {res['params']:>12,} | {compression:>10.1f}x | {res['error']:>10.2e}")
    
    return results

# Test
tensor = np.random.randn(20, 25, 30)
ranks = {'cp': 10, 'tucker': [8, 8, 8], 'tt': [10, 10]}
compare_decompositions(tensor, ranks)
```

---

## Points Clés à Retenir

> 📌 **CP donne la représentation la plus compacte mais est NP-hard à calculer optimalement**

> 📌 **Tucker est flexible mais le noyau peut être grand (curse of dimensionality)**

> 📌 **TT évite la malédiction de la dimensionnalité avec une complexité linéaire en d**

> 📌 **Le choix dépend de la structure des données et des contraintes de l'application**

---

*Chapitre suivant : [Chapitre 6 - Réseaux de Tenseurs en Physique Quantique](../Chapitre_06_Physique_Quantique/06_introduction.md)*

