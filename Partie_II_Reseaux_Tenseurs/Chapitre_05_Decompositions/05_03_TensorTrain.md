# 5.3 Tensor Train (TT) / Matrix Product States (MPS)

---

## Introduction

Le **Tensor Train** (TT) ou **Matrix Product State** (MPS) évite la malédiction de la dimensionnalité en représentant un tenseur comme un produit de tenseurs 3D. C'est particulièrement efficace pour les tenseurs de grande dimension.

---

## Définition Formelle

Pour un tenseur $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$, la décomposition TT est :

$$\mathcal{T}[i_1, i_2, \ldots, i_N] = \sum_{r_0=1}^{R_0} \sum_{r_1=1}^{R_1} \cdots \sum_{r_{N-1}=1}^{R_{N-1}} G_1[r_0, i_1, r_1] \cdot G_2[r_1, i_2, r_2] \cdots G_N[r_{N-1}, i_N, r_N]$$

En notation matricielle :

$$\mathcal{T}[i_1, \ldots, i_N] = G_1[i_1] \cdot G_2[i_2] \cdots G_N[i_N]$$

où $G_k[i_k]$ est une matrice de taille $(R_{k-1}, R_k)$.

---

## Structure TT

```python
import numpy as np

class TensorTrain:
    """
    Représentation Tensor Train d'un tenseur
    """
    
    def __init__(self, cores):
        """
        Args:
            cores: Liste de tenseurs 3D [G₁, G₂, ..., Gₙ]
                   Gₖ.shape = (r_{k-1}, i_k, r_k)
        """
        self.cores = cores
        self.n_modes = len(cores)
        self.local_dims = [core.shape[1] for core in cores]
        self.ranks = [cores[0].shape[0]] + [core.shape[2] for core in cores]
        
    def reconstruct(self):
        """
        Reconstruit le tenseur complet (coûteux!)
        """
        result = self.cores[0]  # Shape: (r₀, i₁, r₁)
        
        for core in self.cores[1:]:
            # Contracte: (..., r_{k-1}) × (r_{k-1}, i_k, r_k) → (..., i_k, r_k)
            result = np.tensordot(result, core, axes=([-1], [0]))
        
        # Squeeze les dimensions de liaison aux bords
        return result.squeeze()
    
    def count_parameters(self):
        """Compte le nombre de paramètres"""
        return sum(core.size for core in self.cores)
    
    def full_tensor_size(self):
        """Taille du tenseur complet"""
        return np.prod(self.local_dims)

# Exemple
cores = [
    np.random.randn(1, 5, 4),   # G₁: (1, 5, 4)
    np.random.randn(4, 6, 3),   # G₂: (4, 6, 3)
    np.random.randn(3, 7, 1)    # G₃: (3, 7, 1)
]

tt = TensorTrain(cores)
print(f"Tensor Train:")
print(f"  Cores: {len(cores)}")
print(f"  TT-ranks: {tt.ranks}")
print(f"  Paramètres: {tt.count_parameters():,}")
print(f"  Taille complète: {tt.full_tensor_size():,}")
print(f"  Compression: {tt.full_tensor_size() / tt.count_parameters():.1f}x")
```

---

## TT-SVD

```python
def tt_svd(tensor, max_rank=None, epsilon=1e-10):
    """
    Décomposition TT via SVD (TT-SVD)
    
    Algorithm:
    Répète pour chaque mode:
    1. Reshape en matrice
    2. SVD
    3. Tronque selon max_rank ou epsilon
    4. Continue avec le reste
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    cores = []
    remainder = tensor.copy()
    rank_left = 1
    
    for k in range(n_modes - 1):
        # Reshape en matrice: (r_{k-1} × i_k, i_{k+1} × ... × i_N)
        remainder = remainder.reshape(rank_left * shape[k], -1)
        
        # SVD
        U, S, Vt = np.linalg.svd(remainder, full_matrices=False)
        
        # Détermine le rang
        if max_rank is not None:
            if isinstance(max_rank, (list, tuple)):
                rank = min(max_rank[k] if k < len(max_rank) else len(S), len(S))
            else:
                rank = min(max_rank, len(S))
        else:
            # Par epsilon: garde les valeurs singulières > epsilon * S[0]
            rank = np.sum(S > epsilon * S[0])
            rank = max(1, rank)
        
        # Tronque
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Core k: (r_{k-1}, i_k, r_k)
        core = U.reshape(rank_left, shape[k], rank)
        cores.append(core)
        
        # Prépare pour l'itération suivante
        remainder = np.diag(S) @ Vt
        rank_left = rank
    
    # Dernier core
    cores.append(remainder.reshape(rank_left, shape[-1], 1))
    
    return cores

# Exemple
tensor = np.random.randn(10, 12, 8, 6)
cores = tt_svd(tensor, max_rank=[5, 5, 5])

tt = TensorTrain(cores)
print(f"TT-SVD décomposition:")
print(f"  Tenseur original: {tensor.shape} ({tensor.size:,} éléments)")
print(f"  TT cores: {[c.shape for c in cores]}")
print(f"  Paramètres TT: {tt.count_parameters():,}")
print(f"  Compression: {tt.full_tensor_size() / tt.count_parameters():.1f}x")

# Reconstruction
reconstructed = tt.reconstruct()
error = np.linalg.norm(tensor - reconstructed, 'fro') / np.linalg.norm(tensor, 'fro')
print(f"  Erreur relative: {error:.6f}")
```

---

## Opérations sur TT

### Addition

```python
def tt_add(tt1, tt2):
    """
    Addition de deux Tensor Trains
    
    Les rangs peuvent être différents
    """
    # Concatène les cores (augmente les rangs)
    new_cores = []
    
    for i, (c1, c2) in enumerate(zip(tt1.cores, tt2.cores)):
        if i == 0:
            # Premier core: concatène verticalement
            new_core = np.concatenate([c1, c2], axis=2)  # Concatène sur r_k
        elif i == len(tt1.cores) - 1:
            # Dernier core: concatène horizontalement
            new_core = np.concatenate([c1, c2], axis=0)  # Concatène sur r_{k-1}
        else:
            # Cores intermédiaires: bloc diagonal
            r1_left, i, r1_right = c1.shape
            r2_left, _, r2_right = c2.shape
            
            new_core = np.zeros((r1_left + r2_left, i, r1_right + r2_right))
            new_core[:r1_left, :, :r1_right] = c1
            new_core[r1_left:, :, r1_right:] = c2
        
        new_cores.append(new_core)
    
    return TensorTrain(new_cores)
```

### Contraction

```python
def tt_contract(tt1, tt2, modes_to_contract):
    """
    Contraction de deux Tensor Trains
    
    Contracte sur certains modes spécifiés
    """
    # Implémentation simplifiée
    # La contraction TT est complexe car les rangs doivent être compatibles
    pass
```

---

## Applications en ML

### Compression de Matrices en TT

```python
class TTLinear(nn.Module):
    """
    Couche linéaire avec poids en format Tensor Train
    """
    
    def __init__(self, input_dims, output_dims, tt_ranks):
        """
        Args:
            input_dims: tuple (d₁, d₂, ..., dₙ) pour reshape de l'input
            output_dims: tuple (d'₁, d'₂, ..., d'ₘ) pour reshape de l'output
            tt_ranks: rangs TT
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_size = np.prod(input_dims)
        self.output_size = np.prod(output_dims)
        
        # Crée les cores TT pour représenter la matrice de poids
        # W: (output_size, input_size) = (∏d'ᵢ, ∏dⱼ)
        # Reshape en tenseur: (d'₁, ..., d'ₘ, d₁, ..., dₙ)
        
        # Cores pour les dimensions d'entrée
        self.input_cores = nn.ModuleList()
        prev_rank = 1
        for i, dim in enumerate(input_dims):
            next_rank = tt_ranks[i] if i < len(tt_ranks) else 1
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            self.input_cores.append(core)
            prev_rank = next_rank
        
        # Cores pour les dimensions de sortie
        self.output_cores = nn.ModuleList()
        for i, dim in enumerate(output_dims):
            next_rank = tt_ranks[len(input_dims) + i] if (len(input_dims) + i) < len(tt_ranks) else 1
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            self.output_cores.append(core)
            prev_rank = next_rank
        
    def forward(self, x):
        """
        Forward pass avec contraction TT
        """
        batch_size = x.shape[0]
        
        # Reshape input
        x = x.view(batch_size, *self.input_dims)
        
        # Contracte avec les cores d'entrée
        result = x
        for i, core in enumerate(self.input_cores):
            # Contracte sur la dimension i+1 (après batch)
            result = torch.tensordot(result, core, dims=([i+1], [1]))
            # Réarrange les dimensions
        
        # Contracte avec les cores de sortie
        for core in self.output_cores:
            result = torch.tensordot(result, core, dims=([1], [0]))
        
        # Reshape final
        result = result.view(batch_size, self.output_size)
        
        return result

# Exemple: compresser 1024 → 512
# Reshape: 1024 = 16×16×4, 512 = 16×16×2
tt_linear = TTLinear(
    input_dims=(16, 16, 4),
    output_dims=(16, 16, 2),
    tt_ranks=[4, 4, 4, 4]
)

print(f"TT-Linear:")
print(f"  Input: {tt_linear.input_size}, Output: {tt_linear.output_size}")
total_params = sum(c.numel() for cores in [tt_linear.input_cores, tt_linear.output_cores] for c in cores)
print(f"  Paramètres: {total_params:,}")
print(f"  vs dense: {tt_linear.input_size * tt_linear.output_size:,}")
print(f"  Compression: {tt_linear.input_size * tt_linear.output_size / total_params:.2f}x")
```

---

## Optimisation et Rounding

```python
def tt_rounding(tt, max_rank):
    """
    Arrondit un TT pour réduire les rangs
    
    Utilise SVD pour compresser chaque core
    """
    cores = tt.cores.copy()
    n_modes = len(cores)
    
    # Passe gauche-à-droite: réduit les rangs
    for k in range(n_modes - 1):
        core = cores[k]  # (r_{k-1}, i_k, r_k)
        
        # Reshape en matrice: (r_{k-1} * i_k, r_k)
        matrix = core.reshape(-1, core.shape[2])
        
        # SVD
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        
        # Tronque
        rank = min(max_rank, len(S))
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        # Mise à jour
        cores[k] = U.reshape(core.shape[0], core.shape[1], rank)
        # S * Vt va dans le core suivant
        if k < n_modes - 1:
            cores[k+1] = np.tensordot(
                np.diag(S) @ Vt,
                cores[k+1],
                axes=([1], [0])
            )
    
    return TensorTrain(cores)
```

---

## Exercices

### Exercice 5.3.1
Implémentez TT-SVD pour un tenseur d'ordre 5 et comparez l'erreur de reconstruction avec différents rangs.

### Exercice 5.3.2
Créez une couche linéaire TT et comparez ses performances avec une couche dense standard après entraînement.

### Exercice 5.3.3
Implémentez le rounding TT et testez son effet sur l'erreur de reconstruction.

---

## Points Clés à Retenir

> 📌 **TT évite la malédiction de la dimensionnalité avec une complexité linéaire en d**

> 📌 **TT-SVD donne la meilleure approximation de rang donné**

> 📌 **Les opérations sur TT (addition, contraction) sont efficaces**

> 📌 **TT est particulièrement adapté pour compresser les grandes matrices**

---

*Section suivante : [5.4 Hierarchical Tucker](./05_04_HierarchicalTucker.md)*

