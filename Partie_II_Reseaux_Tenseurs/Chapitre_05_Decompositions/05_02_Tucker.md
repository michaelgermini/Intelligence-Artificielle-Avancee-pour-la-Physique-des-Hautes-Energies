# 5.2 Décomposition Tucker

---

## Introduction

La **décomposition Tucker** (aussi appelée Higher-Order SVD - HOSVD) factorise un tenseur en un tenseur noyau et des matrices facteurs pour chaque mode. Elle est plus flexible que CP mais le noyau peut être grand.

---

## Définition Formelle

Pour un tenseur $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$, la décomposition Tucker est :

$$\mathcal{T} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \cdots \times_N \mathbf{U}^{(N)}$$

où :
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}$ : tenseur noyau (core tensor)
- $\mathbf{U}^{(n)} \in \mathbb{R}^{I_n \times R_n}$ : matrice facteur du mode $n$
- $\times_n$ : produit mode-$n$

---

## Propriétés

```python
import numpy as np

class TuckerDecomposition:
    """
    Décomposition Tucker d'un tenseur
    """
    
    def __init__(self, tensor, ranks):
        """
        Args:
            tensor: Tenseur à décomposer
            ranks: tuple (R₁, R₂, ..., Rₙ) - rangs par mode
        """
        self.tensor = np.array(tensor)
        self.ranks = ranks
        self.n_modes = tensor.ndim
        self.shape = tensor.shape
        
        # Noyau et facteurs (initialisés plus tard)
        self.core = None
        self.factors = None
    
    def reconstruct(self):
        """
        Reconstruit le tenseur: G ×₁ U₁ ×₂ U₂ ×₃ ...
        """
        result = self.core.copy()
        
        for mode, factor in enumerate(self.factors):
            result = mode_n_product(result, factor, mode)
        
        return result
    
    def compression_ratio(self):
        """Calcule le ratio de compression"""
        original_size = np.prod(self.shape)
        
        core_size = np.prod(self.core.shape)
        factors_size = sum(f.shape[0] * f.shape[1] for f in self.factors)
        
        compressed_size = core_size + factors_size
        return original_size / compressed_size

def mode_n_product(tensor, matrix, mode):
    """
    Produit mode-n : T ×ₙ M
    
    Args:
        tensor: Tenseur (I₁, ..., Iₙ, ..., I_N)
        matrix: Matrice (J, Iₙ)
        mode: Mode sur lequel contracter
    """
    # Déplace le mode en première position
    tensor = np.moveaxis(tensor, mode, 0)
    
    # Reshape pour multiplication matricielle
    original_shape = tensor.shape
    tensor = tensor.reshape(original_shape[0], -1)
    
    # Multiplication: M @ tensor
    result = matrix @ tensor
    
    # Nouvelle shape
    new_shape = (matrix.shape[0],) + original_shape[1:]
    result = result.reshape(new_shape)
    
    # Remet le mode à sa place originale
    result = np.moveaxis(result, 0, mode)
    
    return result
```

---

## HOSVD (Higher-Order SVD)

```python
def hosvd(tensor, ranks=None, full=False):
    """
    Higher-Order SVD: méthode standard pour la décomposition Tucker
    
    Algorithm:
    1. Pour chaque mode n:
       - Matricise selon le mode n
       - Fait SVD
       - Garde les R_n premières composantes
    2. Calcule le noyau via projection
    """
    shape = tensor.shape
    n_modes = len(shape)
    
    if ranks is None:
        ranks = shape  # Pas de compression
    
    # SVD de chaque mode
    factors = []
    for mode in range(n_modes):
        # Matricisation
        tensor_mat = unfold_tensor(tensor, mode)
        
        # SVD
        U, S, Vt = np.linalg.svd(tensor_mat, full_matrices=False)
        
        # Garde les R_n premières colonnes
        rank = ranks[mode]
        if full:
            rank = min(rank, len(S))
        
        factors.append(U[:, :rank])
    
    # Calcule le noyau
    core = tensor.copy()
    for mode, factor in enumerate(factors):
        # Projection: G = T ×₁ U₁^T ×₂ U₂^T ×₃ ...
        core = mode_n_product(core, factor.T, mode)
    
    return core, factors

# Exemple
tensor = np.random.randn(10, 12, 8)
ranks = (5, 6, 4)  # Rangs réduits

core, factors = hosvd(tensor, ranks)

print(f"Décomposition Tucker:")
print(f"  Tenseur original: {tensor.shape}")
print(f"  Noyau: {core.shape}")
print(f"  Facteurs: {[f.shape for f in factors]}")

# Reconstruction
tucker = TuckerDecomposition(tensor, ranks)
tucker.core = core
tucker.factors = factors

reconstructed = tucker.reconstruct()
error = np.linalg.norm(tensor - reconstructed, 'fro') / np.linalg.norm(tensor, 'fro')
print(f"  Erreur relative: {error:.6f}")
print(f"  Compression: {tucker.compression_ratio():.2f}x")
```

---

## HOOI (Higher-Order Orthogonal Iteration)

```python
def hooi(tensor, ranks, max_iter=100, tol=1e-6):
    """
    Higher-Order Orthogonal Iteration
    
    Itératif, meilleur que HOSVD pour la compression
    """
    n_modes = len(tensor.shape)
    
    # Initialisation avec HOSVD
    core, factors = hosvd(tensor, ranks, full=False)
    
    prev_error = float('inf')
    
    for iteration in range(max_iter):
        # Mise à jour itérative de chaque facteur
        for mode in range(n_modes):
            # Contracte tous les autres modes
            temp = tensor.copy()
            for n in range(n_modes):
                if n != mode:
                    temp = mode_n_product(temp, factors[n].T, n)
            
            # Matricise selon le mode
            temp_mat = unfold_tensor(temp, mode)
            
            # SVD pour trouver le meilleur facteur
            U, S, Vt = np.linalg.svd(temp_mat, full_matrices=False)
            factors[mode] = U[:, :ranks[mode]]
        
        # Recalcule le noyau
        core = tensor.copy()
        for mode, factor in enumerate(factors):
            core = mode_n_product(core, factor.T, mode)
        
        # Calcule l'erreur
        reconstructed = reconstruct_tucker(core, factors)
        error = np.linalg.norm(tensor - reconstructed, 'fro')
        rel_error = error / np.linalg.norm(tensor, 'fro')
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Error = {rel_error:.6f}")
        
        if abs(prev_error - error) < tol:
            print(f"Converged after {iteration+1} iterations")
            break
        
        prev_error = error
    
    return core, factors, rel_error

def reconstruct_tucker(core, factors):
    """Reconstruit depuis la décomposition Tucker"""
    result = core.copy()
    for mode, factor in enumerate(factors):
        result = mode_n_product(result, factor, mode)
    return result

# Comparaison HOSVD vs HOOI
tensor = np.random.randn(15, 20, 10)
ranks = (8, 10, 5)

print("Comparaison HOSVD vs HOOI:")
core_hosvd, factors_hosvd = hosvd(tensor, ranks)
recon_hosvd = reconstruct_tucker(core_hosvd, factors_hosvd)
error_hosvd = np.linalg.norm(tensor - recon_hosvd, 'fro') / np.linalg.norm(tensor, 'fro')

core_hooi, factors_hooi, error_hooi = hooi(tensor, ranks, max_iter=50)

print(f"  HOSVD error: {error_hosvd:.6f}")
print(f"  HOOI error: {error_hooi:.6f}")
print(f"  Amélioration: {(error_hosvd - error_hooi) / error_hosvd * 100:.1f}%")
```

---

## Relation avec CP

### CP comme Tucker Décomposé

CP est un cas spécial de Tucker où le noyau est super-diagonal :

```python
def tucker_to_cp(core, factors):
    """
    Convertit une décomposition Tucker en CP (si possible)
    
    Nécessite que le noyau soit super-diagonal
    """
    # Vérifie si le noyau est super-diagonal
    # (uniquement des valeurs non-nulles sur la diagonale principale)
    
    # Extraction des facteurs CP depuis Tucker
    # (Simplification)
    pass

def cp_to_tucker(cp_factors, cp_weights):
    """
    Convertit CP en Tucker (toujours possible)
    
    Le noyau Tucker est super-diagonal avec les poids CP
    """
    n_modes = len(cp_factors)
    rank = cp_factors[0].shape[1]
    
    # Crée le noyau super-diagonal
    core = np.zeros([f.shape[1] for f in cp_factors])
    for r in range(rank):
        indices = tuple([r] * n_modes)
        core[indices] = cp_weights[r]
    
    return core, cp_factors
```

---

## Applications en Compression

### Compression de Tenseurs de Poids

```python
class TuckerCompressedConv(nn.Module):
    """
    Couche convolutionnelle compressée avec Tucker
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 ranks=(None, None, None, None)):
        super().__init__()
        
        # Tucker pour les poids 4D: (out_ch, in_ch, kH, kW)
        if ranks[0] is None:
            ranks = (out_channels, in_channels, kernel_size, kernel_size)
        
        self.ranks = ranks
        
        # Noyau Tucker
        self.core = nn.Parameter(
            torch.randn(*ranks)
        )
        
        # Facteurs (un par mode)
        self.factor_out = nn.Parameter(torch.randn(out_channels, ranks[0]))
        self.factor_in = nn.Parameter(torch.randn(in_channels, ranks[1]))
        self.factor_h = nn.Parameter(torch.randn(kernel_size, ranks[2]))
        self.factor_w = nn.Parameter(torch.randn(kernel_size, ranks[3]))
        
    def forward(self, x):
        """Forward avec poids reconstruits depuis Tucker"""
        # Reconstruit les poids (coûteux, peut être pré-calculé)
        weights = self.reconstruct_weights()
        
        return F.conv2d(x, weights, stride=1, padding=1)
    
    def reconstruct_weights(self):
        """Reconstruit les poids 4D depuis la décomposition"""
        # Contracte le noyau avec les facteurs
        weights = self.core
        
        # Mode 0: out_channels
        weights = torch.tensordot(self.factor_out, weights, dims=([1], [0]))
        
        # Mode 1: in_channels
        weights = torch.tensordot(self.factor_in, weights, dims=([1], [1]))
        
        # Mode 2: height
        weights = torch.tensordot(self.factor_h, weights, dims=([1], [2]))
        
        # Mode 3: width
        weights = torch.tensordot(self.factor_w, weights, dims=([1], [3]))
        
        return weights
    
    def compression_ratio(self):
        """Ratio de compression"""
        original = (self.factor_out.shape[0] * self.factor_in.shape[0] * 
                   self.factor_h.shape[0] * self.factor_w.shape[0])
        
        compressed = (self.core.numel() + 
                     sum([self.factor_out.numel(), self.factor_in.numel(),
                          self.factor_h.numel(), self.factor_w.numel()]))
        
        return original / compressed

# Exemple
conv_original = nn.Conv2d(64, 128, 3, padding=1)
conv_compressed = TuckerCompressedConv(64, 128, 3, ranks=(32, 32, 2, 2))

print(f"Compression Tucker pour Conv:")
print(f"  Paramètres originaux: {conv_original.weight.numel():,}")
print(f"  Paramètres compressés: {sum(p.numel() for p in conv_compressed.parameters()):,}")
print(f"  Compression: {conv_compressed.compression_ratio():.2f}x")
```

---

## Avantages et Inconvénients

### Avantages

- **Flexibilité** : Rangs différents par mode
- **Compression efficace** : Si les rangs sont bien choisis
- **Stabilité numérique** : HOSVD est stable

### Inconvénients

- **Taille du noyau** : Peut être grand si les rangs ne sont pas bien choisis
- **Curse of dimensionality** : Le noyau a $R_1 \times R_2 \times \cdots \times R_N$ éléments

```python
def tucker_core_size(ranks):
    """Calcule la taille du noyau Tucker"""
    return np.prod(ranks)

# Exemple: curse of dimensionality
for n_modes in [3, 4, 5]:
    ranks = tuple([10] * n_modes)
    core_size = tucker_core_size(ranks)
    print(f"Noyau {n_modes}D avec rangs {ranks}: {core_size:,} éléments")
```

---

## Exercices

### Exercice 5.2.1
Comparez HOSVD et HOOI sur un tenseur de rang faible. Lequel donne la meilleure reconstruction ?

### Exercice 5.2.2
Implémentez une fonction qui trouve automatiquement les rangs optimaux pour une décomposition Tucker avec contrainte d'erreur.

### Exercice 5.2.3
Utilisez Tucker pour compresser une couche convolutionnelle 3×3 et mesurez la perte de performance.

---

## Points Clés à Retenir

> 📌 **Tucker est plus flexible que CP mais le noyau peut être grand**

> 📌 **HOSVD est rapide mais HOOI donne de meilleurs résultats**

> 📌 **Le curse of dimensionality affecte la taille du noyau**

> 📌 **Tucker est efficace pour compresser les convolutions**

---

*Section suivante : [5.3 Tensor Train / Matrix Product States](./05_03_TensorTrain.md)*

