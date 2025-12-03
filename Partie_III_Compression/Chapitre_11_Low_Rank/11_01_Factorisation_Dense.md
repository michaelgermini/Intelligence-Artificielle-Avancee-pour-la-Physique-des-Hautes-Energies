# 11.1 Factorisation Matricielle des Couches Denses

---

## Introduction

La **factorisation matricielle** décompose une grande matrice de poids en plusieurs matrices plus petites. Pour une couche dense, cela réduit drastiquement le nombre de paramètres tout en préservant une grande partie de l'expressivité.

---

## Principe de Base

### Décomposition d'une Matrice

Pour une matrice $\mathbf{W} \in \mathbb{R}^{m \times n}$ de rang $r$, la décomposition est :

$$\mathbf{W} = \mathbf{U} \mathbf{V}^T$$

où :
- $\mathbf{U} \in \mathbb{R}^{m \times r}$
- $\mathbf{V} \in \mathbb{R}^{n \times r}$

**Réduction** : $mn$ paramètres → $r(m + n)$ paramètres

### Condition d'Intérêt

La compression est bénéfique si :

$$r < \frac{mn}{m+n}$$

Pour $m=1024, n=512$ : compression bénéfique si $r < 341$

---

## Implémentation de Base

```python
import torch
import torch.nn as nn
import numpy as np

class FactorizedLinear(nn.Module):
    """
    Couche linéaire factorisée
    
    W = U @ V^T
    """
    
    def __init__(self, in_features, out_features, rank):
        """
        Args:
            in_features: Dimension d'entrée
            out_features: Dimension de sortie
            rank: Rang de la factorisation (r)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(in_features, rank))
        
        # Initialisation
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
    
    def forward(self, x):
        """
        Forward: y = x @ W^T = x @ (U @ V^T)^T = x @ V @ U^T
        
        Plus efficace: x @ V puis résultat @ U^T
        """
        # x: (batch, in_features)
        # V: (in_features, rank)
        # U: (out_features, rank)
        
        # (batch, in_features) @ (in_features, rank) = (batch, rank)
        intermediate = x @ self.V
        
        # (batch, rank) @ (rank, out_features) = (batch, out_features)
        output = intermediate @ self.U.T
        
        return output
    
    def reconstruct_weight(self):
        """Reconstruit la matrice de poids complète"""
        return self.U @ self.V.T
    
    def count_parameters(self):
        """Nombre de paramètres"""
        return self.U.numel() + self.V.numel()
    
    def compression_ratio(self):
        """Ratio de compression"""
        original = self.in_features * self.out_features
        compressed = self.count_parameters()
        return original / compressed

# Exemple
factorized = FactorizedLinear(in_features=1024, out_features=512, rank=64)

print("Factorisation Matricielle:")
print(f"  Dimensions: {factorized.in_features} → {factorized.out_features}")
print(f"  Rang: {factorized.rank}")
print(f"  Paramètres originaux: {factorized.in_features * factorized.out_features:,}")
print(f"  Paramètres compressés: {factorized.count_parameters():,}")
print(f"  Compression: {factorized.compression_ratio():.2f}x")

# Test forward
x = torch.randn(32, 1024)
y = factorized(x)
print(f"  Output shape: {y.shape}")
```

---

## Conversion depuis une Couche Standard

### Méthode 1 : Initialisation Aléatoire

```python
@staticmethod
def from_linear_random(linear_layer, rank):
    """
    Crée une couche factorisée avec initialisation aléatoire
    """
    factorized = FactorizedLinear(
        linear_layer.in_features,
        linear_layer.out_features,
        rank
    )
    
    # Copie le bias si présent
    if linear_layer.bias is not None:
        factorized.register_parameter('bias', 
                                     nn.Parameter(linear_layer.bias.data.clone()))
    
    return factorized
```

### Méthode 2 : SVD Initialisation

```python
@classmethod
def from_linear_svd(cls, linear_layer, rank):
    """
    Crée une couche factorisée via SVD
    
    Optimal: minimise l'erreur de reconstruction
    """
    W = linear_layer.weight.data  # (out_features, in_features)
    
    # SVD: W = U @ diag(S) @ V^T
    U, S, Vt = torch.svd(W)
    
    # Troncature au rang
    U_r = U[:, :rank]  # (out_features, rank)
    S_r = S[:rank]     # (rank,)
    Vt_r = Vt[:rank, :]  # (rank, in_features)
    
    # Factorisation: W ≈ (U_r @ sqrt(S_r)) @ (sqrt(S_r) @ Vt_r)
    # U = U_r @ sqrt(S_r), V = (sqrt(S_r) @ Vt_r)^T
    sqrt_S = torch.sqrt(S_r)
    
    factorized = cls(linear_layer.in_features, linear_layer.out_features, rank)
    
    factorized.U.data = U_r @ torch.diag(sqrt_S)
    factorized.V.data = (torch.diag(sqrt_S) @ Vt_r).T
    
    # Bias
    if linear_layer.bias is not None:
        factorized.register_parameter('bias',
                                     nn.Parameter(linear_layer.bias.data.clone()))
    
    return factorized

# Test
original = nn.Linear(1024, 512)
factorized_svd = FactorizedLinear.from_linear_svd(original, rank=64)

# Erreur de reconstruction
W_reconstructed = factorized_svd.reconstruct_weight()
reconstruction_error = torch.norm(original.weight.data - W_reconstructed, 'fro')
relative_error = reconstruction_error / torch.norm(original.weight.data, 'fro')

print(f"Conversion SVD:")
print(f"  Erreur relative de reconstruction: {relative_error:.6f}")
```

---

## Optimisation du Rang

### Analyse du Rang Effectif

```python
def find_optimal_rank(weight_matrix, max_rank=None, target_error=0.01):
    """
    Trouve le rang optimal pour une erreur cible
    
    Args:
        weight_matrix: Matrice de poids
        max_rank: Rang maximal à tester
        target_error: Erreur relative maximale acceptée
    """
    if max_rank is None:
        max_rank = min(weight_matrix.shape)
    
    U, S, Vt = torch.svd(weight_matrix)
    
    # Calcule l'erreur pour chaque rang
    ranks = []
    errors = []
    
    for r in range(1, max_rank + 1):
        # Reconstruction avec rang r
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        
        W_reconstructed = U_r @ torch.diag(S_r) @ Vt_r
        
        # Erreur
        error = torch.norm(weight_matrix - W_reconstructed, 'fro')
        relative_error = error / torch.norm(weight_matrix, 'fro')
        
        ranks.append(r)
        errors.append(relative_error.item())
        
        # Vérifie si l'erreur cible est atteinte
        if relative_error <= target_error:
            print(f"Rang optimal trouvé: {r} (erreur: {relative_error:.6f})")
            return r
    
    # Trouve le rang avec erreur la plus proche de la cible
    best_rank_idx = np.argmin(np.abs(np.array(errors) - target_error))
    best_rank = ranks[best_rank_idx]
    
    print(f"Rang optimal (approximatif): {best_rank} "
          f"(erreur: {errors[best_rank_idx]:.6f})")
    
    return best_rank

# Exemple
W = torch.randn(1024, 512)
optimal_rank = find_optimal_rank(W, target_error=0.05)

print(f"\nAnalyse pour W {W.shape}:")
print(f"  Rang optimal (5% erreur): {optimal_rank}")
print(f"  Compression: {(W.numel()) / (optimal_rank * (W.shape[0] + W.shape[1])):.2f}x")
```

### Analyse d'Énergie

```python
def energy_analysis(weight_matrix, energy_threshold=0.99):
    """
    Trouve le rang nécessaire pour capturer energy_threshold% de l'énergie
    """
    U, S, Vt = torch.svd(weight_matrix)
    
    # Énergie totale
    total_energy = (S ** 2).sum().item()
    
    # Énergie cumulative
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    
    # Trouve le rang
    energy_ratios = cumulative_energy / total_energy
    rank = torch.searchsorted(energy_ratios, 
                              torch.tensor(energy_threshold)).item() + 1
    
    rank = min(rank, len(S))
    
    energy_captured = (cumulative_energy[rank-1] / total_energy).item()
    
    print(f"Analyse d'Énergie ({energy_threshold*100:.0f}%):")
    print(f"  Rang nécessaire: {rank}")
    print(f"  Énergie capturée: {energy_captured:.4%}")
    
    return rank
```

---

## Factorisation Multi-Stage

### Décomposition Récursive

```python
class MultiStageFactorizedLinear(nn.Module):
    """
    Factorisation en plusieurs étapes
    
    W = U1 @ U2 @ ... @ Uk
    """
    
    def __init__(self, in_features, out_features, ranks):
        """
        Args:
            ranks: Liste de rangs pour chaque étape
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        
        # Crée les facteurs
        self.factors = nn.ModuleList()
        
        prev_dim = in_features
        for i, rank in enumerate(ranks):
            if i == len(ranks) - 1:
                # Dernière étape: vers out_features
                next_dim = out_features
            else:
                next_dim = rank
            
            factor = nn.Linear(prev_dim, next_dim, bias=False)
            self.factors.append(factor)
            prev_dim = next_dim
    
    def forward(self, x):
        """Forward: x @ F1 @ F2 @ ... @ Fk"""
        result = x
        for factor in self.factors:
            result = factor(result)
        return result
    
    def count_parameters(self):
        """Total parameters"""
        return sum(f.weight.numel() for f in self.factors)

# Exemple: 1024 → 256 → 512
multi_stage = MultiStageFactorizedLinear(
    in_features=1024,
    out_features=512,
    ranks=[256, 256]
)

print(f"Multi-stage Factorization:")
print(f"  Stages: {len(multi_stage.factors)}")
print(f"  Paramètres: {multi_stage.count_parameters():,}")
print(f"  vs Standard: {1024 * 512:,}")
print(f"  Compression: {1024 * 512 / multi_stage.count_parameters():.2f}x")
```

---

## Comparaison avec Couche Standard

```python
def compare_factorized_vs_standard(in_features, out_features, rank, num_tests=100):
    """
    Compare les performances d'une couche factorisée vs standard
    """
    standard = nn.Linear(in_features, out_features)
    factorized = FactorizedLinear.from_linear_svd(standard, rank=rank)
    
    # Test avec mêmes données
    x = torch.randn(num_tests, in_features)
    
    with torch.no_grad():
        y_standard = standard(x)
        y_factorized = factorized(x)
        
        # Erreur relative
        error = torch.norm(y_standard - y_factorized, 'fro')
        relative_error = error / torch.norm(y_standard, 'fro')
    
    print(f"Comparaison Factorisée vs Standard:")
    print(f"  Standard params: {standard.weight.numel():,}")
    print(f"  Factorized params: {factorized.count_parameters():,}")
    print(f"  Compression: {factorized.compression_ratio():.2f}x")
    print(f"  Erreur relative output: {relative_error:.6f}")
    
    return {
        'compression': factorized.compression_ratio(),
        'error': relative_error.item()
    }

compare_factorized_vs_standard(1024, 512, rank=64)
```

---

## Exercices

### Exercice 11.1.1
Implémentez une fonction qui trouve automatiquement le rang optimal en fonction d'une contrainte de compression.

### Exercice 11.1.2
Comparez différentes stratégies d'initialisation (aléatoire, SVD, Xavier) pour les facteurs.

### Exercice 11.1.3
Analysez comment la factorisation affecte les gradients pendant l'entraînement.

---

## Points Clés à Retenir

> 📌 **La factorisation W = U @ V^T réduit mn → r(m+n) paramètres**

> 📌 **SVD donne la meilleure initialisation (erreur minimale)**

> 📌 **Le rang optimal dépend du compromis compression/précision**

> 📌 **L'analyse d'énergie aide à choisir le rang**

> 📌 **La compression est bénéfique si r < mn/(m+n)**

---

*Section suivante : [11.2 Décomposition SVD Tronquée](./11_02_SVD_Tronquee.md)*

