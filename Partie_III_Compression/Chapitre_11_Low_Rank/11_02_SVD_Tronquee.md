# 11.2 Décomposition SVD Tronquée

---

## Introduction

La **SVD tronquée** (Truncated SVD) est la méthode optimale pour obtenir la meilleure approximation de rang faible d'une matrice au sens de la norme de Frobenius. Elle est théoriquement justifiée par le théorème d'Eckart-Young.

---

## Théorème d'Eckart-Young

### Énoncé

Pour une matrice $\mathbf{W} \in \mathbb{R}^{m \times n}$ avec SVD $\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, la meilleure approximation de rang $k$ (au sens de la norme de Frobenius) est :

$$\mathbf{W}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

où $\mathbf{U}_k$, $\mathbf{\Sigma}_k$, $\mathbf{V}_k$ sont les $k$ premières colonnes/éléments.

---

## Implémentation

```python
import torch
import torch.nn as nn
import numpy as np

class TruncatedSVDLinear(nn.Module):
    """
    Couche linéaire via SVD tronquée
    """
    
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs SVD
        # W ≈ U @ diag(S) @ V^T
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.S = nn.Parameter(torch.ones(rank))  # Valeurs singulières
        self.V = nn.Parameter(torch.randn(in_features, rank))
    
    @classmethod
    def from_weight_matrix(cls, W, rank):
        """
        Crée depuis une matrice de poids via SVD
        """
        # SVD
        U, S, Vt = torch.svd(W)
        
        # Troncature
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        # Crée la couche
        layer = cls(W.shape[1], W.shape[0], rank)
        
        # Initialise
        layer.U.data = U_r
        layer.S.data = S_r
        layer.V.data = Vt_r.T
        
        return layer
    
    def forward(self, x):
        """
        y = x @ (U @ diag(S) @ V^T)^T
          = x @ V @ diag(S) @ U^T
        """
        # x: (batch, in_features)
        # V: (in_features, rank)
        # S: (rank,)
        # U: (out_features, rank)
        
        # x @ V: (batch, rank)
        xv = x @ self.V
        
        # xv @ diag(S): (batch, rank)
        xvs = xv * self.S.unsqueeze(0)
        
        # xvs @ U^T: (batch, out_features)
        output = xvs @ self.U.T
        
        return output
    
    def reconstruct_weight(self):
        """Reconstruit W"""
        return self.U @ torch.diag(self.S) @ self.V.T
    
    def compression_ratio(self):
        """Ratio de compression"""
        original = self.in_features * self.out_features
        compressed = self.U.numel() + self.S.numel() + self.V.numel()
        return original / compressed

# Exemple
W = torch.randn(512, 1024)
svd_layer = TruncatedSVDLinear.from_weight_matrix(W, rank=64)

print("SVD Tronquée:")
print(f"  Original: {W.shape} ({W.numel():,} params)")
print(f"  Compressed: rank {svd_layer.rank} ({svd_layer.U.numel() + svd_layer.S.numel() + svd_layer.V.numel():,} params)")
print(f"  Compression: {svd_layer.compression_ratio():.2f}x")

# Erreur de reconstruction
W_recon = svd_layer.reconstruct_weight()
error = torch.norm(W - W_recon, 'fro') / torch.norm(W, 'fro')
print(f"  Erreur relative: {error:.6f}")
```

---

## Propriétés de la SVD Tronquée

### Erreur Minimale

```python
def analyze_svd_approximation_error(W, max_rank=None):
    """
    Analyse l'erreur d'approximation pour différents rangs
    """
    if max_rank is None:
        max_rank = min(W.shape)
    
    U, S, Vt = torch.svd(W)
    
    errors = []
    ranks = list(range(1, max_rank + 1, max(1, max_rank // 20)))
    
    for rank in ranks:
        # Reconstruction
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        W_approx = U_r @ torch.diag(S_r) @ Vt_r
        
        # Erreur
        error = torch.norm(W - W_approx, 'fro')
        relative_error = error / torch.norm(W, 'fro')
        
        # Énergie capturée
        total_energy = (S ** 2).sum()
        captured_energy = (S_r ** 2).sum()
        energy_ratio = captured_energy / total_energy
        
        errors.append({
            'rank': rank,
            'error': relative_error.item(),
            'energy': energy_ratio.item()
        })
    
    return errors

# Visualisation
W_test = torch.randn(256, 512)
errors = analyze_svd_approximation_error(W_test, max_rank=128)

print("Erreur SVD vs Rang:")
for e in errors[::5]:  # Affiche tous les 5
    print(f"  Rank {e['rank']:3d}: Error={e['error']:.4f}, Energy={e['energy']:.4f}")
```

---

## Variantes de Factorisation SVD

### Forme Compacte

Au lieu de stocker U, S, V séparément, on peut utiliser :

$$\mathbf{W}_k = (\mathbf{U}_k \sqrt{\mathbf{\Sigma}_k}) (\sqrt{\mathbf{\Sigma}_k} \mathbf{V}_k^T) = \mathbf{A} \mathbf{B}^T$$

```python
class CompactSVDLinear(nn.Module):
    """
    SVD sous forme compacte: W = A @ B^T où A = U @ sqrt(S), B = V @ sqrt(S)
    """
    
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs compacts
        self.A = nn.Parameter(torch.randn(out_features, rank))
        self.B = nn.Parameter(torch.randn(in_features, rank))
    
    @classmethod
    def from_weight_matrix(cls, W, rank):
        """Crée depuis une matrice via SVD compacte"""
        U, S, Vt = torch.svd(W)
        
        # Troncature
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        # Forme compacte
        sqrt_S = torch.sqrt(S_r)
        
        layer = cls(W.shape[1], W.shape[0], rank)
        layer.A.data = U_r @ torch.diag(sqrt_S)
        layer.B.data = (torch.diag(sqrt_S) @ Vt_r).T
        
        return layer
    
    def forward(self, x):
        """y = x @ B @ A^T"""
        return x @ self.B @ self.A.T
    
    def compression_ratio(self):
        original = self.in_features * self.out_features
        compressed = self.A.numel() + self.B.numel()
        return original / compressed

# Comparaison
W = torch.randn(512, 1024)
compact_svd = CompactSVDLinear.from_weight_matrix(W, rank=64)

print(f"Forme Compacte SVD:")
print(f"  Paramètres: {compact_svd.A.numel() + compact_svd.B.numel():,}")
print(f"  vs Forme étendue: {64 * (512 + 1024) + 64:,}")
print(f"  Compression: {compact_svd.compression_ratio():.2f}x")
```

---

## SVD pour Couches Convolutionnelles

```python
class SVDConv2d(nn.Module):
    """
    Convolution compressée via SVD
    
    Pour un kernel 4D (out_ch, in_ch, H, W), applique SVD sur
    la dimension (out_ch, in_ch*H*W)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 rank, stride=1, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        
        # Facteurs SVD
        # Kernel reshaped: (out_ch, in_ch * H * W)
        self.U = nn.Parameter(torch.randn(out_channels, rank))
        self.V = nn.Parameter(torch.randn(rank, in_channels * kernel_size * kernel_size))
    
    @classmethod
    def from_conv2d(cls, conv_layer, rank):
        """Crée depuis une couche Conv2d via SVD"""
        W = conv_layer.weight.data  # (out_ch, in_ch, H, W)
        
        # Reshape: (out_ch, in_ch * H * W)
        W_reshaped = W.view(W.shape[0], -1)
        
        # SVD
        U, S, Vt = torch.svd(W_reshaped)
        
        # Troncature
        sqrt_S = torch.sqrt(S[:rank])
        U_r = U[:, :rank]
        Vt_r = Vt[:rank, :]
        
        # Crée la couche
        svd_conv = cls(
            conv_layer.in_channels,
            conv_layer.out_channels,
            conv_layer.kernel_size[0],
            rank=rank,
            stride=conv_layer.stride[0],
            padding=conv_layer.padding[0]
        )
        
        # Initialise
        svd_conv.U.data = U_r @ torch.diag(sqrt_S)
        svd_conv.V.data = (torch.diag(sqrt_S) @ Vt_r).view(
            rank, conv_layer.in_channels, 
            conv_layer.kernel_size[0], conv_layer.kernel_size[0]
        )
        
        return svd_conv
    
    def forward(self, x):
        """
        Forward: 2 convolutions séquentielles
        """
        # Première conv: x @ V^T (reshape V)
        V_reshaped = self.V.view(self.rank, self.in_channels, 
                                 self.kernel_size, self.kernel_size)
        intermediate = F.conv2d(x, V_reshaped, stride=self.stride, 
                               padding=self.padding)
        
        # Deuxième conv: intermediate @ U^T
        # U doit être reshapé pour conv (rank, 1, 1) @ (out_ch, rank)
        # Simplifié: utilise conv1x1
        U_reshaped = self.U.T.unsqueeze(-1).unsqueeze(-1)  # (rank, out_ch, 1, 1)
        U_reshaped = U_reshaped.permute(1, 0, 2, 3)  # (out_ch, rank, 1, 1)
        
        output = F.conv2d(intermediate, U_reshaped)
        
        return output
```

---

## Optimisations Numériques

### Calcul Efficace de la SVD

```python
def efficient_truncated_svd(W, rank, method='default'):
    """
    Calcul efficace de SVD tronquée pour grandes matrices
    """
    if method == 'default':
        # PyTorch utilise une implémentation optimisée
        U, S, Vt = torch.svd(W)
        return U[:, :rank], S[:rank], Vt[:rank, :]
    
    elif method == 'randomized':
        # SVD randomisée pour très grandes matrices
        # (Simplifié - nécessite implémentation complète)
        # Utilise projection aléatoire pour réduire la taille
        pass

# Pour très grandes matrices
def approximate_svd_large(W, rank, oversample=10):
    """
    SVD approximative pour matrices très grandes
    Utilise projection aléatoire
    """
    m, n = W.shape
    
    # Matrice aléatoire de projection
    Omega = torch.randn(n, rank + oversample, device=W.device)
    
    # Projection: Y = W @ Omega
    Y = W @ Omega
    
    # QR de Y
    Q, _ = torch.qr(Y)
    
    # Projette W sur l'espace de Q
    B = Q.T @ W
    
    # SVD de B (plus petit)
    U_B, S, Vt_B = torch.svd(B)
    
    # U final
    U = Q @ U_B
    
    # Troncature
    return U[:, :rank], S[:rank], Vt_B[:rank, :]
```

---

## Exercices

### Exercice 11.2.1
Comparez SVD tronquée vs factorisation simple (W = U @ V^T) en termes d'erreur de reconstruction.

### Exercice 11.2.2
Implémentez une version optimisée de SVD pour matrices très grandes (>10K x 10K).

### Exercice 11.2.3
Analysez comment l'erreur de reconstruction SVD affecte la performance d'un réseau complet.

---

## Points Clés à Retenir

> 📌 **SVD tronquée donne la meilleure approximation de rang k (théorème d'Eckart-Young)**

> 📌 **L'erreur d'approximation est la somme des valeurs singulières au-delà du rang**

> 📌 **La forme compacte (A @ B^T) réduit le stockage**

> 📌 **Pour très grandes matrices, SVD randomisée est plus efficace**

> 📌 **SVD peut être appliquée aux convolutions via reshape**

---

*Section suivante : [11.3 Low-Rank Adaptation (LoRA)](./11_03_LoRA.md)*

