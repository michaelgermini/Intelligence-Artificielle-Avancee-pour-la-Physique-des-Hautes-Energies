# 7.2 Tensorisation des Couches Convolutionnelles

---

## Introduction

La tensorisation des couches convolutionnelles est plus complexe que pour les couches denses car les poids sont des tenseurs 4D. Plusieurs approches existent : Tucker, CP, ou Tensor Train généralisé.

---

## Structure des Poids Convolutifs

### Tenseur de Poids

Une couche `Conv2d(in_channels, out_channels, kernel_size)` a des poids :
- Shape : `(out_channels, in_channels, kernel_height, kernel_width)`
- Ordre 4 : chaque dimension peut être tensorisée

---

## Méthode 1 : Décomposition Tucker

### Principe

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TuckerCompressedConv(nn.Module):
    """
    Convolution compressée avec décomposition Tucker
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 ranks=(None, None, None, None)):
        """
        Args:
            in_channels: C_in
            out_channels: C_out
            kernel_size: k (ou (k_h, k_w))
            ranks: tuple (r_C_out, r_C_in, r_h, r_w) - rangs Tucker
        """
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if ranks[0] is None:
            # Rangs par défaut: fraction de la dimension originale
            ranks = (
                out_channels // 2,
                in_channels // 2,
                kernel_size[0] // 2 if kernel_size[0] > 1 else 1,
                kernel_size[1] // 2 if kernel_size[1] > 1 else 1
            )
        
        self.ranks = ranks
        r_C_out, r_C_in, r_h, r_w = ranks
        
        # Noyau Tucker: (r_C_out, r_C_in, r_h, r_w)
        self.core = nn.Parameter(
            torch.randn(r_C_out, r_C_in, r_h, r_w)
        )
        
        # Facteurs
        self.factor_C_out = nn.Parameter(
            torch.randn(out_channels, r_C_out)
        )
        self.factor_C_in = nn.Parameter(
            torch.randn(in_channels, r_C_in)
        )
        self.factor_h = nn.Parameter(
            torch.randn(kernel_size[0], r_h)
        )
        self.factor_w = nn.Parameter(
            torch.randn(kernel_size[1], r_w)
        )
        
        # Initialisation
        nn.init.xavier_uniform_(self.core)
        nn.init.xavier_uniform_(self.factor_C_out)
        nn.init.xavier_uniform_(self.factor_C_in)
        nn.init.xavier_uniform_(self.factor_h)
        nn.init.xavier_uniform_(self.factor_w)
        
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    def reconstruct_weights(self):
        """
        Reconstruit les poids complets depuis la décomposition Tucker
        
        W = Core ×₁ F_C_out ×₂ F_C_in ×₃ F_h ×₄ F_w
        """
        # Contracte le noyau avec les facteurs
        weights = self.core
        
        # Mode 0: out_channels
        weights = torch.tensordot(self.factor_C_out, weights, dims=([1], [0]))
        # (out_ch, r_C_in, r_h, r_w)
        
        # Mode 1: in_channels
        weights = torch.tensordot(self.factor_C_in, weights, dims=([1], [1]))
        # (out_ch, in_ch, r_h, r_w)
        
        # Mode 2: height
        weights = torch.tensordot(self.factor_h, weights, dims=([1], [2]))
        # (out_ch, in_ch, kernel_h, r_w)
        
        # Mode 3: width
        weights = torch.tensordot(self.factor_w, weights, dims=([1], [3]))
        # (out_ch, in_ch, kernel_h, kernel_w)
        
        return weights
    
    def forward(self, x):
        """
        Forward avec poids reconstruits
        """
        weights = self.reconstruct_weights()
        return F.conv2d(x, weights, padding=self.padding)
    
    def count_parameters(self):
        """Nombre de paramètres compressés"""
        return (self.core.numel() + 
               self.factor_C_out.numel() + 
               self.factor_C_in.numel() +
               self.factor_h.numel() +
               self.factor_w.numel())
    
    def compression_ratio(self):
        """Ratio de compression"""
        original = self.factor_C_out.shape[0] * self.factor_C_in.shape[0] * \
                   self.factor_h.shape[0] * self.factor_w.shape[0]
        return original / self.count_parameters()

# Exemple
tucker_conv = TuckerCompressedConv(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    ranks=(32, 32, 2, 2)
)

print("Convolution Tucker:")
print(f"  Original: {64 * 128 * 3 * 3:,} paramètres")
print(f"  Compressé: {tucker_conv.count_parameters():,} paramètres")
print(f"  Compression: {tucker_conv.compression_ratio():.2f}x")
```

---

## Méthode 2 : Décomposition CP

### Principe

```python
class CPCompressedConv(nn.Module):
    """
    Convolution compressée avec décomposition CP
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, rank):
        """
        Args:
            rank: rang CP (nombre de composantes)
        """
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.rank = rank
        
        # Facteurs CP
        # W ≈ Σᵣ A_r ⊗ B_r ⊗ C_r ⊗ D_r
        self.factor_out = nn.Parameter(
            torch.randn(out_channels, rank)
        )
        self.factor_in = nn.Parameter(
            torch.randn(in_channels, rank)
        )
        self.factor_h = nn.Parameter(
            torch.randn(kernel_size[0], rank)
        )
        self.factor_w = nn.Parameter(
            torch.randn(kernel_size[1], rank)
        )
        
        # Poids (optionnel)
        self.weights = nn.Parameter(torch.ones(rank))
        
        nn.init.xavier_uniform_(self.factor_out)
        nn.init.xavier_uniform_(self.factor_in)
        nn.init.xavier_uniform_(self.factor_h)
        nn.init.xavier_uniform_(self.factor_w)
        
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    def reconstruct_weights(self):
        """
        Reconstruit les poids depuis CP
        """
        weights = torch.zeros(
            self.factor_out.shape[0],
            self.factor_in.shape[0],
            self.factor_h.shape[0],
            self.factor_w.shape[0]
        )
        
        for r in range(self.rank):
            component = torch.outer(
                self.factor_out[:, r],
                self.factor_in[:, r]
            ).unsqueeze(-1).unsqueeze(-1)
            component = component * self.factor_h[:, r].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            component = component * self.factor_w[:, r].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
            weights += self.weights[r] * component
        
        return weights
    
    def forward(self, x):
        weights = self.reconstruct_weights()
        return F.conv2d(x, weights, padding=self.padding)
    
    def count_parameters(self):
        return (self.factor_out.numel() + 
               self.factor_in.numel() +
               self.factor_h.numel() +
               self.factor_w.numel() +
               self.weights.numel())
```

---

## Méthode 3 : Tensor Train pour Convolutions

### Principe

```python
class TTCompressedConv(nn.Module):
    """
    Convolution compressée avec Tensor Train
    
    Flatten les dimensions spatiales et utilise TT
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, tt_rank):
        """
        Args:
            tt_rank: rang Tensor Train
        """
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Factorise les dimensions
        # (out_ch, in_ch, k_h, k_w) → (d₁, d₂, d₃, d₄)
        # Factorisation simplifiée
        input_dims = self._factorize_dimensions(in_channels)
        output_dims = self._factorize_dimensions(out_channels)
        spatial_dims = (kernel_size[0], kernel_size[1])
        
        # Crée les cores TT
        # Ordre: input_dims + spatial_dims + output_dims
        self.cores = nn.ModuleList()
        
        # (Simplifié - implémentation complète est complexe)
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    def _factorize_dimensions(self, n):
        """Factorise une dimension"""
        # Heuristique simple
        if n == 64:
            return (8, 8)
        elif n == 128:
            return (8, 16)
        elif n == 256:
            return (16, 16)
        else:
            # Factorisation par défaut
            sqrt_n = int(np.sqrt(n))
            return (sqrt_n, n // sqrt_n)
```

---

## Conversion Automatique

### Depuis une Convolution Standard

```python
def conv_to_tucker(conv_layer, ranks):
    """
    Convertit une convolution en format Tucker via HOSVD
    """
    W = conv_layer.weight.data.numpy()  # (C_out, C_in, k_h, k_w)
    
    # HOSVD
    from scipy.linalg import svd
    
    factors = []
    core = W.copy()
    
    # SVD pour chaque mode
    for mode in range(4):
        # Matricise selon le mode
        shape = W.shape
        W_mat = np.moveaxis(W, mode, 0).reshape(shape[mode], -1)
        
        # SVD
        U, S, Vt = svd(W_mat, full_matrices=False)
        
        # Garde les rangs spécifiés
        rank = ranks[mode]
        U_trunc = U[:, :rank]
        factors.append(U_trunc)
        
        # Projette le noyau
        core = np.tensordot(core, U_trunc.T, axes=([mode], [0]))
    
    # Crée la couche compressée
    compressed = TuckerCompressedConv(
        conv_layer.in_channels,
        conv_layer.out_channels,
        conv_layer.kernel_size,
        ranks=ranks
    )
    
    # Initialise
    compressed.core.data = torch.from_numpy(core).float()
    for i, factor in enumerate(factors):
        if i == 0:
            compressed.factor_C_out.data = torch.from_numpy(factor).float()
        elif i == 1:
            compressed.factor_C_in.data = torch.from_numpy(factor).float()
        elif i == 2:
            compressed.factor_h.data = torch.from_numpy(factor).float()
        elif i == 3:
            compressed.factor_w.data = torch.from_numpy(factor).float()
    
    return compressed

# Test
original_conv = nn.Conv2d(64, 128, 3, padding=1)
compressed_conv = conv_to_tucker(original_conv, ranks=(32, 32, 2, 2))

print(f"Compression Conv:")
print(f"  Original: {original_conv.weight.numel():,}")
print(f"  Compressé: {compressed_conv.count_parameters():,}")
print(f"  Ratio: {original_conv.weight.numel() / compressed_conv.count_parameters():.2f}x")
```

---

## Convolution en Format TT

### Approche Directe

```python
def conv_with_tt_weights(x, tt_cores, padding=1):
    """
    Applique une convolution avec poids en format TT
    
    (Approche simplifiée - nécessite implémentation optimisée)
    """
    # Reconstruit les poids depuis TT
    weights = reconstruct_from_tt(tt_cores)
    
    # Applique la convolution standard
    return F.conv2d(x, weights, padding=padding)

def reconstruct_from_tt(cores):
    """Reconstruit les poids complets depuis les cores TT"""
    result = cores[0]
    for core in cores[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.squeeze()
```

---

## Optimisations

### Pré-calcul des Poids

Pour éviter de reconstruire les poids à chaque forward :

```python
class CachedTuckerConv(TuckerCompressedConv):
    """
    Version avec cache des poids reconstruits
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_weights = None
        self.weights_dirty = True
    
    def _invalidate_cache(self):
        """Marque le cache comme invalide"""
        self.weights_dirty = True
    
    def forward(self, x):
        if self.weights_dirty or self.cached_weights is None:
            self.cached_weights = self.reconstruct_weights()
            self.weights_dirty = False
        
        return F.conv2d(x, self.cached_weights, padding=self.padding)
```

### Calcul Direct sans Reconstruction

Pour une efficacité maximale, on peut éviter la reconstruction complète :

```python
def direct_tucker_conv(x, core, factors, padding=1):
    """
    Applique la convolution directement depuis les facteurs Tucker
    sans reconstruire les poids complets
    
    (Plus complexe mais plus efficace)
    """
    # Applique les facteurs séquentiellement
    # (Implémentation optimisée)
    pass
```

---

## Comparaison des Méthodes

```python
def compare_conv_compression_methods(in_ch, out_ch, kernel_size=3):
    """
    Compare différentes méthodes de compression
    """
    original = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1)
    original_params = original.weight.numel()
    
    methods = {
        'Tucker': TuckerCompressedConv(in_ch, out_ch, kernel_size, 
                                      ranks=(out_ch//2, in_ch//2, 2, 2)),
        'CP': CPCompressedConv(in_ch, out_ch, kernel_size, rank=16),
    }
    
    print(f"Comparaison méthodes compression (Conv {in_ch}→{out_ch}, k={kernel_size}):")
    print(f"  Original: {original_params:,} paramètres")
    print()
    
    for name, compressed in methods.items():
        comp_params = compressed.count_parameters()
        ratio = original_params / comp_params
        print(f"  {name}:")
        print(f"    Paramètres: {comp_params:,}")
        print(f"    Compression: {ratio:.2f}x")
    
    return methods

compare_conv_compression_methods(64, 128, 3)
```

---

## Exercices

### Exercice 7.2.1
Implémentez une fonction qui trouve automatiquement les rangs Tucker optimaux pour une compression cible.

### Exercice 7.2.2
Comparez CP vs Tucker pour différentes architectures de convolutions (VGG, ResNet).

### Exercice 7.2.3
Implémentez une version optimisée de convolution Tucker qui évite la reconstruction complète des poids.

---

## Points Clés à Retenir

> 📌 **Tucker est particulièrement adapté aux convolutions (structure 4D naturelle)**

> 📌 **CP est plus compact mais peut avoir des problèmes de stabilité**

> 📌 **La reconstruction des poids peut être coûteuse - considérer le cache**

> 📌 **Les rangs optimaux dépendent de la structure des données**

> 📌 **TT peut être utilisé mais nécessite un flattening des dimensions**

---

*Section suivante : [7.3 Formats TT pour les Poids de Réseaux](./07_03_Formats_TT.md)*

