# Chapitre 11 : Approximations de Rang Faible pour la Compression

---

## Introduction

Les **approximations de rang faible** exploitent le fait que les matrices de poids dans les réseaux de neurones ont souvent un rang effectif bien inférieur à leur rang théorique. Cette propriété permet des compressions significatives tout en préservant les performances.

---

## Plan du Chapitre

1. [Factorisation Matricielle des Couches Denses](./11_01_Factorisation_Dense.md)
2. [Décomposition SVD Tronquée](./11_02_SVD_Tronquee.md)
3. [Low-Rank Adaptation (LoRA)](./11_03_LoRA.md)
4. [Combinaison Rang Faible + Quantification](./11_04_Combinaison.md)
5. [Analyse Théorique des Erreurs d'Approximation](./11_05_Erreurs.md)

---

## Principe Fondamental

Toute matrice $\mathbf{W} \in \mathbb{R}^{m \times n}$ de rang $r$ peut être factorisée :

$$\mathbf{W} = \mathbf{U} \mathbf{V}^T$$

où $\mathbf{U} \in \mathbb{R}^{m \times r}$ et $\mathbf{V} \in \mathbb{R}^{n \times r}$.

Réduction de paramètres : $mn \rightarrow r(m+n)$

```python
import torch
import torch.nn as nn
import numpy as np

class LowRankAnalysis:
    """
    Analyse le potentiel de compression par rang faible
    """
    
    @staticmethod
    def analyze_weight_matrix(W, threshold=0.99):
        """
        Analyse une matrice de poids pour déterminer son rang effectif
        """
        # SVD
        U, S, Vt = torch.svd(W)
        
        # Rang effectif (capture threshold% de l'énergie)
        total_energy = (S ** 2).sum()
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        effective_rank = torch.searchsorted(
            cumulative_energy / total_energy, 
            torch.tensor(threshold)
        ).item() + 1
        
        effective_rank = min(effective_rank, len(S))
        
        # Compression possible
        original_params = W.numel()
        compressed_params = effective_rank * (W.shape[0] + W.shape[1])
        compression_ratio = original_params / compressed_params
        
        return {
            'original_params': original_params,
            'effective_rank': effective_rank,
            'full_rank': min(W.shape),
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'singular_values': S.numpy(),
            'energy_captured': (cumulative_energy[effective_rank-1] / total_energy).item()
        }

# Exemple
W = torch.randn(1024, 512)
analysis = LowRankAnalysis.analyze_weight_matrix(W, threshold=0.99)

print("Analyse de rang faible:")
print(f"  Matrice: {W.shape}")
print(f"  Rang effectif (99%): {analysis['effective_rank']} / {analysis['full_rank']}")
print(f"  Compression: {analysis['compression_ratio']:.2f}x")
print(f"  Énergie capturée: {analysis['energy_captured']:.2%}")
```

---

## Compression par SVD Tronquée

```python
class SVDCompressedLinear(nn.Module):
    """
    Couche linéaire compressée par SVD
    """
    
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs de rang faible
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))
        
        # Initialisation
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        
    def forward(self, x):
        """
        y = (U @ V) @ x = U @ (V @ x)
        """
        return x @ self.V.T @ self.U.T
    
    @classmethod
    def from_linear(cls, linear_layer, rank):
        """
        Crée une couche compressée depuis une couche standard
        via SVD
        """
        W = linear_layer.weight.data
        
        # SVD
        U, S, Vt = torch.svd(W)
        
        # Troncature
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        # Crée la nouvelle couche
        compressed = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank
        )
        
        # Initialise avec SVD tronquée
        compressed.U.data = U_r @ torch.diag(torch.sqrt(S_r))
        compressed.V.data = torch.diag(torch.sqrt(S_r)) @ Vt_r
        
        return compressed
    
    def reconstruction_error(self, original_weight):
        """Calcule l'erreur de reconstruction"""
        W_reconstructed = self.U @ self.V
        error = torch.norm(original_weight - W_reconstructed, 'fro')
        relative_error = error / torch.norm(original_weight, 'fro')
        return relative_error.item()

# Exemple
original_layer = nn.Linear(1024, 512)
compressed_layer = SVDCompressedLinear.from_linear(original_layer, rank=64)

print(f"Paramètres originaux: {original_layer.weight.numel():,}")
print(f"Paramètres compressés: {compressed_layer.U.numel() + compressed_layer.V.numel():,}")
print(f"Compression: {original_layer.weight.numel() / (compressed_layer.U.numel() + compressed_layer.V.numel()):.2f}x")
print(f"Erreur relative: {compressed_layer.reconstruction_error(original_layer.weight.data):.4f}")
```

---

## LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation: adaptation efficace de grandes couches
    
    W_new = W_original + α * (B @ A)
    """
    
    def __init__(self, original_layer, rank, alpha=1.0):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Gèle les poids originaux
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Matrices de rang faible
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # Sortie originale
        original_out = self.original(x)
        
        # Adaptation LoRA
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        
        return original_out + self.alpha * lora_out
    
    def merge_weights(self):
        """
        Fusionne les poids LoRA avec les poids originaux
        """
        delta_W = self.alpha * (self.lora_B @ self.lora_A)
        self.original.weight.data += delta_W
    
    def trainable_parameters(self):
        """Nombre de paramètres entraînables"""
        return self.lora_A.numel() + self.lora_B.numel()

# Exemple: fine-tuning de BERT avec LoRA
base_layer = nn.Linear(768, 768)  # Couche type BERT
lora_layer = LoRALayer(base_layer, rank=16, alpha=1.0)

print(f"Paramètres totaux: {base_layer.weight.numel() + base_layer.bias.numel():,}")
print(f"Paramètres entraînables: {lora_layer.trainable_parameters():,}")
print(f"Réduction: {1 - lora_layer.trainable_parameters() / (base_layer.weight.numel() + base_layer.bias.numel()):.2%}")
```

---

## Combinaison avec Quantification

```python
class QuantizedLowRankLayer(nn.Module):
    """
    Combine rang faible et quantification
    """
    
    def __init__(self, in_features, out_features, rank, n_bits=8):
        super().__init__()
        
        self.rank = rank
        self.n_bits = n_bits
        
        # Facteurs de rang faible (seront quantifiés)
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(rank, in_features))
        
    def quantize(self):
        """Quantifie les poids"""
        def quantize_tensor(t, n_bits):
            scale = t.abs().max() / (2 ** (n_bits - 1) - 1)
            q = torch.round(t / scale)
            q = torch.clamp(q, -2**(n_bits-1), 2**(n_bits-1)-1)
            return q, scale
        
        self.U_q, self.U_scale = quantize_tensor(self.U.data, self.n_bits)
        self.V_q, self.V_scale = quantize_tensor(self.V.data, self.n_bits)
        
    def forward_quantized(self, x):
        """Forward pass avec poids quantifiés"""
        # Déquantifie
        U_deq = self.U_q.float() * self.U_scale
        V_deq = self.V_q.float() * self.V_scale
        
        # Forward
        return x @ V_deq.T @ U_deq.T

# Compression totale
layer = QuantizedLowRankLayer(1024, 512, rank=64, n_bits=8)
layer.quantize()

original_size = 1024 * 512 * 4  # float32 bytes
compressed_size = (64 * (1024 + 512)) * 1  # int8 bytes

print(f"Taille originale: {original_size / 1024:.1f} KB")
print(f"Taille compressée: {compressed_size / 1024:.1f} KB")
print(f"Compression totale: {original_size / compressed_size:.1f}x")
```

---

## Exercices

### Exercice 11.1
Implémentez une fonction qui trouve automatiquement le rang optimal pour une couche donnée avec une contrainte d'erreur maximale.

### Exercice 11.2
Comparez LoRA avec fine-tuning complet sur un modèle pré-entraîné. Mesurez la réduction de paramètres et la perte de performance.

### Exercice 11.3
Créez une couche qui combine SVD, quantification et pruning pour maximiser la compression.

---

## Points Clés à Retenir

> 📌 **Les matrices de poids ont souvent un rang effectif bien inférieur au rang théorique**

> 📌 **La SVD tronquée donne la meilleure approximation de rang k**

> 📌 **LoRA permet un fine-tuning efficace en n'entraînant que les adaptateurs**

> 📌 **La combinaison rang faible + quantification multiplie les gains**

---

*Section suivante : [11.1 Factorisation Matricielle des Couches Denses](./11_01_Factorisation_Dense.md)*

