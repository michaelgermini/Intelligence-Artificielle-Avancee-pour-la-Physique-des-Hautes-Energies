# 9.1 Fondements de la Quantification

---

## Introduction

La **quantification** est une technique fondamentale de compression qui réduit la précision numérique des paramètres d'un réseau de neurones. Comprendre ses fondements mathématiques est essentiel pour l'appliquer efficacement.

---

## Principe de Base

### Définition

La quantification transforme des valeurs continues (float32) en valeurs discrètes (int8, int4, etc.) :

$$\mathbb{R} \rightarrow \{q_{\min}, q_{\min}+1, \ldots, q_{\max}\}$$

où $q_{\min}$ et $q_{\max}$ sont déterminés par le nombre de bits.

---

## Quantification Uniforme

### Formule Mathématique

Pour une valeur $x \in \mathbb{R}$, la quantification uniforme est :

$$q = \text{round}\left(\frac{x - \text{zero\_point}}{\text{scale}}\right)$$

Et la déquantification :

$$\hat{x} = q \times \text{scale} + \text{zero\_point}$$

où :
- **scale** : pas de quantification (quantum)
- **zero_point** : valeur quantifiée correspondant à 0

---

## Implémentation de Base

```python
import torch
import torch.nn as nn
import numpy as np

class UniformQuantizer:
    """
    Quantificateur uniforme générique
    """
    
    def __init__(self, n_bits=8, symmetric=True, signed=True):
        """
        Args:
            n_bits: nombre de bits (8 pour INT8, 4 pour INT4)
            symmetric: si True, quantification symétrique autour de 0
            signed: si True, utilise des entiers signés
        """
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.signed = signed
        
        # Détermine la plage de valeurs quantifiées
        if signed:
            self.q_min = -(2 ** (n_bits - 1))
            self.q_max = 2 ** (n_bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** n_bits - 1
        
        # Pour quantification symétrique, zero_point est toujours 0
        if symmetric and signed:
            self.zero_point = 0
        elif symmetric and not signed:
            raise ValueError("Symmetric quantization requires signed integers")
    
    def compute_scale_zero_point(self, x):
        """
        Calcule scale et zero_point optimaux pour quantifier x
        
        Args:
            x: tenseur à quantifier
        
        Returns:
            scale: pas de quantification
            zero_point: point zéro quantifié
        """
        x_min = x.min().item()
        x_max = x.max().item()
        
        if self.symmetric:
            # Symétrique: plage centrée sur 0
            abs_max = max(abs(x_min), abs(x_max))
            
            if abs_max == 0:
                scale = 1.0
            else:
                # Scale pour couvrir [-abs_max, abs_max]
                scale = abs_max / self.q_max
            
            zero_point = 0
        else:
            # Asymétrique: utilise toute la plage
            if x_max == x_min:
                scale = 1.0
                zero_point = 0
            else:
                scale = (x_max - x_min) / (self.q_max - self.q_min)
                zero_point = self.q_min - round(x_min / scale)
                # Clamp zero_point dans la plage valide
                zero_point = int(np.clip(zero_point, self.q_min, self.q_max))
        
        return scale, zero_point
    
    def quantize(self, x, scale, zero_point):
        """
        Quantifie un tenseur
        
        Args:
            x: tenseur float32
            scale: pas de quantification
            zero_point: point zéro
        
        Returns:
            q: tenseur quantifié (int8/int4)
        """
        # Quantification
        q = torch.round(x / scale + zero_point)
        
        # Clamp dans la plage valide
        q = torch.clamp(q, self.q_min, self.q_max)
        
        # Convertit en type entier approprié
        if self.n_bits == 8:
            return q.to(torch.int8)
        elif self.n_bits == 4:
            # INT4 nécessite un encodage spécial
            return q.to(torch.int8)  # Simplifié
        else:
            return q.to(torch.int32)
    
    def dequantize(self, q, scale, zero_point):
        """
        Déquantifie un tenseur
        
        Args:
            q: tenseur quantifié (int)
            scale: pas de quantification
            zero_point: point zéro
        
        Returns:
            x_recon: tenseur déquantifié (float32)
        """
        return (q.float() - zero_point) * scale
    
    def quantize_dequantize(self, x):
        """
        Simule la quantification (quantify + dequantify)
        
        Utile pour la simulation pendant l'entraînement
        """
        scale, zp = self.compute_scale_zero_point(x)
        q = self.quantize(x, scale, zp)
        return self.dequantize(q, scale, zp)
    
    def quantization_error(self, x, scale, zero_point):
        """
        Calcule l'erreur de quantification
        
        Returns:
            erreur absolue moyenne
            erreur relative moyenne
            erreur maximale
        """
        q = self.quantize(x, scale, zero_point)
        x_recon = self.dequantize(q, scale, zero_point)
        
        abs_error = (x - x_recon).abs()
        
        return {
            'mean_abs_error': abs_error.mean().item(),
            'mean_rel_error': (abs_error / (x.abs() + 1e-8)).mean().item(),
            'max_error': abs_error.max().item(),
            'mse': (x - x_recon).pow(2).mean().item()
        }

# Exemple
quantizer = UniformQuantizer(n_bits=8, symmetric=True)

# Test sur des données aléatoires
x = torch.randn(1000) * 5.0  # Données dans [-5, 5] environ

scale, zp = quantizer.compute_scale_zero_point(x)
q = quantizer.quantize(x, scale, zp)
x_recon = quantizer.dequantize(q, scale, zp)

errors = quantizer.quantization_error(x, scale, zp)

print("Quantification Uniforme (INT8, Symétrique):")
print(f"  Scale: {scale:.6f}, Zero Point: {zp}")
print(f"  Erreur absolue moyenne: {errors['mean_abs_error']:.6f}")
print(f"  Erreur relative moyenne: {errors['mean_rel_error']:.2%}")
print(f"  Erreur maximale: {errors['max_error']:.6f}")
print(f"  MSE: {errors['mse']:.6f}")
```

---

## Quantification Symétrique vs Asymétrique

### Comparaison

```python
def compare_symmetric_vs_asymmetric(x):
    """
    Compare quantification symétrique et asymétrique
    """
    # Symétrique
    quantizer_sym = UniformQuantizer(n_bits=8, symmetric=True)
    scale_sym, zp_sym = quantizer_sym.compute_scale_zero_point(x)
    q_sym = quantizer_sym.quantize(x, scale_sym, zp_sym)
    x_recon_sym = quantizer_sym.dequantize(q_sym, scale_sym, zp_sym)
    error_sym = quantizer_sym.quantization_error(x, scale_sym, zp_sym)
    
    # Asymétrique
    quantizer_asym = UniformQuantizer(n_bits=8, symmetric=False)
    scale_asym, zp_asym = quantizer_asym.compute_scale_zero_point(x)
    q_asym = quantizer_asym.quantize(x, scale_asym, zp_asym)
    x_recon_asym = quantizer_asym.dequantize(q_asym, scale_asym, zp_asym)
    error_asym = quantizer_asym.quantization_error(x, scale_asym, zp_asym)
    
    print("Comparaison Symétrique vs Asymétrique:")
    print(f"  Symétrique: Scale={scale_sym:.6f}, ZP={zp_sym}, "
          f"Erreur={error_sym['mean_abs_error']:.6f}")
    print(f"  Asymétrique: Scale={scale_asym:.6f}, ZP={zp_asym}, "
          f"Erreur={error_asym['mean_abs_error']:.6f}")
    
    # Avantages/Inconvénients
    print("\n  Avantages symétrique:")
    print("    - Zero point = 0 simplifie les calculs")
    print("    - Meilleur pour hardware (pas de soustraction de ZP)")
    print("  Avantages asymétrique:")
    print("    - Utilise toute la plage de bits efficacement")
    print("    - Meilleur pour distributions asymétriques")
    
    return error_sym, error_asym

# Test avec distribution asymétrique
x_asym = torch.cat([torch.randn(500) * 0.5, torch.randn(500) * 2.0 + 1.0])
compare_symmetric_vs_asymmetric(x_asym)
```

---

## Granularité de Quantification

### Per-Tensor (Par Tenseur)

```python
class PerTensorQuantizer:
    """
    Quantification per-tensor: un scale/zero_point pour tout le tenseur
    """
    
    def __init__(self, n_bits=8, symmetric=True):
        self.quantizer = UniformQuantizer(n_bits, symmetric)
    
    def quantize_tensor(self, x):
        """Quantifie tout le tenseur avec les mêmes paramètres"""
        scale, zp = self.quantizer.compute_scale_zero_point(x)
        q = self.quantizer.quantize(x, scale, zp)
        return q, scale, zp

# Exemple: matrice de poids
W = torch.randn(128, 256) * 0.1
per_tensor = PerTensorQuantizer(n_bits=8)

q_W, scale, zp = per_tensor.quantize_tensor(W)
print(f"Per-Tensor: Scale={scale:.6f}, ZP={zp}")
print(f"  Taille paramètres: 1 scale + 1 zero_point = 2 valeurs")
```

### Per-Channel (Par Canal)

```python
class PerChannelQuantizer:
    """
    Quantification per-channel: un scale/zero_point par canal
    """
    
    def __init__(self, n_bits=8, symmetric=True, channel_dim=0):
        """
        Args:
            channel_dim: dimension le long de laquelle quantifier
        """
        self.quantizer = UniformQuantizer(n_bits, symmetric)
        self.channel_dim = channel_dim
    
    def quantize_tensor(self, x):
        """
        Quantifie avec des paramètres par canal
        
        Pour un tenseur (C, H, W), calcule C scales et zero_points
        """
        scales = []
        zero_points = []
        quantized_channels = []
        
        # Itère sur chaque canal
        num_channels = x.shape[self.channel_dim]
        
        for c in range(num_channels):
            # Extrait le canal
            if self.channel_dim == 0:
                channel = x[c]
            elif self.channel_dim == 1:
                channel = x[:, c]
            else:
                # Généralisation pour autres dimensions
                indices = [slice(None)] * x.ndim
                indices[self.channel_dim] = c
                channel = x[tuple(indices)]
            
            # Quantifie le canal
            scale, zp = self.quantizer.compute_scale_zero_point(channel)
            q_channel = self.quantizer.quantize(channel, scale, zp)
            
            scales.append(scale)
            zero_points.append(zp)
            quantized_channels.append(q_channel)
        
        # Reassemble
        q_x = torch.stack(quantized_channels, dim=self.channel_dim)
        scales = torch.tensor(scales)
        zero_points = torch.tensor(zero_points, dtype=torch.int32)
        
        return q_x, scales, zero_points

# Exemple: poids convolutionnels (out_channels, in_channels, H, W)
W_conv = torch.randn(64, 32, 3, 3) * 0.1
per_channel = PerChannelQuantizer(n_bits=8, channel_dim=0)

q_W_conv, scales, zps = per_channel.quantize_tensor(W_conv)
print(f"Per-Channel (dim=0): {scales.shape[0]} scales")
print(f"  Taille paramètres: {scales.shape[0]} scales + {zps.shape[0]} zero_points = "
      f"{scales.shape[0] * 2} valeurs")
print(f"  vs Per-Tensor: 2 valeurs")
```

---

## Erreur de Quantification

### Analyse Théorique

L'erreur de quantification suit une distribution uniforme dans l'intervalle $[-\frac{\text{scale}}{2}, \frac{\text{scale}}{2}]$ :

$$\text{Erreur} = \hat{x} - x \sim \mathcal{U}(-\text{scale}/2, \text{scale}/2)$$

Variance de l'erreur :

$$\text{Var}(\text{erreur}) = \frac{\text{scale}^2}{12}$$

```python
def analyze_quantization_error_distribution(x, n_bits=8, n_samples=10000):
    """
    Analyse la distribution de l'erreur de quantification
    """
    quantizer = UniformQuantizer(n_bits=n_bits, symmetric=True)
    scale, zp = quantizer.compute_scale_zero_point(x)
    
    errors = []
    for _ in range(n_samples):
        # Quantifie/déquantifie
        x_recon = quantizer.quantize_dequantize(x)
        error = (x_recon - x).abs().mean().item()
        errors.append(error)
    
    errors = np.array(errors)
    
    # Comparaison avec théorie
    theoretical_var = scale ** 2 / 12
    theoretical_std = np.sqrt(theoretical_var)
    
    empirical_std = np.std(errors)
    empirical_mean = np.mean(errors)
    
    print(f"Analyse Erreur Quantification (INT{n_bits}):")
    print(f"  Scale: {scale:.6f}")
    print(f"  Variance théorique: {theoretical_var:.6f}")
    print(f"  Écart-type théorique: {theoretical_std:.6f}")
    print(f"  Écart-type empirique: {empirical_std:.6f}")
    print(f"  Moyenne erreur: {empirical_mean:.6f}")
    
    return errors

# analyze_quantization_error_distribution(torch.randn(1000))
```

---

## Impact sur les Performances

### Compression Mémoire

```python
def memory_compression_analysis():
    """
    Analyse la compression mémoire pour différentes précisions
    """
    original_size_mb = 100  # Modèle FP32 en MB
    
    precisions = {
        'FP32': 32,
        'FP16': 16,
        'INT8': 8,
        'INT4': 4,
        'Binary': 1,
    }
    
    print("Compression Mémoire:")
    print(f"  Original (FP32): {original_size_mb:.2f} MB")
    print()
    
    for name, bits in precisions.items():
        if name == 'FP32':
            continue
        
        compression_ratio = 32 / bits
        compressed_size = original_size_mb / compression_ratio
        
        print(f"  {name:8} ({bits:2} bits): "
              f"{compressed_size:6.2f} MB, "
              f"Compression {compression_ratio:.1f}x")

memory_compression_analysis()
```

### Vitesse d'Inférence

```python
def inference_speed_analysis():
    """
    Analyse théorique de l'accélération
    """
    print("Accélération Inférence (théorique):")
    print("  FP32: 1.0x (référence)")
    print("  FP16: ~2x (GPU modernes)")
    print("  INT8: ~2-4x (CPU/GPU avec support)")
    print("  INT4: ~4-8x (hardware spécialisé)")
    print()
    print("  Note: Dépend fortement du hardware et de l'implémentation")

inference_speed_analysis()
```

---

## Exercices

### Exercice 9.1.1
Implémentez une quantification per-group où les éléments sont divisés en groupes et chaque groupe a son propre scale/zero_point.

### Exercice 9.1.2
Analysez l'erreur de quantification pour différentes distributions (gaussienne, uniforme, bimodale).

### Exercice 9.1.3
Comparez quantitativement quantification symétrique vs asymétrique sur un réseau réel (mesurer accuracy, compression, vitesse).

---

## Points Clés à Retenir

> 📌 **La quantification uniforme utilise un scale et un zero_point pour mapper continu → discret**

> 📌 **Symétrique: zero_point=0, meilleur pour hardware. Asymétrique: meilleure utilisation de la plage**

> 📌 **Per-channel donne généralement de meilleurs résultats que per-tensor**

> 📌 **L'erreur de quantification a une variance théorique de scale²/12**

> 📌 **INT8 réduit la mémoire de 4x et peut accélérer de 2-4x**

---

*Section suivante : [9.2 Post-Training Quantization (PTQ)](./09_02_PTQ.md)*

