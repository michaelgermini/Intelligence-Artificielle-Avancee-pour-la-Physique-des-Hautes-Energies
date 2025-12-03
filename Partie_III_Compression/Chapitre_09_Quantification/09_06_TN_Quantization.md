# 9.6 Quantification pour Réseaux de Tenseurs

---

## Introduction

La quantification des **réseaux de tenseurs** (Tensor Train, Tucker, etc.) nécessite des approches spécifiques car les poids sont structurés en cores tensoriels plutôt qu'en matrices denses.

---

## Défis Spécifiques

### Problèmes

1. **Structure des cores** : Les cores ont des shapes différentes
2. **Dépendances** : Les cores sont interdépendants
3. **Sensibilité** : La quantification peut affecter la structure TT

---

## Quantification des Cores TT

```python
import torch
import numpy as np

class TTQuantizer:
    """
    Quantificateur pour Tensor Train
    """
    
    def __init__(self, n_bits=8, symmetric=True):
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.quantizers = {}  # Un quantifier par core
    
    def quantize_tt_cores(self, cores):
        """
        Quantifie les cores TT
        
        Args:
            cores: liste de tenseurs 3D [(r₀,d₁,r₁), (r₁,d₂,r₂), ...]
        """
        quantized_cores = []
        scales = []
        zero_points = []
        
        for i, core in enumerate(cores):
            # Quantifie chaque core indépendamment
            quantizer = UniformQuantizer(n_bits=self.n_bits, 
                                        symmetric=self.symmetric)
            scale, zp = quantizer.compute_scale_zero_point(core)
            q_core = quantizer.quantize(core, scale, zp)
            
            quantized_cores.append(q_core)
            scales.append(scale)
            zero_points.append(zp)
        
        return quantized_cores, scales, zero_points
    
    def dequantize_tt_cores(self, quantized_cores, scales, zero_points):
        """Déquantifie les cores TT"""
        cores = []
        for q_core, scale, zp in zip(quantized_cores, scales, zero_points):
            quantizer = UniformQuantizer(n_bits=self.n_bits,
                                        symmetric=self.symmetric)
            core = quantizer.dequantize(q_core, scale, zp)
            cores.append(core)
        return cores

# Exemple
cores_tt = [
    torch.randn(1, 16, 8) * 0.1,
    torch.randn(8, 16, 4) * 0.1,
    torch.randn(4, 16, 1) * 0.1,
]

tt_quantizer = TTQuantizer(n_bits=8)
q_cores, scales, zps = tt_quantizer.quantize_tt_cores(cores_tt)

# Compression
original_size = sum(c.numel() * 32 for c in cores_tt)
quantized_size = sum(c.numel() * 8 for c in q_cores) + len(scales) * 32
print(f"Compression TT: {original_size / quantized_size:.2f}x")
```

---

## QAT pour Tensor Train

```python
class QATTTLinear(nn.Module):
    """
    Couche TT avec Quantization-Aware Training
    """
    
    def __init__(self, input_dims, output_dims, tt_ranks, n_bits=8):
        super().__init__()
        
        # Cores TT
        self.cores = nn.ModuleList()
        prev_rank = 1
        
        for i, dim in enumerate(input_dims):
            next_rank = tt_ranks[i] if i < len(tt_ranks) else 1
            core = nn.Parameter(torch.randn(prev_rank, dim, next_rank))
            self.cores.append(core)
            prev_rank = next_rank
        
        # Fake quantization pour chaque core
        self.fake_quants = nn.ModuleList()
        for core in self.cores:
            fq = FakeQuantizeModule(n_bits=n_bits, symmetric=True)
            self.fake_quants.append(fq)
    
    def forward(self, x):
        # Quantifie chaque core
        quantized_cores = []
        for core, fq in zip(self.cores, self.fake_quants):
            q_core = fq(core)
            quantized_cores.append(q_core)
        
        # Forward avec cores quantifiés
        # (Utilise la contraction TT)
        result = x.view(x.shape[0], *self.input_dims)
        
        for i, core in enumerate(quantized_cores):
            result = torch.tensordot(result, core, dims=([i+1], [1]))
        
        return result.view(x.shape[0], -1)
```

---

## Quantification Tucker

```python
class TuckerQuantizer:
    """
    Quantificateur pour décomposition Tucker
    """
    
    def quantize_tucker(self, core, factors, n_bits=8):
        """
        Quantifie le noyau Tucker et les facteurs
        
        Args:
            core: tenseur noyau (r₁, r₂, ..., rₙ)
            factors: liste de matrices facteurs [U₁, U₂, ..., Uₙ]
        """
        # Quantifie le noyau
        core_quantizer = UniformQuantizer(n_bits=n_bits, symmetric=True)
        scale_core, zp_core = core_quantizer.compute_scale_zero_point(core)
        q_core = core_quantizer.quantize(core, scale_core, zp_core)
        
        # Quantifie chaque facteur
        q_factors = []
        factor_scales = []
        factor_zps = []
        
        for factor in factors:
            factor_quantizer = UniformQuantizer(n_bits=n_bits, symmetric=True)
            scale_f, zp_f = factor_quantizer.compute_scale_zero_point(factor)
            q_f = factor_quantizer.quantize(factor, scale_f, zp_f)
            
            q_factors.append(q_f)
            factor_scales.append(scale_f)
            factor_zps.append(zp_f)
        
        return {
            'quantized_core': q_core,
            'quantized_factors': q_factors,
            'core_scale': scale_core,
            'core_zp': zp_core,
            'factor_scales': factor_scales,
            'factor_zps': factor_zps
        }
```

---

## Impact sur la Compression

```python
def analyze_tn_quantization_impact(original_tn, quantized_tn):
    """
    Analyse l'impact de la quantification sur un réseau de tenseurs
    """
    # Compression
    original_params = sum(p.numel() for p in original_tn.parameters())
    quantized_params = sum(p.numel() for p in quantized_tn.parameters())
    
    # En termes de bits
    original_bits = original_params * 32
    quantized_bits = quantized_params * 8  # INT8
    
    compression_ratio = original_bits / quantized_bits
    
    print(f"Impact Quantification TN:")
    print(f"  Paramètres: {original_params:,} → {quantized_params:,}")
    print(f"  Bits: {original_bits:,} → {quantized_bits:,}")
    print(f"  Compression: {compression_ratio:.2f}x")
    
    return compression_ratio
```

---

## Exercices

### Exercice 9.6.1
Implémentez QAT pour une couche TT et comparez avec PTQ.

### Exercice 9.6.2
Analysez comment la quantification affecte les rangs TT (peut-on réduire les rangs après quantification?).

---

## Points Clés à Retenir

> 📌 **Quantifier les cores TT indépendamment est une approche simple**

> 📌 **QAT peut améliorer les résultats pour les réseaux tensoriels**

> 📌 **La quantification Tucker nécessite de quantifier le noyau et les facteurs**

> 📌 **La compression totale combine compression tensorielle + quantification**

---

*Chapitre suivant : [Chapitre 10 - Knowledge Distillation](../Chapitre_10_Knowledge_Distillation/10_introduction.md)*

