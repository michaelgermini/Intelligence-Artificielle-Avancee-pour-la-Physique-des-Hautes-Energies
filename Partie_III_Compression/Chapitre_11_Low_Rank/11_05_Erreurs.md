# 11.5 Analyse Théorique des Erreurs d'Approximation

---

## Introduction

Comprendre les erreurs d'approximation est crucial pour choisir les paramètres de compression. Cette section présente l'analyse théorique des erreurs pour les approximations de rang faible.

---

## Erreur de Reconstruction SVD

### Théorème d'Eckart-Young (Forme Générale)

Pour une matrice $\mathbf{W} \in \mathbb{R}^{m \times n}$ avec SVD $\mathbf{W} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, la meilleure approximation de rang $k$ est :

$$\mathbf{W}_k = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

L'erreur (norme de Frobenius) est :

$$||\mathbf{W} - \mathbf{W}_k||_F = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}$$

où $\sigma_i$ sont les valeurs singulières.

---

## Implémentation d'Analyse

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

class ApproximationErrorAnalysis:
    """
    Analyse théorique des erreurs d'approximation
    """
    
    @staticmethod
    def svd_reconstruction_error(W, rank):
        """
        Calcule l'erreur de reconstruction SVD pour un rang donné
        """
        U, S, Vt = torch.svd(W)
        
        # Reconstruction avec rang
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        W_approx = U_r @ torch.diag(S_r) @ Vt_r
        
        # Erreurs
        frobenius_error = torch.norm(W - W_approx, 'fro')
        relative_error = frobenius_error / torch.norm(W, 'fro')
        
        # Erreur théorique (somme des valeurs singulières au-delà du rang)
        theoretical_error = torch.sqrt((S[rank:] ** 2).sum())
        theoretical_relative = theoretical_error / torch.norm(W, 'fro')
        
        return {
            'empirical_frobenius': frobenius_error.item(),
            'empirical_relative': relative_error.item(),
            'theoretical_frobenius': theoretical_error.item(),
            'theoretical_relative': theoretical_relative.item(),
            'singular_values': S.numpy()
        }
    
    @staticmethod
    def analyze_error_vs_rank(W, max_rank=None):
        """
        Analyse l'erreur pour différents rangs
        """
        if max_rank is None:
            max_rank = min(W.shape)
        
        U, S, Vt = torch.svd(W)
        norms_sq = S ** 2
        total_norm_sq = norms_sq.sum()
        
        errors = []
        ranks = list(range(1, max_rank + 1, max(1, max_rank // 50)))
        
        for rank in ranks:
            # Erreur théorique
            error_norm_sq = norms_sq[rank:].sum()
            relative_error = torch.sqrt(error_norm_sq / total_norm_sq).item()
            
            # Énergie capturée
            energy_captured = norms_sq[:rank].sum() / total_norm_sq
            
            errors.append({
                'rank': rank,
                'relative_error': relative_error,
                'energy_captured': energy_captured.item(),
                'compression_ratio': (W.numel()) / (rank * (W.shape[0] + W.shape[1]))
            })
        
        return errors

# Exemple
W = torch.randn(512, 1024)
errors = ApproximationErrorAnalysis.analyze_error_vs_rank(W, max_rank=256)

print("Analyse Erreur vs Rang:")
for e in errors[::20]:  # Affiche tous les 20
    print(f"  Rank {e['rank']:3d}: Error={e['relative_error']:.4f}, "
          f"Energy={e['energy_captured']:.4f}, Compression={e['compression_ratio']:.2f}x")
```

---

## Borne d'Erreur

### Inégalité de Weyl

Pour des perturbations de la matrice originale :

$$|\sigma_i(\mathbf{W} + \mathbf{E}) - \sigma_i(\mathbf{W})| \leq ||\mathbf{E}||_2$$

où $\sigma_i$ sont les valeurs singulières et $||\cdot||_2$ est la norme spectrale.

---

## Propagation d'Erreur

### Erreur dans le Forward Pass

Pour une couche factorisée $y = (\mathbf{U}\mathbf{V}^T) x$ :

Si $\mathbf{W} = \mathbf{U}\mathbf{V}^T + \mathbf{E}$ (erreur d'approximation), alors :

$$||y - y_{\text{true}}||_2 \leq ||\mathbf{E}||_2 \cdot ||x||_2$$

```python
def analyze_forward_error(layer, test_inputs, original_weight):
    """
    Analyse l'erreur dans le forward pass
    """
    # Forward avec couche factorisée
    outputs_approx = layer(test_inputs)
    
    # Forward avec poids originaux
    original_layer = nn.Linear(layer.in_features, layer.out_features)
    original_layer.weight.data = original_weight
    outputs_true = original_layer(test_inputs)
    
    # Erreur
    error_per_sample = torch.norm(outputs_approx - outputs_true, p=2, dim=1)
    relative_error = error_per_sample / torch.norm(outputs_true, p=2, dim=1)
    
    # Borne théorique
    W_recon = layer.reconstruct_weight()
    E = original_weight - W_recon
    error_bound = torch.norm(E, p=2) * torch.norm(test_inputs, p=2, dim=1)
    
    return {
        'mean_error': error_per_sample.mean().item(),
        'mean_relative_error': relative_error.mean().item(),
        'error_bound': error_bound.mean().item()
    }

# Test
factorized = FactorizedLinear(1024, 512, rank=64)
original_W = torch.randn(512, 1024)
factorized.U.data, factorized.V.data = initialize_from_svd(original_W, rank=64)

test_inputs = torch.randn(100, 1024)
errors = analyze_forward_error(factorized, test_inputs, original_W)

print(f"Erreur Forward Pass:")
print(f"  Erreur moyenne: {errors['mean_error']:.6f}")
print(f"  Erreur relative: {errors['mean_relative_error']:.4f}")
print(f"  Borne théorique: {errors['error_bound']:.6f}")
```

---

## Erreur Cumulée dans un Réseau

```python
def analyze_cumulative_error(model_original, model_compressed, test_inputs):
    """
    Analyse l'erreur cumulée à travers toutes les couches
    """
    # Forward pass complet
    with torch.no_grad():
        outputs_original = model_original(test_inputs)
        outputs_compressed = model_compressed(test_inputs)
    
    # Erreur finale
    final_error = torch.norm(outputs_original - outputs_compressed, 'fro')
    relative_error = final_error / torch.norm(outputs_original, 'fro')
    
    # Analyse par couche
    layer_errors = []
    # (Simplifié - nécessite hooks pour capturer les erreurs par couche)
    
    return {
        'final_error': final_error.item(),
        'relative_error': relative_error.item(),
        'layer_errors': layer_errors
    }
```

---

## Erreur avec Quantification

### Erreur Combinée

Pour low-rank + quantisation, l'erreur totale est approximativement :

$$\text{Erreur}_{\text{total}} \approx \text{Erreur}_{\text{SVD}} + \text{Erreur}_{\text{quant}}$$

```python
def combined_error_analysis(W, rank, n_bits):
    """
    Analyse l'erreur combinée low-rank + quantification
    """
    # Erreur SVD
    svd_analysis = ApproximationErrorAnalysis.svd_reconstruction_error(W, rank)
    svd_error = svd_analysis['relative_error']
    
    # Erreur quantification
    quantizer = UniformQuantizer(n_bits=n_bits)
    scale, zp = quantizer.compute_scale_zero_point(W)
    W_quantized = quantizer.quantize_dequantize(W)
    quant_error = torch.norm(W - W_quantized, 'fro') / torch.norm(W, 'fro')
    
    # Erreur combinée (approximation)
    combined_error = svd_error + quant_error
    
    # Erreur réelle (low-rank puis quantifier)
    U, S, Vt = torch.svd(W)
    sqrt_S = torch.sqrt(S[:rank])
    U_r = U[:, :rank]
    Vt_r = Vt[:rank, :]
    
    W_low_rank = (U_r @ torch.diag(sqrt_S)) @ (torch.diag(sqrt_S) @ Vt_r)
    W_combined = quantizer.quantize_dequantize(W_low_rank)
    
    actual_combined_error = torch.norm(W - W_combined, 'fro') / torch.norm(W, 'fro')
    
    return {
        'svd_error': svd_error,
        'quant_error': quant_error.item(),
        'estimated_combined': combined_error,
        'actual_combined': actual_combined_error.item()
    }

W_test = torch.randn(512, 1024)
errors = combined_error_analysis(W_test, rank=64, n_bits=8)

print("Erreurs Combinées:")
print(f"  SVD seule: {errors['svd_error']:.6f}")
print(f"  Quant seule: {errors['quant_error']:.6f}")
print(f"  Estimée combinée: {errors['estimated_combined']:.6f}")
print(f"  Réelle combinée: {errors['actual_combined']:.6f}")
```

---

## Trade-off Compression vs Erreur

```python
def compression_error_tradeoff(W, rank_range=None, n_bits_range=[4, 6, 8]):
    """
    Analyse le trade-off compression vs erreur
    """
    if rank_range is None:
        rank_range = list(range(16, min(W.shape) + 1, 16))
    
    results = []
    
    for rank in rank_range:
        for n_bits in n_bits_range:
            # Compression
            original_size = W.numel() * 4  # FP32 bytes
            compressed_size = rank * (W.shape[0] + W.shape[1]) * (n_bits / 8)
            compression = original_size / compressed_size
            
            # Erreur
            error_analysis = combined_error_analysis(W, rank, n_bits)
            error = error_analysis['actual_combined']
            
            results.append({
                'rank': rank,
                'n_bits': n_bits,
                'compression': compression,
                'error': error
            })
    
    return results

# Visualisation du trade-off
W = torch.randn(1024, 512)
tradeoffs = compression_error_tradeoff(W)

print("Trade-off Compression vs Erreur:")
for t in tradeoffs[::3]:
    print(f"  Rank {t['rank']:3d}, {t['n_bits']}bits: "
          f"Compression={t['compression']:.1f}x, Error={t['error']:.4f}")
```

---

## Exercices

### Exercice 11.5.1
Vérifiez empiriquement le théorème d'Eckart-Young sur différentes matrices.

### Exercice 11.5.2
Analysez comment l'erreur d'approximation se propage dans un réseau multi-couches.

### Exercice 11.5.3
Trouvez le rang optimal qui maximise la compression sous contrainte d'erreur maximale.

---

## Points Clés à Retenir

> 📌 **L'erreur SVD théorique est la racine de la somme des carrés des valeurs singulières ignorées**

> 📌 **Le théorème d'Eckart-Young garantit l'optimalité de SVD tronquée**

> 📌 **L'erreur se propage linéairement dans le forward pass**

> 📌 **L'erreur combinée (low-rank + quant) est approximativement additive**

> 📌 **Il existe un trade-off optimal compression/erreur**

---

*Chapitre suivant : [Chapitre 12 - pQuant Library](../Chapitre_12_pQuant/12_introduction.md)*

