# 5.6 Comparaison et Choix de Décomposition

---

## Introduction

Ce chapitre compare les différentes décompositions tensorielles et guide le choix selon l'application.

---

## Tableau Comparatif

```python
import numpy as np

def comparison_table():
    """
    Comparaison complète des décompositions
    """
    
    decompositions = {
        'CP': {
            'Complexité params': 'O(N × I × R)',
            'Unicité': 'Sous conditions',
            'Stabilité': 'Moyenne (dégénérescence)',
            'Facilité calcul': 'Moyenne (ALS)',
            'Meilleur pour': 'Rang faible, interprétabilité'
        },
        'Tucker': {
            'Complexité params': 'O(R^N + N × I × R)',
            'Unicité': 'Oui (sous orthonormalité)',
            'Stabilité': 'Bonne (HOSVD stable)',
            'Facilité calcul': 'Bonne (HOSVD rapide)',
            'Meilleur pour': 'Compression modérée, flexibilité'
        },
        'Tensor Train': {
            'Complexité params': 'O(N × I × R²)',
            'Unicité': 'Non (sauf conditions)',
            'Stabilité': 'Bonne',
            'Facilité calcul': 'Bonne (TT-SVD)',
            'Meilleur pour': 'Grandes dimensions, compression forte'
        },
        'Tensor Ring': {
            'Complexité params': 'O(N × I × R²)',
            'Unicité': 'Non',
            'Stabilité': 'Bonne',
            'Facilité calcul': 'Bonne',
            'Meilleur pour': 'Rang circulaire, symétrie'
        }
    }
    
    print("Comparaison des Décompositions Tensorielles")
    print("=" * 80)
    print(f"{'Décomposition':<15} | {'Complexité':<20} | {'Unicité':<10} | {'Stabilité':<10}")
    print("-" * 80)
    
    for name, info in decompositions.items():
        print(f"{name:<15} | {info['Complexité params']:<20} | "
              f"{info['Unicité']:<10} | {info['Stabilité']:<10}")
    
    return decompositions

comparison_table()
```

---

## Choix selon l'Application

### Compression de Modèles ML

```python
class DecompositionSelector:
    """
    Sélectionne la meilleure décomposition selon le cas
    """
    
    @staticmethod
    def select_for_linear_layer(in_features, out_features, target_compression):
        """
        Sélection pour une couche linéaire
        """
        original_size = in_features * out_features
        
        # Teste différentes décompositions
        options = {}
        
        # CP (factorisation matricielle)
        cp_rank = int(np.sqrt(original_size / (in_features + out_features) / target_compression))
        cp_size = cp_rank * (in_features + out_features)
        options['CP'] = {
            'params': cp_size,
            'compression': original_size / cp_size,
            'complexity': 'O(n)'
        }
        
        # TT (si on peut factoriser les dimensions)
        # Nécessite de factoriser in_features et out_features
        # Ex: 1024 = 32×32, 512 = 16×32
        # Simplification
        tt_rank = 8
        tt_size = estimate_tt_params([32, 32], [16, 32], tt_rank)
        options['TT'] = {
            'params': tt_size,
            'compression': original_size / tt_size,
            'complexity': 'O(d)'
        }
        
        # Sélectionne la meilleure
        best = max(options.items(), key=lambda x: x[1]['compression'])
        
        return best[0], options
    
    @staticmethod
    def select_for_conv_layer(in_ch, out_ch, kernel_size, target_compression):
        """
        Sélection pour une couche convolutionnelle
        """
        original_size = in_ch * out_ch * kernel_size * kernel_size
        
        # Tucker est souvent meilleur pour les convolutions
        # (structure 4D naturelle)
        tucker_ranks = estimate_tucker_ranks(
            (out_ch, in_ch, kernel_size, kernel_size),
            target_compression
        )
        
        tucker_size = estimate_tucker_params(
            (out_ch, in_ch, kernel_size, kernel_size),
            tucker_ranks
        )
        
        return 'Tucker', {
            'params': tucker_size,
            'compression': original_size / tucker_size
        }

def estimate_tt_params(input_dims, output_dims, rank):
    """Estime les paramètres TT"""
    # Simplifié
    return rank * (sum(input_dims) + sum(output_dims))

def estimate_tucker_params(shape, ranks):
    """Estime les paramètres Tucker"""
    core_size = np.prod(ranks)
    factors_size = sum(shape[i] * ranks[i] for i in range(len(shape)))
    return core_size + factors_size

def estimate_tucker_ranks(shape, target_compression):
    """Estime les rangs Tucker pour une compression cible"""
    # Heuristique simplifiée
    original_size = np.prod(shape)
    target_size = original_size / target_compression
    
    # Approximation: rangs uniformes
    avg_rank = int((target_size / len(shape)) ** (1/len(shape)))
    return tuple([avg_rank] * len(shape))

# Exemple
selector = DecompositionSelector()
best, options = selector.select_for_linear_layer(1024, 512, target_compression=10)
print(f"Meilleure décomposition pour Linear(1024, 512): {best}")
print(f"Options: {options}")
```

---

## Critères de Sélection

### 1. Structure des Données

```python
def choose_by_structure(tensor_shape, data_structure):
    """
    Choisit selon la structure des données
    """
    if data_structure == 'sequential':
        # TT est naturel pour les séquences
        return 'Tensor Train'
    
    elif data_structure == 'hierarchical':
        # HT pour structures hiérarchiques
        return 'Hierarchical Tucker'
    
    elif data_structure == 'matrix_like':
        # CP ou low-rank pour matrices
        return 'CP'
    
    elif data_structure == 'multidimensional':
        # Tucker pour dimensions multiples
        return 'Tucker'
```

### 2. Contraintes de Compression

```python
def choose_by_compression_target(original_size, target_size, n_modes):
    """
    Choisit selon l'objectif de compression
    """
    compression_ratio = original_size / target_size
    
    if compression_ratio < 5:
        return 'CP'  # Compression légère
    elif compression_ratio < 50:
        return 'Tucker'  # Compression modérée
    else:
        return 'Tensor Train'  # Compression forte
```

### 3. Contraintes de Calcul

```python
def choose_by_compute_budget(max_params, n_modes, dim_per_mode):
    """
    Choisit selon le budget computationnel
    """
    # TT: O(N × I × R²)
    # Peut contrôler R pour tenir dans le budget
    
    # Tucker: O(R^N) - curse of dimensionality
    # Limité pour N grand
    
    if n_modes > 5:
        return 'Tensor Train'  # Évite curse
    else:
        return 'Tucker'  # Plus flexible
```

---

## Benchmarks Empiriques

```python
def benchmark_decompositions(tensor, ranks_dict, target_error=0.01):
    """
    Benchmark les différentes décompositions
    """
    results = {}
    
    # CP
    try:
        cp_factors, cp_weights, cp_error = cp_als(
            tensor, ranks_dict['cp'], max_iter=50
        )
        cp_params = sum(f.size for f in cp_factors) + len(cp_weights)
        results['CP'] = {
            'error': cp_error,
            'params': cp_params,
            'meets_target': cp_error < target_error
        }
    except:
        results['CP'] = {'error': np.inf, 'params': 0}
    
    # Tucker
    try:
        tucker_core, tucker_factors = hosvd(tensor, ranks_dict['tucker'])
        tucker_error = compute_reconstruction_error(
            tensor, reconstruct_tucker(tucker_core, tucker_factors)
        )
        tucker_params = tucker_core.size + sum(f.size for f in tucker_factors)
        results['Tucker'] = {
            'error': tucker_error,
            'params': tucker_params,
            'meets_target': tucker_error < target_error
        }
    except:
        results['Tucker'] = {'error': np.inf, 'params': 0}
    
    # TT
    try:
        tt_cores = tt_svd(tensor, max_rank=ranks_dict['tt'])
        tt_error = compute_tt_error(tensor, tt_cores)
        tt_params = sum(c.size for c in tt_cores)
        results['TT'] = {
            'error': tt_error,
            'params': tt_params,
            'meets_target': tt_error < target_error
        }
    except:
        results['TT'] = {'error': np.inf, 'params': 0}
    
    return results

def compute_reconstruction_error(original, reconstructed):
    """Calcule l'erreur relative"""
    return np.linalg.norm(original - reconstructed, 'fro') / \
           np.linalg.norm(original, 'fro')

def compute_tt_error(tensor, cores):
    """Erreur pour TT"""
    tt = TensorTrain(cores)
    reconstructed = tt.reconstruct()
    return compute_reconstruction_error(tensor, reconstructed)

# Test
tensor_bench = np.random.randn(10, 12, 8)
ranks = {'cp': 5, 'tucker': (5, 6, 4), 'tt': [5, 5]}

results = benchmark_decompositions(tensor_bench, ranks)

print("\nBenchmark des décompositions:")
for name, res in results.items():
    print(f"  {name}:")
    print(f"    Erreur: {res.get('error', np.inf):.6f}")
    print(f"    Paramètres: {res.get('params', 0):,}")
    print(f"    Objectif atteint: {res.get('meets_target', False)}")
```

---

## Guide de Décision

```
┌─────────────────────────────────────────────────────────────────┐
│              Arbre de Décision                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  N > 5 dimensions?                                              │
│    ├─ Oui → Tensor Train (évite curse)                        │
│    └─ Non → Continue...                                        │
│                                                                 │
│  Compression > 50x?                                             │
│    ├─ Oui → Tensor Train                                       │
│    └─ Non → Continue...                                        │
│                                                                 │
│  Structure hiérarchique?                                        │
│    ├─ Oui → Hierarchical Tucker                                │
│    └─ Non → Continue...                                        │
│                                                                 │
│  Rang faible et connu?                                          │
│    ├─ Oui → CP                                                 │
│    └─ Non → Tucker                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Exercices

### Exercice 5.6.1
Testez toutes les décompositions sur le même tenseur et comparez l'erreur vs le nombre de paramètres.

### Exercice 5.6.2
Créez un système automatique qui sélectionne la meilleure décomposition selon des critères (erreur, compression, vitesse).

### Exercice 5.6.3
Analysez quelle décomposition est la meilleure pour compresser une couche ResNet.

---

## Points Clés à Retenir

> 📌 **CP : Simple mais peut être instable, bon pour rang faible connu**

> 📌 **Tucker : Flexible mais souffre de curse of dimensionality**

> 📌 **TT : Excellent pour grandes dimensions, structure séquentielle**

> 📌 **HT : Bon compromis, structure hiérarchique naturelle**

> 📌 **Le choix dépend de la structure des données et des contraintes**

---

*Chapitre suivant : [Chapitre 6 - Réseaux de Tenseurs en Physique Quantique](../Chapitre_06_Physique_Quantique/06_introduction.md)*

