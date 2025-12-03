# 4.4 Complexité Computationnelle

---

## Introduction

La **complexité computationnelle** des opérations tensorielles est cruciale pour l'optimisation des algorithmes. Comprendre le coût des contractions et des décompositions permet de choisir les meilleures stratégies pour les applications à grande échelle.

---

## Métriques de Complexité

### Notation Big-O

```python
import numpy as np
import time

def analyze_contraction_complexity():
    """
    Analyse la complexité d'une contraction tensorielle
    """
    print("Complexité des Opérations Tensorielles")
    print("=" * 60)
    
    operations = {
        'Addition élément-par-élément': 'O(N)',
        'Produit Hadamard': 'O(N)',
        'Contraction simple (2 tenseurs)': 'O(N × M)',
        'Contraction multiple': 'O(∏ dimensions)',
        'SVD d\'une matrice': 'O(min(mn², m²n))',
        'Décomposition CP (ALS)': 'O(I × R × iterations)',
        'Décomposition Tucker (HOSVD)': 'O(∏I_i + Σ I_i R_i)',
    }
    
    for op, complexity in operations.items():
        print(f"  {op:35} : {complexity}")

analyze_contraction_complexity()
```

---

## Complexité des Contractions

### Contraction Simple

```python
def contraction_flops(shape1, shape2, contracted_dims):
    """
    Calcule le nombre de FLOPs (Floating Point Operations) pour une contraction
    
    Args:
        shape1, shape2: Shapes des tenseurs
        contracted_dims: (axis1, axis2) indices à contracter
    
    Returns:
        Nombre de FLOPs (multiplications + additions)
    """
    # Dimensions contractées
    contracted_size1 = shape1[contracted_dims[0]]
    contracted_size2 = shape2[contracted_dims[1]]
    
    assert contracted_size1 == contracted_size2, \
        "Dimensions contractées doivent être égales"
    
    # Dimensions libres
    free_dims1 = [d for i, d in enumerate(shape1) if i != contracted_dims[0]]
    free_dims2 = [d for i, d in enumerate(shape2) if i != contracted_dims[1]]
    
    # Taille du résultat
    result_size = np.prod(free_dims1) * np.prod(free_dims2)
    
    # Pour chaque élément du résultat:
    #   - contracted_size multiplications
    #   - contracted_size - 1 additions
    flops = result_size * (2 * contracted_size1 - 1)
    
    return flops

# Exemples
examples = [
    ((100, 50), (50, 80), (1, 0)),  # Produit matriciel
    ((10, 20, 30), (30, 40), (2, 0)),  # Contraction tenseur-matrice
    ((5, 6, 7), (7, 8, 9), (2, 0)),  # Contraction tenseur-tenseur
]

print("\nComplexité de Contractions:")
print("-" * 60)
for shape1, shape2, dims in examples:
    flops = contraction_flops(shape1, shape2, dims)
    print(f"  {shape1} × {shape2} (dims {dims})")
    print(f"    FLOPs: {flops:,}")
    print(f"    Mémoire résultat: {np.prod([s for i, s in enumerate(shape1) if i != dims[0]]) * np.prod([s for i, s in enumerate(shape2) if i != dims[1]]):,} éléments")
```

---

## Ordre Optimal de Contraction

### Problème

Pour contracter plusieurs tenseurs, l'ordre des contractions affecte drastiquement la complexité :

```python
def matrix_chain_multiplication_cost(dims):
    """
    Calcule le coût optimal pour multiplier une chaîne de matrices
    
    dims: liste [d0, d1, d2, ..., dn] où la matrice i a shape (d_i, d_{i+1})
    
    Returns: coût minimal et ordre optimal
    """
    n = len(dims) - 1  # Nombre de matrices
    if n <= 1:
        return 0, []
    
    # m[i][j] = coût minimal pour multiplier A_i ... A_j
    m = np.full((n, n), np.inf)
    split = np.zeros((n, n), dtype=int)
    
    # Coût = 0 pour une seule matrice
    for i in range(n):
        m[i, i] = 0
    
    # Programmation dynamique
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            for k in range(i, j):
                # Coût de multiplier (A_i ... A_k) et (A_{k+1} ... A_j)
                cost = (m[i, k] + m[k+1, j] + 
                       dims[i] * dims[k+1] * dims[j+1])
                
                if cost < m[i, j]:
                    m[i, j] = cost
                    split[i, j] = k
    
    # Reconstruction de l'ordre optimal
    def get_order(i, j):
        if i == j:
            return f"A_{i}"
        k = split[i, j]
        left = get_order(i, k)
        right = get_order(k+1, j)
        return f"({left} × {right})"
    
    optimal_order = get_order(0, n-1)
    
    return m[0, n-1], optimal_order

# Exemple: Multiplier A(10×100) × B(100×5) × C(5×50)
dims = [10, 100, 5, 50]
cost, order = matrix_chain_multiplication_cost(dims)

print("\nOptimisation de Chaîne de Matrices:")
print("-" * 60)
print(f"  Dimensions: {dims}")
print(f"  Coût optimal: {int(cost):,} FLOPs")
print(f"  Ordre optimal: {order}")

# Comparaison avec ordre naïf
naive_cost = dims[0] * dims[1] * dims[2] + dims[0] * dims[2] * dims[3]
print(f"  Ordre naïf (A×(B×C)): {naive_cost:,} FLOPs")
print(f"  Amélioration: {naive_cost / cost:.2f}x")
```

---

## Complexité des Décompositions

### CP (CANDECOMP/PARAFAC)

```python
def cp_als_complexity(shape, rank, n_iterations):
    """
    Complexité de CP-ALS
    
    Args:
        shape: tuple (I₁, I₂, ..., Iₙ)
        rank: R (rang CP)
        n_iterations: nombre d'itérations
    
    Complexity per iteration:
        Pour chaque mode n:
        - Matricisation: O(∏I_i)
        - Produit Khatri-Rao: O(R × ∏_{i≠n} I_i)
        - Moindres carrés: O(I_n × R² + R³)
        Total: O(∏I_i + R × Σ I_i + N × R² × Σ I_i + N × R³)
    """
    n_modes = len(shape)
    total_elements = np.prod(shape)
    
    # Par itération
    matricization = total_elements
    kr_product = rank * sum(np.prod([shape[j] for j in range(n_modes) if j != i]) 
                           for i in range(n_modes))
    least_squares = sum(shape[i] * rank * rank + rank * rank * rank 
                       for i in range(n_modes))
    
    cost_per_iteration = matricization + kr_product + least_squares
    total_cost = cost_per_iteration * n_iterations
    
    return {
        'per_iteration': cost_per_iteration,
        'total': total_cost,
        'dominant_term': f'O({n_iterations} × (∏I_i + R × Σ I_i + R³))'
    }

shape = (50, 60, 70)
rank = 20
iterations = 100

cp_complexity = cp_als_complexity(shape, rank, iterations)
print("\nComplexité CP-ALS:")
print("-" * 60)
print(f"  Shape: {shape}, Rank: {rank}, Iterations: {iterations}")
print(f"  Coût par itération: {cp_complexity['per_iteration']:,} FLOPs")
print(f"  Coût total: {cp_complexity['total']:,} FLOPs")
print(f"  Terme dominant: {cp_complexity['dominant_term']}")
```

### Tucker (HOSVD)

```python
def tucker_hosvd_complexity(shape, ranks):
    """
    Complexité de HOSVD (Higher-Order SVD)
    
    Complexity:
        Pour chaque mode:
        - Matricisation: O(∏I_i)
        - SVD: O(I_n × (∏_{i≠n} I_i)²) ou O(min(I_n, ∏_{i≠n} I_i) × I_n × ∏_{i≠n} I_i)
        - Calcul noyau: O(∏I_i)
    """
    n_modes = len(shape)
    total_elements = np.prod(shape)
    
    # Coût SVD pour chaque mode
    svd_costs = []
    for mode in range(n_modes):
        mode_size = shape[mode]
        other_sizes = np.prod([shape[i] for i in range(n_modes) if i != mode])
        
        # SVD: O(min(m, n) × m × n) pour matrice m×n
        svd_cost = min(mode_size, other_sizes) * mode_size * other_sizes
        svd_costs.append(svd_cost)
    
    total_svd = sum(svd_costs)
    core_computation = total_elements
    
    total_cost = n_modes * total_elements + total_svd + core_computation
    
    return {
        'matricization': n_modes * total_elements,
        'svd': total_svd,
        'core': core_computation,
        'total': total_cost,
        'dominant_term': f'O(Σ SVD(I_n × ∏_{i≠n} I_i))'
    }

shape = (50, 60, 70)
ranks = (30, 40, 50)

tucker_complexity = tucker_hosvd_complexity(shape, ranks)
print("\nComplexité Tucker (HOSVD):")
print("-" * 60)
print(f"  Shape: {shape}, Ranks: {ranks}")
print(f"  Matricisations: {tucker_complexity['matricization']:,} FLOPs")
print(f"  SVDs: {tucker_complexity['svd']:,} FLOPs")
print(f"  Calcul noyau: {tucker_complexity['core']:,} FLOPs")
print(f"  Total: {tucker_complexity['total']:,} FLOPs")
```

### Tensor Train (TT-SVD)

```python
def tt_svd_complexity(shape, max_ranks):
    """
    Complexité de TT-SVD
    
    Complexity:
        Pour chaque mode k (sauf le dernier):
        - Reshape: O(1)
        - SVD: O(min(r_{k-1} × I_k, ∏_{i>k} I_i) × r_{k-1} × I_k × ∏_{i>k} I_i)
        Total: O(Σ SVD_k)
    """
    n_modes = len(shape)
    costs = []
    
    rank_left = 1
    for k in range(n_modes - 1):
        # Taille de la matrice pour SVD
        rows = rank_left * shape[k]
        cols = np.prod(shape[k+1:])
        
        # SVD cost
        svd_cost = min(rows, cols) * rows * cols
        costs.append(svd_cost)
        
        # Rank pour l'itération suivante
        rank_left = min(max_ranks[k] if k < len(max_ranks) else min(rows, cols),
                       min(rows, cols))
    
    total_cost = sum(costs)
    
    return {
        'per_mode': costs,
        'total': total_cost,
        'dominant_term': f'O(Σ SVD(r_{k-1} × I_k × ∏_{i>k} I_i))'
    }

shape = (10, 20, 30, 40)
max_ranks = [5, 5, 5]

tt_complexity = tt_svd_complexity(shape, max_ranks)
print("\nComplexité Tensor Train (TT-SVD):")
print("-" * 60)
print(f"  Shape: {shape}, Max ranks: {max_ranks}")
for i, cost in enumerate(tt_complexity['per_mode']):
    print(f"  Mode {i+1}: {cost:,} FLOPs")
print(f"  Total: {tt_complexity['total']:,} FLOPs")
```

---

## Optimisation de la Mémoire

### Analyse de Mémoire

```python
def memory_analysis(tensors):
    """
    Analyse l'utilisation mémoire pour des opérations tensorielles
    """
    print("\nAnalyse Mémoire:")
    print("-" * 60)
    
    for name, tensor in tensors.items():
        size_bytes = tensor.nbytes
        size_mb = size_bytes / (1024 * 1024)
        print(f"  {name:15}: {size_mb:8.2f} MB ({size_bytes:,} bytes)")
    
    total_memory = sum(t.nbytes for t in tensors.values())
    print(f"  {'TOTAL':15}: {total_memory / (1024*1024):8.2f} MB")

# Exemple
tensors = {
    'Input': np.random.randn(1000, 784).astype(np.float32),
    'Weight': np.random.randn(784, 256).astype(np.float32),
    'Output': np.random.randn(1000, 256).astype(np.float32),
}

memory_analysis(tensors)
```

### Complexité Mémoire vs Calcul

```python
def memory_vs_compute_tradeoff():
    """
    Trade-off entre mémoire et calcul
    
    Exemple: Calculer A @ B @ C
    """
    A = np.random.randn(100, 200)
    B = np.random.randn(200, 300)
    C = np.random.randn(300, 50)
    
    # Stratégie 1: Calculer tout d'un coup (mémoire élevée)
    strategy1_memory = A.nbytes + B.nbytes + C.nbytes + (A.shape[0] * C.shape[1] * 4)
    strategy1_compute = A.shape[0] * A.shape[1] * B.shape[1] + \
                       A.shape[0] * B.shape[1] * C.shape[1]
    
    # Stratégie 2: Calculer par blocs (mémoire réduite, calcul similaire)
    strategy2_memory = A.nbytes + B.nbytes + C.nbytes + \
                      (A.shape[0] * B.shape[1] * 4)  # Résultat intermédiaire
    strategy2_compute = strategy1_compute  # Même calcul
    
    print("\nTrade-off Mémoire vs Calcul:")
    print("-" * 60)
    print(f"  Stratégie 1 (tout d'un coup):")
    print(f"    Mémoire: {strategy1_memory / (1024*1024):.2f} MB")
    print(f"    Calcul: {strategy1_compute:,} FLOPs")
    print(f"  Stratégie 2 (par blocs):")
    print(f"    Mémoire: {strategy2_memory / (1024*1024):.2f} MB")
    print(f"    Calcul: {strategy2_compute:,} FLOPs")
    print(f"  Économie mémoire: {(strategy1_memory - strategy2_memory) / (1024*1024):.2f} MB")

memory_vs_compute_tradeoff()
```

---

## Benchmarking Pratique

```python
def benchmark_operations():
    """
    Benchmark pratique des opérations tensorielles
    """
    import time
    
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    print("\nBenchmark Opérations Tensorielles:")
    print("-" * 60)
    print(f"{'Size':<15} | {'Matmul (ms)':<15} | {'SVD (ms)':<15} | {'Einsum (ms)':<15}")
    print("-" * 60)
    
    for size in sizes:
        A = np.random.randn(*size)
        B = np.random.randn(size[1], size[0])
        
        # Matmul
        start = time.time()
        C = A @ B
        matmul_time = (time.time() - start) * 1000
        
        # SVD
        start = time.time()
        U, S, Vt = np.linalg.svd(A)
        svd_time = (time.time() - start) * 1000
        
        # Einsum
        start = time.time()
        D = np.einsum('ij,jk->ik', A, B)
        einsum_time = (time.time() - start) * 1000
        
        print(f"{str(size):<15} | {matmul_time:>13.2f} | {svd_time:>13.2f} | {einsum_time:>13.2f}")

# Décommentez pour exécuter (peut prendre du temps)
# benchmark_operations()
```

---

## Complexité Asymptotique

### Comparaison des Décompositions

```python
def asymptotic_complexity_comparison():
    """
    Compare la complexité asymptotique des différentes décompositions
    """
    print("\nComplexité Asymptotique (N = taille, R = rang):")
    print("-" * 60)
    
    complexities = {
        'CP-ALS': 'O(I × R × iterations)',
        'Tucker (HOSVD)': 'O(∏I_i + Σ SVD(I_n × ∏_{i≠n} I_i))',
        'TT-SVD': 'O(Σ SVD(r_{k-1} × I_k × ∏_{i>k} I_i))',
        'SVD Matriciel': 'O(min(mn², m²n))',
    }
    
    for method, complexity in complexities.items():
        print(f"  {method:20}: {complexity}")
    
    print("\nPour un tenseur d'ordre d avec dimensions I:")
    print("  - CP: linéaire en I, mais itératif")
    print("  - Tucker: exponentiel en d (curse of dimensionality)")
    print("  - TT: linéaire en d")

asymptotic_complexity_comparison()
```

---

## Optimisations Pratiques

### Utilisation d'einsum Optimisé

```python
def einsum_optimization_tips():
    """
    Conseils pour optimiser einsum
    """
    tips = [
        "Utiliser einsum_path pour trouver l'ordre optimal",
        "Éviter les contractions inutiles",
        "Réduire l'ordre des tenseurs intermédiaires",
        "Utiliser des opérations batch quand possible",
        "Préférer des opérations natives (matmul) pour cas simples"
    ]
    
    print("\nOptimisations pour einsum:")
    print("-" * 60)
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

einsum_optimization_tips()

# Exemple: trouver le chemin optimal
def optimize_einsum_path():
    A = np.random.randn(50, 100)
    B = np.random.randn(100, 80)
    C = np.random.randn(80, 60)
    
    # Trouve le chemin optimal
    path, info = np.einsum_path('ij,jk,kl->il', A, B, C, optimize='optimal')
    
    print("\nOptimisation einsum path:")
    print(f"  Chemin optimal: {path}")
    print(f"  Info: {info}")

optimize_einsum_path()
```

---

## Exercices

### Exercice 4.4.1
Calculez la complexité exacte (en FLOPs) d'une décomposition CP pour un tenseur (100, 200, 150) avec rang 20 sur 50 itérations.

### Exercice 4.4.2
Implémentez un algorithme qui trouve l'ordre optimal de contraction pour 5 tenseurs.

### Exercice 4.4.3
Comparez empiriquement (benchmark) la complexité de différentes décompositions tensorielles.

---

## Points Clés à Retenir

> 📌 **La complexité d'une contraction dépend de l'ordre des opérations**

> 📌 **L'ordre optimal de contraction peut réduire la complexité de plusieurs ordres de grandeur**

> 📌 **Les décompositions tensoriennes ont des complexités très différentes**

> 📌 **Il y a souvent un trade-off entre mémoire et temps de calcul**

> 📌 **Utiliser einsum_path pour optimiser automatiquement les contractions**

---

*Chapitre suivant : [Chapitre 5 - Décompositions Tensorielles Fondamentales](../Chapitre_05_Decompositions/05_introduction.md)*

