# 4.2 Contraction de Tenseurs

---

## Introduction

La **contraction** est l'opération fondamentale sur les tenseurs. Elle généralise le produit matriciel aux dimensions supérieures en sommant sur des indices partagés.

---

## Définition de la Contraction

### Contraction Simple

La contraction de deux tenseurs sur des indices spécifiés :

$$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$$

```python
import numpy as np

def tensor_contraction(T1, T2, axes):
    """
    Contraction de deux tenseurs
    
    Args:
        T1: Tenseur d'ordre n
        T2: Tenseur d'ordre m
        axes: tuple (axes_T1, axes_T2) des indices à contracter
    
    Returns:
        Tenseur contracté
    """
    return np.tensordot(T1, T2, axes=axes)

# Exemple: Produit matriciel
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Contraction sur l'indice commun (dimension 4)
C = tensor_contraction(A, B, axes=(1, 0))
print(f"Produit matriciel: {A.shape} × {B.shape} = {C.shape}")
print(f"Vérification: {(A @ B).shape}")

# Exemple: Contraction de tenseurs d'ordre supérieur
T1 = np.random.randn(3, 4, 5)
T2 = np.random.randn(5, 6, 7)

# Contraction sur le dernier indice de T1 et le premier de T2
C2 = tensor_contraction(T1, T2, axes=(2, 0))
print(f"\nContraction tensorielle: {T1.shape} contracted with {T2.shape}")
print(f"Résultat: {C2.shape}")
```

---

## Types de Contractions

### Contraction sur Un Seul Indice

```python
def single_index_contraction():
    """
    Contraction sur un seul indice partagé
    """
    # Tenseurs d'ordre 3
    T1 = np.random.randn(4, 5, 6)
    T2 = np.random.randn(6, 7, 8)
    
    # Contraction sur l'indice j
    # T1[i, j, k] * T2[j, l, m] → R[i, k, l, m]
    R = np.tensordot(T1, T2, axes=(2, 0))
    print(f"T1 shape: {T1.shape}")
    print(f"T2 shape: {T2.shape}")
    print(f"Résultat R shape: {R.shape}")
    print(f"Ordre: {T1.ndim + T2.ndim - 2}")

single_index_contraction()
```

### Contraction sur Plusieurs Indices

```python
def multi_index_contraction():
    """
    Contraction sur plusieurs indices simultanément
    """
    T1 = np.random.randn(3, 4, 5)
    T2 = np.random.randn(5, 4, 6)
    
    # Contraction sur les indices 1 et 2 de T1 avec 1 et 0 de T2
    # Somme sur les dimensions (4, 5)
    R = np.tensordot(T1, T2, axes=((1, 2), (1, 0)))
    print(f"T1 shape: {T1.shape}")
    print(f"T2 shape: {T2.shape}")
    print(f"Contraction sur {(1, 2)} et {(1, 0)}")
    print(f"Résultat R shape: {R.shape}")

multi_index_contraction()
```

### Contraction Trace (Auto-contraction)

```python
def trace_contraction():
    """
    Contraction d'un tenseur avec lui-même (trace)
    """
    # Matrice: trace = somme des éléments diagonaux
    M = np.random.randn(5, 5)
    trace_matrix = np.trace(M)
    trace_einsum = np.einsum('ii->', M)
    print(f"Trace (matrice): {trace_matrix:.4f} = {trace_einsum:.4f}")
    
    # Tenseur d'ordre 4: trace sur certains indices
    T = np.random.randn(3, 4, 3, 4)
    # Trace sur indices 0 et 2
    trace_tensor = np.einsum('ijik->jk', T)
    print(f"\nTrace tensorielle: {T.shape} → {trace_tensor.shape}")

trace_contraction()
```

---

## Notation d'Einstein pour les Contractions

```python
def einsum_contractions():
    """
    Exemples de contractions avec einsum
    """
    # Produit matriciel
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C1 = np.einsum('ij,jk->ik', A, B)
    C2 = A @ B
    print(f"Produit matriciel: {np.allclose(C1, C2)}")
    
    # Contraction tensorielle générale
    T = np.random.randn(3, 4, 5)
    U = np.random.randn(5, 6)
    V = np.einsum('ijk,kl->ijl', T, U)
    print(f"T{tuple(T.shape)} × U{tuple(U.shape)} = V{tuple(V.shape)}")
    
    # Diagonale
    D = np.einsum('ii->i', np.random.randn(5, 5))
    print(f"Diagonale: shape {D.shape}")
    
    # Batch matrix multiplication
    A_batch = np.random.randn(10, 3, 4)
    B_batch = np.random.randn(10, 4, 5)
    C_batch = np.einsum('bij,bjk->bik', A_batch, B_batch)
    print(f"Batch matmul: {C_batch.shape}")
    
    # Contraction complexe: attention mechanism
    Q = np.random.randn(32, 8, 64)  # (batch, heads, seq, d_model)
    K = np.random.randn(32, 8, 64)
    # Attention = softmax(Q @ K^T / sqrt(d))
    scores = np.einsum('bhd,bhe->bhe', Q, K)  # Simplifié
    print(f"Attention scores: {scores.shape}")

einsum_contractions()
```

---

## Complexité des Contractions

### Calcul de la Complexité

```python
def contraction_complexity(T1_shape, T2_shape, contracted_axes):
    """
    Calcule la complexité d'une contraction
    
    Returns:
        flops: Nombre d'opérations flottantes
        memory: Mémoire nécessaire
    """
    # Dimensions contractées
    contract_dims_T1 = [T1_shape[i] for i in contracted_axes[0]]
    contract_dims_T2 = [T2_shape[i] for i in contracted_axes[1]]
    
    contract_size = np.prod(contract_dims_T1)
    
    # Vérifie que les dimensions correspondent
    assert contract_size == np.prod(contract_dims_T2)
    
    # Dimensions libres (non contractées)
    free_dims_T1 = [T1_shape[i] for i in range(len(T1_shape)) 
                    if i not in contracted_axes[0]]
    free_dims_T2 = [T2_shape[i] for i in range(len(T2_shape))
                    if i not in contracted_axes[1]]
    
    result_size = np.prod(free_dims_T1) * np.prod(free_dims_T2)
    
    # FLOPs: contract_size multiplications + (contract_size - 1) additions
    # par élément du résultat
    flops = result_size * (2 * contract_size - 1)
    
    # Mémoire: entrées + résultat
    memory = np.prod(T1_shape) + np.prod(T2_shape) + result_size
    
    return {
        'flops': flops,
        'memory': memory,
        'contract_size': contract_size,
        'result_size': result_size
    }

# Analyse
T1 = np.random.randn(100, 50, 30)
T2 = np.random.randn(30, 40)

complexity = contraction_complexity(T1.shape, T2.shape, ((2,), (0,)))
print("Complexité de la contraction:")
print(f"  FLOPs: {complexity['flops']:,}")
print(f"  Mémoire: {complexity['memory']:,} éléments")
print(f"  Taille de contraction: {complexity['contract_size']}")
print(f"  Taille du résultat: {complexity['result_size']}")
```

---

## Optimisation de l'Ordre de Contraction

### Problème du Contraction Path Optimal

Pour contracter plusieurs tenseurs, l'ordre affecte drastiquement la complexité :

```python
def contraction_path_complexity(chain_shapes):
    """
    Compare différentes stratégies pour contracter une chaîne de tenseurs
    
    chain_shapes: liste de shapes [s1, s2, ..., sn]
    où chaque tenseur Ti a shape (s_{i-1}, s_i) (chaîne matricielle)
    """
    n = len(chain_shapes) - 1
    
    # Stratégie 1: Contraction gauche à droite
    def left_to_right_cost():
        cost = 0
        # Commence avec le premier produit
        current_shape = (chain_shapes[0], chain_shapes[1])
        for i in range(2, n + 1):
            next_shape = (chain_shapes[i-1], chain_shapes[i])
            # Coût du produit matriciel
            cost += current_shape[0] * current_shape[1] * next_shape[1]
            current_shape = (current_shape[0], next_shape[1])
        return cost
    
    # Stratégie 2: Contraction droite à gauche
    def right_to_left_cost():
        cost = 0
        current_shape = (chain_shapes[n-1], chain_shapes[n])
        for i in range(n - 2, -1, -1):
            prev_shape = (chain_shapes[i], chain_shapes[i+1])
            cost += prev_shape[0] * prev_shape[1] * current_shape[1]
            current_shape = (prev_shape[0], current_shape[1])
        return cost
    
    cost_lr = left_to_right_cost()
    cost_rl = right_to_left_cost()
    
    print(f"Chaîne de {n} matrices: {chain_shapes}")
    print(f"  Coût gauche→droite: {cost_lr:,}")
    print(f"  Coût droite→gauche: {cost_rl:,}")
    print(f"  Meilleur: {'G→D' if cost_lr < cost_rl else 'D→G'}")
    
    return min(cost_lr, cost_rl)

# Exemple: Multiplication de chaîne de matrices
shapes = [10, 100, 50, 20, 30]
contraction_path_complexity(shapes)
```

### Algorithme de Programmation Dynamique

```python
def optimal_chain_contraction(shapes):
    """
    Trouve l'ordre optimal de contraction pour une chaîne de matrices
    via programmation dynamique
    
    shapes: [n0, n1, n2, ..., nk] pour matrices M1(n0×n1), M2(n1×n2), ...
    """
    n = len(shapes) - 1  # Nombre de matrices
    
    # dp[i][j] = coût minimal pour multiplier Mi ... Mj
    dp = [[0] * n for _ in range(n)]
    # parent[i][j] = indice optimal pour la dernière multiplication
    parent = [[0] * n for _ in range(n)]
    
    # Longueur de la chaîne
    for L in range(2, n + 1):
        for i in range(n - L + 1):
            j = i + L - 1
            dp[i][j] = float('inf')
            
            # Essaie tous les points de coupure
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] + 
                       shapes[i] * shapes[k+1] * shapes[j+1])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    parent[i][j] = k
    
    return dp[0][n-1], parent

# Test
shapes = [10, 100, 50, 20, 30]
optimal_cost, parent = optimal_chain_contraction(shapes)
print(f"\nCoût optimal: {optimal_cost:,}")
print(f"Coût naïf (gauche→droite): {sum(shapes[i]*shapes[i+1]*shapes[i+2] for i in range(len(shapes)-2)):,}")
```

---

## Applications aux Réseaux de Tenseurs

### Contraction de Tensor Train

```python
def tensor_train_contraction(cores):
    """
    Contracte un Tensor Train (chaîne de tenseurs 3D)
    
    cores: liste de tenseurs 3D [G1, G2, ..., Gn]
          où Gi a shape (r_{i-1}, n_i, r_i)
    """
    result = cores[0]
    
    for i in range(1, len(cores)):
        # Contraction: (..., r_{i-1}) × (r_{i-1}, n_i, r_i) → (..., n_i, r_i)
        result = np.tensordot(result, cores[i], axes=([-1], [0]))
    
    return result

# Exemple
cores = [
    np.random.randn(1, 5, 4),   # G1
    np.random.randn(4, 6, 3),   # G2
    np.random.randn(3, 7, 1)    # G3
]

result = tensor_train_contraction(cores)
print(f"Tensor Train contraction:")
print(f"  Cores shapes: {[c.shape for c in cores]}")
print(f"  Résultat shape: {result.shape}")
```

---

## Exercices

### Exercice 4.2.1
Implémentez une fonction qui contracte trois tenseurs T1, T2, T3 de manière optimale en termes de FLOPs.

### Exercice 4.2.2
Calculez la complexité de la contraction d'un MPS (Matrix Product State) de 10 sites, chacun de dimension locale 2 et rang de liaison 5.

### Exercice 4.2.3
Comparez différentes stratégies de contraction pour un réseau tensoriel avec la topologie d'un arbre.

---

## Points Clés à Retenir

> 📌 **La contraction généralise le produit matriciel aux dimensions supérieures**

> 📌 **L'ordre de contraction affecte exponentiellement la complexité**

> 📌 **La programmation dynamique trouve le chemin optimal pour les chaînes**

> 📌 **La notation einsum simplifie l'écriture des contractions complexes**

---

*Section suivante : [4.3 Diagrammes Tensoriels](./04_03_Diagrammes.md)*

