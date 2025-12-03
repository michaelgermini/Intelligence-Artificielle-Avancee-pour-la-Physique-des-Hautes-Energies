# 4.1 Définition et Notation des Tenseurs

---

## Introduction

Cette section définit rigoureusement les tenseurs, leur notation, et leurs propriétés fondamentales. Les tenseurs généralisent les concepts de scalaires, vecteurs et matrices à des dimensions arbitraires.

---

## Définitions Formelles

### Tenseur d'Ordre d

Un **tenseur d'ordre d** (ou de rang d) sur un espace vectoriel $V$ de dimension $n$ est un élément du produit tensoriel :

$$\mathcal{T} \in \underbrace{V \otimes V \otimes \cdots \otimes V}_{d \text{ fois}}$$

En coordonnées, c'est un tableau multidimensionnel :

$$T_{i_1, i_2, \ldots, i_d} \quad \text{où} \quad i_k \in \{1, 2, \ldots, n_k\}$$

```python
import numpy as np

class Tensor:
    """
    Classe de base pour manipuler des tenseurs
    """
    
    def __init__(self, data):
        """
        Crée un tenseur à partir d'un array NumPy
        
        Args:
            data: array NumPy de dimension arbitraire
        """
        self.data = np.array(data)
        self.order = self.data.ndim
        self.shape = self.data.shape
        self.size = self.data.size
        
    def __repr__(self):
        return (f"Tensor(order={self.order}, shape={self.shape}, "
                f"size={self.size})")
    
    def __getitem__(self, indices):
        """Accès aux éléments"""
        return self.data[indices]
    
    def __setitem__(self, indices, value):
        """Modification des éléments"""
        self.data[indices] = value
    
    def reshape(self, new_shape):
        """Change la forme du tenseur"""
        return Tensor(self.data.reshape(new_shape))
    
    def transpose(self, axes=None):
        """Transpose le tenseur"""
        return Tensor(self.data.transpose(axes))
    
    def norm(self, p=2):
        """Calcule la norme Lp"""
        if p == 'fro':
            return np.linalg.norm(self.data, 'fro')
        return np.linalg.norm(self.data.flatten(), p)

# Exemples
T1 = Tensor(np.random.randn(3, 4))      # Ordre 2 (matrice)
T2 = Tensor(np.random.randn(2, 3, 4))   # Ordre 3
T3 = Tensor(np.random.randn(2, 3, 4, 5)) # Ordre 4

print("Exemples de tenseurs:")
print(f"T1: {T1}")
print(f"T2: {T2}")
print(f"T3: {T3}")
```

---

## Indexation et Notation

### Notation Indicielle

Les tenseurs utilisent la notation indicielle avec la convention d'Einstein :

```python
def einstein_sum_example():
    """
    Exemples de la notation d'Einstein
    
    Convention: indices répétés impliquent sommation
    """
    # Exemple 1: Produit scalaire
    # v_i w_i = Σᵢ vᵢ wᵢ
    v = np.array([1, 2, 3])
    w = np.array([4, 5, 6])
    dot_product = np.einsum('i,i->', v, w)
    print(f"Produit scalaire: {dot_product}")
    print(f"Vérification: {np.dot(v, w)}")
    
    # Exemple 2: Produit matriciel
    # C_ij = A_ik B_kj = Σₖ Aᵢₖ Bₖⱼ
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C = np.einsum('ik,kj->ij', A, B)
    print(f"\nProduit matriciel: {C.shape}")
    print(f"Vérification: {(A @ B).shape}")
    
    # Exemple 3: Contraction tensorielle
    # T_ijk U_jl = V_ikl = Σⱼ Tᵢⱼₖ Uⱼₗ
    T = np.random.randn(3, 4, 5)
    U = np.random.randn(4, 6)
    V = np.einsum('ijk,jl->ikl', T, U)
    print(f"\nContraction: T{tuple(T.shape)} × U{tuple(U.shape)} = V{tuple(V.shape)}")
    
    # Exemple 4: Trace
    # tr(A) = A_ii = Σᵢ Aᵢᵢ
    M = np.random.randn(4, 4)
    trace = np.einsum('ii->', M)
    print(f"\nTrace: {trace:.4f}")
    print(f"Vérification: {np.trace(M):.4f}")

einstein_sum_example()
```

### Accès aux Éléments

```python
class TensorIndexing:
    """
    Techniques d'indexation avancées pour tenseurs
    """
    
    @staticmethod
    def basic_indexing(tensor):
        """Indexation basique"""
        print("Indexation basique:")
        print(f"  tensor[0]: {tensor[0].shape if tensor.ndim > 1 else tensor[0]}")
        print(f"  tensor[0, 1]: {tensor[0, 1] if tensor.ndim >= 2 else 'N/A'}")
        print(f"  tensor[0, :, 2]: {tensor[0, :, 2].shape if tensor.ndim >= 3 else 'N/A'}")
    
    @staticmethod
    def advanced_indexing(tensor):
        """Indexation avancée"""
        # Boolean indexing
        mask = tensor > 0
        positive_values = tensor[mask]
        print(f"\nValeurs positives: {len(positive_values)}/{tensor.size}")
        
        # Fancy indexing
        indices = [0, 2, 4]
        if tensor.ndim >= 2:
            selected = tensor[indices, :]
            print(f"Lignes sélectionnées: {selected.shape}")
    
    @staticmethod
    def slicing(tensor):
        """Slicing multidimensionnel"""
        if tensor.ndim >= 2:
            slice_2d = tensor[1:3, :]
            print(f"\nSlice 2D: {slice_2d.shape}")
        
        if tensor.ndim >= 3:
            slice_3d = tensor[:, 1:3, ::2]
            print(f"Slice 3D: {slice_3d.shape}")

# Test
T = np.random.randn(5, 6, 7)
TensorIndexing.basic_indexing(T)
TensorIndexing.advanced_indexing(T)
TensorIndexing.slicing(T)
```

---

## Rang Tensoriel

### Définition du Rang

Le **rang tensoriel** (ou rang CP) est le nombre minimal $r$ tel que :

$$\mathcal{T} = \sum_{k=1}^{r} \lambda_k \mathbf{a}_k^{(1)} \otimes \mathbf{a}_k^{(2)} \otimes \cdots \otimes \mathbf{a}_k^{(d)}$$

où chaque $\mathbf{a}_k^{(i)}$ est un vecteur.

```python
def tensor_rank_analysis(tensor):
    """
    Analyse le rang d'un tenseur
    
    Le rang tensoriel est difficile à calculer exactement (NP-hard),
    mais on peut estimer des bornes via les rangs des matricisations
    """
    # Matricisations (unfolding selon chaque mode)
    mode_ranks = []
    for mode in range(tensor.ndim):
        # Déplie selon le mode
        shape = tensor.shape
        dim_mode = shape[mode]
        other_dims = np.prod([shape[i] for i in range(len(shape)) if i != mode])
        
        # Matricisation
        tensor_reshaped = np.moveaxis(tensor, mode, 0)
        matrix = tensor_reshaped.reshape(dim_mode, other_dims)
        
        # Rang de la matrice
        rank = np.linalg.matrix_rank(matrix)
        mode_ranks.append(rank)
    
    # Bornes sur le rang tensoriel
    lower_bound = max(mode_ranks)
    upper_bound = min(tensor.shape)
    
    return {
        'mode_ranks': mode_ranks,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mode_with_min_rank': np.argmin(mode_ranks),
        'mode_with_max_rank': np.argmax(mode_ranks)
    }

# Exemple
T = np.random.randn(10, 12, 8)
analysis = tensor_rank_analysis(T)
print("Analyse du rang tensoriel:")
print(f"  Rangs des modes: {analysis['mode_ranks']}")
print(f"  Borne inférieure: {analysis['lower_bound']}")
print(f"  Borne supérieure: {analysis['upper_bound']}")
print(f"  Rang tensoriel ∈ [{analysis['lower_bound']}, {analysis['upper_bound']}]")
```

---

## Propriétés Fondamentales

### Symétrie

Un tenseur est **symétrique** si ses composantes sont invariantes sous permutation des indices :

```python
def check_symmetry(tensor):
    """
    Vérifie si un tenseur est symétrique
    """
    if tensor.ndim < 2:
        return True  # Scalaires et vecteurs sont trivialement symétriques
    
    # Pour un tenseur d'ordre 2 (matrice)
    if tensor.ndim == 2:
        return np.allclose(tensor, tensor.T)
    
    # Pour les tenseurs d'ordre supérieur
    # Vérifie toutes les permutations possibles
    n = tensor.shape[0]
    if not all(s == n for s in tensor.shape):
        return False  # Doit être cubique pour être symétrique
    
    # Teste quelques permutations
    for perm in [(1, 0), (2, 1, 0)] if tensor.ndim >= 3 else [(1, 0)]:
        if tensor.ndim >= len(perm):
            permuted = np.transpose(tensor, perm)
            if not np.allclose(tensor, permuted):
                return False
    
    return True

# Test
symmetric_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
print(f"Matrice symétrique: {check_symmetry(symmetric_matrix)}")

asymmetric_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrice asymétrique: {check_symmetry(asymmetric_matrix)}")
```

### Décomposabilité

Un tenseur est **séparable** (décomposable) s'il peut s'écrire comme un produit tensoriel :

```python
def is_separable(tensor):
    """
    Vérifie si un tenseur est séparable (rang 1)
    """
    # Pour un tenseur d'ordre 2, c'est une matrice de rang 1
    if tensor.ndim == 2:
        rank = np.linalg.matrix_rank(tensor)
        return rank == 1
    
    # Pour les tenseurs d'ordre supérieur, vérifie le rang tensoriel
    # Approximation: vérifie si toutes les matricisations sont de rang 1
    for mode in range(tensor.ndim):
        shape = tensor.shape
        dim_mode = shape[mode]
        other_dims = np.prod([shape[i] for i in range(len(shape)) if i != mode])
        
        tensor_reshaped = np.moveaxis(tensor, mode, 0)
        matrix = tensor_reshaped.reshape(dim_mode, other_dims)
        
        if np.linalg.matrix_rank(matrix) > 1:
            return False
    
    return True

# Test
# Tenseur séparable (produit extérieur)
u = np.array([1, 2, 3])
v = np.array([4, 5])
T_sep = np.outer(u, v)
print(f"Tenseur séparable: {is_separable(T_sep)}")

# Tenseur non-séparable
T_nonsep = np.random.randn(3, 4)
print(f"Tenseur non-séparable: {is_separable(T_nonsep)}")
```

---

## Opérations de Base

### Addition et Multiplication

```python
def tensor_operations():
    """
    Opérations élémentaires sur les tenseurs
    """
    T1 = np.random.randn(3, 4, 5)
    T2 = np.random.randn(3, 4, 5)
    
    # Addition élément par élément
    T_sum = T1 + T2
    print(f"Addition: {T_sum.shape}")
    
    # Multiplication élément par élément (Hadamard)
    T_hadamard = T1 * T2
    print(f"Produit de Hadamard: {T_hadamard.shape}")
    
    # Multiplication par scalaire
    T_scaled = 2.5 * T1
    print(f"Multiplication par scalaire: {T_scaled.shape}")
    
    # Normes
    frobenius = np.linalg.norm(T1, 'fro')
    l1_norm = np.abs(T1).sum()
    l2_norm = np.linalg.norm(T1.flatten())
    
    print(f"\nNormes:")
    print(f"  Frobenius: {frobenius:.4f}")
    print(f"  L1: {l1_norm:.4f}")
    print(f"  L2: {l2_norm:.4f}")

tensor_operations()
```

### Produit Tensoriel (Outer Product)

```python
def outer_product(*vectors):
    """
    Produit tensoriel de plusieurs vecteurs
    
    Le résultat a un ordre égal à la somme des ordres
    """
    result = vectors[0]
    for v in vectors[1:]:
        result = np.tensordot(result, v, axes=0)
    return result

# Exemple
u = np.array([1, 2])
v = np.array([3, 4, 5])
w = np.array([6, 7])

T = outer_product(u, v, w)
print(f"u ⊗ v ⊗ w: shape = {T.shape}, order = {T.ndim}")
print(f"  u: shape {u.shape}")
print(f"  v: shape {v.shape}")
print(f"  w: shape {w.shape}")
print(f"  Résultat: shape {T.shape}")

# Vérification: T[i, j, k] = u[i] * v[j] * w[k]
print(f"\nVérification:")
print(f"  T[0, 0, 0] = {T[0, 0, 0]}, u[0]*v[0]*w[0] = {u[0]*v[0]*w[0]}")
```

---

## Applications en Machine Learning

### Tenseurs dans les Réseaux de Neurones

```python
class NeuralNetworkTensors:
    """
    Exemples de tenseurs dans les réseaux de neurones
    """
    
    @staticmethod
    def weight_tensors():
        """Tenseurs de poids dans différents types de couches"""
        
        # Couche Fully-Connected (Linear)
        # W: (out_features, in_features) - ordre 2
        W_fc = np.random.randn(256, 784)
        print(f"FC Layer weight: order {W_fc.ndim}, shape {W_fc.shape}")
        
        # Couche Convolutionnelle
        # W: (out_channels, in_channels, kernel_h, kernel_w) - ordre 4
        W_conv = np.random.randn(64, 32, 3, 3)
        print(f"Conv Layer weight: order {W_conv.ndim}, shape {W_conv.shape}")
        
        # Batch de données
        # Input: (batch_size, channels, height, width) - ordre 4
        X = np.random.randn(32, 3, 224, 224)
        print(f"Batch input: order {X.ndim}, shape {X.shape}")
        
        # Attention (Transformer)
        # Attention weights: (batch, heads, seq_len, seq_len) - ordre 4
        A = np.random.randn(8, 12, 512, 512)
        print(f"Attention weights: order {A.ndim}, shape {A.shape}")
    
    @staticmethod
    def parameter_count_example():
        """Compte les paramètres en fonction de la représentation"""
        
        # Représentation dense
        W_dense = np.random.randn(1024, 512)
        params_dense = W_dense.size
        print(f"Représentation dense: {params_dense:,} paramètres")
        
        # Représentation factorisée (low-rank)
        U = np.random.randn(1024, 64)
        V = np.random.randn(64, 512)
        W_factorized = U @ V
        params_factorized = U.size + V.size
        print(f"Représentation factorisée: {params_factorized:,} paramètres")
        print(f"  Compression: {params_dense / params_factorized:.2f}x")
        print(f"  Erreur relative: {np.linalg.norm(W_dense - W_factorized) / np.linalg.norm(W_dense):.4f}")

NeuralNetworkTensors.weight_tensors()
print()
NeuralNetworkTensors.parameter_count_example()
```

---

## Exercices

### Exercice 4.1.1
Créez un tenseur d'ordre 5 de shape (2, 3, 4, 5, 6). Calculez toutes ses matricisations et leurs rangs.

### Exercice 4.1.2
Implémentez une fonction qui vérifie si un tenseur d'ordre 3 est symétrique par rapport à toutes ses permutations.

### Exercice 4.1.3
Créez un tenseur séparable (rang 1) et vérifiez que toutes ses matricisations sont de rang 1.

---

## Points Clés à Retenir

> 📌 **Un tenseur d'ordre d est un tableau à d dimensions**

> 📌 **Le rang tensoriel est plus complexe que le rang matriciel**

> 📌 **La notation d'Einstein simplifie les expressions avec indices répétés**

> 📌 **Les tenseurs dans les réseaux de neurones peuvent être très grands**

---

*Section suivante : [4.2 Contraction de Tenseurs](./04_02_Contraction.md)*

