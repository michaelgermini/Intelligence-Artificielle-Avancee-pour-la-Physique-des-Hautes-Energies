# 2.4 Produits Tensoriels et Espaces de Hilbert

---

## Introduction

Les **produits tensoriels** généralisent le concept de produit extérieur aux espaces de dimension arbitraire. Ils sont fondamentaux pour comprendre les réseaux de tenseurs et leur connexion avec la mécanique quantique.

---

## Produit Tensoriel d'Espaces Vectoriels

### Définition

Le **produit tensoriel** de deux espaces vectoriels $V$ et $W$ est un espace vectoriel $V \otimes W$ tel que :

1. Il existe une application bilinéaire $\otimes : V \times W \rightarrow V \otimes W$
2. Tout élément de $V \otimes W$ est une combinaison linéaire de produits $\mathbf{v} \otimes \mathbf{w}$

### Propriétés Fondamentales

```
┌─────────────────────────────────────────────────────────────────┐
│              Propriétés du Produit Tensoriel                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Bilinéarité:                                               │
│     (αv₁ + βv₂) ⊗ w = α(v₁ ⊗ w) + β(v₂ ⊗ w)                  │
│     v ⊗ (αw₁ + βw₂) = α(v ⊗ w₁) + β(v ⊗ w₂)                  │
│                                                                 │
│  2. Dimension:                                                  │
│     dim(V ⊗ W) = dim(V) × dim(W)                               │
│                                                                 │
│  3. Base:                                                       │
│     Si {eᵢ} base de V et {fⱼ} base de W,                       │
│     alors {eᵢ ⊗ fⱼ} est base de V ⊗ W                          │
│                                                                 │
│  4. Associativité:                                              │
│     (U ⊗ V) ⊗ W ≅ U ⊗ (V ⊗ W)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implémentation

```python
import numpy as np
from itertools import product

def tensor_product_vectors(v, w):
    """
    Produit tensoriel de deux vecteurs
    
    v ⊗ w est équivalent au produit extérieur (outer product)
    """
    return np.outer(v, w)

def tensor_product_matrices(A, B):
    """
    Produit de Kronecker (produit tensoriel de matrices)
    
    (A ⊗ B)_{(i,k),(j,l)} = A_{i,j} × B_{k,l}
    """
    return np.kron(A, B)

def tensor_product_general(*arrays):
    """
    Produit tensoriel de plusieurs tableaux
    
    Résultat: tenseur de rang sum(ranks)
    """
    result = arrays[0]
    for arr in arrays[1:]:
        # Ajoute de nouvelles dimensions
        result = np.tensordot(result, arr, axes=0)
    return result

# Exemples
v = np.array([1, 2])
w = np.array([3, 4, 5])

print("Produit tensoriel de vecteurs:")
print(f"v = {v}, dim = {len(v)}")
print(f"w = {w}, dim = {len(w)}")
print(f"v ⊗ w shape: {tensor_product_vectors(v, w).shape}")
print(tensor_product_vectors(v, w))

# Produit de Kronecker
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])

print("\nProduit de Kronecker:")
print(f"A shape: {A.shape}, B shape: {B.shape}")
print(f"A ⊗ B shape: {tensor_product_matrices(A, B).shape}")
print(tensor_product_matrices(A, B))
```

---

## Tenseurs et Notation Indicielle

### Définition d'un Tenseur

Un **tenseur d'ordre n** sur un espace vectoriel $V$ est un élément de :

$$\underbrace{V \otimes V \otimes \cdots \otimes V}_{n \text{ fois}}$$

```python
class Tensor:
    """
    Classe représentant un tenseur avec notation indicielle
    """
    
    def __init__(self, data, index_names=None):
        self.data = np.array(data)
        self.order = len(self.data.shape)
        self.shape = self.data.shape
        
        if index_names is None:
            index_names = [f'i{k}' for k in range(self.order)]
        self.indices = index_names
        
    def __repr__(self):
        return f"Tensor(order={self.order}, shape={self.shape}, indices={self.indices})"
    
    def __getitem__(self, indices):
        return self.data[indices]
    
    def contract(self, other, self_idx, other_idx):
        """
        Contraction de deux tenseurs sur des indices spécifiés
        
        T^{i,j,k} × S^{k,l,m} → R^{i,j,l,m} (contraction sur k)
        """
        # Trouve les positions des indices
        self_pos = self.indices.index(self_idx)
        other_pos = other.indices.index(other_idx)
        
        # Contraction via tensordot
        result_data = np.tensordot(self.data, other.data, axes=(self_pos, other_pos))
        
        # Nouveaux indices
        new_indices = (self.indices[:self_pos] + self.indices[self_pos+1:] +
                      other.indices[:other_pos] + other.indices[other_pos+1:])
        
        return Tensor(result_data, new_indices)
    
    def reshape_to_matrix(self, left_indices, right_indices):
        """
        Reshape le tenseur en matrice en groupant les indices
        """
        # Calcule les dimensions
        left_dims = [self.shape[self.indices.index(idx)] for idx in left_indices]
        right_dims = [self.shape[self.indices.index(idx)] for idx in right_indices]
        
        # Réarrange les axes
        left_axes = [self.indices.index(idx) for idx in left_indices]
        right_axes = [self.indices.index(idx) for idx in right_indices]
        
        permuted = np.transpose(self.data, left_axes + right_axes)
        
        return permuted.reshape(np.prod(left_dims), np.prod(right_dims))

# Exemple
T = Tensor(np.random.randn(3, 4, 5), ['i', 'j', 'k'])
S = Tensor(np.random.randn(5, 6), ['k', 'l'])

print(f"T: {T}")
print(f"S: {S}")

# Contraction sur l'indice k
R = T.contract(S, 'k', 'k')
print(f"T contracted with S on k: {R}")
```

### Convention de Sommation d'Einstein

```python
def einsum_examples():
    """
    Exemples de la notation Einstein avec np.einsum
    """
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C = np.random.randn(3, 4, 5)
    v = np.random.randn(4)
    
    # Produit matriciel: C_ik = A_ij B_jk
    result1 = np.einsum('ij,jk->ik', A, B)
    print(f"Produit matriciel: {result1.shape}")
    
    # Trace: tr(A) = A_ii
    A_square = np.random.randn(4, 4)
    trace = np.einsum('ii->', A_square)
    print(f"Trace: {trace:.4f} (vérif: {np.trace(A_square):.4f})")
    
    # Produit extérieur: T_ij = u_i v_j
    u = np.random.randn(3)
    outer = np.einsum('i,j->ij', u, v)
    print(f"Produit extérieur: {outer.shape}")
    
    # Contraction de tenseur: R_il = C_ijk A_jk... non, plutôt
    # Somme sur tous les indices: s = C_ijk
    total = np.einsum('ijk->', C)
    print(f"Somme totale: {total:.4f}")
    
    # Produit de Hadamard: C_ij = A_ij * B_ij (si même taille)
    A2 = np.random.randn(3, 4)
    B2 = np.random.randn(3, 4)
    hadamard = np.einsum('ij,ij->ij', A2, B2)
    print(f"Hadamard: {hadamard.shape}")
    
    # Batch matrix multiplication
    batch_A = np.random.randn(10, 3, 4)
    batch_B = np.random.randn(10, 4, 5)
    batch_result = np.einsum('bij,bjk->bik', batch_A, batch_B)
    print(f"Batch matmul: {batch_result.shape}")

einsum_examples()
```

---

## Espaces de Hilbert

### Définition

Un **espace de Hilbert** est un espace vectoriel $\mathcal{H}$ muni d'un produit scalaire $\langle \cdot, \cdot \rangle$ qui est complet pour la norme induite.

### Produit Scalaire et Norme

```python
class HilbertSpace:
    """
    Représentation d'un espace de Hilbert de dimension finie
    """
    
    def __init__(self, dimension, field='real'):
        self.dim = dimension
        self.field = field  # 'real' ou 'complex'
        
    def inner_product(self, u, v):
        """
        Produit scalaire standard
        <u, v> = u† · v (conjugué pour complexe)
        """
        if self.field == 'complex':
            return np.vdot(u, v)  # Conjugué automatique
        return np.dot(u, v)
    
    def norm(self, v):
        """Norme induite par le produit scalaire"""
        return np.sqrt(np.real(self.inner_product(v, v)))
    
    def distance(self, u, v):
        """Distance entre deux vecteurs"""
        return self.norm(u - v)
    
    def is_orthogonal(self, u, v, tol=1e-10):
        """Vérifie l'orthogonalité"""
        return np.abs(self.inner_product(u, v)) < tol
    
    def projection(self, v, subspace_basis):
        """
        Projection orthogonale sur un sous-espace
        """
        proj = np.zeros_like(v)
        for basis_vec in subspace_basis:
            coef = self.inner_product(basis_vec, v) / self.inner_product(basis_vec, basis_vec)
            proj = proj + coef * basis_vec
        return proj
    
    def tensor_product(self, other):
        """Produit tensoriel d'espaces de Hilbert"""
        return HilbertSpace(self.dim * other.dim, self.field)

# Exemple
H = HilbertSpace(3, 'real')
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([1, 1, 1]) / np.sqrt(3)

print(f"<u, v> = {H.inner_product(u, v)}")
print(f"||w|| = {H.norm(w):.4f}")
print(f"u ⊥ v ? {H.is_orthogonal(u, v)}")

# Projection
arbitrary = np.array([1, 2, 3])
proj_xy = H.projection(arbitrary, [u, v])
print(f"Projection de {arbitrary} sur le plan xy: {proj_xy}")
```

### Opérateurs sur les Espaces de Hilbert

```python
class LinearOperator:
    """
    Opérateur linéaire sur un espace de Hilbert
    """
    
    def __init__(self, matrix, hilbert_space):
        self.matrix = np.array(matrix)
        self.H = hilbert_space
        
    def __call__(self, v):
        """Applique l'opérateur"""
        return self.matrix @ v
    
    def adjoint(self):
        """
        Opérateur adjoint A†
        <Au, v> = <u, A†v>
        """
        if self.H.field == 'complex':
            return LinearOperator(self.matrix.conj().T, self.H)
        return LinearOperator(self.matrix.T, self.H)
    
    def is_hermitian(self, tol=1e-10):
        """Vérifie si A = A†"""
        adj = self.adjoint().matrix
        return np.allclose(self.matrix, adj, atol=tol)
    
    def is_unitary(self, tol=1e-10):
        """Vérifie si A†A = AA† = I"""
        adj = self.adjoint().matrix
        return (np.allclose(adj @ self.matrix, np.eye(len(self.matrix)), atol=tol) and
                np.allclose(self.matrix @ adj, np.eye(len(self.matrix)), atol=tol))
    
    def is_positive(self):
        """Vérifie si A est semi-définie positive"""
        if not self.is_hermitian():
            return False
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return np.all(eigenvalues >= -1e-10)
    
    def spectral_decomposition(self):
        """
        Décomposition spectrale pour opérateurs hermitiens
        A = Σ λᵢ |vᵢ⟩⟨vᵢ|
        """
        if not self.is_hermitian():
            raise ValueError("L'opérateur doit être hermitien")
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        return eigenvalues, eigenvectors

# Exemple avec un opérateur hermitien
H_space = HilbertSpace(3, 'complex')
A_matrix = np.array([
    [2, 1-1j, 0],
    [1+1j, 3, 1],
    [0, 1, 1]
])

A = LinearOperator(A_matrix, H_space)
print(f"A est hermitien: {A.is_hermitian()}")
print(f"A est positif: {A.is_positive()}")

eigenvals, eigenvecs = A.spectral_decomposition()
print(f"Valeurs propres: {eigenvals}")
```

---

## Produit Tensoriel d'Espaces de Hilbert

### Construction

Le produit tensoriel $\mathcal{H}_1 \otimes \mathcal{H}_2$ de deux espaces de Hilbert est un espace de Hilbert avec le produit scalaire :

$$\langle u_1 \otimes u_2, v_1 \otimes v_2 \rangle = \langle u_1, v_1 \rangle_1 \cdot \langle u_2, v_2 \rangle_2$$

```python
class TensorProductHilbertSpace:
    """
    Produit tensoriel d'espaces de Hilbert
    """
    
    def __init__(self, *spaces):
        self.spaces = spaces
        self.dims = [s.dim for s in spaces]
        self.total_dim = np.prod(self.dims)
        
    def tensor_product_state(self, *states):
        """
        Crée un état produit tensoriel |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ...
        """
        result = states[0]
        for state in states[1:]:
            result = np.kron(result, state)
        return result
    
    def inner_product(self, psi, phi):
        """Produit scalaire dans l'espace produit"""
        return np.vdot(psi, phi)
    
    def partial_trace(self, rho, keep_indices):
        """
        Trace partielle d'une matrice densité
        
        rho: matrice densité sur l'espace total
        keep_indices: indices des sous-espaces à garder
        """
        n_spaces = len(self.dims)
        trace_indices = [i for i in range(n_spaces) if i not in keep_indices]
        
        # Reshape en tenseur
        rho_tensor = rho.reshape(self.dims + self.dims)
        
        # Trace sur les indices spécifiés
        for idx in sorted(trace_indices, reverse=True):
            # Trace sur les indices idx et idx + n_spaces
            rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + n_spaces - len(trace_indices))
        
        # Reshape en matrice
        keep_dims = [self.dims[i] for i in keep_indices]
        return rho_tensor.reshape(np.prod(keep_dims), np.prod(keep_dims))
    
    def is_separable(self, state, tol=1e-10):
        """
        Vérifie si un état est séparable (produit tensoriel)
        Pour 2 sous-espaces seulement
        """
        if len(self.spaces) != 2:
            raise NotImplementedError("Seulement pour 2 sous-espaces")
        
        # Reshape en matrice
        matrix = state.reshape(self.dims[0], self.dims[1])
        
        # Un état est séparable ssi la matrice est de rang 1
        rank = np.linalg.matrix_rank(matrix, tol=tol)
        return rank == 1

# Exemple : deux qubits
H_qubit = HilbertSpace(2, 'complex')
H_2qubits = TensorProductHilbertSpace(H_qubit, H_qubit)

# États de base
ket_0 = np.array([1, 0], dtype=complex)
ket_1 = np.array([0, 1], dtype=complex)

# État produit |00⟩
state_00 = H_2qubits.tensor_product_state(ket_0, ket_0)
print(f"|00⟩ = {state_00}")
print(f"Séparable: {H_2qubits.is_separable(state_00)}")

# État de Bell (intriqué) |Φ+⟩ = (|00⟩ + |11⟩)/√2
bell_state = (H_2qubits.tensor_product_state(ket_0, ket_0) + 
              H_2qubits.tensor_product_state(ket_1, ket_1)) / np.sqrt(2)
print(f"\n|Φ+⟩ = {bell_state}")
print(f"Séparable: {H_2qubits.is_separable(bell_state)}")
```

---

## Applications aux Réseaux de Tenseurs

### États MPS (Matrix Product States)

```python
def create_mps(local_dims, bond_dims):
    """
    Crée un MPS aléatoire
    
    |ψ⟩ = Σ A[1]^{i₁} A[2]^{i₂} ... A[n]^{iₙ} |i₁ i₂ ... iₙ⟩
    
    local_dims: dimensions locales (d₁, d₂, ..., dₙ)
    bond_dims: dimensions de liaison (χ₁, χ₂, ..., χₙ₋₁)
    """
    n_sites = len(local_dims)
    tensors = []
    
    for i in range(n_sites):
        # Dimensions: (bond_left, physical, bond_right)
        bond_left = 1 if i == 0 else bond_dims[i-1]
        bond_right = 1 if i == n_sites-1 else bond_dims[i]
        physical = local_dims[i]
        
        A = np.random.randn(bond_left, physical, bond_right)
        tensors.append(A)
    
    return tensors

def contract_mps(tensors):
    """
    Contracte un MPS pour obtenir le vecteur d'état complet
    """
    result = tensors[0]
    
    for A in tensors[1:]:
        # Contraction sur l'indice de liaison
        result = np.tensordot(result, A, axes=(-1, 0))
    
    # Squeeze les dimensions de liaison aux bords
    return result.squeeze()

def mps_norm(tensors):
    """Calcule la norme d'un MPS"""
    state = contract_mps(tensors)
    return np.linalg.norm(state.flatten())

# Exemple : 4 sites, dimension locale 2 (qubits)
local_dims = [2, 2, 2, 2]
bond_dims = [2, 4, 2]  # Dimensions de liaison

mps = create_mps(local_dims, bond_dims)

print("Structure du MPS:")
for i, A in enumerate(mps):
    print(f"  Site {i}: shape = {A.shape}")

state = contract_mps(mps)
print(f"\nÉtat contracté: shape = {state.shape}")
print(f"Dimension totale: {np.prod(local_dims)}")
print(f"Norme: {mps_norm(mps):.4f}")

# Comparaison des paramètres
full_params = np.prod(local_dims)
mps_params = sum(A.size for A in mps)
print(f"\nParamètres état complet: {full_params}")
print(f"Paramètres MPS: {mps_params}")
print(f"Ratio: {full_params / mps_params:.2f}x")
```

---

## Exercices

### Exercice 2.4.1
Calculez le produit tensoriel des vecteurs $\mathbf{u} = (1, 2)$ et $\mathbf{v} = (3, 4, 5)$. Quelle est la dimension de l'espace résultant ?

### Exercice 2.4.2
Montrez que l'état de Bell $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$ ne peut pas s'écrire comme un produit tensoriel de deux états à un qubit.

### Exercice 2.4.3
Implémentez une fonction qui vérifie si une matrice est une matrice densité valide (hermitienne, semi-définie positive, trace 1).

---

## Points Clés à Retenir

> 📌 **Le produit tensoriel multiplie les dimensions des espaces**

> 📌 **La notation d'Einstein simplifie les contractions tensorielles**

> 📌 **Les états intriqués ne sont pas séparables (rang > 1)**

> 📌 **Les MPS représentent efficacement les états à faible intrication**

---

## Références

1. Nielsen, M., Chuang, I. "Quantum Computation and Quantum Information." Cambridge, 2010
2. Schollwöck, U. "The density-matrix renormalization group in the age of matrix product states." Ann. Phys. 326, 2011
3. Bridgeman, J., Chubb, C. "Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks." J. Phys. A, 2017

---

*Section suivante : [2.5 Normes Matricielles et Erreurs d'Approximation](./02_05_Normes.md)*

