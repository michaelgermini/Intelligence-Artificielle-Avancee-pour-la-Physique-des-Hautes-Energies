# Chapitre 4 : Introduction aux Réseaux de Tenseurs

---

## Introduction

Les **réseaux de tenseurs** sont des structures mathématiques puissantes qui généralisent les décompositions matricielles aux dimensions supérieures. Originellement développés en physique quantique pour représenter des états quantiques, ils trouvent aujourd'hui des applications majeures en machine learning pour la compression de modèles.

---

## Objectifs d'Apprentissage

À la fin de ce chapitre, vous serez capable de :

- Définir et manipuler des tenseurs de différents ordres
- Effectuer des contractions tensorielles
- Utiliser la notation diagrammatique
- Analyser la complexité des opérations tensorielles

---

## Plan du Chapitre

1. [Définition et Notation des Tenseurs](./04_01_Definition_Tenseurs.md)
2. [Contraction de Tenseurs](./04_02_Contraction.md)
3. [Diagrammes Tensoriels](./04_03_Diagrammes.md)
4. [Complexité Computationnelle](./04_04_Complexite.md)

---

## Qu'est-ce qu'un Tenseur ?

### Définition Intuitive

Un **tenseur** est une généralisation multidimensionnelle des scalaires, vecteurs et matrices :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hiérarchie des Tenseurs                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ordre 0 : Scalaire        a                    1 nombre       │
│  Ordre 1 : Vecteur         v_i                  n nombres      │
│  Ordre 2 : Matrice         M_ij                 n×m nombres    │
│  Ordre 3 : Tenseur 3D      T_ijk                n×m×p nombres  │
│  Ordre d : Tenseur général X_{i₁...i_d}         ∏ nₖ nombres   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Définition Formelle

Un tenseur d'ordre $d$ sur un espace vectoriel $V$ est un élément du produit tensoriel :

$$\mathcal{T} \in V_1 \otimes V_2 \otimes \cdots \otimes V_d$$

En coordonnées, c'est un tableau multidimensionnel $T_{i_1, i_2, \ldots, i_d}$ où chaque indice $i_k$ varie de 1 à $n_k$.

```python
import numpy as np
import torch

# Création de tenseurs
scalar = 5.0                           # Ordre 0
vector = np.array([1, 2, 3])           # Ordre 1, shape (3,)
matrix = np.array([[1, 2], [3, 4]])    # Ordre 2, shape (2, 2)
tensor_3d = np.random.randn(3, 4, 5)   # Ordre 3, shape (3, 4, 5)
tensor_4d = np.random.randn(2, 3, 4, 5) # Ordre 4, shape (2, 3, 4, 5)

print("Tenseurs créés:")
print(f"  Scalaire: ordre 0")
print(f"  Vecteur: ordre {vector.ndim}, shape {vector.shape}")
print(f"  Matrice: ordre {matrix.ndim}, shape {matrix.shape}")
print(f"  Tenseur 3D: ordre {tensor_3d.ndim}, shape {tensor_3d.shape}")
print(f"  Tenseur 4D: ordre {tensor_4d.ndim}, shape {tensor_4d.shape}")
```

---

## Terminologie

### Ordre, Rang et Mode

```python
class TensorInfo:
    """
    Analyse les propriétés d'un tenseur
    """
    
    def __init__(self, tensor):
        self.tensor = np.array(tensor)
        
    @property
    def order(self):
        """
        Ordre (ou degré) : nombre d'indices
        Aussi appelé "nombre de modes"
        """
        return self.tensor.ndim
    
    @property
    def shape(self):
        """Shape : dimensions le long de chaque mode"""
        return self.tensor.shape
    
    @property
    def size(self):
        """Nombre total d'éléments"""
        return self.tensor.size
    
    def mode_n_fibers(self, n):
        """
        Fibres du mode n : vecteurs obtenus en fixant tous les indices sauf un
        """
        # Déplace le mode n en première position
        tensor_transposed = np.moveaxis(self.tensor, n, 0)
        # Reshape pour avoir les fibres comme colonnes
        return tensor_transposed.reshape(self.shape[n], -1)
    
    def mode_n_unfolding(self, n):
        """
        Dépliage (matricisation) selon le mode n
        
        Le tenseur est réarrangé en matrice où:
        - Les lignes correspondent au mode n
        - Les colonnes correspondent à tous les autres modes
        """
        return self.mode_n_fibers(n)
    
    def analyze(self):
        """Analyse complète"""
        print(f"Ordre: {self.order}")
        print(f"Shape: {self.shape}")
        print(f"Taille totale: {self.size}")
        print(f"\nMatricisations:")
        for n in range(self.order):
            unfolding = self.mode_n_unfolding(n)
            print(f"  Mode-{n}: shape {unfolding.shape}, rang {np.linalg.matrix_rank(unfolding)}")

# Exemple
T = np.random.randn(4, 5, 6)
info = TensorInfo(T)
info.analyze()
```

### Rang Tensoriel

Le **rang tensoriel** est plus complexe que le rang matriciel :

```python
def tensor_rank_bounds(tensor):
    """
    Estime les bornes du rang tensoriel via les rangs des matricisations
    
    Le rang tensoriel r satisfait:
    max_n rank(T_(n)) ≤ r ≤ min_n (product of other dims)
    """
    info = TensorInfo(tensor)
    
    mode_ranks = []
    for n in range(info.order):
        unfolding = info.mode_n_unfolding(n)
        mode_ranks.append(np.linalg.matrix_rank(unfolding))
    
    lower_bound = max(mode_ranks)
    upper_bound = min(info.shape)
    
    return {
        'mode_ranks': mode_ranks,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

# Test
T = np.random.randn(4, 5, 6)
bounds = tensor_rank_bounds(T)
print(f"Rangs des modes: {bounds['mode_ranks']}")
print(f"Rang tensoriel ∈ [{bounds['lower_bound']}, {bounds['upper_bound']}]")
```

---

## Opérations de Base

### Produit Tensoriel (Outer Product)

```python
def tensor_outer_product(*tensors):
    """
    Produit tensoriel (outer product) de plusieurs tenseurs
    
    Le résultat a un ordre égal à la somme des ordres
    """
    result = tensors[0]
    for t in tensors[1:]:
        result = np.tensordot(result, t, axes=0)
    return result

# Exemple
u = np.array([1, 2])
v = np.array([3, 4, 5])
w = np.array([6, 7])

# u ⊗ v ⊗ w
outer = tensor_outer_product(u, v, w)
print(f"u shape: {u.shape}")
print(f"v shape: {v.shape}")
print(f"w shape: {w.shape}")
print(f"u ⊗ v ⊗ w shape: {outer.shape}")
print(f"Ordre résultant: {outer.ndim}")
```

### Produit de Hadamard (Element-wise)

```python
def hadamard_product(T1, T2):
    """
    Produit de Hadamard : multiplication élément par élément
    Les tenseurs doivent avoir la même shape
    """
    assert T1.shape == T2.shape, "Shapes incompatibles"
    return T1 * T2

# Exemple
T1 = np.random.randn(3, 4, 5)
T2 = np.random.randn(3, 4, 5)
H = hadamard_product(T1, T2)
print(f"Hadamard product shape: {H.shape}")
```

### Produit de Kronecker

```python
def kronecker_product(A, B):
    """
    Produit de Kronecker pour matrices
    (A ⊗ B)_{(i,k),(j,l)} = A_{i,j} × B_{k,l}
    """
    return np.kron(A, B)

# Exemple
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])
K = kronecker_product(A, B)
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"A ⊗ B shape: {K.shape}")
print(f"A ⊗ B:\n{K}")
```

---

## Contraction de Tenseurs

### Définition

La **contraction** somme sur des indices partagés :

$$C_{i,k} = \sum_j A_{i,j} \cdot B_{j,k}$$

C'est la généralisation du produit matriciel.

```python
def tensor_contraction(T1, T2, axes):
    """
    Contraction de deux tenseurs sur les axes spécifiés
    
    axes: tuple (axes_T1, axes_T2) des indices à contracter
    """
    return np.tensordot(T1, T2, axes=axes)

# Exemples de contractions
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Produit matriciel : contraction sur l'indice commun
C = tensor_contraction(A, B, axes=(1, 0))
print(f"Produit matriciel: {A.shape} × {B.shape} = {C.shape}")

# Contraction de tenseurs d'ordre supérieur
T1 = np.random.randn(2, 3, 4)
T2 = np.random.randn(4, 5, 6)

# Contraction sur le dernier axe de T1 et le premier de T2
C2 = tensor_contraction(T1, T2, axes=(2, 0))
print(f"Tenseur contraction: {T1.shape} contracted with {T2.shape} = {C2.shape}")
```

### Notation d'Einstein

```python
# La notation d'Einstein avec np.einsum
def einsum_examples():
    """
    Exemples de notation Einstein pour les contractions
    """
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    T = np.random.randn(2, 3, 4)
    
    # Produit matriciel
    C = np.einsum('ij,jk->ik', A, B)
    print(f"Produit matriciel: {C.shape}")
    
    # Trace
    M = np.random.randn(4, 4)
    trace = np.einsum('ii->', M)
    print(f"Trace: {trace:.4f}")
    
    # Produit extérieur
    u = np.random.randn(3)
    v = np.random.randn(4)
    outer = np.einsum('i,j->ij', u, v)
    print(f"Produit extérieur: {outer.shape}")
    
    # Contraction de tenseur
    S = np.random.randn(4, 5)
    result = np.einsum('ijk,kl->ijl', T, S)
    print(f"Contraction tensorielle: {result.shape}")
    
    # Norme de Frobenius
    frob_sq = np.einsum('ijk,ijk->', T, T)
    print(f"||T||_F² = {frob_sq:.4f}")

einsum_examples()
```

---

## Diagrammes Tensoriels

### Notation Graphique

Les diagrammes tensoriels représentent visuellement les tenseurs et leurs contractions :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Notation Diagrammatique                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Tenseur d'ordre n = nœud avec n "pattes" (indices libres)     │
│                                                                 │
│  Scalaire:     ○                  (0 patte)                    │
│                                                                 │
│  Vecteur:      ○───               (1 patte)                    │
│                                                                 │
│  Matrice:    ───○───              (2 pattes)                   │
│                                                                 │
│  Tenseur 3:    │                                               │
│              ──○──                (3 pattes)                   │
│                                                                 │
│  Contraction = pattes connectées (indices sommés)              │
│                                                                 │
│  Produit matriciel:  ───○───○───                               │
│                        A   B                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implémentation de Diagrammes

```python
class TensorDiagram:
    """
    Représentation d'un diagramme tensoriel
    """
    
    def __init__(self):
        self.tensors = {}  # name -> tensor
        self.edges = []    # (tensor1, idx1, tensor2, idx2) for contractions
        self.free_indices = {}  # name -> list of free index positions
        
    def add_tensor(self, name, tensor, free_idx=None):
        """Ajoute un tenseur au diagramme"""
        self.tensors[name] = tensor
        if free_idx is None:
            free_idx = list(range(tensor.ndim))
        self.free_indices[name] = free_idx
        
    def contract(self, name1, idx1, name2, idx2):
        """Spécifie une contraction entre deux tenseurs"""
        self.edges.append((name1, idx1, name2, idx2))
        
        # Met à jour les indices libres
        if idx1 in self.free_indices[name1]:
            self.free_indices[name1].remove(idx1)
        if idx2 in self.free_indices[name2]:
            self.free_indices[name2].remove(idx2)
    
    def evaluate(self):
        """
        Évalue le diagramme (contracte tous les tenseurs)
        Utilise np.einsum pour l'efficacité
        """
        # Construit la chaîne einsum
        # (Implémentation simplifiée)
        pass
    
    def complexity(self):
        """
        Estime la complexité de l'évaluation
        """
        total_ops = 1
        for name, tensor in self.tensors.items():
            total_ops *= tensor.size
        return total_ops

# Exemple : produit matriciel comme diagramme
diagram = TensorDiagram()
A = np.random.randn(100, 50)
B = np.random.randn(50, 80)

diagram.add_tensor('A', A)
diagram.add_tensor('B', B)
diagram.contract('A', 1, 'B', 0)

print(f"Tenseurs: {list(diagram.tensors.keys())}")
print(f"Contractions: {diagram.edges}")
```

---

## Complexité Computationnelle

### Coût des Opérations

```python
def contraction_cost(shape1, shape2, contracted_axes):
    """
    Calcule le coût (en FLOPs) d'une contraction tensorielle
    
    Coût = product(toutes les dimensions) × 2
    (multiplication + addition pour chaque élément)
    """
    all_dims = list(shape1) + [shape2[i] for i in range(len(shape2)) 
                                if i not in contracted_axes]
    
    # Chaque élément du résultat nécessite sum_size multiplications et additions
    contracted_size = np.prod([shape1[ax] for ax in contracted_axes])
    result_size = np.prod(all_dims) // (contracted_size ** 2)
    
    flops = 2 * result_size * contracted_size
    
    return flops

# Exemple
shape1 = (100, 50, 30)
shape2 = (30, 40)

cost = contraction_cost(shape1, shape2, [2])
print(f"Coût de contraction {shape1} avec {shape2}: {cost:,} FLOPs")
```

### Ordre Optimal de Contraction

L'ordre des contractions affecte drastiquement la complexité :

```python
def compare_contraction_orders():
    """
    Démontre l'importance de l'ordre de contraction
    """
    # Trois matrices à multiplier: A(10×100) × B(100×5) × C(5×50)
    A = np.random.randn(10, 100)
    B = np.random.randn(100, 5)
    C = np.random.randn(5, 50)
    
    # Ordre 1: (A × B) × C
    # Coût AB: 10 × 100 × 5 = 5,000
    # Coût (AB)C: 10 × 5 × 50 = 2,500
    # Total: 7,500
    cost_1 = 10 * 100 * 5 + 10 * 5 * 50
    
    # Ordre 2: A × (B × C)
    # Coût BC: 100 × 5 × 50 = 25,000
    # Coût A(BC): 10 × 100 × 50 = 50,000
    # Total: 75,000
    cost_2 = 100 * 5 * 50 + 10 * 100 * 50
    
    print("Comparaison des ordres de contraction:")
    print(f"  (A × B) × C: {cost_1:,} FLOPs")
    print(f"  A × (B × C): {cost_2:,} FLOPs")
    print(f"  Ratio: {cost_2 / cost_1:.1f}x")
    
    # Vérification que le résultat est le même
    result_1 = (A @ B) @ C
    result_2 = A @ (B @ C)
    print(f"  Résultats identiques: {np.allclose(result_1, result_2)}")

compare_contraction_orders()
```

---

## Applications aux Réseaux de Neurones

### Tenseurs dans le Deep Learning

```python
class TensorInNN:
    """
    Exemples de tenseurs dans les réseaux de neurones
    """
    
    @staticmethod
    def conv_kernel():
        """Noyau de convolution : tenseur d'ordre 4"""
        # (out_channels, in_channels, height, width)
        kernel = torch.randn(64, 32, 3, 3)
        print(f"Conv kernel: ordre {kernel.dim()}, shape {tuple(kernel.shape)}")
        print(f"  Paramètres: {kernel.numel():,}")
        return kernel
    
    @staticmethod
    def attention_tensor():
        """Tenseur d'attention : ordre 4"""
        # (batch, heads, seq_len, seq_len)
        batch, heads, seq = 32, 8, 512
        attention = torch.randn(batch, heads, seq, seq)
        print(f"Attention: ordre {attention.dim()}, shape {tuple(attention.shape)}")
        print(f"  Éléments: {attention.numel():,}")
        return attention
    
    @staticmethod
    def batch_activations():
        """Activations avec batch : ordre variable"""
        # FC: (batch, features)
        fc_act = torch.randn(32, 256)
        print(f"FC activations: ordre {fc_act.dim()}")
        
        # Conv: (batch, channels, height, width)
        conv_act = torch.randn(32, 64, 28, 28)
        print(f"Conv activations: ordre {conv_act.dim()}")
        
        # Transformer: (batch, seq, features)
        trans_act = torch.randn(32, 512, 768)
        print(f"Transformer activations: ordre {trans_act.dim()}")

TensorInNN.conv_kernel()
TensorInNN.attention_tensor()
TensorInNN.batch_activations()
```

---

## Exercices

### Exercice 4.1
Créez un tenseur d'ordre 5 et calculez toutes ses matricisations. Vérifiez que le rang de chaque matricisation est cohérent.

### Exercice 4.2
Implémentez une fonction qui trouve l'ordre optimal de contraction pour une chaîne de matrices (algorithme de programmation dynamique).

### Exercice 4.3
Représentez le produit matriciel par blocs comme un diagramme tensoriel.

---

## Points Clés à Retenir

> 📌 **Un tenseur d'ordre d a d indices et vit dans un espace de dimension ∏nᵢ**

> 📌 **La contraction généralise le produit matriciel aux dimensions supérieures**

> 📌 **L'ordre de contraction affecte exponentiellement la complexité**

> 📌 **Les diagrammes tensoriels visualisent intuitivement les opérations**

---

## Références

1. Kolda, T., Bader, B. "Tensor Decompositions and Applications." SIAM Review, 2009
2. Bridgeman, J., Chubb, C. "Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks." J. Phys. A, 2017
3. Cichocki, A. et al. "Tensor Networks for Dimensionality Reduction and Large-scale Optimization." Found. Trends ML, 2016

---

*Section suivante : [4.1 Définition et Notation des Tenseurs](./04_01_Definition_Tenseurs.md)*

