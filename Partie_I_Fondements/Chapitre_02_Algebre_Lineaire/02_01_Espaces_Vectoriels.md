# 2.1 Espaces Vectoriels et Transformations Linéaires

---

## Espaces Vectoriels

### Définition Formelle

Un **espace vectoriel** $V$ sur un corps $\mathbb{K}$ (typiquement $\mathbb{R}$ ou $\mathbb{C}$) est un ensemble muni de deux opérations :

1. **Addition** : $+ : V \times V \rightarrow V$
2. **Multiplication scalaire** : $\cdot : \mathbb{K} \times V \rightarrow V$

satisfaisant les axiomes suivants pour tous $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ et $\alpha, \beta \in \mathbb{K}$ :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Axiomes d'Espace Vectoriel                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Commutativité :     u + v = v + u                          │
│  2. Associativité :     (u + v) + w = u + (v + w)              │
│  3. Élément neutre :    ∃ 0 : u + 0 = u                        │
│  4. Inverse additif :   ∃ (-u) : u + (-u) = 0                  │
│  5. Compatibilité :     α(βu) = (αβ)u                          │
│  6. Élément unité :     1·u = u                                │
│  7. Distributivité :    α(u + v) = αu + αv                     │
│  8. Distributivité :    (α + β)u = αu + βu                     │
└─────────────────────────────────────────────────────────────────┘
```

### Exemples Importants

```python
import numpy as np

# 1. R^n : espace des n-uplets de réels
R3 = np.array([1.0, 2.0, 3.0])  # Vecteur dans R³

# 2. Matrices m×n : espace des matrices
M_2x3 = np.array([[1, 2, 3],
                   [4, 5, 6]])  # Matrice dans R^(2×3)

# 3. Polynômes de degré ≤ n
# P(x) = a₀ + a₁x + a₂x² représenté par [a₀, a₁, a₂]
poly_coeffs = np.array([1, -2, 1])  # 1 - 2x + x²

# 4. Fonctions continues sur [a,b] (espace de dimension infinie)
# Représentation discrète
x = np.linspace(0, 1, 100)
f = np.sin(2 * np.pi * x)  # Approximation discrète
```

### Dimension et Base

La **dimension** d'un espace vectoriel est le nombre de vecteurs dans une base :

```python
def find_basis(vectors):
    """
    Trouve une base à partir d'un ensemble de vecteurs
    en utilisant la décomposition QR
    """
    V = np.array(vectors).T  # Colonnes = vecteurs
    
    # Décomposition QR
    Q, R = np.linalg.qr(V)
    
    # Rang = nombre de pivots non nuls
    tol = 1e-10
    rank = np.sum(np.abs(np.diag(R)) > tol)
    
    # Base = premières colonnes de Q
    basis = Q[:, :rank]
    
    return basis, rank

# Exemple : 3 vecteurs dans R³
vectors = [
    [1, 0, 0],
    [1, 1, 0],
    [2, 1, 0]  # Combinaison linéaire des deux premiers
]

basis, dim = find_basis(vectors)
print(f"Dimension: {dim}")
print(f"Base:\n{basis}")
```

---

## Sous-Espaces Vectoriels

### Définition

Un sous-ensemble $W \subseteq V$ est un **sous-espace** si :
1. $\mathbf{0} \in W$
2. $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
3. $\alpha \in \mathbb{K}, \mathbf{u} \in W \Rightarrow \alpha\mathbf{u} \in W$

### Sous-Espaces Importants

```python
def column_space(A):
    """
    Espace colonne (image) de A
    Im(A) = {Ax : x ∈ R^n}
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rank = np.sum(S > 1e-10)
    return U[:, :rank]

def null_space(A):
    """
    Noyau de A
    Ker(A) = {x : Ax = 0}
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(S > 1e-10)
    return Vt[rank:, :].T

def row_space(A):
    """
    Espace ligne de A
    """
    return column_space(A.T)

# Démonstration
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrice A:")
print(A)
print(f"\nRang de A: {np.linalg.matrix_rank(A)}")

col_space = column_space(A)
print(f"\nDimension de l'espace colonne: {col_space.shape[1]}")

ker = null_space(A)
print(f"Dimension du noyau: {ker.shape[1]}")

# Vérification du théorème du rang
# dim(Im(A)) + dim(Ker(A)) = n (nombre de colonnes)
print(f"\nVérification: {col_space.shape[1]} + {ker.shape[1]} = {A.shape[1]}")
```

---

## Transformations Linéaires

### Définition

Une fonction $T: V \rightarrow W$ est une **transformation linéaire** si :

$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

pour tous $\mathbf{u}, \mathbf{v} \in V$ et $\alpha, \beta \in \mathbb{K}$.

### Représentation Matricielle

Toute transformation linéaire entre espaces de dimension finie peut être représentée par une matrice :

```python
class LinearTransformation:
    """
    Représentation d'une transformation linéaire
    """
    
    def __init__(self, matrix):
        self.A = np.array(matrix, dtype=float)
        self.input_dim = self.A.shape[1]
        self.output_dim = self.A.shape[0]
        
    def __call__(self, v):
        """Applique la transformation"""
        v = np.array(v)
        return self.A @ v
    
    def compose(self, other):
        """Composition de transformations : (T ∘ S)(v) = T(S(v))"""
        return LinearTransformation(self.A @ other.A)
    
    def kernel(self):
        """Calcule le noyau"""
        return null_space(self.A)
    
    def image(self):
        """Calcule l'image"""
        return column_space(self.A)
    
    def rank(self):
        """Rang de la transformation"""
        return np.linalg.matrix_rank(self.A)
    
    def is_injective(self):
        """Vérifie si T est injective (noyau trivial)"""
        return self.rank() == self.input_dim
    
    def is_surjective(self):
        """Vérifie si T est surjective (image = codomaine)"""
        return self.rank() == self.output_dim
    
    def is_bijective(self):
        """Vérifie si T est bijective (isomorphisme)"""
        return self.is_injective() and self.is_surjective()

# Exemples de transformations
# Rotation de 45° dans R²
theta = np.pi / 4
rotation = LinearTransformation([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Projection sur l'axe x
projection_x = LinearTransformation([
    [1, 0],
    [0, 0]
])

# Application
v = [1, 1]
print(f"Vecteur original: {v}")
print(f"Après rotation: {rotation(v)}")
print(f"Après projection: {projection_x(v)}")
```

### Propriétés des Transformations

```python
def analyze_transformation(T):
    """Analyse complète d'une transformation linéaire"""
    A = T.A
    
    analysis = {
        'dimensions': f'{T.input_dim} → {T.output_dim}',
        'rank': T.rank(),
        'nullity': T.input_dim - T.rank(),
        'is_injective': T.is_injective(),
        'is_surjective': T.is_surjective(),
        'is_bijective': T.is_bijective(),
    }
    
    # Valeurs propres (si matrice carrée)
    if T.input_dim == T.output_dim:
        eigenvalues = np.linalg.eigvals(A)
        analysis['eigenvalues'] = eigenvalues
        analysis['determinant'] = np.linalg.det(A)
        analysis['trace'] = np.trace(A)
    
    return analysis

# Analyse de la rotation
print("Analyse de la rotation:")
for key, value in analyze_transformation(rotation).items():
    print(f"  {key}: {value}")
```

---

## Changement de Base

### Matrice de Passage

Si $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ et $\mathcal{B}' = \{\mathbf{b}'_1, \ldots, \mathbf{b}'_n\}$ sont deux bases, la **matrice de passage** $P$ satisfait :

$$[\mathbf{v}]_{\mathcal{B}'} = P^{-1} [\mathbf{v}]_{\mathcal{B}}$$

```python
def change_of_basis_matrix(old_basis, new_basis):
    """
    Calcule la matrice de passage de old_basis vers new_basis
    
    Si v = old_basis @ [v]_old, alors [v]_new = P @ [v]_old
    où P = new_basis^(-1) @ old_basis
    """
    old_B = np.array(old_basis).T  # Colonnes = vecteurs de base
    new_B = np.array(new_basis).T
    
    # P = new_B^(-1) @ old_B
    P = np.linalg.solve(new_B, old_B)
    
    return P

# Exemple : base canonique vers base tournée
canonical_basis = [[1, 0], [0, 1]]
rotated_basis = [[np.cos(np.pi/4), np.sin(np.pi/4)],
                 [-np.sin(np.pi/4), np.cos(np.pi/4)]]

P = change_of_basis_matrix(canonical_basis, rotated_basis)
print("Matrice de passage:")
print(P)

# Vérification
v_canonical = np.array([1, 0])
v_rotated = P @ v_canonical
print(f"\n[1,0] en base canonique = {v_rotated} en base tournée")
```

### Similitude de Matrices

Deux matrices $A$ et $B$ sont **similaires** s'il existe $P$ inversible tel que :

$$B = P^{-1} A P$$

```python
def are_similar(A, B, tol=1e-10):
    """
    Vérifie si deux matrices sont similaires
    (mêmes valeurs propres avec mêmes multiplicités)
    """
    eig_A = np.sort(np.linalg.eigvals(A))
    eig_B = np.sort(np.linalg.eigvals(B))
    
    return np.allclose(eig_A, eig_B, atol=tol)

def diagonalize(A):
    """
    Diagonalise une matrice si possible
    A = P D P^(-1)
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    P = eigenvectors
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    
    # Vérification
    A_reconstructed = P @ D @ P_inv
    
    return {
        'P': P,
        'D': D,
        'P_inv': P_inv,
        'eigenvalues': eigenvalues,
        'reconstruction_error': np.linalg.norm(A - A_reconstructed)
    }

# Exemple
A = np.array([[4, -2], [1, 1]])
result = diagonalize(A)

print("Matrice A:")
print(A)
print(f"\nValeurs propres: {result['eigenvalues']}")
print(f"\nMatrice diagonale D:")
print(result['D'])
print(f"\nErreur de reconstruction: {result['reconstruction_error']:.2e}")
```

---

## Applications en Deep Learning

### Couches Linéaires

```python
import torch
import torch.nn as nn

class AnalyzableLinear(nn.Module):
    """
    Couche linéaire avec outils d'analyse
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x):
        return self.linear(x)
    
    def weight_matrix(self):
        return self.linear.weight.detach().numpy()
    
    def analyze_weight_space(self):
        """Analyse l'espace des poids"""
        W = self.weight_matrix()
        
        return {
            'shape': W.shape,
            'rank': np.linalg.matrix_rank(W),
            'condition_number': np.linalg.cond(W),
            'frobenius_norm': np.linalg.norm(W, 'fro'),
            'spectral_norm': np.linalg.norm(W, 2)
        }
    
    def effective_dimension(self, threshold=0.99):
        """
        Dimension effective basée sur l'énergie des valeurs singulières
        """
        W = self.weight_matrix()
        _, S, _ = np.linalg.svd(W)
        
        # Énergie cumulative
        energy = np.cumsum(S**2) / np.sum(S**2)
        
        # Nombre de composantes pour atteindre le seuil
        return np.searchsorted(energy, threshold) + 1

# Démonstration
layer = AnalyzableLinear(512, 256)

# Initialisation avec structure de rang faible
with torch.no_grad():
    # W = UV^T où U ∈ R^(256×32), V ∈ R^(512×32)
    U = torch.randn(256, 32)
    V = torch.randn(512, 32)
    layer.linear.weight = nn.Parameter(U @ V.T)

analysis = layer.analyze_weight_space()
print("Analyse de la couche:")
for key, value in analysis.items():
    print(f"  {key}: {value}")

print(f"\nDimension effective (99% énergie): {layer.effective_dimension()}")
```

### Réduction de Dimensionnalité

```python
class PCALayer(nn.Module):
    """
    Couche de réduction de dimensionnalité par PCA
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Paramètres appris lors du fit
        self.register_buffer('mean', torch.zeros(input_dim))
        self.register_buffer('components', torch.zeros(output_dim, input_dim))
        
    def fit(self, X):
        """
        Calcule les composantes principales
        
        X: [n_samples, input_dim]
        """
        # Centrage
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        
        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Garde les top-k composantes
        self.components = Vt[:self.output_dim]
        
        # Variance expliquée
        explained_variance = (S[:self.output_dim]**2) / (S**2).sum()
        return explained_variance
        
    def forward(self, X):
        """Projette sur les composantes principales"""
        X_centered = X - self.mean
        return X_centered @ self.components.T
    
    def inverse_transform(self, X_reduced):
        """Reconstruction depuis l'espace réduit"""
        return X_reduced @ self.components + self.mean
```

---

## Exercices

### Exercice 2.1.1
Montrez que l'ensemble des matrices symétriques $n \times n$ forme un sous-espace vectoriel de $\mathbb{R}^{n \times n}$. Quelle est sa dimension ?

### Exercice 2.1.2
Soit $T: \mathbb{R}^3 \rightarrow \mathbb{R}^2$ définie par $T(x, y, z) = (x + y, y + z)$.
- Trouvez la matrice de $T$ dans les bases canoniques
- Calculez le noyau et l'image de $T$
- $T$ est-elle injective ? Surjective ?

### Exercice 2.1.3
Implémentez une fonction qui vérifie si un ensemble de vecteurs forme une base d'un espace vectoriel donné.

---

## Points Clés à Retenir

> 📌 **Les transformations linéaires sont entièrement caractérisées par leur action sur une base**

> 📌 **Le théorème du rang : dim(Im) + dim(Ker) = dim(domaine)**

> 📌 **Le changement de base est fondamental pour la diagonalisation et la compression**

> 📌 **Les couches de réseaux de neurones sont des transformations linéaires (+ non-linéarité)**

---

## Références

1. Strang, G. "Linear Algebra and Its Applications." 4th Edition, 2006
2. Axler, S. "Linear Algebra Done Right." 3rd Edition, 2015
3. Goodfellow, I. et al. "Deep Learning." Chapter 2: Linear Algebra, 2016

---

*Section suivante : [2.2 Décomposition en Valeurs Singulières (SVD)](./02_02_SVD.md)*

