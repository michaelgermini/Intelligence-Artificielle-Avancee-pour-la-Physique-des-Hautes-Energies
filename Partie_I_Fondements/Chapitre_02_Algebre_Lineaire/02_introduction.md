# Chapitre 2 : Fondements Mathématiques - Algèbre Linéaire Avancée

---

## Introduction

L'algèbre linéaire constitue le langage fondamental des réseaux de tenseurs et des techniques de compression de modèles. Ce chapitre établit les bases mathématiques nécessaires pour comprendre les décompositions tensorielles, les approximations de rang faible, et leur application à la compression de réseaux de neurones.

---

## Objectifs d'Apprentissage

À la fin de ce chapitre, vous serez capable de :

- Manipuler les espaces vectoriels et les transformations linéaires
- Comprendre et appliquer la décomposition en valeurs singulières (SVD)
- Utiliser les approximations de rang faible pour la compression
- Maîtriser les produits tensoriels et leur notation
- Analyser les erreurs d'approximation avec les normes matricielles

---

## Plan du Chapitre

1. [Espaces Vectoriels et Transformations Linéaires](./02_01_Espaces_Vectoriels.md)
2. [Décomposition en Valeurs Singulières (SVD)](./02_02_SVD.md)
3. [Approximations de Rang Faible](./02_03_Low_Rank.md)
4. [Produits Tensoriels et Espaces de Hilbert](./02_04_Produits_Tensoriels.md)
5. [Normes Matricielles et Erreurs d'Approximation](./02_05_Normes.md)

---

## Pourquoi l'Algèbre Linéaire ?

### Connexion avec le Deep Learning

Chaque opération dans un réseau de neurones peut être exprimée en termes d'algèbre linéaire :

```python
import numpy as np

# Une couche dense est une transformation linéaire + non-linéarité
def dense_layer(x, W, b, activation='relu'):
    """
    y = activation(Wx + b)
    
    - W : matrice de poids (transformation linéaire)
    - b : biais (translation)
    - activation : non-linéarité
    """
    linear_output = np.dot(W, x) + b
    
    if activation == 'relu':
        return np.maximum(0, linear_output)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-linear_output))
    return linear_output
```

### Connexion avec la Compression

La compression de modèles repose sur des propriétés fondamentales :

1. **Rang faible** : Les matrices de poids sont souvent approximables par des matrices de rang inférieur
2. **Structure tensorielle** : Les tenseurs de haute dimension ont une structure exploitable
3. **Redondance** : L'information est distribuée de manière non uniforme

---

## Notation et Conventions

### Scalaires, Vecteurs, Matrices, Tenseurs

| Objet | Notation | Exemple |
|-------|----------|---------|
| Scalaire | Minuscule italique | $a, b, \lambda$ |
| Vecteur | Minuscule gras | $\mathbf{v}, \mathbf{x}$ |
| Matrice | Majuscule gras | $\mathbf{A}, \mathbf{W}$ |
| Tenseur | Majuscule calligraphique | $\mathcal{T}, \mathcal{X}$ |

### Indices et Composantes

```python
# Conventions d'indexation
# Vecteur : v[i]
# Matrice : A[i,j] où i = ligne, j = colonne
# Tenseur d'ordre 3 : T[i,j,k]

import numpy as np

# Création d'exemples
v = np.array([1, 2, 3])           # Vecteur (ordre 1)
A = np.array([[1, 2], [3, 4]])    # Matrice (ordre 2)
T = np.random.randn(3, 4, 5)      # Tenseur ordre 3

print(f"Vecteur shape: {v.shape}")   # (3,)
print(f"Matrice shape: {A.shape}")   # (2, 2)
print(f"Tenseur shape: {T.shape}")   # (3, 4, 5)
```

---

## Concepts Clés Préliminaires

### Rang d'une Matrice

Le **rang** d'une matrice est le nombre de lignes (ou colonnes) linéairement indépendantes :

```python
def matrix_rank_analysis(A):
    """Analyse le rang d'une matrice"""
    # Rang numérique
    rank = np.linalg.matrix_rank(A)
    
    # Valeurs singulières
    U, S, Vt = np.linalg.svd(A)
    
    # Rang effectif (avec tolérance)
    tol = 1e-10
    effective_rank = np.sum(S > tol)
    
    return {
        'rank': rank,
        'effective_rank': effective_rank,
        'singular_values': S,
        'condition_number': S[0] / S[-1] if S[-1] > 0 else np.inf
    }

# Exemple : matrice de rang faible
A_low_rank = np.outer([1, 2, 3], [4, 5, 6])  # Rang 1
print(f"Rang de A: {np.linalg.matrix_rank(A_low_rank)}")
```

### Produit Scalaire et Orthogonalité

```python
def inner_product(u, v):
    """Produit scalaire standard"""
    return np.dot(u, v)

def are_orthogonal(u, v, tol=1e-10):
    """Vérifie si deux vecteurs sont orthogonaux"""
    return np.abs(inner_product(u, v)) < tol

def gram_schmidt(vectors):
    """Orthonormalisation de Gram-Schmidt"""
    orthonormal = []
    
    for v in vectors:
        # Soustrait les projections sur les vecteurs précédents
        for u in orthonormal:
            v = v - inner_product(v, u) * u
        
        # Normalise
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            orthonormal.append(v / norm)
    
    return np.array(orthonormal)
```

---

## Applications en Compression de Modèles

### Aperçu des Techniques

| Technique | Base Mathématique | Application |
|-----------|-------------------|-------------|
| SVD tronquée | Approximation de rang faible | Compression de couches denses |
| Décomposition CP | Factorisation tensorielle | Compression de convolutions |
| Tensor Train | Produits matriciels | Réduction de dimensionnalité |
| Quantification | Discrétisation | Réduction de précision |

### Exemple : Compression par SVD

```python
def compress_weight_matrix(W, target_rank):
    """
    Compresse une matrice de poids par SVD tronquée
    
    W ≈ U_r @ S_r @ V_r^T
    
    Réduction de paramètres : m×n → r×(m+n+1)
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    # Troncature au rang cible
    U_r = U[:, :target_rank]
    S_r = S[:target_rank]
    Vt_r = Vt[:target_rank, :]
    
    # Reconstruction
    W_approx = U_r @ np.diag(S_r) @ Vt_r
    
    # Métriques
    original_params = W.shape[0] * W.shape[1]
    compressed_params = target_rank * (W.shape[0] + W.shape[1] + 1)
    compression_ratio = original_params / compressed_params
    
    reconstruction_error = np.linalg.norm(W - W_approx, 'fro') / np.linalg.norm(W, 'fro')
    
    return {
        'U': U_r,
        'S': S_r,
        'Vt': Vt_r,
        'W_approx': W_approx,
        'compression_ratio': compression_ratio,
        'relative_error': reconstruction_error
    }

# Démonstration
W = np.random.randn(1000, 500)  # Matrice de poids
result = compress_weight_matrix(W, target_rank=50)

print(f"Ratio de compression: {result['compression_ratio']:.2f}x")
print(f"Erreur relative: {result['relative_error']:.4f}")
```

---

## Prérequis Mathématiques

Ce chapitre suppose une familiarité avec :

- Calcul matriciel de base (multiplication, transposition, inverse)
- Systèmes d'équations linéaires
- Notions de base sur les espaces vectoriels
- Dérivation et intégration

Pour les lecteurs nécessitant une révision, l'Annexe A fournit un rappel complet.

---

## Exercices Préliminaires

### Exercice 2.0.1
Calculez le rang des matrices suivantes :

$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix}, \quad \mathbf{B} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### Exercice 2.0.2
Montrez que le produit de deux matrices de rang $r_1$ et $r_2$ a un rang au plus $\min(r_1, r_2)$.

### Exercice 2.0.3
Une matrice de poids $\mathbf{W} \in \mathbb{R}^{1024 \times 512}$ est approximée par une décomposition de rang 64. Calculez le facteur de compression.

---

*Commençons par la première section : [2.1 Espaces Vectoriels et Transformations Linéaires](./02_01_Espaces_Vectoriels.md)*

