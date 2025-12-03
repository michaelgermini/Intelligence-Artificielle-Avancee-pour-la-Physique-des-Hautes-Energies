# 22.2 NumPy et Manipulation de Tenseurs

---

## Introduction

**NumPy** est la bibliothèque fondamentale pour le calcul scientifique en Python. Elle fournit des tableaux multidimensionnels (tenseurs) efficaces et des opérations mathématiques optimisées. Tous les frameworks de deep learning (PyTorch, TensorFlow) s'appuient sur NumPy pour leurs opérations de base.

Cette section présente les opérations essentielles de NumPy pour la manipulation de tenseurs.

---

## Création de Tableaux

### Initialisation

```python
import numpy as np

# Création de tableaux
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"1D array: {arr1d.shape}")  # (5,)
print(f"2D array: {arr2d.shape}")  # (2, 3)
print(f"3D array: {arr3d.shape}")  # (2, 2, 2)

# Tableaux spéciaux
zeros = np.zeros((3, 4))  # Matrice 3×4 remplie de zéros
ones = np.ones((2, 3))    # Matrice 2×3 remplie de uns
identity = np.eye(4)      # Matrice identité 4×4
random = np.random.randn(5, 3)  # Matrice 5×3 avec valeurs aléatoires

# Range et linspace
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_arr = np.linspace(0, 1, 5)  # [0., 0.25, 0.5, 0.75, 1.]

# Meshgrid (utile pour visualisation)
x = np.linspace(-5, 5, 11)
y = np.linspace(-5, 5, 11)
X, Y = np.meshgrid(x, y)
```

---

## Propriétés et Attributs

### Shape, dtype, size

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"Shape: {arr.shape}")          # (2, 3)
print(f"Dimensions: {arr.ndim}")      # 2
print(f"Size: {arr.size}")            # 6 (nombre total d'éléments)
print(f"dtype: {arr.dtype}")          # int64
print(f"Itemsize: {arr.itemsize}")    # 8 bytes par élément

# Modifier shape
reshaped = arr.reshape(3, 2)  # (3, 2)
flattened = arr.flatten()      # (6,) - copie
raveled = arr.ravel()          # (6,) - vue (pas de copie)
```

---

## Indexation et Slicing

### Accès aux Éléments

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Indexation
element = arr[1, 2]          # 7
row = arr[1]                 # [5, 6, 7, 8]
column = arr[:, 2]           # [3, 7, 11]

# Slicing
slice_2d = arr[1:3, 0:2]     # [[5, 6], [9, 10]]
every_other = arr[::2]       # [[1, 2, 3, 4], [9, 10, 11, 12]]

# Indexation avancée
mask = arr > 5
filtered = arr[mask]         # [6, 7, 8, 9, 10, 11, 12]

indices = [0, 2]
selected = arr[indices]      # Première et troisième ligne

# Indexation booléenne
arr[arr > 5] = 0             # Remplace valeurs > 5 par 0
```

---

## Opérations Mathématiques

### Opérations Élément par Élément

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Opérations élément par élément
sum_arr = a + b              # [[6, 8], [10, 12]]
diff_arr = a - b             # [[-4, -4], [-4, -4]]
prod_arr = a * b             # [[5, 12], [21, 32]]
div_arr = a / b              # [[0.2, 0.333...], [0.429..., 0.5]]
power_arr = a ** 2           # [[1, 4], [9, 16]]

# Opérations mathématiques
sin_arr = np.sin(a)          # Sinus de chaque élément
exp_arr = np.exp(a)          # Exponentielle
log_arr = np.log(a)          # Logarithme naturel
sqrt_arr = np.sqrt(a)        # Racine carrée

# Agrégations
sum_total = np.sum(a)        # 10 (somme de tous les éléments)
sum_axis_0 = np.sum(a, axis=0)  # [4, 6] (somme le long de l'axe 0)
mean_arr = np.mean(a)        # 2.5
std_arr = np.std(a)          # 1.118...
max_val = np.max(a)          # 4
min_val = np.min(a)          # 1
```

---

## Algèbre Linéaire

### Matrices et Vecteurs

```python
# Multiplication matricielle
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# @ opérateur (Python 3.5+)
C = A @ B                    # Multiplication matricielle standard
# ou
C = np.dot(A, B)             # Même chose

# Multiplication élément par élément
C_elem = A * B

# Transposition
A_T = A.T                    # [[1, 3], [2, 4]]

# Déterminant
det = np.linalg.det(A)       # -2.0

# Inverse
A_inv = np.linalg.inv(A)     # [[-2., 1.], [1.5, -0.5]]

# Valeurs propres et vecteurs propres
eigenvals, eigenvecs = np.linalg.eig(A)

# Décomposition SVD
U, s, Vt = np.linalg.svd(A)

# Norme
norm_L2 = np.linalg.norm(A)  # Norme de Frobenius
norm_row = np.linalg.norm(A, axis=1)  # Norme de chaque ligne
```

---

## Broadcasting

### Opérations avec Formes Différentes

```python
# Broadcasting permet opérations entre tableaux de formes différentes

# Exemple 1: Scalaire + tableau
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10            # Additionne 10 à chaque élément

# Exemple 2: Vecteur + matrice
row = np.array([1, 2, 3])
result = arr + row           # Ajoute row à chaque ligne

col = np.array([[1], [2]])
result = arr + col           # Ajoute col à chaque colonne

# Exemple 3: Broadcasting 3D
arr_3d = np.random.randn(3, 4, 5)
arr_2d = np.random.randn(4, 5)
result = arr_3d + arr_2d     # Broadcast sur première dimension

# Règles de broadcasting:
# 1. Aligner dimensions à droite
# 2. Dimensions compatibles si égales ou l'une = 1
# 3. Dimension 1 est étendue pour correspondre
```

---

## Opérations Avancées

### Fonctions Utiles

```python
# Concaténation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_vertical = np.vstack([a, b])      # Empile verticalement
concat_horizontal = np.hstack([a, b])    # Empile horizontalement
concat_axis = np.concatenate([a, b], axis=0)  # Général

# Split
split_arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
parts = np.split(split_arr, 2, axis=0)   # Divise en 2 parties

# Répétition
repeated = np.repeat(a, 2, axis=0)       # Répète chaque ligne 2 fois
tiled = np.tile(a, (2, 2))               # Répète le tableau

# Tri
arr = np.array([3, 1, 4, 1, 5, 9, 2])
sorted_arr = np.sort(arr)                # Tri (copie)
arr.sort()                               # Tri en place
indices = np.argsort(arr)                # Indices de tri

# Unique
unique_vals = np.unique(arr)             # Valeurs uniques
unique_vals, counts = np.unique(arr, return_counts=True)

# Où (conditionnel)
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, arr, -1)      # [ -1, -1, -1, 4, 5]

# Sélection
selected = np.select([arr < 2, arr < 4, arr >= 4], 
                     [0, 1, 2])           # [0, 1, 1, 2, 2]
```

---

## Performance et Optimisation

### Tips pour Performance

```python
import time

# Éviter boucles Python, utiliser opérations vectorisées
def slow_sum(arr):
    """Lent: boucle Python"""
    result = 0
    for x in arr:
        result += x
    return result

def fast_sum(arr):
    """Rapide: opération NumPy vectorisée"""
    return np.sum(arr)

# Comparaison
large_arr = np.random.randn(1000000)

start = time.time()
slow_result = slow_sum(large_arr)
slow_time = time.time() - start

start = time.time()
fast_result = fast_sum(large_arr)
fast_time = time.time() - start

print(f"Slow: {slow_time:.4f}s")
print(f"Fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}×")

# Utiliser vues au lieu de copies quand possible
arr = np.array([[1, 2, 3], [4, 5, 6]])
view = arr[:2, :2]          # Vue (pas de copie)
copy = arr[:2, :2].copy()   # Copie explicite

# Pré-allouer tableaux
# Mauvais: redimensionner à chaque itération
result = np.array([])
for i in range(10):
    result = np.append(result, i)

# Bon: pré-allouer
result = np.zeros(10)
for i in range(10):
    result[i] = i
```

---

## Exercices

### Exercice 22.2.1
Créez une matrice 5×5 avec valeurs aléatoires, calculez sa décomposition SVD, et reconstruisez-la.

### Exercice 22.2.2
Implémentez une fonction qui calcule le produit matriciel en utilisant seulement des opérations NumPy vectorisées (sans boucles Python).

### Exercice 22.2.3
Créez deux tableaux de formes (3, 4, 5) et (4, 5) et utilisez broadcasting pour effectuer des opérations entre eux.

### Exercice 22.2.4
Comparez performance entre opérations vectorisées NumPy et boucles Python pour un calcul complexe (ex: norme de chaque ligne d'une matrice).

---

## Points Clés à Retenir

> 📌 **NumPy fournit tableaux multidimensionnels efficaces**

> 📌 **Les opérations vectorisées sont beaucoup plus rapides que boucles Python**

> 📌 **Le broadcasting permet opérations entre tableaux de formes différentes**

> 📌 **L'indexation avancée permet sélection complexe d'éléments**

> 📌 **Utiliser vues au lieu de copies quand possible pour performance**

> 📌 **NumPy est base pour tous frameworks deep learning**

---

*Section précédente : [22.1 Environnement](./22_01_Environnement.md) | Section suivante : [22.3 PyTorch](./22_03_PyTorch.md)*

