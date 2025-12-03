# 2.3 Approximations de Rang Faible (Low-Rank Approximations)

---

## Introduction

Les **approximations de rang faible** constituent le fondement théorique de nombreuses techniques de compression. L'idée centrale est que les matrices de grande dimension peuvent souvent être bien approximées par des matrices de rang beaucoup plus petit, permettant des gains significatifs en stockage et en calcul.

---

## Fondements Théoriques

### Définition du Rang

Le **rang** d'une matrice $\mathbf{A}$ est le nombre maximum de colonnes (ou lignes) linéairement indépendantes :

$$\text{rank}(\mathbf{A}) = \dim(\text{Im}(\mathbf{A})) = \dim(\text{Row}(\mathbf{A}))$$

### Factorisation de Rang Faible

Une matrice $\mathbf{A} \in \mathbb{R}^{m \times n}$ de rang $r$ peut toujours s'écrire :

$$\mathbf{A} = \mathbf{B}\mathbf{C}$$

où $\mathbf{B} \in \mathbb{R}^{m \times r}$ et $\mathbf{C} \in \mathbb{R}^{r \times n}$.

```python
import numpy as np

def verify_rank_factorization(A):
    """
    Vérifie que toute matrice de rang r peut être factorisée en BC
    """
    m, n = A.shape
    r = np.linalg.matrix_rank(A)
    
    # Méthode 1: Via SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    B_svd = U[:, :r] * np.sqrt(S[:r])
    C_svd = np.sqrt(S[:r])[:, np.newaxis] * Vt[:r, :]
    
    # Méthode 2: Via décomposition QR
    Q, R = np.linalg.qr(A)
    # Trouve les colonnes pivots
    
    # Vérification
    A_reconstructed = B_svd @ C_svd
    error = np.linalg.norm(A - A_reconstructed)
    
    print(f"Matrice originale: {m}×{n}, rang = {r}")
    print(f"Facteurs: B ({m}×{r}), C ({r}×{n})")
    print(f"Paramètres originaux: {m*n}")
    print(f"Paramètres factorisés: {m*r + r*n}")
    print(f"Erreur de reconstruction: {error:.2e}")
    
    return B_svd, C_svd

# Exemple avec une matrice de rang faible
np.random.seed(42)
# Crée une matrice de rang 5
U_true = np.random.randn(100, 5)
V_true = np.random.randn(5, 50)
A = U_true @ V_true

B, C = verify_rank_factorization(A)
```

---

## Théorème d'Eckart-Young-Mirsky

### Énoncé

Pour toute matrice $\mathbf{A}$ et tout entier $k < \text{rank}(\mathbf{A})$ :

$$\mathbf{A}_k = \arg\min_{\text{rank}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|$$

où $\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ est la SVD tronquée.

Ce résultat est valide pour :
- La norme de Frobenius : $\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}$
- La norme spectrale : $\|\mathbf{A} - \mathbf{A}_k\|_2 = \sigma_{k+1}$

```python
def eckart_young_demonstration(A, k):
    """
    Démontre le théorème d'Eckart-Young
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Approximation optimale de rang k
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # Erreurs théoriques
    error_frob_theory = np.sqrt(np.sum(S[k:]**2))
    error_spec_theory = S[k] if k < len(S) else 0
    
    # Erreurs calculées
    error_frob_actual = np.linalg.norm(A - A_k, 'fro')
    error_spec_actual = np.linalg.norm(A - A_k, 2)
    
    print(f"Approximation de rang {k}:")
    print(f"  Erreur Frobenius (théorique): {error_frob_theory:.6f}")
    print(f"  Erreur Frobenius (calculée):  {error_frob_actual:.6f}")
    print(f"  Erreur spectrale (théorique): {error_spec_theory:.6f}")
    print(f"  Erreur spectrale (calculée):  {error_spec_actual:.6f}")
    
    # Comparaison avec une approximation aléatoire de rang k
    random_B = np.random.randn(A.shape[0], k)
    random_C = np.random.randn(k, A.shape[1])
    A_random = random_B @ random_C
    
    # Optimise la projection (meilleure approximation dans cet espace)
    # A_random_opt = random_B @ (random_B.T @ A @ random_C.T @ np.linalg.inv(random_C @ random_C.T))
    
    error_random = np.linalg.norm(A - A_random, 'fro')
    print(f"  Erreur approximation aléatoire: {error_random:.6f}")
    print(f"  → SVD est {error_random/error_frob_actual:.1f}x meilleure")
    
    return A_k

# Test
A_test = np.random.randn(100, 80)
for k in [5, 10, 20]:
    eckart_young_demonstration(A_test, k)
    print()
```

---

## Méthodes de Factorisation

### 1. Factorisation Non-Négative (NMF)

```python
def nmf_multiplicative_update(A, k, n_iter=200, tol=1e-6):
    """
    Non-negative Matrix Factorization
    A ≈ WH où W, H ≥ 0
    
    Utile quand les données sont naturellement non-négatives
    (images, spectres, comptages)
    """
    m, n = A.shape
    
    # Assure que A est non-négatif
    A = np.maximum(A, 0)
    
    # Initialisation aléatoire positive
    W = np.abs(np.random.randn(m, k)) + 0.1
    H = np.abs(np.random.randn(k, n)) + 0.1
    
    for iteration in range(n_iter):
        # Mise à jour de H
        H = H * (W.T @ A) / (W.T @ W @ H + 1e-10)
        
        # Mise à jour de W
        W = W * (A @ H.T) / (W @ H @ H.T + 1e-10)
        
        # Calcul de l'erreur
        if iteration % 50 == 0:
            error = np.linalg.norm(A - W @ H, 'fro')
            print(f"Iteration {iteration}: erreur = {error:.4f}")
    
    return W, H

# Exemple avec une matrice non-négative
A_nonneg = np.abs(np.random.randn(50, 40))
W, H = nmf_multiplicative_update(A_nonneg, k=10)
print(f"\nErreur finale NMF: {np.linalg.norm(A_nonneg - W @ H, 'fro'):.4f}")
```

### 2. CUR Decomposition

```python
def cur_decomposition(A, k, n_samples=None):
    """
    Décomposition CUR : A ≈ CUR
    
    Sélectionne k colonnes (C) et k lignes (R) de A
    Avantage: préserve la sparsité et l'interprétabilité
    """
    m, n = A.shape
    if n_samples is None:
        n_samples = 2 * k
    
    # Calcul des probabilités de sélection basées sur les normes
    col_norms = np.linalg.norm(A, axis=0)**2
    col_probs = col_norms / np.sum(col_norms)
    
    row_norms = np.linalg.norm(A, axis=1)**2
    row_probs = row_norms / np.sum(row_norms)
    
    # Sélection des colonnes et lignes
    col_indices = np.random.choice(n, size=n_samples, replace=False, p=col_probs)
    row_indices = np.random.choice(m, size=n_samples, replace=False, p=row_probs)
    
    # Construction de C et R
    C = A[:, col_indices]
    R = A[row_indices, :]
    
    # Construction de U (pseudo-inverse de l'intersection)
    W = A[np.ix_(row_indices, col_indices)]
    U = np.linalg.pinv(W)
    
    # Reconstruction
    A_approx = C @ U @ R
    
    return {
        'C': C,
        'U': U,
        'R': R,
        'col_indices': col_indices,
        'row_indices': row_indices,
        'approximation': A_approx,
        'error': np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    }

# Comparaison CUR vs SVD
A_test = np.random.randn(100, 80)
cur_result = cur_decomposition(A_test, k=20)
svd_result = truncated_svd(A_test, k=20)

print(f"Erreur relative CUR: {cur_result['error']:.4f}")
print(f"Erreur relative SVD: {svd_result['relative_error']:.4f}")
```

### 3. Interpolative Decomposition (ID)

```python
def interpolative_decomposition(A, k):
    """
    Décomposition interpolative: A ≈ A[:, J] @ Z
    
    Sélectionne k colonnes de A et exprime les autres comme
    combinaisons linéaires de celles-ci
    """
    from scipy.linalg import qr
    
    m, n = A.shape
    
    # QR avec pivotage
    Q, R, P = qr(A, pivoting=True)
    
    # Sélectionne les k premières colonnes pivots
    J = P[:k]
    
    # Calcule la matrice d'interpolation
    R11 = R[:k, :k]
    R12 = R[:k, k:]
    
    # Z = [I; R11^(-1) @ R12] réarrangé selon P
    T = np.linalg.solve(R11, R12)
    
    # Reconstruction
    C = A[:, J]  # Colonnes sélectionnées
    
    # Matrice de coefficients
    Z = np.zeros((k, n))
    Z[:, J] = np.eye(k)
    Z[:, P[k:]] = T
    
    A_approx = C @ Z
    
    return {
        'C': C,
        'Z': Z,
        'column_indices': J,
        'approximation': A_approx,
        'error': np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
    }
```

---

## Applications en Deep Learning

### Compression de Couches Denses

```python
import torch
import torch.nn as nn

class LowRankLinear(nn.Module):
    """
    Couche linéaire de rang faible
    
    Au lieu de W (out × in), utilise:
    W = U @ V où U (out × rank), V (rank × in)
    """
    
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs de rang faible
        self.U = nn.Parameter(torch.randn(out_features, rank) / np.sqrt(rank))
        self.V = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x):
        # y = (U @ V) @ x + b = U @ (V @ x) + b
        out = x @ self.V.T  # [batch, rank]
        out = out @ self.U.T  # [batch, out_features]
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def get_full_weight(self):
        return self.U @ self.V
    
    @classmethod
    def from_linear(cls, linear_layer, rank):
        """
        Crée une couche de rang faible à partir d'une couche standard
        """
        W = linear_layer.weight.data.numpy()
        
        # SVD pour initialisation optimale
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        
        # Crée la nouvelle couche
        low_rank = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None
        )
        
        # Initialise avec SVD tronquée
        low_rank.U.data = torch.from_numpy(U[:, :rank] * np.sqrt(S[:rank])).float()
        low_rank.V.data = torch.from_numpy(np.sqrt(S[:rank])[:, np.newaxis] * Vt[:rank, :]).float()
        
        if linear_layer.bias is not None:
            low_rank.bias.data = linear_layer.bias.data.clone()
            
        return low_rank
    
    def compression_ratio(self):
        original = self.in_features * self.out_features
        compressed = self.rank * (self.in_features + self.out_features)
        return original / compressed

# Démonstration
original_layer = nn.Linear(1024, 512)
compressed_layer = LowRankLinear.from_linear(original_layer, rank=64)

print(f"Paramètres originaux: {1024 * 512 + 512:,}")
print(f"Paramètres compressés: {64 * (1024 + 512) + 512:,}")
print(f"Ratio de compression: {compressed_layer.compression_ratio():.2f}x")

# Test
x = torch.randn(32, 1024)
y_original = original_layer(x)
y_compressed = compressed_layer(x)

error = torch.norm(y_original - y_compressed) / torch.norm(y_original)
print(f"Erreur relative: {error.item():.4f}")
```

### LoRA (Low-Rank Adaptation)

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation pour le fine-tuning efficace
    
    W_new = W_original + α * (B @ A)
    
    Seuls A et B sont entraînés, W_original est gelé
    """
    
    def __init__(self, original_layer, rank, alpha=1.0):
        super().__init__()
        
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Gèle les poids originaux
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Adaptateurs de rang faible
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # Sortie originale
        original_out = self.original(x)
        
        # Adaptation LoRA
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        
        return original_out + self.alpha * lora_out
    
    def merge_weights(self):
        """Fusionne les poids LoRA avec les poids originaux"""
        delta_W = self.alpha * (self.lora_B @ self.lora_A)
        self.original.weight.data += delta_W
        
    def trainable_parameters(self):
        """Nombre de paramètres entraînables"""
        return self.rank * (self.original.in_features + self.original.out_features)
    
    def total_parameters(self):
        """Nombre total de paramètres"""
        return (self.original.in_features * self.original.out_features + 
                self.trainable_parameters())

# Démonstration de LoRA
base_layer = nn.Linear(4096, 4096)  # Grande couche (16M params)
lora_layer = LoRALayer(base_layer, rank=16, alpha=1.0)

print(f"Paramètres totaux: {lora_layer.total_parameters():,}")
print(f"Paramètres entraînables: {lora_layer.trainable_parameters():,}")
print(f"Ratio: {lora_layer.trainable_parameters() / lora_layer.total_parameters() * 100:.2f}%")
```

---

## Sélection Automatique du Rang

### Méthode de l'Énergie

```python
def select_rank_by_energy(S, threshold=0.99):
    """
    Sélectionne le rang pour capturer un pourcentage de l'énergie
    
    Énergie = sum(σᵢ²) / sum(σⱼ²)
    """
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    rank = np.searchsorted(cumulative_energy, threshold) + 1
    
    return rank, cumulative_energy

def select_rank_by_error(S, max_relative_error):
    """
    Sélectionne le rang pour une erreur relative maximale
    
    Erreur = ||A - A_k||_F / ||A||_F = sqrt(sum_{i>k} σᵢ²) / sqrt(sum σⱼ²)
    """
    total_norm_sq = np.sum(S**2)
    
    for k in range(len(S)):
        residual_norm_sq = np.sum(S[k:]**2)
        relative_error = np.sqrt(residual_norm_sq / total_norm_sq)
        
        if relative_error <= max_relative_error:
            return k, relative_error
    
    return len(S), 0.0

# Exemple
A = np.random.randn(500, 300)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

print("Sélection du rang:")
for threshold in [0.90, 0.95, 0.99, 0.999]:
    rank, _ = select_rank_by_energy(S, threshold)
    print(f"  {threshold*100:.1f}% énergie → rang {rank}")

print("\nPar erreur maximale:")
for max_error in [0.1, 0.05, 0.01, 0.001]:
    rank, actual_error = select_rank_by_error(S, max_error)
    print(f"  Erreur max {max_error:.3f} → rang {rank} (erreur réelle: {actual_error:.4f})")
```

### Méthode du Coude (Elbow Method)

```python
def elbow_method(S, plot=False):
    """
    Détecte le "coude" dans la courbe des valeurs singulières
    """
    # Normalise les valeurs singulières
    S_norm = S / S[0]
    
    # Calcule la courbure
    # κ = |f''| / (1 + f'²)^(3/2)
    
    # Approximation discrète des dérivées
    d1 = np.diff(S_norm)
    d2 = np.diff(d1)
    
    # Courbure (simplifiée)
    curvature = np.abs(d2) / (1 + d1[:-1]**2)**1.5
    
    # Le coude est au maximum de courbure
    elbow_idx = np.argmax(curvature) + 1
    
    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.semilogy(S_norm, 'b-o', markersize=3)
        ax1.axvline(x=elbow_idx, color='r', linestyle='--', label=f'Coude: k={elbow_idx}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Valeur singulière (normalisée)')
        ax1.set_title('Spectre des valeurs singulières')
        ax1.legend()
        
        ax2.plot(curvature, 'g-')
        ax2.axvline(x=elbow_idx-1, color='r', linestyle='--')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Courbure')
        ax2.set_title('Courbure du spectre')
        
        plt.tight_layout()
        return elbow_idx, fig
    
    return elbow_idx

# Test
elbow_rank = elbow_method(S)
print(f"Rang suggéré par la méthode du coude: {elbow_rank}")
```

---

## Exercices

### Exercice 2.3.1
Comparez les performances (erreur et temps de calcul) de la SVD, NMF et CUR pour approximer une matrice 1000×500 avec un rang cible de 50.

### Exercice 2.3.2
Implémentez une couche convolutionnelle de rang faible en factorisant les filtres.

### Exercice 2.3.3
Étant donné un réseau avec 10 couches denses de taille 1024×1024, calculez le nombre de paramètres avant et après compression par rang faible (rang 64 pour chaque couche).

---

## Points Clés à Retenir

> 📌 **L'approximation de rang faible optimale est donnée par la SVD tronquée**

> 📌 **Le rang effectif est souvent bien inférieur au rang théorique**

> 📌 **LoRA permet un fine-tuning efficace en n'entraînant que les adaptateurs**

> 📌 **Le choix du rang est un compromis compression/précision**

---

## Références

1. Eckart, C., Young, G. "The approximation of one matrix by another of lower rank." Psychometrika, 1936
2. Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022
3. Mahoney, M., Drineas, P. "CUR matrix decompositions for improved data analysis." PNAS, 2009

---

*Section suivante : [2.4 Produits Tensoriels et Espaces de Hilbert](./02_04_Produits_Tensoriels.md)*

