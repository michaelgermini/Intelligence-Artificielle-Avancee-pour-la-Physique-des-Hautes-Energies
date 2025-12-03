# 2.5 Normes Matricielles et Erreurs d'Approximation

---

## Introduction

Les **normes matricielles** sont essentielles pour quantifier les erreurs d'approximation en compression de modèles. Ce chapitre présente les différentes normes, leurs propriétés, et leur utilisation pour analyser la qualité des approximations.

---

## Normes Vectorielles

### Normes $\ell^p$

Pour un vecteur $\mathbf{x} \in \mathbb{R}^n$, la norme $\ell^p$ est définie par :

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$$

```python
import numpy as np

def lp_norm(x, p):
    """
    Calcule la norme Lp d'un vecteur
    
    Cas spéciaux:
    - p = 1 : norme Manhattan
    - p = 2 : norme euclidienne
    - p = inf : norme max
    """
    x = np.array(x)
    
    if p == np.inf:
        return np.max(np.abs(x))
    elif p == 0:
        return np.sum(x != 0)  # "Norme" L0 (pas une vraie norme)
    else:
        return np.sum(np.abs(x)**p)**(1/p)

# Comparaison des normes
x = np.array([3, -4, 0, 2, -1])

print("Comparaison des normes Lp:")
for p in [0, 1, 2, 3, np.inf]:
    if p == 0:
        print(f"  'L0' (support): {lp_norm(x, p)}")
    elif p == np.inf:
        print(f"  L∞: {lp_norm(x, p)}")
    else:
        print(f"  L{p}: {lp_norm(x, p):.4f}")
```

### Propriétés des Normes

```python
def verify_norm_properties(x, y, alpha, norm_func):
    """
    Vérifie les propriétés d'une norme:
    1. ||x|| ≥ 0 et ||x|| = 0 ssi x = 0
    2. ||αx|| = |α| ||x||
    3. ||x + y|| ≤ ||x|| + ||y|| (inégalité triangulaire)
    """
    results = {}
    
    # Non-négativité
    results['non_negative'] = norm_func(x) >= 0
    results['zero_iff_zero'] = (norm_func(np.zeros_like(x)) == 0)
    
    # Homogénéité
    lhs = norm_func(alpha * x)
    rhs = np.abs(alpha) * norm_func(x)
    results['homogeneity'] = np.isclose(lhs, rhs)
    
    # Inégalité triangulaire
    results['triangle'] = norm_func(x + y) <= norm_func(x) + norm_func(y) + 1e-10
    
    return results

# Test
x = np.random.randn(10)
y = np.random.randn(10)
alpha = -2.5

for p in [1, 2, np.inf]:
    norm_func = lambda v, p=p: lp_norm(v, p)
    props = verify_norm_properties(x, y, alpha, norm_func)
    print(f"L{p} norm properties: {all(props.values())}")
```

---

## Normes Matricielles

### Norme de Frobenius

La **norme de Frobenius** est l'analogue matriciel de la norme $\ell^2$ :

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{tr}(\mathbf{A}^T\mathbf{A})} = \sqrt{\sum_i \sigma_i^2}$$

```python
def frobenius_norm(A):
    """
    Norme de Frobenius
    
    Propriétés:
    - Invariante par rotation: ||UAV|| = ||A|| pour U, V orthogonales
    - ||A||_F² = sum(σᵢ²)
    """
    return np.sqrt(np.sum(A**2))

def frobenius_via_svd(A):
    """Calcul via les valeurs singulières"""
    _, S, _ = np.linalg.svd(A, full_matrices=False)
    return np.sqrt(np.sum(S**2))

# Vérification
A = np.random.randn(50, 30)
print(f"Frobenius (direct): {frobenius_norm(A):.6f}")
print(f"Frobenius (SVD): {frobenius_via_svd(A):.6f}")
print(f"Frobenius (numpy): {np.linalg.norm(A, 'fro'):.6f}")
```

### Norme Spectrale (Norme d'Opérateur)

La **norme spectrale** est la plus grande valeur singulière :

$$\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \max_{\|\mathbf{x}\|_2=1} \|\mathbf{A}\mathbf{x}\|_2$$

```python
def spectral_norm(A):
    """
    Norme spectrale (norme d'opérateur L2)
    
    = plus grande valeur singulière
    = racine de la plus grande valeur propre de A^T A
    """
    S = np.linalg.svd(A, compute_uv=False)
    return S[0]

def spectral_norm_power_iteration(A, n_iter=100, tol=1e-10):
    """
    Calcul de la norme spectrale par itération de puissance
    Utile pour les très grandes matrices
    """
    m, n = A.shape
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(n_iter):
        # Calcule A^T A v
        Av = A @ v
        AtAv = A.T @ Av
        
        # Normalise
        v_new = AtAv / np.linalg.norm(AtAv)
        
        # Vérifie convergence
        if np.linalg.norm(v - v_new) < tol:
            break
        v = v_new
    
    # Valeur singulière = ||Av||
    return np.linalg.norm(A @ v)

# Comparaison
print(f"Spectrale (SVD): {spectral_norm(A):.6f}")
print(f"Spectrale (power): {spectral_norm_power_iteration(A):.6f}")
print(f"Spectrale (numpy): {np.linalg.norm(A, 2):.6f}")
```

### Norme Nucléaire (Norme Trace)

La **norme nucléaire** est la somme des valeurs singulières :

$$\|\mathbf{A}\|_* = \sum_i \sigma_i = \text{tr}(\sqrt{\mathbf{A}^T\mathbf{A}})$$

```python
def nuclear_norm(A):
    """
    Norme nucléaire (trace norm, Schatten 1-norm)
    
    = somme des valeurs singulières
    = meilleure approximation convexe du rang
    """
    S = np.linalg.svd(A, compute_uv=False)
    return np.sum(S)

print(f"Nucléaire: {nuclear_norm(A):.6f}")
print(f"Nucléaire (numpy): {np.linalg.norm(A, 'nuc'):.6f}")
```

### Relations entre Normes

```python
def norm_inequalities(A):
    """
    Relations entre les différentes normes matricielles
    """
    m, n = A.shape
    r = np.linalg.matrix_rank(A)
    
    frob = np.linalg.norm(A, 'fro')
    spec = np.linalg.norm(A, 2)
    nuc = np.linalg.norm(A, 'nuc')
    
    print("Inégalités entre normes:")
    print(f"  ||A||_2 ≤ ||A||_F ≤ √r ||A||_2")
    print(f"  {spec:.4f} ≤ {frob:.4f} ≤ {np.sqrt(r) * spec:.4f}")
    
    print(f"\n  ||A||_F ≤ ||A||_* ≤ √r ||A||_F")
    print(f"  {frob:.4f} ≤ {nuc:.4f} ≤ {np.sqrt(r) * frob:.4f}")
    
    print(f"\n  ||A||_2 ≤ ||A||_* ≤ r ||A||_2")
    print(f"  {spec:.4f} ≤ {nuc:.4f} ≤ {r * spec:.4f}")

norm_inequalities(A)
```

---

## Erreurs d'Approximation

### Erreur Absolue vs Relative

```python
class ApproximationError:
    """
    Analyse des erreurs d'approximation
    """
    
    def __init__(self, A, A_approx):
        self.A = A
        self.A_approx = A_approx
        self.error_matrix = A - A_approx
        
    def absolute_error(self, norm='fro'):
        """Erreur absolue ||A - A_approx||"""
        return np.linalg.norm(self.error_matrix, norm)
    
    def relative_error(self, norm='fro'):
        """Erreur relative ||A - A_approx|| / ||A||"""
        return self.absolute_error(norm) / np.linalg.norm(self.A, norm)
    
    def elementwise_error(self):
        """Statistiques d'erreur élément par élément"""
        abs_errors = np.abs(self.error_matrix)
        return {
            'max': np.max(abs_errors),
            'mean': np.mean(abs_errors),
            'std': np.std(abs_errors),
            'median': np.median(abs_errors)
        }
    
    def snr(self):
        """Signal-to-Noise Ratio en dB"""
        signal_power = np.sum(self.A**2)
        noise_power = np.sum(self.error_matrix**2)
        return 10 * np.log10(signal_power / noise_power)
    
    def report(self):
        """Rapport complet d'erreur"""
        print("=" * 50)
        print("Rapport d'Erreur d'Approximation")
        print("=" * 50)
        
        print("\nErreurs par norme:")
        for norm in ['fro', 2, 'nuc']:
            norm_name = {'fro': 'Frobenius', 2: 'Spectrale', 'nuc': 'Nucléaire'}[norm]
            print(f"  {norm_name}:")
            print(f"    Absolue: {self.absolute_error(norm):.6f}")
            print(f"    Relative: {self.relative_error(norm):.4%}")
        
        print(f"\nSNR: {self.snr():.2f} dB")
        
        elem = self.elementwise_error()
        print("\nErreurs élément par élément:")
        for key, value in elem.items():
            print(f"  {key}: {value:.6f}")

# Démonstration avec SVD tronquée
A = np.random.randn(100, 80)
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Approximation de rang 20
k = 20
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

error_analysis = ApproximationError(A, A_approx)
error_analysis.report()
```

### Bornes d'Erreur Théoriques

```python
def theoretical_error_bounds(S, k):
    """
    Bornes d'erreur théoriques pour l'approximation de rang k
    
    Théorème d'Eckart-Young:
    ||A - A_k||_F = sqrt(sum_{i>k} σᵢ²)
    ||A - A_k||_2 = σ_{k+1}
    """
    # Erreur Frobenius
    error_frob = np.sqrt(np.sum(S[k:]**2))
    
    # Erreur spectrale
    error_spec = S[k] if k < len(S) else 0
    
    # Erreur relative
    total_frob = np.sqrt(np.sum(S**2))
    rel_error_frob = error_frob / total_frob
    
    # Énergie capturée
    energy_captured = np.sum(S[:k]**2) / np.sum(S**2)
    
    return {
        'error_frobenius': error_frob,
        'error_spectral': error_spec,
        'relative_error_frobenius': rel_error_frob,
        'energy_captured': energy_captured
    }

# Analyse pour différents rangs
_, S, _ = np.linalg.svd(A, full_matrices=False)

print("Bornes d'erreur théoriques:")
print("k  | Err. Frob | Err. Spec | Err. Rel | Énergie")
print("-" * 55)

for k in [5, 10, 20, 30, 40, 50]:
    bounds = theoretical_error_bounds(S, k)
    print(f"{k:2} | {bounds['error_frobenius']:9.4f} | {bounds['error_spectral']:9.4f} | "
          f"{bounds['relative_error_frobenius']:8.4%} | {bounds['energy_captured']:7.4%}")
```

---

## Conditionnement et Stabilité

### Nombre de Condition

Le **nombre de condition** mesure la sensibilité aux perturbations :

$$\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

```python
def condition_analysis(A):
    """
    Analyse du conditionnement d'une matrice
    """
    S = np.linalg.svd(A, compute_uv=False)
    
    # Nombre de condition
    if S[-1] > 1e-15:
        cond = S[0] / S[-1]
    else:
        cond = np.inf
    
    # Rang numérique
    tol = S[0] * max(A.shape) * np.finfo(float).eps
    numerical_rank = np.sum(S > tol)
    
    # Analyse de stabilité
    print("Analyse du Conditionnement")
    print("=" * 40)
    print(f"Dimensions: {A.shape}")
    print(f"Rang théorique: {min(A.shape)}")
    print(f"Rang numérique: {numerical_rank}")
    print(f"\nValeurs singulières:")
    print(f"  σ_max = {S[0]:.6e}")
    print(f"  σ_min = {S[-1]:.6e}")
    print(f"\nNombre de condition: {cond:.6e}")
    
    # Interprétation
    print("\nInterprétation:")
    if cond < 10:
        print("  → Très bien conditionné")
    elif cond < 100:
        print("  → Bien conditionné")
    elif cond < 1e6:
        print("  → Modérément mal conditionné")
    else:
        print("  → Très mal conditionné (problème numériquement instable)")
    
    return cond, numerical_rank

# Test avec différentes matrices
print("\n--- Matrice aléatoire ---")
condition_analysis(A)

print("\n--- Matrice mal conditionnée ---")
A_bad = np.vander(np.linspace(0, 1, 10))  # Matrice de Vandermonde
condition_analysis(A_bad)
```

### Impact sur les Approximations

```python
def perturbation_analysis(A, epsilon=1e-6):
    """
    Analyse de l'impact des perturbations sur l'approximation
    """
    # Perturbation aléatoire
    E = np.random.randn(*A.shape)
    E = E / np.linalg.norm(E, 'fro') * epsilon * np.linalg.norm(A, 'fro')
    
    A_perturbed = A + E
    
    # SVD des deux matrices
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    U_p, S_p, Vt_p = np.linalg.svd(A_perturbed, full_matrices=False)
    
    # Comparaison des valeurs singulières
    print("Impact de la perturbation sur les valeurs singulières:")
    print(f"Perturbation relative: {epsilon:.2e}")
    print("\nIndice | σ original | σ perturbé | Différence relative")
    print("-" * 60)
    
    for i in range(min(10, len(S))):
        rel_diff = np.abs(S[i] - S_p[i]) / S[i] if S[i] > 0 else 0
        print(f"{i:5} | {S[i]:10.6f} | {S_p[i]:10.6f} | {rel_diff:18.6e}")
    
    # Borne théorique (théorème de Weyl)
    print(f"\nBorne de Weyl: |σᵢ - σᵢ'| ≤ ||E||_2 = {np.linalg.norm(E, 2):.6e}")

perturbation_analysis(A, epsilon=1e-4)
```

---

## Applications Pratiques

### Sélection du Rang Optimal

```python
def optimal_rank_selection(A, methods=['energy', 'elbow', 'gap']):
    """
    Différentes méthodes pour sélectionner le rang optimal
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    results = {}
    
    # Méthode de l'énergie
    if 'energy' in methods:
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        for threshold in [0.90, 0.95, 0.99]:
            rank = np.searchsorted(cumulative_energy, threshold) + 1
            results[f'energy_{threshold}'] = rank
    
    # Méthode du coude
    if 'elbow' in methods:
        # Courbure discrète
        d1 = np.diff(np.log(S + 1e-10))
        d2 = np.diff(d1)
        elbow = np.argmax(np.abs(d2)) + 1
        results['elbow'] = elbow
    
    # Méthode du gap
    if 'gap' in methods:
        ratios = S[:-1] / S[1:]
        gap = np.argmax(ratios) + 1
        results['gap'] = gap
    
    return results

ranks = optimal_rank_selection(A)
print("Rangs optimaux suggérés:")
for method, rank in ranks.items():
    print(f"  {method}: {rank}")
```

### Compression avec Contrainte d'Erreur

```python
def compress_with_error_bound(A, max_relative_error, norm='fro'):
    """
    Trouve le rang minimal pour respecter une contrainte d'erreur
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    if norm == 'fro':
        total_norm = np.sqrt(np.sum(S**2))
        max_error = max_relative_error * total_norm
        
        # Trouve k tel que sqrt(sum_{i>k} σᵢ²) ≤ max_error
        cumsum_sq = np.cumsum(S[::-1]**2)[::-1]
        errors = np.sqrt(np.append(cumsum_sq[1:], 0))
        
        k = np.searchsorted(-errors, -max_error)
        
    elif norm == 2:
        total_norm = S[0]
        max_error = max_relative_error * total_norm
        
        # Trouve k tel que σ_{k+1} ≤ max_error
        k = np.searchsorted(-S, -max_error)
    
    # Compression
    A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    actual_error = np.linalg.norm(A - A_compressed, norm) / np.linalg.norm(A, norm)
    
    return {
        'rank': k,
        'target_error': max_relative_error,
        'actual_error': actual_error,
        'compression_ratio': A.size / (k * sum(A.shape) + k)
    }

# Test
for target in [0.1, 0.05, 0.01, 0.001]:
    result = compress_with_error_bound(A, target)
    print(f"Erreur cible: {target:.3f} → rang {result['rank']}, "
          f"erreur réelle: {result['actual_error']:.4f}, "
          f"compression: {result['compression_ratio']:.1f}x")
```

---

## Exercices

### Exercice 2.5.1
Montrez que pour toute matrice $\mathbf{A}$, $\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F \leq \sqrt{r}\|\mathbf{A}\|_2$ où $r = \text{rank}(\mathbf{A})$.

### Exercice 2.5.2
Une matrice de poids a des valeurs singulières $\sigma_i = 10/i$ pour $i = 1, \ldots, 100$. Quel rang faut-il pour une erreur relative de 1% en norme de Frobenius ?

### Exercice 2.5.3
Implémentez une fonction qui compresse une matrice en minimisant la norme nucléaire sous contrainte d'erreur de Frobenius.

---

## Points Clés à Retenir

> 📌 **La norme de Frobenius mesure l'erreur "totale", la norme spectrale l'erreur "maximale"**

> 📌 **Le nombre de condition indique la sensibilité aux perturbations**

> 📌 **L'erreur de la SVD tronquée est exactement $\sqrt{\sum_{i>k}\sigma_i^2}$ en norme de Frobenius**

> 📌 **Le choix de la norme dépend de l'application (reconstruction vs prédiction)**

---

## Références

1. Golub, G., Van Loan, C. "Matrix Computations." Chapter 2: Matrix Analysis
2. Trefethen, L., Bau, D. "Numerical Linear Algebra." Lecture 3-5
3. Stewart, G.W. "Matrix Perturbation Theory." Academic Press, 1990

---

*Chapitre suivant : [Chapitre 3 - Deep Learning : Architectures et Principes](../Chapitre_03_Deep_Learning/03_introduction.md)*

