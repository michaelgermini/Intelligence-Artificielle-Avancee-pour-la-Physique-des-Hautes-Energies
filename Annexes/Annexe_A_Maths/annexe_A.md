# Annexe A : Rappels Mathématiques

---

## Introduction

Cette annexe présente les concepts mathématiques fondamentaux utilisés tout au long de ce livre. Elle sert de référence rapide pour l'algèbre tensorielle, l'optimisation, la théorie de l'information, et les probabilités et statistiques.

---

## Plan de l'Annexe

1. [A.1 Algèbre Tensorielle](#a1-algèbre-tensorielle)
2. [A.2 Optimisation Convexe](#a2-optimisation-convexe)
3. [A.3 Théorie de l'Information](#a3-théorie-de-linformation)
4. [A.4 Probabilités et Statistiques](#a4-probabilités-et-statistiques)

---

## A.1 Algèbre Tensorielle

### Produits Tensoriels

Le **produit tensoriel** de deux espaces vectoriels $V$ et $W$ :

$$V \otimes W = \text{span}\{v \otimes w : v \in V, w \in W\}$$

#### Propriétés

```python
import numpy as np

def outer_product(v, w):
    """
    Produit tensoriel externe (outer product)
    
    Args:
        v: Vecteur de dimension m
        w: Vecteur de dimension n
    
    Returns:
        Matrice m × n
    """
    return np.outer(v, w)

# Exemple
v = np.array([1, 2, 3])
w = np.array([4, 5])
result = outer_product(v, w)
print(f"Outer product shape: {result.shape}")  # (3, 2)
```

#### Produit de Kronecker

Pour matrices $A \in \mathbb{R}^{m \times n}$ et $B \in \mathbb{R}^{p \times q}$ :

$$A \otimes B = \begin{bmatrix}
a_{11}B & \cdots & a_{1n}B \\
\vdots & \ddots & \vdots \\
a_{m1}B & \cdots & a_{mn}B
\end{bmatrix}$$

```python
def kronecker_product(A, B):
    """Produit de Kronecker"""
    return np.kron(A, B)

# Exemple
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
K = kronecker_product(A, B)
print(f"Kronecker product shape: {K.shape}")  # (4, 4)
```

### Contraction

**Contraction** sur indices répétés (convention d'Einstein) :

$$C_{ik} = A_{ij} B_{jk} = \sum_j A_{ij} B_{jk}$$

#### Types de Contractions

```python
def tensor_contraction(T, axes):
    """
    Contraction tensorielle
    
    Args:
        T: Tenseur à contracter
        axes: Tuple d'indices à contracter
    """
    return np.tensordot(T, T, axes=axes)

# Exemple: contraction sur dernière dimension
T1 = np.random.randn(3, 4, 5)
T2 = np.random.randn(5, 6)
result = np.tensordot(T1, T2, axes=([2], [0]))
print(f"Contraction result shape: {result.shape}")  # (3, 4, 6)
```

### Rang Tensoriel

Le **rang tensoriel** est le nombre minimal de termes de rang 1 nécessaires :

$$\text{rank}(\mathcal{T}) = \min\{r : \mathcal{T} = \sum_{k=1}^r \mathbf{v}_k^{(1)} \otimes \cdots \otimes \mathbf{v}_k^{(d)}\}$$

#### Rang vs Mode-n Rank

```python
def compute_mode_rank(tensor, mode):
    """
    Calcule mode-n rank (rang matriciel après unfold)
    """
    # Unfold tenseur le long du mode
    shape = tensor.shape
    n_modes = len(shape)
    
    # Permutation pour mettre mode en premier
    perm = list(range(n_modes))
    perm[0], perm[mode] = perm[mode], perm[0]
    tensor_perm = np.transpose(tensor, perm)
    
    # Reshape en matrice
    I_mode = shape[mode]
    I_other = np.prod([shape[i] for i in range(n_modes) if i != mode])
    matrix = tensor_perm.reshape(I_mode, I_other)
    
    # Rang matriciel
    rank = np.linalg.matrix_rank(matrix)
    return rank

# Exemple
T = np.random.randn(5, 6, 7)
for mode in range(3):
    rank = compute_mode_rank(T, mode)
    print(f"Mode-{mode} rank: {rank}")
```

### Normes Tensorielles

#### Norme de Frobenius

$$||\mathcal{T}||_F = \sqrt{\sum_{i_1,\ldots,i_d} T_{i_1\ldots i_d}^2}$$

```python
def frobenius_norm(tensor):
    """Norme de Frobenius d'un tenseur"""
    return np.linalg.norm(tensor.flatten())

T = np.random.randn(3, 4, 5)
norm_F = frobenius_norm(T)
print(f"Frobenius norm: {norm_F}")
```

---

## A.2 Optimisation Convexe

### Problème d'Optimisation

$$\min_{\mathbf{x} \in \mathcal{D}} f(\mathbf{x})$$

où $f$ est **convexe** si :

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$$

pour tout $\lambda \in [0,1]$ et $\mathbf{x}, \mathbf{y} \in \mathcal{D}$.

#### Vérifier Convexité

```python
def is_convex(f, domain, n_samples=1000):
    """
    Vérifie convexité d'une fonction
    
    Note: Approche numérique, pas démonstration formelle
    """
    for _ in range(n_samples):
        x = np.random.uniform(domain[0], domain[1])
        y = np.random.uniform(domain[0], domain[1])
        lambda_val = np.random.uniform(0, 1)
        
        # Point sur ligne entre x et y
        z = lambda_val * x + (1 - lambda_val) * y
        
        # Vérifier inégalité
        lhs = f(z)
        rhs = lambda_val * f(x) + (1 - lambda_val) * f(y)
        
        if lhs > rhs + 1e-6:  # Petite tolérance numérique
            return False
    
    return True

# Exemple: f(x) = x^2 est convexe
f_quad = lambda x: x**2
print(f"x^2 is convex: {is_convex(f_quad, (-10, 10))}")
```

### Conditions d'Optimalité

#### Minimum Local

Pour un minimum local :

$$\nabla f(\mathbf{x}^*) = 0$$

#### Problème avec Contraintes (Lagrange)

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \sum_i \lambda_i g_i(\mathbf{x})$$

Conditions KKT (Karush-Kuhn-Tucker) :

1. Stationnarité : $\nabla_x \mathcal{L} = 0$
2. Primal feasibility : $g_i(\mathbf{x}) \leq 0$
3. Dual feasibility : $\lambda_i \geq 0$
4. Complementary slackness : $\lambda_i g_i(\mathbf{x}) = 0$

```python
from scipy.optimize import minimize

def optimize_with_constraints():
    """Exemple optimisation avec contraintes"""
    # Fonction objectif
    def objective(x):
        return x[0]**2 + x[1]**2
    
    # Contrainte: x[0] + x[1] >= 1
    constraints = {
        'type': 'ineq',
        'fun': lambda x: x[0] + x[1] - 1
    }
    
    # Point initial
    x0 = [0.5, 0.5]
    
    # Optimisation
    result = minimize(objective, x0, constraints=constraints)
    return result

result = optimize_with_constraints()
print(f"Optimal point: {result.x}")
```

### Algorithmes d'Optimisation

#### Gradient Descent

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.01, n_iter=100):
    """Gradient descent simple"""
    x = x0.copy()
    history = [x.copy()]
    
    for _ in range(n_iter):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x.copy())
    
    return x, history

# Exemple: minimiser f(x) = x^2
f = lambda x: x**2
grad_f = lambda x: 2 * x

x_opt, history = gradient_descent(f, grad_f, np.array([10.0]), learning_rate=0.1)
print(f"Optimal x: {x_opt[0]}")
```

#### Newton's Method

$$\mathbf{x}_{t+1} = \mathbf{x}_t - [\nabla^2 f(\mathbf{x}_t)]^{-1} \nabla f(\mathbf{x}_t)$$

---

## A.3 Théorie de l'Information

### Entropie de Shannon

L'**entropie** mesure l'incertitude d'une variable aléatoire :

$$H(X) = -\sum_{i} p_i \log_2(p_i)$$

#### Propriétés

```python
def shannon_entropy(probabilities, base=2):
    """
    Calcule entropie de Shannon
    
    Args:
        probabilities: Array de probabilités (doit sommer à 1)
        base: Base du logarithme (2 pour bits, e pour nats)
    """
    # Normaliser
    probs = np.array(probabilities)
    probs = probs / probs.sum()
    
    # Éliminer zéros (log(0) = -inf)
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log(probs) / np.log(base))
    return entropy

# Exemple: Distribution uniforme (entropie maximale)
uniform_probs = np.ones(4) / 4
H_uniform = shannon_entropy(uniform_probs)
print(f"Entropy of uniform distribution: {H_uniform:.4f} bits")

# Distribution déterministe (entropie minimale)
deterministic_probs = [1.0, 0.0, 0.0, 0.0]
H_deterministic = shannon_entropy(deterministic_probs)
print(f"Entropy of deterministic distribution: {H_deterministic:.4f} bits")
```

### Information Mutuelle

L'**information mutuelle** mesure dépendance entre variables :

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

```python
def mutual_information(pxy, px, py, base=2):
    """
    Calcule information mutuelle I(X;Y)
    
    Args:
        pxy: Distribution jointe p(x,y)
        px: Distribution marginale p(x)
        py: Distribution marginale p(y)
    """
    # Éviter division par zéro
    eps = 1e-10
    pxy = pxy + eps
    px = px + eps
    py = py + eps
    
    # Normaliser
    pxy = pxy / pxy.sum()
    px = px / px.sum()
    py = py / py.sum()
    
    # Calculer MI
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > eps:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    
    return mi / np.log(base)

# Exemple: X et Y indépendants
px = np.array([0.5, 0.5])
py = np.array([0.5, 0.5])
pxy_indep = np.outer(px, py)  # Produit indépendant
MI_indep = mutual_information(pxy_indep, px, py)
print(f"MI for independent variables: {MI_indep:.4f}")

# Exemple: X et Y dépendants
pxy_dep = np.array([[0.4, 0.1], [0.1, 0.4]])  # Corrélés
MI_dep = mutual_information(pxy_dep, px, py)
print(f"MI for dependent variables: {MI_dep:.4f}")
```

### Divergence de Kullback-Leibler

La **KL divergence** mesure différence entre distributions :

$$D_{KL}(P||Q) = \sum_i p_i \log\left(\frac{p_i}{q_i}\right)$$

$$D_{KL}(P||Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

#### Propriétés

- $D_{KL}(P||Q) \geq 0$ (égalité si $P = Q$)
- Pas symétrique : $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

```python
def kl_divergence(p, q, base=2):
    """
    Calcule KL divergence D_KL(P||Q)
    
    Args:
        p, q: Distributions discrètes
    """
    p = np.array(p)
    q = np.array(q)
    
    # Normaliser
    p = p / p.sum()
    q = q / q.sum()
    
    # Éviter log(0)
    eps = 1e-10
    q = q + eps
    q = q / q.sum()
    
    # Calculer KL
    kl = np.sum(p * np.log(p / q) / np.log(base))
    return kl

# Exemple
p = np.array([0.5, 0.3, 0.2])
q = np.array([0.33, 0.33, 0.34])
kl_pq = kl_divergence(p, q)
print(f"D_KL(P||Q): {kl_pq:.4f}")

# KL divergence n'est pas symétrique
kl_qp = kl_divergence(q, p)
print(f"D_KL(Q||P): {kl_qp:.4f}")
print(f"Difference: {abs(kl_pq - kl_qp):.4f}")
```

---

## A.4 Probabilités et Statistiques

### Distributions Utiles

#### Distribution Gaussienne (Normale)

$$p(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Propriétés** :
- Moyenne : $\mu$
- Variance : $\sigma^2$
- Symétrique autour de $\mu$

```python
from scipy import stats

def gaussian_example():
    """Exemples avec distribution gaussienne"""
    # Distribution normale standard
    mu, sigma = 0, 1
    x = np.linspace(-5, 5, 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    
    # Propriétés
    mean = stats.norm.mean(mu, sigma)
    var = stats.norm.var(mu, sigma)
    
    print(f"Gaussian mean: {mean}, variance: {var}")
    return x, pdf

x, pdf = gaussian_example()
```

#### Distribution Multivariée Gaussienne

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

```python
def multivariate_gaussian(mean, cov):
    """Distribution gaussienne multivariée"""
    d = len(mean)
    
    def pdf(x):
        diff = x - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        
        exponent = -0.5 * diff.T @ inv_cov @ diff
        normalization = 1.0 / np.sqrt((2 * np.pi)**d * det_cov)
        
        return normalization * np.exp(exponent)
    
    return pdf

# Exemple 2D
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
pdf_2d = multivariate_gaussian(mean, cov)

x = np.array([0, 0])
print(f"PDF at origin: {pdf_2d(x):.4f}")
```

#### Distribution Bernoulli

$$p(x) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}$$

- Espérance : $\mathbb{E}[X] = p$
- Variance : $\text{Var}(X) = p(1-p)$

#### Distribution Categorique (Multinomiale)

$$p(x=i) = p_i, \quad \sum_i p_i = 1$$

### Moments Statistiques

#### Moyenne (Espérance)

$$\mu = \mathbb{E}[X] = \int x p(x) dx$$

Pour échantillon : $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$

```python
def moments_example():
    """Exemples calcul moments"""
    data = np.random.randn(1000)
    
    # Moyenne
    mean = np.mean(data)
    
    # Variance
    variance = np.var(data, ddof=1)  # ddof=1 pour échantillon
    
    # Skewness (asymétrie)
    from scipy.stats import skew
    skewness = skew(data)
    
    # Kurtosis (aplatissement)
    from scipy.stats import kurtosis
    kurt = kurtosis(data)
    
    return {
        'mean': mean,
        'variance': variance,
        'std': np.sqrt(variance),
        'skewness': skewness,
        'kurtosis': kurt
    }

moments = moments_example()
print(f"Moments: {moments}")
```

#### Variance et Écart-Type

**Variance** : $\sigma^2 = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - \mu^2$

**Écart-type** : $\sigma = \sqrt{\sigma^2}$

#### Skewness et Kurtosis

**Skewness** (asymétrie) : $\gamma_1 = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$

**Kurtosis** (aplatissement) : $\gamma_2 = \mathbb{E}\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$

### Tests Statistiques

#### Test du Chi-deux

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

où $O_i$ sont valeurs observées et $E_i$ valeurs attendues.

```python
from scipy.stats import chisquare

def chi_square_test(observed, expected):
    """
    Test du Chi-deux
    
    Args:
        observed: Fréquences observées
        expected: Fréquences attendues
    """
    statistic, p_value = chisquare(observed, expected)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Exemple: Test uniformité dé
observed = [18, 16, 15, 17, 16, 18]  # Résultats lancés dé
expected = [16.67] * 6  # Attendu uniforme

result = chi_square_test(observed, expected)
print(f"Chi-square test: {result}")
```

#### Test t de Student

Pour comparer moyennes de deux groupes :

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

où $s_p$ est écart-type poolé.

```python
from scipy.stats import ttest_ind

def t_test_example():
    """Exemple test t"""
    group1 = np.random.normal(10, 2, 30)
    group2 = np.random.normal(12, 2, 30)
    
    statistic, p_value = ttest_ind(group1, group2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'groups_different': p_value < 0.05
    }

t_result = t_test_example()
print(f"t-test result: {t_result}")
```

### Intervalles de Confiance

#### Intervalle de Confiance pour la Moyenne

$$\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

```python
from scipy.stats import t

def confidence_interval(data, confidence=0.95):
    """
    Calcule intervalle de confiance pour moyenne
    
    Args:
        data: Échantillon
        confidence: Niveau de confiance (0.95 pour 95%)
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_critical * std / np.sqrt(n)
    
    return {
        'mean': mean,
        'lower': mean - margin,
        'upper': mean + margin,
        'confidence': confidence
    }

data = np.random.normal(10, 2, 100)
ci = confidence_interval(data)
print(f"95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
```

---

## Exercices

### Exercice A.1
Calculez produit de Kronecker de deux matrices et vérifiez propriétés.

### Exercice A.2
Implémentez fonction qui calcule entropie de Shannon pour distribution discrète.

### Exercice A.3
Calculez KL divergence entre deux distributions gaussiennes.

### Exercice A.4
Effectuez test statistique pour comparer deux échantillons et interprétez résultats.

---

## Références Rapides

### Formules Clés

| Concept | Formule |
|---------|---------|
| Produit tensoriel externe | $v \otimes w = v w^T$ |
| Norme de Frobenius | $||T||_F = \sqrt{\sum T_{i_1...i_d}^2}$ |
| Entropie de Shannon | $H(X) = -\sum p_i \log p_i$ |
| Information mutuelle | $I(X;Y) = H(X) - H(X\|Y)$ |
| KL divergence | $D_{KL}(P\|Q) = \sum p_i \log(p_i/q_i)$ |
| Variance | $\sigma^2 = \mathbb{E}[(X-\mu)^2]$ |

---

*Retour à la [Table des Matières](../../INDEX.md)*
