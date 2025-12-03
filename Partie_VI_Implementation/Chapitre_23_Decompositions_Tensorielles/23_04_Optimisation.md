# 23.4 Optimisation et Convergence

---

## Introduction

L'optimisation des algorithmes de décomposition tensorielle est cruciale pour obtenir de bonnes performances en pratique. Cette section couvre les techniques d'optimisation, la gestion de la convergence, et les stratégies pour améliorer stabilité et vitesse.

---

## Problèmes de Convergence

### Difficultés Courantes

```python
"""
Problèmes fréquents:

1. Convergence lente
   - ALS peut converger lentement
   - Nécessite beaucoup d'itérations

2. Convergence vers minima locaux
   - Fonction non-convexe
   - Dépendance initialisation

3. Instabilité numérique
   - Matrices mal conditionnées
   - Overflow/underflow

4. Rank sous-estimé
   - Choix rank trop petit
   - Perte d'information
"""
```

---

## Amélioration de la Convergence

### Initialisation Intelligente

```python
import numpy as np

class ImprovedCPDecomposition:
    """
    CP avec initialisation améliorée
    """
    
    def __init__(self, rank):
        self.rank = rank
    
    def _initialize_svd(self, tensor):
        """Initialisation basée sur SVD"""
        factors = []
        n_modes = len(tensor.shape)
        
        for mode in range(n_modes):
            # Unfold le long du mode
            unfolded = self._unfold(tensor, mode)
            
            # SVD
            U, s, Vt = np.linalg.svd(unfolded, full_matrices=False)
            
            # Prendre r premières colonnes
            factor = U[:, :self.rank]
            
            # Normaliser
            factor = factor / (np.linalg.norm(factor, axis=0) + 1e-8)
            
            factors.append(factor)
        
        return factors
    
    def _unfold(self, tensor, mode):
        """Unfold tenseur"""
        shape = tensor.shape
        n_modes = len(shape)
        
        perm = list(range(n_modes))
        perm[0], perm[mode] = perm[mode], perm[0]
        tensor_perm = np.transpose(tensor, perm)
        
        I_mode = shape[mode]
        I_other = np.prod([shape[i] for i in range(n_modes) if i != mode])
        
        return tensor_perm.reshape(I_mode, int(I_other))

# Comparaison initialisation
def compare_initializations(tensor, rank=5, n_iter=50):
    """Compare initialisations aléatoire vs SVD"""
    
    # Aléatoire
    cp_random = CPDecoposition(rank)
    cp_random.factors = cp_random._initialize_factors()
    errors_random = cp_random.decompose(tensor, n_iter=n_iter, verbose=False)
    
    # SVD
    cp_svd = ImprovedCPDecomposition(rank)
    cp_svd.tensor_shape = tensor.shape
    cp_svd.n_modes = len(tensor.shape)
    cp_svd.factors = cp_svd._initialize_svd(tensor)
    
    cp_svd.decompose(tensor, n_iter=n_iter, verbose=False)
    
    print(f"Initialisation aléatoire - Erreur finale: {errors_random[-1]:.6f}")
    print(f"Initialisation SVD - Erreur finale: {cp_svd.reconstruct()}")
    
    return errors_random
```

---

## Régularisation

### Techniques de Régularisation

```python
class RegularizedCPDecomposition(CPDecoposition):
    """
    CP avec régularisation
    """
    
    def __init__(self, rank, reg_l2=1e-6, reg_sparse=0.0):
        super().__init__(rank)
        self.reg_l2 = reg_l2
        self.reg_sparse = reg_sparse
    
    def _update_factor_regularized(self, tensor, mode):
        """Mise à jour avec régularisation"""
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_product(other_factors)
        unfolded = self._unfold(tensor, mode)
        
        # Gram matrix avec régularisation L2
        gram = kr_product.T @ kr_product
        gram += self.reg_l2 * np.eye(gram.shape[0])
        
        # Résoudre
        rhs = kr_product.T @ unfolded.T
        factor = np.linalg.solve(gram, rhs).T
        
        # Régularisation sparse (L1)
        if self.reg_sparse > 0:
            factor = self._soft_threshold(factor, self.reg_sparse)
        
        self.factors[mode] = factor
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding pour L1"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
```

---

## Line Search et Learning Rate

### Adaptation du Pas

```python
class AdaptiveCPDecomposition(CPDecoposition):
    """
    CP avec adaptive learning rate
    """
    
    def __init__(self, rank, lr_init=1.0, lr_decay=0.95):
        super().__init__(rank)
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.lr = lr_init
    
    def _update_factor_adaptive(self, tensor, mode):
        """Mise à jour avec learning rate adaptatif"""
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_product(other_factors)
        unfolded = self._unfold(tensor, mode)
        
        # Calcul direction
        gram = kr_product.T @ kr_product
        gram += 1e-8 * np.eye(gram.shape[0])
        rhs = kr_product.T @ unfolded.T
        direction = np.linalg.solve(gram, rhs).T - self.factors[mode]
        
        # Line search
        best_lr = self._line_search(tensor, mode, direction)
        
        # Mise à jour
        self.factors[mode] += best_lr * direction
        
        # Décroissance LR
        self.lr *= self.lr_decay
    
    def _line_search(self, tensor, mode, direction, n_steps=10):
        """Line search pour meilleur learning rate"""
        lrs = np.linspace(0, self.lr * 2, n_steps)
        best_lr = 0
        best_error = float('inf')
        
        original_factor = self.factors[mode].copy()
        
        for lr in lrs:
            self.factors[mode] = original_factor + lr * direction
            reconstructed = self.reconstruct()
            error = np.linalg.norm(tensor - reconstructed)
            
            if error < best_error:
                best_error = error
                best_lr = lr
        
        return best_lr
```

---

## Monitoring Convergence

### Métriques et Critères

```python
class ConvergenceMonitor:
    """
    Monitoring de convergence
    """
    
    def __init__(self, tol=1e-6, patience=10, min_improvement=1e-8):
        self.tol = tol
        self.patience = patience
        self.min_improvement = min_improvement
        self.errors = []
        self.best_error = float('inf')
        self.patience_counter = 0
    
    def update(self, error):
        """Met à jour avec nouvelle erreur"""
        self.errors.append(error)
        
        improvement = self.best_error - error
        
        if improvement > self.min_improvement:
            self.best_error = error
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def should_stop(self):
        """Vérifie si doit arrêter"""
        # Critère 1: Tolérance absolue
        if len(self.errors) > 1:
            relative_improvement = abs(self.errors[-2] - self.errors[-1]) / (self.errors[-2] + 1e-10)
            if relative_improvement < self.tol:
                return True
        
        # Critère 2: Patience (pas d'amélioration)
        if self.patience_counter >= self.patience:
            return True
        
        return False
    
    def get_stats(self):
        """Retourne statistiques"""
        if len(self.errors) < 2:
            return {}
        
        return {
            'n_iterations': len(self.errors),
            'final_error': self.errors[-1],
            'best_error': self.best_error,
            'initial_error': self.errors[0],
            'improvement': self.errors[0] - self.errors[-1],
            'relative_improvement': (self.errors[0] - self.errors[-1]) / self.errors[0],
            'converged': self.should_stop()
        }

# Utilisation
monitor = ConvergenceMonitor(tol=1e-6, patience=10)
cp = CPDecoposition(rank=5)

for iteration in range(100):
    # ... update factors ...
    reconstructed = cp.reconstruct()
    error = np.linalg.norm(tensor - reconstructed)
    
    monitor.update(error)
    
    if monitor.should_stop():
        print(f"Stopped at iteration {iteration}")
        break

stats = monitor.get_stats()
print(f"Convergence stats: {stats}")
```

---

## Exercices

### Exercice 23.4.1
Implémentez initialisation SVD et comparez convergence vs initialisation aléatoire.

### Exercice 23.4.2
Testez impact de différentes valeurs de régularisation sur stabilité et qualité.

### Exercice 23.4.3
Implémentez line search et comparez avec pas fixe.

### Exercice 23.4.4
Créez système de monitoring convergence avec critères multiples.

---

## Points Clés à Retenir

> 📌 **L'initialisation influence fortement qualité et vitesse convergence**

> 📌 **Régularisation améliore stabilité mais peut réduire précision**

> 📌 **Learning rate adaptatif peut accélérer convergence**

> 📌 **Monitoring approprié permet détecter convergence et problèmes**

> 📌 **Critères d'arrêt multiples améliorent robustesse**

---

*Section précédente : [23.3 Tensor Train](./23_03_Tensor_Train.md) | Section suivante : [23.5 Intégration PyTorch](./23_05_Integration_PyTorch.md)*

