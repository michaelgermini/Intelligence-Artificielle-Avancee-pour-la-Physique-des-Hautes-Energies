# 23.2 Implémentation de la Décomposition CP

---

## Introduction

La **décomposition CP** (Canonical Polyadic / PARAFAC) représente un tenseur comme somme de produits tensoriels de vecteurs. Cette section présente l'implémentation manuelle de la décomposition CP, incluant l'algorithme ALS (Alternating Least Squares) et ses variantes.

---

## Formulation Mathématique

### Définition

```python
"""
Décomposition CP d'un tenseur T de shape (I₁, I₂, ..., Iₙ):

T ≈ Σᵣ₌₁ᴿ λᵣ · aᵣ⁽¹⁾ ○ aᵣ⁽²⁾ ○ ... ○ aᵣ⁽ⁿ⁾

où:
- R est le rank CP
- λᵣ sont les poids (optionnel)
- aᵣ⁽ᵏ⁾ sont les facteurs (vecteurs)
- ○ est le produit tensoriel externe
"""
```

---

## Implémentation Basique

### Structure de Base

```python
import numpy as np
import torch

class CPDecoposition:
    """
    Décomposition CP d'un tenseur
    """
    
    def __init__(self, rank, use_weights=True):
        """
        Args:
            rank: Rank de la décomposition CP
            use_weights: Si True, inclut poids λ
        """
        self.rank = rank
        self.use_weights = use_weights
        self.factors = None
        self.weights = None
    
    def decompose(self, tensor, n_iter=100, tol=1e-6, verbose=False):
        """
        Décompose tenseur avec ALS
        
        Args:
            tensor: Tenseur à décomposer (numpy ou torch)
            n_iter: Nombre maximum d'itérations
            tol: Tolérance pour convergence
            verbose: Afficher progression
        """
        self.tensor_shape = tensor.shape
        self.n_modes = len(tensor.shape)
        
        # Initialiser facteurs aléatoirement
        self.factors = self._initialize_factors()
        
        if self.use_weights:
            self.weights = np.ones(self.rank)
        
        # ALS iterations
        errors = []
        for iteration in range(n_iter):
            # Mettre à jour chaque facteur
            for mode in range(self.n_modes):
                self._update_factor(tensor, mode)
            
            # Calculer erreur
            reconstructed = self.reconstruct()
            error = np.linalg.norm(tensor - reconstructed)
            errors.append(error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Error = {error:.6f}")
            
            # Vérifier convergence
            if iteration > 0 and abs(errors[-2] - errors[-1]) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration+1}")
                break
        
        return errors
    
    def _initialize_factors(self):
        """Initialise facteurs aléatoirement"""
        factors = []
        for mode in range(self.n_modes):
            factor = np.random.randn(self.tensor_shape[mode], self.rank)
            factors.append(factor)
        return factors
    
    def _update_factor(self, tensor, mode):
        """Met à jour facteur pour mode donné (ALS)"""
        # Khatri-Rao product des autres facteurs
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_product(other_factors)
        
        # Unfold tensor le long du mode
        unfolded = self._unfold(tensor, mode)
        
        # Résoudre système linéaire
        # unfolded ≈ factor @ kr_product.T
        # factor = unfolded @ (kr_product.T)^+
        kr_pseudo_inv = np.linalg.pinv(kr_product.T)
        self.factors[mode] = unfolded @ kr_pseudo_inv
    
    def _khatri_rao_product(self, matrices):
        """
        Produit de Khatri-Rao (colonne-wise Kronecker)
        
        Pour matrices A (I×R) et B (J×R):
        KR = [A[:,0] ⊗ B[:,0], ..., A[:,R-1] ⊗ B[:,R-1]]
        """
        if len(matrices) == 1:
            return matrices[0]
        
        result = matrices[0]
        for mat in matrices[1:]:
            # Produit de Khatri-Rao
            I, R = result.shape
            J, R2 = mat.shape
            
            assert R == R2, "Matrices must have same number of columns"
            
            # Calcul colonne par colonne
            kr_cols = []
            for r in range(R):
                kr_cols.append(np.kron(result[:, r], mat[:, r]))
            result = np.column_stack(kr_cols)
        
        return result
    
    def _unfold(self, tensor, mode):
        """Unfold tenseur le long d'un mode"""
        shape = tensor.shape
        n_modes = len(shape)
        
        # Permuter pour mettre mode en première position
        perm = list(range(n_modes))
        perm[0], perm[mode] = perm[mode], perm[0]
        tensor_perm = np.transpose(tensor, perm)
        
        # Reshape
        I_mode = shape[mode]
        I_other = np.prod([shape[i] for i in range(n_modes) if i != mode])
        
        return tensor_perm.reshape(I_mode, I_other)
    
    def reconstruct(self):
        """Reconstruit tenseur depuis facteurs"""
        if self.factors is None:
            raise ValueError("Must decompose first")
        
        # Produit tensoriel externe pour chaque composante
        components = []
        for r in range(self.rank):
            component = self.factors[0][:, r]
            for mode in range(1, self.n_modes):
                component = np.outer(component, self.factors[mode][:, r])
            
            if self.use_weights:
                component = self.weights[r] * component
            
            components.append(component)
        
        # Somme des composantes
        reconstructed = np.sum(components, axis=0)
        return reconstructed

# Test
cp = CPDecoposition(rank=5, use_weights=True)
tensor = np.random.randn(10, 15, 20)
errors = cp.decompose(tensor, n_iter=50, verbose=True)

reconstructed = cp.reconstruct()
final_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
print(f"\nErreur finale relative: {final_error:.6f}")
```

---

## Optimisations

### Améliorations Performance

```python
class OptimizedCPDecomposition(CPDecoposition):
    """
    Version optimisée avec améliorations
    """
    
    def _update_factor_optimized(self, tensor, mode):
        """
        Version optimisée avec calcul direct
        """
        # Calcul direct sans pseudo-inverse (plus rapide)
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_product(other_factors)
        
        unfolded = self._unfold(tensor, mode)
        
        # Système normal: A^T A x = A^T b
        A = kr_product.T
        b = unfolded.T
        
        # Gram matrix
        gram = A @ A.T
        
        # Regularization pour stabilité
        gram += 1e-8 * np.eye(gram.shape[0])
        
        # Résoudre
        rhs = A @ b
        self.factors[mode] = np.linalg.solve(gram, rhs).T
    
    def _update_with_regularization(self, tensor, mode, reg=1e-6):
        """Mise à jour avec régularisation"""
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_product(other_factors)
        unfolded = self._unfold(tensor, mode)
        
        # Régularisation L2
        gram = kr_product.T @ kr_product
        gram += reg * np.eye(gram.shape[0])
        
        rhs = kr_product.T @ unfolded.T
        self.factors[mode] = np.linalg.solve(gram, rhs).T
```

---

## Version PyTorch

### Avec Support GPU

```python
import torch

class PyTorchCPDecomposition:
    """
    Décomposition CP avec PyTorch (support GPU)
    """
    
    def __init__(self, rank, device='cpu'):
        self.rank = rank
        self.device = device
        self.factors = None
    
    def decompose(self, tensor, n_iter=100, tol=1e-6):
        """Décompose avec PyTorch"""
        tensor = tensor.to(self.device)
        self.tensor_shape = tensor.shape
        self.n_modes = len(tensor.shape)
        
        # Initialiser facteurs
        self.factors = [
            torch.randn(size, self.rank, device=self.device, requires_grad=False)
            for size in self.tensor_shape
        ]
        
        errors = []
        for iteration in range(n_iter):
            for mode in range(self.n_modes):
                self._update_factor_torch(tensor, mode)
            
            reconstructed = self.reconstruct()
            error = torch.norm(tensor - reconstructed).item()
            errors.append(error)
            
            if iteration > 0 and abs(errors[-2] - errors[-1]) < tol:
                break
        
        return errors
    
    def _update_factor_torch(self, tensor, mode):
        """Mise à jour avec PyTorch"""
        other_factors = [self.factors[i] for i in range(self.n_modes) if i != mode]
        kr_product = self._khatri_rao_torch(other_factors)
        
        unfolded = self._unfold_torch(tensor, mode)
        
        gram = torch.matmul(kr_product.T, kr_product)
        gram += 1e-8 * torch.eye(gram.shape[0], device=self.device)
        
        rhs = torch.matmul(kr_product.T, unfolded.T)
        self.factors[mode] = torch.linalg.solve(gram, rhs).T
    
    def _khatri_rao_torch(self, matrices):
        """Produit Khatri-Rao avec PyTorch"""
        if len(matrices) == 1:
            return matrices[0]
        
        result = matrices[0]
        for mat in matrices[1:]:
            I, R = result.shape
            J, R2 = mat.shape
            
            kr_cols = []
            for r in range(R):
                kr_cols.append(torch.kron(result[:, r], mat[:, r]))
            result = torch.stack(kr_cols, dim=1)
        
        return result
    
    def _unfold_torch(self, tensor, mode):
        """Unfold avec PyTorch"""
        shape = tensor.shape
        n_modes = len(shape)
        
        perm = list(range(n_modes))
        perm[0], perm[mode] = perm[mode], perm[0]
        tensor_perm = tensor.permute(*perm)
        
        I_mode = shape[mode]
        I_other = torch.prod(torch.tensor([shape[i] for i in range(n_modes) if i != mode]))
        
        return tensor_perm.reshape(I_mode, int(I_other))
    
    def reconstruct(self):
        """Reconstruit avec PyTorch"""
        reconstructed = torch.zeros(self.tensor_shape, device=self.device)
        
        for r in range(self.rank):
            component = self.factors[0][:, r]
            for mode in range(1, self.n_modes):
                component = torch.outer(component, self.factors[mode][:, r])
            
            reconstructed += component
        
        return reconstructed

# Test avec GPU si disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

cp_torch = PyTorchCPDecomposition(rank=5, device=device)
tensor_torch = torch.randn(10, 15, 20, device=device)
errors_torch = cp_torch.decompose(tensor_torch, n_iter=50)

print(f"Erreur finale: {errors_torch[-1]:.6f}")
```

---

## Exercices

### Exercice 23.2.1
Implémentez décomposition CP complète avec ALS et testez sur tenseur 3D et 4D.

### Exercice 23.2.2
Comparez performance entre version NumPy et PyTorch de décomposition CP.

### Exercice 23.2.3
Ajoutez régularisation et observons impact sur stabilité et qualité décomposition.

### Exercice 23.2.4
Implémentez version avec poids λ et comparons avec version sans poids.

---

## Points Clés à Retenir

> 📌 **ALS est algorithme standard pour décomposition CP**

> 📌 **Khatri-Rao product est opération clé pour mise à jour facteurs**

> 📌 **Régularisation améliore stabilité convergence**

> 📌 **PyTorch permet utilisation GPU pour accélération**

> 📌 **Initialisation influence qualité décomposition finale**

---

*Section précédente : [23.1 Bibliothèques](./23_01_Bibliotheques.md) | Section suivante : [23.3 Tensor Train](./23_03_Tensor_Train.md)*

