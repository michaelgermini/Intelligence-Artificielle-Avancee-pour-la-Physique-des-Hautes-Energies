# 23.3 Implémentation du Tensor Train

---

## Introduction

Le **Tensor Train (TT)** est une décomposition particulièrement efficace pour les tenseurs de haute dimension. Cette section présente l'implémentation du Tensor Train, incluant l'algorithme de construction, les opérations sur les TT, et les optimisations.

---

## Formulation Mathématique

### Structure Tensor Train

```python
"""
Tensor Train d'un tenseur T de shape (I₁, I₂, ..., Iₙ):

T[i₁, i₂, ..., iₙ] = G₁[i₁] G₂[i₂] ... Gₙ[iₙ]

où Gₖ[iₖ] est une matrice (rₖ₋₁ × rₖ) pour chaque valeur iₖ.

Les cores sont des tenseurs:
- G₁: (I₁ × r₁)
- Gₖ: (rₖ₋₁ × Iₖ × rₖ) pour k = 2, ..., n-1
- Gₙ: (rₙ₋₁ × Iₙ)

Bond dimensions: [1, r₁, r₂, ..., rₙ₋₁, 1]
"""
```

---

## Implémentation Basique

### Construction depuis Tenseur

```python
import numpy as np

class TensorTrain:
    """
    Représentation Tensor Train
    """
    
    def __init__(self, cores=None, ranks=None):
        """
        Args:
            cores: Liste des cores [G₁, G₂, ..., Gₙ]
            ranks: Bond dimensions [1, r₁, r₂, ..., rₙ₋₁, 1]
        """
        self.cores = cores
        self.ranks = ranks if ranks else self._compute_ranks()
    
    def _compute_ranks(self):
        """Calcule bond dimensions depuis cores"""
        if self.cores is None:
            return None
        
        ranks = [1]
        for i in range(len(self.cores) - 1):
            ranks.append(self.cores[i].shape[-1])
        ranks.append(1)
        return ranks
    
    def from_tensor(self, tensor, max_rank=None, eps=1e-6):
        """
        Construit TT depuis tenseur avec SVD
        
        Args:
            tensor: Tenseur à décomposer
            max_rank: Rank maximum (truncation)
            eps: Tolérance pour truncation
        """
        shape = tensor.shape
        n_modes = len(shape)
        
        cores = []
        remaining = tensor
        
        for mode in range(n_modes - 1):
            # Reshape en matrice
            I_current = shape[mode]
            I_remaining = np.prod(shape[mode+1:])
            
            matrix = remaining.reshape(I_current, I_remaining)
            
            # SVD
            U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            # Truncation
            if max_rank:
                rank = min(max_rank, len(s))
            else:
                # Truncation basée sur eps
                cumsum_sq = np.cumsum(s**2)
                total = cumsum_sq[-1]
                rank = np.searchsorted(cumsum_sq, total * (1 - eps)) + 1
                rank = min(rank, len(s))
            
            # Prendre r premières composantes
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
            
            # Core actuel
            core = U.reshape(I_current, rank)
            cores.append(core)
            
            # Préparer pour prochaine itération
            remaining = (np.diag(s) @ Vt).reshape(rank, *shape[mode+1:])
        
        # Dernier core
        cores.append(remaining.reshape(remaining.shape[0], remaining.shape[1]))
        
        self.cores = cores
        self.ranks = self._compute_ranks()
        
        return self
    
    def to_tensor(self):
        """Reconstruit tenseur depuis TT"""
        if self.cores is None:
            raise ValueError("No cores available")
        
        result = self.cores[0]
        
        for core in self.cores[1:]:
            # Contracter: result (..., r) avec core (r, I, r')
            # Résultat: (..., I, r')
            
            # Reshape pour contraction
            result = np.tensordot(result, core, axes=([-1], [0]))
        
        # Squeeze dimensions unitaires
        result = np.squeeze(result)
        
        return result
    
    def numel(self):
        """Nombre de paramètres dans TT"""
        if self.cores is None:
            return 0
        
        total = 0
        for core in self.cores:
            total += core.size
        return total

# Test
tt = TensorTrain()
tensor = np.random.randn(5, 6, 7, 8)
tt.from_tensor(tensor, max_rank=5)

print(f"Tenseur original: {tensor.size} paramètres")
print(f"Tensor Train: {tt.numel()} paramètres")
print(f"Compression: {tensor.size / tt.numel():.2f}×")

reconstructed = tt.to_tensor()
error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
print(f"Erreur relative: {error:.6f}")
```

---

## Opérations sur Tensor Train

### Addition et Multiplication

```python
class TensorTrainOperations(TensorTrain):
    """
    Opérations sur Tensor Trains
    """
    
    def add(self, other):
        """Addition de deux Tensor Trains"""
        if len(self.cores) != len(other.cores):
            raise ValueError("TT must have same number of modes")
        
        new_cores = []
        for i, (core1, core2) in enumerate(zip(self.cores, other.cores)):
            if i == 0:
                # Premier core: concaténer colonnes
                new_core = np.concatenate([core1, core2], axis=1)
            elif i == len(self.cores) - 1:
                # Dernier core: concaténer lignes
                new_core = np.concatenate([core1, core2], axis=0)
            else:
                # Cores intermédiaires: blocs diagonaux
                r1_prev, I, r1_next = core1.shape
                r2_prev, _, r2_next = core2.shape
                
                # Créer bloc diagonal
                new_core = np.zeros((r1_prev + r2_prev, I, r1_next + r2_next))
                new_core[:r1_prev, :, :r1_next] = core1
                new_core[r1_prev:, :, r1_next:] = core2
                
                new_cores.append(new_core)
        
        return TensorTrain(cores=new_cores)
    
    def multiply_scalar(self, scalar):
        """Multiplication par scalaire"""
        new_cores = [core.copy() for core in self.cores]
        # Multiplier premier ou dernier core
        new_cores[0] *= scalar
        return TensorTrain(cores=new_cores)
    
    def round(self, eps=1e-6, max_rank=None):
        """Compression du TT"""
        # Réappliquer SVD sur chaque liaison
        # Simplifié: reconstruire et redécomposer
        tensor = self.to_tensor()
        compressed = TensorTrain()
        compressed.from_tensor(tensor, max_rank=max_rank, eps=eps)
        return compressed

# Test opérations
tt1 = TensorTrain()
tt1.from_tensor(np.random.randn(5, 6, 7), max_rank=3)

tt2 = TensorTrain()
tt2.from_tensor(np.random.randn(5, 6, 7), max_rank=3)

tt_sum = tt1.add(tt2)
print(f"TT1: {tt1.numel()} paramètres")
print(f"TT2: {tt2.numel()} paramètres")
print(f"TT sum: {tt_sum.numel()} paramètres")
```

---

## Version PyTorch

### Support GPU et Gradients

```python
import torch

class PyTorchTensorTrain:
    """
    Tensor Train avec PyTorch
    """
    
    def __init__(self, cores=None, device='cpu'):
        self.cores = cores
        self.device = device
    
    def from_tensor(self, tensor, max_rank=None, eps=1e-6):
        """Construit TT depuis tenseur PyTorch"""
        tensor = tensor.to(self.device)
        shape = tensor.shape
        n_modes = len(shape)
        
        cores = []
        remaining = tensor
        
        for mode in range(n_modes - 1):
            I_current = shape[mode]
            I_remaining = torch.prod(torch.tensor(shape[mode+1:]))
            
            matrix = remaining.reshape(I_current, int(I_remaining))
            
            # SVD
            U, s, Vt = torch.linalg.svd(matrix, full_matrices=False)
            
            # Truncation
            if max_rank:
                rank = min(max_rank, len(s))
            else:
                cumsum_sq = torch.cumsum(s**2, dim=0)
                total = cumsum_sq[-1]
                rank = torch.searchsorted(cumsum_sq, total * (1 - eps)) + 1
                rank = min(int(rank), len(s))
            
            U = U[:, :rank]
            s = s[:rank]
            Vt = Vt[:rank, :]
            
            core = U.reshape(I_current, rank)
            cores.append(core)
            
            remaining = (torch.diag(s) @ Vt).reshape(rank, *shape[mode+1:])
        
        cores.append(remaining.reshape(remaining.shape[0], remaining.shape[1]))
        
        self.cores = cores
        return self
    
    def to_tensor(self):
        """Reconstruit tenseur"""
        result = self.cores[0]
        
        for core in self.cores[1:]:
            result = torch.tensordot(result, core, dims=([-1], [0]))
        
        return result.squeeze()
    
    def numel(self):
        """Nombre de paramètres"""
        return sum(core.numel() for core in self.cores)

# Test PyTorch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tt_torch = PyTorchTensorTrain(device=device)

tensor_torch = torch.randn(5, 6, 7, 8, device=device, requires_grad=True)
tt_torch.from_tensor(tensor_torch, max_rank=5)

print(f"TT PyTorch: {tt_torch.numel()} paramètres")
print(f"Compression: {tensor_torch.numel() / tt_torch.numel():.2f}×")

# Gradients
reconstructed = tt_torch.to_tensor()
loss = reconstructed.sum()
loss.backward()
print(f"Gradients calculés pour cores")
```

---

## Exercices

### Exercice 23.3.1
Implémentez construction TT depuis tenseur avec SVD et testez sur tenseurs de différentes dimensions.

### Exercice 23.3.2
Comparez compression ratio vs erreur pour différents max_rank.

### Exercice 23.3.3
Implémentez opérations (addition, multiplication) sur Tensor Trains.

### Exercice 23.3.4
Utilisez version PyTorch pour entraîner modèle avec contraintes TT.

---

## Points Clés à Retenir

> 📌 **Tensor Train construit via SVD séquentiel**

> 📌 **Truncation contrôle trade-off compression/précision**

> 📌 **Opérations sur TT peuvent être faites sans reconstruire**

> 📌 **Support PyTorch permet gradients et GPU**

> 📌 **TT est particulièrement efficace pour haute dimension**

---

*Section précédente : [23.2 Décomposition CP](./23_02_Decomposition_CP.md) | Section suivante : [23.4 Optimisation](./23_04_Optimisation.md)*

