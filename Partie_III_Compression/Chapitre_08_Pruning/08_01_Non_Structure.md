# 8.1 Pruning Non Structuré

Ce fichier existe déjà. Voir le contenu complet dans [08_01_Non_Structure.md](./08_01_Non_Structure.md).

---

## 8.1.1 Magnitude Pruning

```python
import torch
import torch.nn as nn
import numpy as np

class MagnitudePruning:
    """
    Pruning basé uniquement sur la magnitude des poids
    """
    
    def __init__(self, model):
        self.model = model
        self.masks = {}
        
    def global_magnitude_pruning(self, sparsity):
        """
        Pruning global: calcule un seuil sur tous les poids
        """
        # Collecte tous les poids
        all_weights = torch.cat([
            param.data.abs().flatten()
            for name, param in self.model.named_parameters()
            if 'weight' in name
        ])
        
        # Seuil global
        threshold = torch.quantile(all_weights, sparsity)
        
        # Crée les masques
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.masks[name] = (param.data.abs() > threshold).float()
        
        return self.masks
    
    def layerwise_magnitude_pruning(self, sparsity_per_layer):
        """
        Pruning par couche: seuil différent pour chaque couche
        """
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if isinstance(sparsity_per_layer, dict):
                    sparsity = sparsity_per_layer.get(name, 0.5)
                else:
                    sparsity = sparsity_per_layer
                
                threshold = torch.quantile(param.data.abs(), sparsity)
                self.masks[name] = (param.data.abs() > threshold).float()
        
        return self.masks
    
    def apply_masks(self):
        """Applique les masques"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

# Comparaison
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.Linear(50, 10)
)

print("Comparaison Global vs Layerwise:")
pruner_global = MagnitudePruning(model)
pruner_layer = MagnitudePruning(model)

pruner_global.global_magnitude_pruning(0.8)
pruner_layer.layerwise_magnitude_pruning(0.8)

print(f"  Global: {sum(m.sum().item() for m in pruner_global.masks.values())} poids actifs")
print(f"  Layerwise: {sum(m.sum().item() for m in pruner_layer.masks.values())} poids actifs")
```

---

## 8.1.2 Gradient-based Pruning

```python
class GradientBasedPruning:
    """
    Pruning utilisant les gradients comme mesure d'importance
    """
    
    def __init__(self, model, train_loader, criterion):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
    
    def compute_gradient_salience(self, n_batches=100):
        """
        Calcule la salience = |w| × |∇w L|
        
        Les poids avec petits gradients sont moins critiques
        """
        self.model.train()
        
        salience = {}
        gradients = {}
        
        # Initialise
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                salience[name] = torch.zeros_like(param)
                gradients[name] = torch.zeros_like(param)
        
        # Accumule les gradients
        count = 0
        for x, y in self.train_loader:
            if count >= n_batches:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if name in gradients and param.grad is not None:
                    gradients[name] += param.grad.abs()
            
            count += len(x)
        
        # Normalise et calcule salience
        for name, param in self.model.named_parameters():
            if name in gradients:
                gradients[name] /= count
                salience[name] = param.data.abs() * gradients[name]
        
        return salience, gradients
    
    def prune_by_salience(self, salience, sparsity):
        """
        Prune basé sur la salience gradient
        """
        # Concatène toutes les saliences
        all_salience = torch.cat([s.flatten() for s in salience.values()])
        
        # Seuil
        threshold = torch.quantile(all_salience, sparsity)
        
        # Masques
        masks = {}
        for name, param in self.model.named_parameters():
            if name in salience:
                masks[name] = (salience[name] > threshold).float()
        
        return masks

# Avantage du gradient-based:
# - Capture mieux l'importance réelle des poids
# - Prend en compte la contribution à la loss
# - Mais nécessite un forward/backward pass (plus coûteux)
```

---

## 8.1.3 Second-order Pruning (OBS, OBD)

```python
class SecondOrderPruning:
    """
    Pruning utilisant les informations de second ordre (Hessienne)
    
    Optimal Brain Surgeon (OBS) et Optimal Brain Damage (OBD)
    """
    
    def compute_optimal_brain_damage(self, model, train_loader, criterion):
        """
        Optimal Brain Damage: utilise la diagonale de la Hessienne
        
        Salience(w) = w² / (2 × H_ii)
        où H_ii est l'élément diagonal de la Hessienne
        """
        # Approximation: H_ii ≈ E[g²] (Fisher Information)
        hessian_diag = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                hessian_diag[name] = torch.zeros_like(param)
        
        model.train()
        for x, y in train_loader:
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            
            for name, param in model.named_parameters():
                if name in hessian_diag and param.grad is not None:
                    hessian_diag[name] += param.grad ** 2
        
        # Normalise
        n = len(train_loader.dataset)
        for name in hessian_diag:
            hessian_diag[name] /= n
        
        # Calcule la salience
        salience = {}
        for name, param in model.named_parameters():
            if name in hessian_diag:
                H = hessian_diag[name] + 1e-10  # Évite division par zéro
                salience[name] = (param.data ** 2) / (2 * H)
        
        return salience, hessian_diag
    
    def compute_optimal_brain_surgeon(self, model, train_loader, criterion):
        """
        Optimal Brain Surgeon: utilise la Hessienne complète (approximation)
        
        Plus précis que OBD mais beaucoup plus coûteux
        """
        # OBS nécessite l'inverse de la Hessienne pour chaque poids
        # Très coûteux, utilise souvent des approximations
        
        # Approximation: utilise seulement la diagonale (comme OBD)
        # pour des raisons de coût computationnel
        
        salience, hessian_diag = self.compute_optimal_brain_damage(
            model, train_loader, criterion
        )
        
        return salience
    
    def iterative_obs_pruning(self, model, train_loader, criterion, 
                             n_weights_to_prune):
        """
        OBS itératif: prune un poids à la fois
        
        Après chaque prune, recalcule la Hessienne (coûteux!)
        """
        pruned_weights = []
        
        for iteration in range(n_weights_to_prune):
            # Calcule la salience
            salience, _ = self.compute_optimal_brain_damage(
                model, train_loader, criterion
            )
            
            # Trouve le poids de plus faible salience
            min_sal = float('inf')
            min_name = None
            min_idx = None
            
            for name, sal in salience.items():
                flat_sal = sal.flatten()
                min_idx_local = flat_sal.argmin().item()
                min_sal_local = flat_sal[min_idx_local].item()
                
                if min_sal_local < min_sal:
                    min_sal = min_sal_local
                    min_name = name
                    min_idx = np.unravel_index(
                        flat_sal.argmin().item(), 
                        sal.shape
                    )
            
            # Prune ce poids
            model.state_dict()[min_name][min_idx] = 0
            pruned_weights.append((min_name, min_idx))
            
            if iteration % 10 == 0:
                print(f"Pruned {iteration+1} weights")
        
        return pruned_weights

# Comparaison OBD vs Magnitude
"""
obd = SecondOrderPruning()
obd_salience, _ = obd.compute_optimal_brain_damage(model, train_loader, criterion)

magnitude_salience = {
    name: param.abs() 
    for name, param in model.named_parameters() 
    if 'weight' in name
}

# Les deux peuvent être utilisées pour le pruning
"""
```

---

## Comparaison des Méthodes

```python
def compare_pruning_methods(model, train_loader, criterion, sparsity=0.9):
    """
    Compare magnitude, gradient et second-order pruning
    """
    results = {}
    
    # 1. Magnitude
    pruner_mag = MagnitudePruning(model)
    masks_mag = pruner_mag.global_magnitude_pruning(sparsity)
    results['magnitude'] = {
        'masks': masks_mag,
        'sparsity': compute_sparsity(masks_mag)
    }
    
    # 2. Gradient-based
    pruner_grad = GradientBasedPruning(model, train_loader, criterion)
    salience, _ = pruner_grad.compute_gradient_salience(n_batches=50)
    masks_grad = pruner_grad.prune_by_salience(salience, sparsity)
    results['gradient'] = {
        'masks': masks_grad,
        'sparsity': compute_sparsity(masks_grad)
    }
    
    # 3. Second-order (OBD)
    pruner_obd = SecondOrderPruning()
    salience_obd, _ = pruner_obd.compute_optimal_brain_damage(
        model, train_loader, criterion
    )
    masks_obd = pruner_grad.prune_by_salience(salience_obd, sparsity)
    results['second_order'] = {
        'masks': masks_obd,
        'sparsity': compute_sparsity(masks_obd)
    }
    
    return results

def compute_sparsity(masks):
    """Calcule la sparsité réelle"""
    total = sum(m.numel() for m in masks.values())
    zeros = sum((m == 0).sum().item() for m in masks.values())
    return zeros / total
```

---

*Voir aussi : [8.2 Pruning Structuré](./08_02_Structure.md)*
