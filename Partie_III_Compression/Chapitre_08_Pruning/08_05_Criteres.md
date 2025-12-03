# 8.5 Critères de Sélection et Scheduling

---

## Introduction

Le choix du **critère de sélection** (quelle structure pruner) et du **scheduling** (quand pruner) est crucial pour obtenir les meilleures performances après compression.

---

## Critères de Sélection

### Magnitude (Standard)

```python
def magnitude_criterion(weights):
    """
    Critère standard: magnitude des poids
    
    Supprime les poids de faible magnitude
    """
    return weights.abs()

# Utilisation
W = torch.randn(100, 50)
importance = magnitude_criterion(W)
threshold = torch.quantile(importance, 0.9)
mask = (importance >= threshold).float()
```

### Gradient-based

```python
def gradient_criterion(weights, gradients):
    """
    Critère basé sur les gradients
    
    Supprime les poids avec petits gradients (moins importants)
    """
    # Salience = |weight| × |gradient|
    salience = weights.abs() * gradients.abs()
    return salience

def compute_gradient_salience(model, loss_fn, data_loader):
    """
    Calcule la salience gradient pour tous les poids
    """
    model.train()
    salience = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            salience[name] = torch.zeros_like(param)
    
    # Accumule les gradients
    for x, y in data_loader:
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if name in salience and param.grad is not None:
                salience[name] += param.grad.abs()
    
    # Normalise
    for name in salience:
        salience[name] /= len(data_loader)
    
    return salience
```

### Second-order (Hessian)

```python
def hessian_criterion(weights, hessian_diag):
    """
    Critère basé sur la Hessienne (OBD/OBS)
    
    Salience = w² / (2 * H_ii)
    """
    salience = (weights ** 2) / (2 * hessian_diag + 1e-10)
    return salience

def approximate_hessian_diag(model, loss_fn, data_loader):
    """
    Approximation diagonale de la Hessienne via Fisher Information
    H_ii ≈ E[g_i²]
    """
    hessian_diag = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            hessian_diag[name] = torch.zeros_like(param)
    
    model.eval()
    for x, y in data_loader:
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        
        for name, param in model.named_parameters():
            if name in hessian_diag and param.grad is not None:
                hessian_diag[name] += param.grad ** 2
    
    # Normalise
    n = len(data_loader.dataset)
    for name in hessian_diag:
        hessian_diag[name] /= n
    
    return hessian_diag
```

### Activations (APoZ)

```python
def apoz_criterion(activations):
    """
    Average Percentage of Zeros (APoZ)
    
    Pour les neurones/channels avec beaucoup de zéros dans les activations,
    ils peuvent être prunés
    """
    # Pourcentage de zéros par channel
    n_zeros = (activations == 0).sum(dim=(0, 2, 3))  # Pour Conv
    n_total = activations.size(0) * activations.size(2) * activations.size(3)
    apoz = n_zeros.float() / n_total
    
    return apoz

def compute_apoz(model, data_loader):
    """
    Calcule APoZ pour toutes les couches
    """
    apoz_values = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                apoz_values[name] = apoz_criterion(output)
        return hook
    
    # Enregistre les hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            _ = model(x)
            break
    
    # Supprime les hooks
    for hook in hooks:
        hook.remove()
    
    return apoz_values
```

---

## Scheduling de Pruning

### One-Shot vs Iterative

```python
class PruningScheduler:
    """
    Différentes stratégies de scheduling
    """
    
    @staticmethod
    def one_shot(model, sparsity, criterion='magnitude'):
        """
        Pruning en une seule étape
        
        Avantage: Rapide
        Inconvénient: Peut dégrader significativement les performances
        """
        masks = create_masks(model, sparsity, criterion)
        apply_masks(model, masks)
        return model
    
    @staticmethod
    def iterative(model, target_sparsity, n_steps=10, 
                 criterion='magnitude', retrain=True):
        """
        Pruning itératif: prune graduellement
        
        Meilleur pour préserver les performances
        """
        current_sparsity = 0
        sparsity_per_step = target_sparsity / n_steps
        
        for step in range(n_steps):
            current_sparsity += sparsity_per_step
            
            # Prune
            masks = create_masks(model, current_sparsity, criterion)
            apply_masks(model, masks)
            
            # Re-entraîne
            if retrain:
                fine_tune_epoch(model)
        
        return model
    
    @staticmethod
    def gradual(model, target_sparsity, start_epoch=0, 
               end_epoch=50, criterion='magnitude'):
        """
        Pruning graduel pendant l'entraînement
        
        Prune progressivement pendant que le modèle apprend
        """
        def pruning_hook(epoch):
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            progress = max(0, min(1, progress))
            current_sparsity = target_sparsity * progress
            
            masks = create_masks(model, current_sparsity, criterion)
            apply_masks(model, masks)
        
        return pruning_hook
```

### Polishing

```python
def polish_pruning(model, masks, train_fn, n_epochs=10):
    """
    Polishing: re-entraîne après pruning pour "polir" le modèle
    
    Peut récupérer une partie de la performance perdue
    """
    # Applique les masques
    apply_masks(model, masks)
    
    # Re-entraîne
    for epoch in range(n_epochs):
        train_fn(model)
        # Réapplique les masques (pour maintenir la sparsité)
        apply_masks(model, masks)
    
    return model
```

### Regrowth

```python
class RegrowthPruning:
    """
    Pruning avec regrowth: permet aux poids de "repousser"
    
    Combine pruning et regrowth pour explorer l'espace des sous-réseaux
    """
    
    def __init__(self, model, sparsity, regrow_fraction=0.1):
        self.model = model
        self.sparsity = sparsity
        self.regrow_fraction = regrow_fraction
        
    def prune_and_regrow_step(self, criterion_fn):
        """
        Une étape: prune certains poids, fait repousser d'autres
        """
        # 1. Prune les poids les moins importants
        importance = criterion_fn(self.model)
        
        # Calcule le nombre à pruner
        total_params = sum(p.numel() for p in self.model.parameters())
        n_to_prune = int(total_params * self.sparsity)
        n_to_regrow = int(n_to_prune * self.regrow_fraction)
        
        # Prune
        flat_importance = torch.cat([imp.flatten() for imp in importance.values()])
        threshold = torch.topk(flat_importance, n_to_prune, largest=False)[0][-1]
        
        masks = {}
        for name, param in self.model.named_parameters():
            if name in importance:
                mask = (importance[name] > threshold).float()
                masks[name] = mask
        
        # 2. Regrow: réactive certains poids prunés (aléatoirement ou par critère)
        for name, mask in masks.items():
            pruned = (mask == 0).sum().item()
            n_regrow = min(n_to_regrow, pruned)
            
            # Sélectionne aléatoirement des poids à regrow
            pruned_indices = torch.where(mask.flatten() == 0)[0]
            regrow_indices = torch.randperm(len(pruned_indices))[:n_regrow]
            
            # Réactive
            mask_flat = mask.flatten()
            mask_flat[pruned_indices[regrow_indices]] = 1
            masks[name] = mask_flat.view(mask.shape)
        
        return masks
```

---

## Comparaison des Critères

```python
def compare_pruning_criteria(model, train_loader, val_loader, 
                            criterion_fn, sparsity=0.9):
    """
    Compare différents critères de pruning
    """
    criteria = {
        'magnitude': lambda m: {
            n: p.abs() for n, p in m.named_parameters() if 'weight' in n
        },
        'gradient': lambda m: compute_gradient_salience(m, train_loader),
        'hessian': lambda m: {
            n: (p**2) / (2 * h + 1e-10)
            for n, (p, h) in zip(
                [n for n, _ in m.named_parameters() if 'weight' in n],
                zip(
                    [p for n, p in m.named_parameters() if 'weight' in n],
                    approximate_hessian_diag(m, val_loader).values()
                )
            )
        }
    }
    
    results = {}
    
    for criterion_name, criterion_func in criteria.items():
        # Prune
        importance = criterion_func(model)
        masks = create_masks_from_importance(importance, sparsity)
        apply_masks(model, masks)
        
        # Fine-tune
        fine_tune_model(model, train_loader, epochs=5)
        
        # Évalue
        accuracy = evaluate(model, val_loader)
        results[criterion_name] = accuracy
    
    return results
```

---

## Scheduling Adaptatif

```python
class AdaptivePruningScheduler:
    """
    Schedule adaptatif: ajuste la vitesse de pruning selon la perte de performance
    """
    
    def __init__(self, initial_sparsity=0.0, target_sparsity=0.9,
                 max_accuracy_drop=0.02):
        self.current_sparsity = initial_sparsity
        self.target_sparsity = target_sparsity
        self.max_drop = max_accuracy_drop
        self.baseline_accuracy = None
    
    def should_prune_more(self, current_accuracy):
        """
        Détermine si on peut continuer à pruner
        """
        if self.baseline_accuracy is None:
            self.baseline_accuracy = current_accuracy
            return True
        
        accuracy_drop = self.baseline_accuracy - current_accuracy
        
        if accuracy_drop < self.max_drop:
            return True  # On peut continuer
        else:
            return False  # Trop de perte, arrêter ou ralentir
    
    def get_next_sparsity(self, current_accuracy):
        """
        Calcule la prochaine sparsité cible
        """
        if not self.should_prune_more(current_accuracy):
            # Ralentit
            increment = (self.target_sparsity - self.current_sparsity) * 0.5
        else:
            # Vitesse normale
            increment = (self.target_sparsity - self.current_sparsity) / 10
        
        self.current_sparsity = min(
            self.current_sparsity + increment,
            self.target_sparsity
        )
        
        return self.current_sparsity
```

---

## Exercices

### Exercice 8.5.1
Comparez magnitude, gradient et hessian criteria sur un réseau pour CIFAR-10. Lequel préserve le mieux les performances ?

### Exercice 8.5.2
Implémentez un scheduler adaptatif qui ajuste automatiquement la vitesse de pruning selon la perte de performance.

### Exercice 8.5.3
Créez une fonction qui combine plusieurs critères (ex: magnitude + gradient) pour la sélection des poids à pruner.

---

## Points Clés à Retenir

> 📌 **Le critère de magnitude est simple et efficace pour la plupart des cas**

> 📌 **Les critères basés sur gradients/hessienne sont plus coûteux mais plus précis**

> 📌 **Le pruning itératif préserve mieux les performances que le one-shot**

> 📌 **Le scheduling adaptatif permet d'optimiser le compromis vitesse/performance**

---

*Chapitre suivant : [Chapitre 9 - Quantification des Réseaux de Neurones](../Chapitre_09_Quantification/09_introduction.md)*

