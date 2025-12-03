# Chapitre 8 : Techniques de Pruning (Élagage)

---

## Introduction

Le **pruning** (élagage) consiste à supprimer les connexions ou neurones les moins importants d'un réseau de neurones. Cette technique peut réduire drastiquement la taille et le coût computationnel des modèles tout en préservant leurs performances.

---

## Plan du Chapitre

1. [Pruning Non Structuré](./08_01_Non_Structure.md)
2. [Pruning Structuré](./08_02_Structure.md)
3. [Pruning Dynamique et Adaptatif](./08_03_Dynamique.md)
4. [Lottery Ticket Hypothesis](./08_04_Lottery_Ticket.md)
5. [Critères de Sélection et Scheduling](./08_05_Criteres.md)

---

## Types de Pruning

```
┌─────────────────────────────────────────────────────────────────┐
│                    Taxonomie du Pruning                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NON STRUCTURÉ (Fine-grained)                                  │
│  ├── Magnitude pruning                                         │
│  ├── Gradient-based pruning                                    │
│  └── Second-order pruning (OBS, OBD)                          │
│                                                                 │
│  STRUCTURÉ (Coarse-grained)                                    │
│  ├── Filter pruning (CNN)                                      │
│  ├── Channel pruning                                           │
│  ├── Head pruning (Transformers)                               │
│  └── Layer pruning                                             │
│                                                                 │
│  DYNAMIQUE                                                      │
│  ├── Pruning pendant l'inférence                               │
│  └── Input-dependent pruning                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pruning par Magnitude

```python
import torch
import torch.nn as nn
import numpy as np

class MagnitudePruning:
    """
    Pruning basé sur la magnitude des poids
    
    Hypothèse: les poids de faible magnitude contribuent peu
    """
    
    def __init__(self, model):
        self.model = model
        self.masks = {}
        
    def compute_threshold(self, sparsity):
        """
        Calcule le seuil pour atteindre une sparsité donnée
        """
        all_weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.abs().flatten())
        
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, sparsity)
        
        return threshold
    
    def create_masks(self, sparsity):
        """
        Crée les masques de pruning
        """
        threshold = self.compute_threshold(sparsity)
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = (param.data.abs() >= threshold).float()
                self.masks[name] = mask
        
        return self.masks
    
    def apply_masks(self):
        """
        Applique les masques aux poids
        """
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def get_sparsity(self):
        """
        Calcule la sparsité réelle du modèle
        """
        total = 0
        zeros = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total += param.numel()
                zeros += (param.data == 0).sum().item()
        
        return zeros / total

# Exemple d'utilisation
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

pruner = MagnitudePruning(model)
pruner.create_masks(sparsity=0.9)  # 90% des poids à zéro
pruner.apply_masks()

print(f"Sparsité atteinte: {pruner.get_sparsity():.1%}")
```

---

## Pruning Structuré

```python
class StructuredPruning:
    """
    Pruning structuré: supprime des structures entières (filtres, neurones)
    
    Avantage: accélération réelle sans hardware spécialisé
    """
    
    @staticmethod
    def prune_filters(conv_layer, n_filters_to_keep, criterion='l1'):
        """
        Élague des filtres entiers d'une couche conv
        """
        weight = conv_layer.weight.data
        n_filters = weight.shape[0]
        
        # Calcul de l'importance de chaque filtre
        if criterion == 'l1':
            importance = weight.abs().sum(dim=(1, 2, 3))
        elif criterion == 'l2':
            importance = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
        
        # Garde les filtres les plus importants
        _, indices = torch.topk(importance, n_filters_to_keep)
        indices = indices.sort()[0]
        
        # Crée une nouvelle couche
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            n_filters_to_keep,
            conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None
        )
        
        new_conv.weight.data = weight[indices]
        if conv_layer.bias is not None:
            new_conv.bias.data = conv_layer.bias.data[indices]
        
        return new_conv, indices
    
    @staticmethod
    def prune_neurons(linear_layer, n_neurons_to_keep, criterion='l1'):
        """
        Élague des neurones entiers d'une couche linéaire
        """
        weight = linear_layer.weight.data
        
        # Importance basée sur les poids sortants
        if criterion == 'l1':
            importance = weight.abs().sum(dim=1)
        elif criterion == 'l2':
            importance = weight.pow(2).sum(dim=1).sqrt()
        
        # Garde les neurones les plus importants
        _, indices = torch.topk(importance, n_neurons_to_keep)
        indices = indices.sort()[0]
        
        # Nouvelle couche
        new_linear = nn.Linear(
            linear_layer.in_features,
            n_neurons_to_keep,
            bias=linear_layer.bias is not None
        )
        
        new_linear.weight.data = weight[indices]
        if linear_layer.bias is not None:
            new_linear.bias.data = linear_layer.bias.data[indices]
        
        return new_linear, indices

# Exemple
conv = nn.Conv2d(64, 128, 3, padding=1)
new_conv, kept_indices = StructuredPruning.prune_filters(conv, 64)

print(f"Filtres: {conv.weight.shape[0]} → {new_conv.weight.shape[0]}")
print(f"Paramètres: {conv.weight.numel():,} → {new_conv.weight.numel():,}")
```

---

## Iterative Pruning avec Fine-tuning

```python
class IterativePruning:
    """
    Pruning itératif: prune graduellement avec fine-tuning
    
    Plus stable que le pruning one-shot pour des sparsités élevées
    """
    
    def __init__(self, model, train_loader, val_loader, criterion):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        
    def prune_and_finetune(self, 
                           target_sparsity, 
                           n_iterations=10,
                           finetune_epochs=5):
        """
        Pruning itératif avec fine-tuning entre chaque étape
        """
        current_sparsity = 0
        sparsity_per_iter = target_sparsity / n_iterations
        
        history = {'sparsity': [], 'accuracy': []}
        
        for iteration in range(n_iterations):
            current_sparsity += sparsity_per_iter
            
            # Prune
            pruner = MagnitudePruning(self.model)
            pruner.create_masks(current_sparsity)
            pruner.apply_masks()
            
            # Fine-tune
            self._finetune(finetune_epochs, pruner.masks)
            
            # Évalue
            accuracy = self._evaluate()
            
            history['sparsity'].append(current_sparsity)
            history['accuracy'].append(accuracy)
            
            print(f"Iteration {iteration+1}: "
                  f"Sparsity={current_sparsity:.1%}, Accuracy={accuracy:.2%}")
        
        return history
    
    def _finetune(self, epochs, masks):
        """Fine-tune le modèle en gardant les masques"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                
                # Applique les masques aux gradients
                for name, param in self.model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad *= masks[name]
                
                optimizer.step()
                
                # Réapplique les masques aux poids
                for name, param in self.model.named_parameters():
                    if name in masks:
                        param.data *= masks[name]
    
    def _evaluate(self):
        """Évalue la précision"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total
```

---

## Lottery Ticket Hypothesis

```python
class LotteryTicketFinder:
    """
    Implémentation de la Lottery Ticket Hypothesis
    
    Hypothèse: Un réseau dense contient un sous-réseau sparse
    qui, entraîné depuis les mêmes poids initiaux, atteint
    une performance comparable.
    """
    
    def __init__(self, model_fn):
        self.model_fn = model_fn
        
    def find_winning_ticket(self, 
                           train_fn,
                           target_sparsity=0.9,
                           n_rounds=10):
        """
        Trouve un "winning ticket" par pruning itératif
        """
        # Crée le modèle et sauvegarde les poids initiaux
        model = self.model_fn()
        initial_weights = {name: param.clone() 
                          for name, param in model.named_parameters()}
        
        masks = {name: torch.ones_like(param) 
                for name, param in model.named_parameters() 
                if 'weight' in name}
        
        sparsity_per_round = 1 - (1 - target_sparsity) ** (1 / n_rounds)
        
        for round_idx in range(n_rounds):
            # Entraîne le modèle
            train_fn(model)
            
            # Prune les poids de plus faible magnitude
            for name, param in model.named_parameters():
                if name in masks:
                    # Poids actuels masqués
                    masked_weights = param.data * masks[name]
                    
                    # Seuil pour ce round
                    nonzero = masked_weights[masks[name] == 1]
                    threshold = torch.quantile(nonzero.abs(), sparsity_per_round)
                    
                    # Met à jour le masque
                    new_mask = (masked_weights.abs() >= threshold).float()
                    masks[name] *= new_mask
            
            # Réinitialise aux poids initiaux
            for name, param in model.named_parameters():
                if name in initial_weights:
                    param.data = initial_weights[name].clone()
                    if name in masks:
                        param.data *= masks[name]
            
            current_sparsity = self._compute_sparsity(masks)
            print(f"Round {round_idx+1}: Sparsity = {current_sparsity:.1%}")
        
        return model, masks, initial_weights
    
    def _compute_sparsity(self, masks):
        total = sum(m.numel() for m in masks.values())
        zeros = sum((m == 0).sum().item() for m in masks.values())
        return zeros / total
```

---

## Pruning pour Physique des Particules

```python
class PhysicsPruning:
    """
    Stratégies de pruning spécifiques à la physique des particules
    """
    
    @staticmethod
    def sensitivity_analysis(model, val_loader, criterion):
        """
        Analyse de sensibilité: identifie les couches les plus importantes
        """
        sensitivities = {}
        
        baseline_loss = evaluate_loss(model, val_loader, criterion)
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Sauvegarde les poids
                original_weight = module.weight.data.clone()
                
                # Teste différents niveaux de pruning
                for sparsity in [0.5, 0.7, 0.9]:
                    # Prune temporairement
                    threshold = torch.quantile(module.weight.abs(), sparsity)
                    mask = (module.weight.abs() >= threshold).float()
                    module.weight.data *= mask
                    
                    # Évalue
                    pruned_loss = evaluate_loss(model, val_loader, criterion)
                    
                    # Restaure
                    module.weight.data = original_weight.clone()
                    
                    key = f"{name}_sparsity_{sparsity}"
                    sensitivities[key] = pruned_loss - baseline_loss
        
        return sensitivities
    
    @staticmethod
    def latency_aware_pruning(model, target_latency, hardware_model):
        """
        Pruning guidé par la latence cible
        
        Utilise un modèle de coût hardware pour optimiser
        """
        # Estime la latence de chaque couche
        layer_latencies = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layer_latencies[name] = hardware_model.estimate_latency(module)
        
        # Optimise la sparsité par couche pour atteindre la latence cible
        # (Algorithme glouton ou programmation dynamique)
        pass
```

---

## Exercices

### Exercice 8.1
Implémentez le pruning basé sur le gradient (gradient-based pruning) et comparez-le au magnitude pruning.

### Exercice 8.2
Créez une fonction qui prune automatiquement un modèle pour atteindre une latence cible sur GPU.

### Exercice 8.3
Reproduisez l'expérience Lottery Ticket sur un petit réseau pour MNIST.

---

## Points Clés à Retenir

> 📌 **Le pruning non structuré atteint de hautes sparsités mais nécessite du hardware spécialisé**

> 📌 **Le pruning structuré donne des accélérations réelles sur hardware standard**

> 📌 **Le fine-tuning après pruning est crucial pour maintenir les performances**

> 📌 **La Lottery Ticket Hypothesis suggère que les réseaux sont sur-paramétrés**

---

*Chapitre suivant : [Chapitre 9 - Quantification des Réseaux de Neurones](../Chapitre_09_Quantification/09_introduction.md)*

