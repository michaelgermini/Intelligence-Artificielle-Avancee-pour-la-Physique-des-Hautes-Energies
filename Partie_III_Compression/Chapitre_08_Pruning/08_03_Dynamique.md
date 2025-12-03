# 8.3 Pruning Dynamique et Adaptatif

---

## Introduction

Le **pruning dynamique** adapte la structure du réseau selon l'input, permettant d'économiser des ressources computationnelles sur les exemples "faciles" tout en conservant la pleine capacité pour les exemples "difficiles".

---

## Pruning Adaptatif par Input

### Principe

Certains inputs nécessitent moins de calculs que d'autres. Le pruning dynamique identifie ces cas et adapte le réseau en conséquence.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicPruningLayer(nn.Module):
    """
    Couche avec pruning dynamique basé sur la difficulté de l'input
    """
    
    def __init__(self, in_features, out_features, n_exits=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_exits = n_exits  # Nombre de niveaux de pruning
        
        # Couches complètes
        self.full_layer = nn.Linear(in_features, out_features)
        
        # Couches prunées progressivement
        self.pruned_layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(n_exits - 1)
        ])
        
        # Gating network: décide quelle version utiliser
        self.gate = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_exits),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, temperature=1.0):
        """
        Forward avec sélection dynamique
        
        Args:
            temperature: Contrôle la "sharpness" de la sélection
        """
        # Prédit la difficulté
        gate_scores = self.gate(x) / temperature
        gate_probs = F.softmax(gate_scores, dim=-1)
        
        # Forward avec toutes les versions
        full_out = self.full_layer(x)
        pruned_outs = [layer(x) for layer in self.pruned_layers]
        
        # Mélange selon les probabilités (soft selection)
        # Ou sélection hard (gargage) pour efficacité
        selected_idx = gate_probs.argmax(dim=-1)
        
        output = torch.zeros_like(full_out)
        for i in range(self.n_exits):
            mask = (selected_idx == i)
            if mask.any():
                if i == 0:
                    output[mask] = full_out[mask]
                else:
                    output[mask] = pruned_outs[i-1][mask]
        
        return output, gate_probs
    
    def compute_expected_flops(self, gate_probs):
        """
        Calcule les FLOPs attendus selon la distribution de la gate
        """
        # FLOPs par version
        flops = [
            self.in_features * self.out_features,  # Full
        ]
        for layer in self.pruned_layers:
            # Approxime avec sparsité
            flops.append(self.in_features * self.out_features * 0.5)
        
        # FLOPs attendus
        expected_flops = sum(p.mean() * f for p, f in zip(gate_probs.T, flops))
        
        return expected_flops

# Test
layer = DynamicPruningLayer(128, 64, n_exits=3)
x = torch.randn(32, 128)

output, gate_probs = layer(x)
print(f"Gate probabilities: {gate_probs.shape}")
print(f"Distribution: {gate_probs.mean(dim=0)}")
print(f"FLOPs attendus: {layer.compute_expected_flops(gate_probs):.0f}")
```

---

## Early Exit (Sortie Précoce)

```python
class EarlyExitNetwork(nn.Module):
    """
    Réseau avec sorties précoces
    
    Les exemples faciles sortent tôt, les difficiles utilisent tout le réseau
    """
    
    def __init__(self, layers, exit_thresholds):
        super().__init__()
        
        self.layers = nn.ModuleList(layers)
        self.exit_classifiers = nn.ModuleList([
            nn.Linear(layer.out_features, 10) for layer in layers
        ])
        self.exit_thresholds = exit_thresholds  # Confiance minimale pour sortir
        
    def forward(self, x, return_exit_info=False):
        """
        Forward avec early exit
        """
        exit_layer = None
        confidence = 0
        
        for i, (layer, classifier) in enumerate(zip(self.layers, self.exit_classifiers)):
            x = layer(x)
            
            # Classification à cette profondeur
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            max_conf = probs.max(dim=-1)[0]
            
            # Détermine si on peut sortir (au moins un exemple suffisamment confiant)
            if max_conf.mean().item() > self.exit_thresholds[i]:
                exit_layer = i
                confidence = max_conf.mean().item()
                break
        
        # Si aucun exit, utilise la dernière couche
        if exit_layer is None:
            exit_layer = len(self.layers) - 1
            logits = self.exit_classifiers[-1](x)
        
        if return_exit_info:
            return logits, exit_layer, confidence
        return logits

# Exemple
layers = [
    nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
    nn.Sequential(nn.Linear(256, 128), nn.ReLU()),
    nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
]

network = EarlyExitNetwork(layers, exit_thresholds=[0.95, 0.90, 0.0])

x = torch.randn(32, 784)
logits, exit_idx, conf = network(x, return_exit_info=True)

print(f"Exit à la couche: {exit_idx}")
print(f"Confiance moyenne: {conf:.2%}")
```

---

## Pruning Conditionnel par Difficulté

```python
class DifficultyAwarePruning(nn.Module):
    """
    Pruning qui s'adapte à la difficulté perçue de l'input
    """
    
    def __init__(self, base_model, pruning_levels=[0.3, 0.5, 0.7]):
        super().__init__()
        
        self.base_model = base_model
        self.pruning_levels = pruning_levels
        
        # Crée plusieurs versions prunées
        self.pruned_models = nn.ModuleList()
        for level in pruning_levels:
            pruned = self._create_pruned_model(level)
            self.pruned_models.append(pruned)
        
        # Estimateur de difficulté
        self.difficulty_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 64),  # Exemple pour CNN
            nn.ReLU(),
            nn.Linear(64, len(pruning_levels) + 1)
        )
        
    def _create_pruned_model(self, sparsity):
        """Crée une version prunée du modèle"""
        # Copie le modèle
        pruned = copy.deepcopy(self.base_model)
        
        # Prune avec magnitude pruning
        # (Implémentation simplifiée)
        return pruned
    
    def forward(self, x):
        """Forward avec sélection de modèle selon difficulté"""
        # Estime la difficulté
        difficulty = self.difficulty_estimator(x)
        model_idx = difficulty.argmax(dim=-1)
        
        # Sélectionne le modèle approprié
        if model_idx == len(self.pruning_levels):
            # Utilise le modèle complet
            return self.base_model(x)
        else:
            # Utilise un modèle pruné
            return self.pruned_models[model_idx](x)
```

---

## Pruning Adaptatif Entraîné

```python
class LearnedPruningMask(nn.Module):
    """
    Masques de pruning appris pendant l'entraînement
    """
    
    def __init__(self, base_layer, temperature=1.0):
        super().__init__()
        
        self.base_layer = base_layer
        self.temperature = temperature
        
        # Masques apprenables (logits)
        self.mask_logits = nn.Parameter(
            torch.ones(base_layer.weight.shape) * 2.0  # Initialise proche de 1
        )
        
    def forward(self, x):
        """
        Forward avec masques stochastiques pendant l'entraînement,
        déterministes pendant l'inférence
        """
        if self.training:
            # Masques stochastiques (Straight-Through Estimator)
            mask_probs = torch.sigmoid(self.mask_logits / self.temperature)
            mask = (torch.rand_like(mask_probs) < mask_probs).float()
        else:
            # Masques déterministes
            mask = (self.mask_logits > 0).float()
        
        # Applique les masques
        masked_weight = self.base_layer.weight * mask
        output = F.linear(x, masked_weight, self.base_layer.bias)
        
        return output, mask
    
    def sparsity(self):
        """Calcule la sparsité actuelle"""
        mask = (self.mask_logits > 0).float()
        return (mask == 0).sum().item() / mask.numel()

# Utilisation
base = nn.Linear(128, 64)
adaptive = LearnedPruningMask(base, temperature=1.0)

x = torch.randn(32, 128)
output, mask = adaptive(x)

print(f"Sparsité: {adaptive.sparsity():.1%}")
print(f"Poids actifs: {mask.sum().item()}/{mask.numel()}")
```

---

## Pruning par Gradient d'Attention

```python
class AttentionBasedPruning(nn.Module):
    """
    Pruning basé sur l'attention: privilégie les parties "importantes"
    """
    
    def __init__(self, base_model):
        super().__init__()
        
        self.base_model = base_model
        
        # Attention weights apprenables
        self.attention = nn.Parameter(torch.ones(100))  # Exemple
        
    def forward(self, x):
        """
        Forward avec attention qui guide le pruning
        """
        # Calcule l'attention
        attn = F.softmax(self.attention, dim=-1)
        
        # Sélectionne les features les plus importantes
        top_k = 50  # Garde top 50%
        _, top_indices = attn.topk(top_k)
        
        # Prune x selon l'attention
        x_pruned = x[:, top_indices]
        
        # Forward avec input pruné
        # (nécessite adaptation du modèle)
        
        return x_pruned
```

---

## Entraînement avec Pruning Dynamique

```python
def train_with_dynamic_pruning(model, train_loader, epochs=10):
    """
    Entraînement avec pruning dynamique
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Loss supplémentaire pour encourager l'utilisation des modèles légers
    def sparsity_loss(model):
        """Pénalise l'utilisation des modèles complets"""
        total_loss = 0
        if hasattr(model, 'gate'):
            # Encourage des probabilités faibles pour le modèle complet
            gate_probs = model.gate(torch.randn(1, model.in_features))
            total_loss += gate_probs[0, 0]  # Probabilité du modèle complet
        return total_loss
    
    for epoch in range(epochs):
        total_loss = 0
        total_flops = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            
            output, gate_probs = model(x)
            loss = criterion(output, y)
            
            # Loss additionnelle pour encourager l'efficacité
            efficiency_loss = sparsity_loss(model)
            total_loss_batch = loss + 0.1 * efficiency_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            # Statistiques
            total_loss += loss.item()
            if hasattr(model, 'compute_expected_flops'):
                total_flops += model.compute_expected_flops(gate_probs)
        
        avg_loss = total_loss / len(train_loader)
        avg_flops = total_flops / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Avg FLOPs={avg_flops:.0f}")
```

---

## Applications pour le Trigger

```python
class TriggerDynamicPruning:
    """
    Pruning dynamique optimisé pour les systèmes de trigger
    
    Objectif: économiser du calcul sur les événements "faciles" à classifier
    """
    
    @staticmethod
    def create_lightweight_classifier(full_model, difficulty_threshold=0.9):
        """
        Crée un classifieur léger pour les événements faciles
        """
        # Version prunée agressive du modèle
        light_model = prune_model(full_model, sparsity=0.95)
        
        return light_model
    
    @staticmethod
    def two_stage_trigger(full_model, light_model, difficulty_estimator):
        """
        Système de trigger à deux étapes:
        1. Estimation rapide de difficulté
        2. Modèle approprié selon difficulté
        """
        def trigger_function(event):
            # Étape 1: Estimation rapide
            difficulty = difficulty_estimator(event)
            
            # Étape 2: Sélection
            if difficulty < 0.5:  # Événement "facile"
                decision = light_model(event)
            else:  # Événement "difficile"
                decision = full_model(event)
            
            return decision
        
        return trigger_function
```

---

## Exercices

### Exercice 8.3.1
Implémentez un réseau avec early exit qui s'entraîne de manière end-to-end avec une loss combinant classification et incitation à sortir tôt.

### Exercice 8.3.2
Créez un système de pruning dynamique qui adapte le nombre de filtres actifs dans une couche convolutionnelle selon la complexité de l'image d'entrée.

### Exercice 8.3.3
Comparez les FLOPs moyens d'un réseau avec et sans early exit sur un dataset de test.

---

## Points Clés à Retenir

> 📌 **Le pruning dynamique adapte la complexité du modèle à la difficulté de l'input**

> 📌 **L'early exit permet d'économiser des calculs sur les exemples faciles**

> 📌 **Les masques apprenables permettent un pruning adaptatif entraîné**

> 📌 **Pour les triggers, le pruning dynamique peut réduire significativement la latence moyenne**

---

*Section suivante : [8.4 Lottery Ticket Hypothesis](./08_04_Lottery_Ticket.md)*

