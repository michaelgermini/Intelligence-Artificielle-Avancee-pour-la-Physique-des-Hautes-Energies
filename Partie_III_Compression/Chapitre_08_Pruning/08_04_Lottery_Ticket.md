# 8.4 Lottery Ticket Hypothesis

---

## Introduction

La **Lottery Ticket Hypothesis** (LTH) énonce qu'un réseau dense contient un sous-réseau sparse (le "ticket gagnant") qui, entraîné depuis les mêmes poids initiaux, peut atteindre des performances comparables au réseau complet.

---

## Énoncé de l'Hypothèse

> Un réseau entraîné contient un sous-réseau (lottery ticket) qui, lorsqu'il est entraîné isolément avec les poids initiaux originaux, atteint une performance similaire au réseau complet en au plus le même nombre d'itérations d'entraînement.

---

## Implémentation de la LTH

```python
import torch
import torch.nn as nn
import numpy as np

class LotteryTicketFinder:
    """
    Trouve les "lottery tickets" dans un réseau
    """
    
    def __init__(self, model_fn, initial_weights_fn):
        """
        Args:
            model_fn: Fonction qui crée le modèle
            initial_weights_fn: Fonction qui initialise les poids
        """
        self.model_fn = model_fn
        self.initial_weights_fn = initial_weights_fn
        
    def iterative_magnitude_pruning(self, train_fn, eval_fn, 
                                   target_sparsity=0.9, 
                                   n_rounds=10,
                                   reset_weights=True):
        """
        Pruning itératif avec réinitialisation aux poids initiaux
        
        C'est l'algorithme standard de la LTH
        """
        # Sauvegarde les poids initiaux
        model = self.model_fn()
        initial_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }
        
        # Masques de pruning
        masks = {
            name: torch.ones_like(param) 
            for name, param in model.named_parameters()
            if 'weight' in name
        }
        
        # Sparsité par round
        sparsity_per_round = 1 - (1 - target_sparsity) ** (1 / n_rounds)
        
        history = {
            'round': [],
            'sparsity': [],
            'accuracy': [],
            'mask_updates': []
        }
        
        for round_idx in range(n_rounds):
            # Entraîne le modèle
            train_fn(model)
            
            # Évalue avant pruning
            accuracy_before = eval_fn(model)
            
            # Prune: supprime les poids de plus faible magnitude
            for name, param in model.named_parameters():
                if name in masks:
                    # Poids masqués
                    masked_weights = param.data * masks[name]
                    
                    # Trouve les poids à pruner
                    flat_weights = masked_weights[masks[name] == 1].abs().flatten()
                    if len(flat_weights) > 0:
                        threshold = torch.quantile(
                            flat_weights, 
                            sparsity_per_round
                        )
                        
                        # Met à jour le masque
                        new_mask = (masked_weights.abs() >= threshold).float()
                        masks[name] *= new_mask
            
            # Réinitialise aux poids initiaux et applique le masque
            if reset_weights:
                for name, param in model.named_parameters():
                    if name in initial_weights:
                        param.data = initial_weights[name].clone()
                        if name in masks:
                            param.data *= masks[name]
            
            # Évalue après pruning
            accuracy_after = eval_fn(model)
            
            # Calcule la sparsité
            total = sum(m.numel() for m in masks.values())
            zeros = sum((m == 0).sum().item() for m in masks.values())
            current_sparsity = zeros / total
            
            history['round'].append(round_idx + 1)
            history['sparsity'].append(current_sparsity)
            history['accuracy'].append(accuracy_after)
            
            print(f"Round {round_idx+1}/{n_rounds}: "
                  f"Sparsity={current_sparsity:.1%}, "
                  f"Accuracy={accuracy_after:.2%}")
        
        return model, masks, initial_weights, history
    
    def find_winning_ticket(self, sparsity, train_fn, eval_fn):
        """
        Trouve un winning ticket à une sparsité donnée
        """
        model, masks, initial_weights, history = \
            self.iterative_magnitude_pruning(
                train_fn, eval_fn, 
                target_sparsity=sparsity,
                reset_weights=True
            )
        
        return {
            'model': model,
            'masks': masks,
            'initial_weights': initial_weights,
            'history': history
        }

# Exemple d'utilisation
"""
def create_model():
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

finder = LotteryTicketFinder(create_model, initialize_weights)

ticket = finder.find_winning_ticket(
    sparsity=0.9,
    train_fn=lambda m: train_one_epoch(m, train_loader),
    eval_fn=lambda m: evaluate(m, val_loader)
)
"""
```

---

## Test de la LTH

### Comparaison des Trois Conditions

```python
def test_lottery_ticket_hypothesis(model_fn, train_fn, eval_fn):
    """
    Teste la Lottery Ticket Hypothesis
    
    Compare trois conditions:
    1. Random reinitialization: masque + poids aléatoires
    2. Random pruning: masque aléatoire + poids initiaux
    3. Winning ticket: masque trouvé + poids initiaux
    """
    
    # Condition 1: Random reinitialization
    model1 = model_fn()
    masks = create_pruning_masks(model1, sparsity=0.9, method='magnitude')
    apply_masks(model1, masks)
    # Réinitialise avec poids aléatoires
    for param in model1.parameters():
        param.data = torch.randn_like(param)
    
    train_fn(model1)
    acc_random_init = eval_fn(model1)
    
    # Condition 2: Random pruning
    model2 = model_fn()
    initial_weights = {n: p.clone() for n, p in model2.named_parameters()}
    random_masks = create_random_masks(model2, sparsity=0.9)
    apply_masks(model2, random_masks)
    # Restaure poids initiaux
    for name, param in model2.named_parameters():
        param.data = initial_weights[name].clone()
        if name in random_masks:
            param.data *= random_masks[name]
    
    train_fn(model2)
    acc_random_mask = eval_fn(model2)
    
    # Condition 3: Winning ticket
    finder = LotteryTicketFinder(model_fn, lambda m: None)
    ticket = finder.find_winning_ticket(0.9, train_fn, eval_fn)
    acc_winning = ticket['history']['accuracy'][-1]
    
    print("Test Lottery Ticket Hypothesis:")
    print(f"  Random reinit: {acc_random_init:.2%}")
    print(f"  Random mask: {acc_random_mask:.2%}")
    print(f"  Winning ticket: {acc_winning:.2%}")
    
    return {
        'random_init': acc_random_init,
        'random_mask': acc_random_mask,
        'winning_ticket': acc_winning
    }
```

---

## LTH pour les Modèles Pré-entraînés

```python
class PretrainedLotteryTicket:
    """
    Extension de la LTH pour les modèles pré-entraînés
    
    Au lieu de réinitialiser aux poids initiaux,
    on réinitialise aux poids pré-entraînés
    """
    
    def __init__(self, pretrained_model):
        self.pretrained_weights = {
            name: param.clone() 
            for name, param in pretrained_model.named_parameters()
        }
        self.pretrained_model = pretrained_model
    
    def find_pretrained_ticket(self, target_sparsity, fine_tune_fn, eval_fn):
        """
        Trouve un ticket dans un modèle pré-entraîné
        """
        model = copy.deepcopy(self.pretrained_model)
        masks = self._iterative_pruning(
            model, target_sparsity, fine_tune_fn
        )
        
        # Restaure les poids pré-entraînés et applique masques
        for name, param in model.named_parameters():
            if name in self.pretrained_weights:
                param.data = self.pretrained_weights[name].clone()
                if name in masks:
                    param.data *= masks[name]
        
        return model, masks
    
    def _iterative_pruning(self, model, target_sparsity, train_fn):
        """Pruning itératif"""
        masks = {
            name: torch.ones_like(param)
            for name, param in model.named_parameters()
            if 'weight' in name
        }
        
        n_rounds = 10
        sparsity_per_round = 1 - (1 - target_sparsity) ** (1 / n_rounds)
        
        for round_idx in range(n_rounds):
            # Fine-tune
            train_fn(model)
            
            # Prune
            for name, param in model.named_parameters():
                if name in masks:
                    masked = param.data * masks[name]
                    flat = masked[masks[name] == 1].abs().flatten()
                    if len(flat) > 0:
                        threshold = torch.quantile(flat, sparsity_per_round)
                        masks[name] *= (masked.abs() >= threshold).float()
            
            # Restaure poids pré-entraînés
            for name, param in model.named_parameters():
                if name in self.pretrained_weights:
                    param.data = self.pretrained_weights[name].clone()
                    if name in masks:
                        param.data *= masks[name]
        
        return masks
```

---

## Early-Bird Tickets

```python
class EarlyBirdTicket:
    """
    Early-Bird Tickets: trouver les tickets plus tôt dans l'entraînement
    
    Les tickets peuvent être identifiés dès les premières époques
    """
    
    def find_early_ticket(self, model, train_loader, 
                         early_epochs=5, target_sparsity=0.9):
        """
        Trouve un ticket après seulement quelques époques
        """
        initial_weights = {
            name: param.clone() 
            for name, param in model.named_parameters()
        }
        
        # Entraîne quelques époques
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(early_epochs):
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        
        # Prune une seule fois (pas itératif)
        masks = create_pruning_masks(model, target_sparsity, 'magnitude')
        
        # Restaure poids initiaux
        for name, param in model.named_parameters():
            if name in initial_weights:
                param.data = initial_weights[name].clone()
                if name in masks:
                    param.data *= masks[name]
        
        return model, masks
```

---

## Transferabilité des Tickets

```python
def test_ticket_transferability(source_model, target_model, masks):
    """
    Teste si un ticket trouvé sur un modèle
    peut être transféré à un autre
    
    Les tickets sont-ils spécifiques à l'architecture ?
    """
    
    # Applique les masques du modèle source au modèle cible
    target_masks = {}
    for name, mask in masks.items():
        # Tente de mapper les masques
        # (nécessite logique de correspondance selon architecture)
        if name in dict(target_model.named_parameters()):
            target_param = dict(target_model.named_parameters())[name]
            if target_param.shape == mask.shape:
                target_masks[name] = mask
    
    # Teste la performance
    for name, param in target_model.named_parameters():
        if name in target_masks:
            param.data *= target_masks[name]
    
    return target_model, target_masks
```

---

## Applications Pratiques

### Compression pour Déploiement

```python
def deploy_with_lottery_ticket(original_model, train_fn, eval_fn):
    """
    Utilise la LTH pour créer un modèle compressé pour déploiement
    """
    finder = LotteryTicketFinder(
        lambda: copy.deepcopy(original_model),
        lambda m: None  # Garde les poids initiaux
    )
    
    # Trouve le ticket
    ticket = finder.find_winning_ticket(
        sparsity=0.9,
        train_fn=train_fn,
        eval_fn=eval_fn
    )
    
    # Le modèle ticket peut être quantifié ou déployé tel quel
    compressed_model = ticket['model']
    
    return compressed_model, ticket['masks']

# Avantages pour le déploiement:
# 1. Sparsité structurée possible
# 2. Peut être combiné avec quantification
# 3. Performance préservée grâce aux poids initiaux
```

---

## Exercices

### Exercice 8.4.1
Reproduisez l'expérience LTH sur un petit réseau pour MNIST. Comparez les trois conditions (random init, random mask, winning ticket).

### Exercice 8.4.2
Testez si les winning tickets sont transférables entre architectures similaires (ex: ResNet18 vs ResNet34).

### Exercice 8.4.3
Implémentez la recherche d'early-bird tickets et comparez leur performance aux tickets trouvés après entraînement complet.

---

## Points Clés à Retenir

> 📌 **La LTH suggère que les réseaux sont sur-paramétrés dès l'initialisation**

> 📌 **Les poids initiaux sont cruciaux : le masque seul ne suffit pas**

> 📌 **Les tickets peuvent être trouvés tôt dans l'entraînement (Early-Bird)**

> 📌 **La LTH offre une méthode systématique pour trouver des sous-réseaux efficaces**

---

*Section suivante : [8.5 Critères de Sélection et Scheduling](./08_05_Criteres.md)*

