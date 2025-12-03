# 7.4 Entraînement Bout-en-Bout avec Contraintes Tensorielles

---

## Introduction

L'entraînement de réseaux avec contraintes tensorielles nécessite des techniques spécialisées pour préserver la structure TT/Tucker tout en optimisant les performances. Cette section couvre les méthodes d'entraînement direct et les optimisations.

---

## Défis de l'Entraînement

### Problèmes Principaux

1. **Gradients** : Le calcul des gradients à travers les contractions TT
2. **Structure** : Maintenir la structure tensorielle pendant l'optimisation
3. **Stabilité** : Éviter l'explosion/vanishing des valeurs dans les cores
4. **Vitesse** : Optimiser les opérations de contraction

---

## Calcul des Gradients

### Backward Pass pour TT

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TTLinearWithGrad(nn.Module):
    """
    Couche TT avec calcul de gradient explicite
    """
    
    def __init__(self, input_dims, output_dims, tt_ranks):
        super().__init__()
        # ... (initialisation comme avant)
        pass
    
    def forward(self, x):
        """Forward avec enregistrement pour backward"""
        # PyTorch calcule automatiquement les gradients
        # Mais on peut optimiser manuellement si nécessaire
        pass

# PyTorch gère automatiquement les gradients
tt_layer = TTLinear((16, 16, 4), (16, 16, 2), tt_rank=8)
x = torch.randn(32, 1024, requires_grad=True)
y = tt_layer(x)

loss = y.sum()
loss.backward()

# Gradients calculés automatiquement
for core in tt_layer.cores:
    if core.grad is not None:
        print(f"Gradient shape: {core.grad.shape}")
```

---

## Techniques d'Optimisation

### 1. Gradient Clipping

```python
def train_tt_model_with_clipping(model, train_loader, epochs=10):
    """
    Entraîne un modèle TT avec gradient clipping
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# Exemple
# train_tt_model_with_clipping(model, train_loader)
```

### 2. Normalisation des Cores

```python
def normalize_tt_cores(model, norm_type='frobenius'):
    """
    Normalise les cores TT pour stabilité
    """
    for core in model.cores:
        if norm_type == 'frobenius':
            norm = torch.norm(core, p='fro')
            if norm > 1.0:
                core.data = core.data / norm
        elif norm_type == 'max':
            max_val = torch.max(torch.abs(core))
            if max_val > 1.0:
                core.data = core.data / max_val

# Applique après chaque backward
# normalize_tt_cores(model)
```

---

## Stratégies d'Entraînement

### Méthode 1 : Entraînement Direct

```python
def train_direct(model, train_loader, val_loader, epochs=50):
    """
    Entraîne directement le modèle tensorisé
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Entraînement
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        val_acc = 100.0 * correct / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_tt_model.pt')
    
    return model
```

### Méthode 2 : Fine-tuning depuis Dense

```python
def fine_tune_from_dense(tt_model, dense_model, train_loader, epochs=20):
    """
    Fine-tune un modèle TT initialisé depuis un modèle dense
    """
    # Initialise le TT depuis le dense
    initialize_tt_from_dense(tt_model, dense_model)
    
    # Utilise un learning rate plus petit
    optimizer = optim.Adam(tt_model.parameters(), lr=0.0001)
    
    # Entraîne
    return train_direct(tt_model, train_loader, epochs=epochs)

def initialize_tt_from_dense(tt_model, dense_model):
    """
    Initialise les cores TT depuis les poids denses via TT-SVD
    """
    for name, module in dense_model.named_modules():
        if isinstance(module, nn.Linear):
            # Trouve le module TT correspondant
            tt_module = find_corresponding_tt_module(tt_model, name)
            
            # Convertit les poids
            W = module.weight.data.numpy()
            input_dims = tt_module.input_dims
            output_dims = tt_module.output_dims
            
            # TT-SVD
            tt_W = matrix_to_tt_svd(W, input_dims, output_dims, max_rank=tt_module.tt_ranks[0])
            
            # Copie les cores
            for i, core in enumerate(tt_W.cores):
                tt_module.cores[i].data = torch.from_numpy(core).float()

def find_corresponding_tt_module(tt_model, name):
    """Trouve le module TT correspondant au nom"""
    # (Simplifié - nécessite mapping approprié)
    pass
```

### Méthode 3 : Progressive Training

```python
def progressive_training(model, train_loader, rank_schedule):
    """
    Entraîne progressivement en augmentant les rangs TT
    
    rank_schedule: liste de rangs à utiliser à chaque étape
    """
    for stage, target_rank in enumerate(rank_schedule):
        print(f"Stage {stage}: Training with rank {target_rank}")
        
        # Ajuste les rangs (nécessite réallocation)
        # adjust_tt_ranks(model, target_rank)
        
        # Entraîne avec ce rang
        train_direct(model, train_loader, epochs=10)
```

---

## Optimisations Avancées

### 1. Reconditionnement Periodique

```python
def periodic_reconditioning(model, frequency=100):
    """
    Reconditionne les cores TT périodiquement
    """
    global step
    step = 0
    
    def recondition_hook(grad):
        global step
        step += 1
        
        if step % frequency == 0:
            # Met en forme canonique (ex: left-canonical)
            # canonicalize_tt_cores(model)
            pass
        
        return grad
    
    # Enregistre le hook (simplifié)
    pass
```

### 2. Adaptive Rank Adjustment

```python
def adaptive_rank_training(model, train_loader, min_rank=4, max_rank=16):
    """
    Ajuste les rangs TT dynamiquement selon les gradients
    """
    for epoch in range(epochs):
        for data, target in train_loader:
            # Forward et backward
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Analyse les gradients pour décider d'augmenter/réduire les rangs
            for i, core in enumerate(model.cores):
                grad_norm = torch.norm(core.grad)
                
                # Si gradient large, peut-être augmenter le rang
                if grad_norm > threshold_high and current_rank < max_rank:
                    # increase_rank(model, i)
                    pass
                # Si gradient petit, peut-être réduire le rang
                elif grad_norm < threshold_low and current_rank > min_rank:
                    # decrease_rank(model, i)
                    pass
            
            optimizer.step()
```

---

## Initialisations Spécialisées

### Initialisation pour TT

```python
def initialize_tt_cores_orthogonal(cores):
    """
    Initialise les cores TT avec des matrices orthogonales
    """
    for core in cores:
        # Pour chaque slice selon la dimension physique
        for i in range(core.shape[1]):
            # Initialise la slice comme matrice orthogonale
            U, _, Vt = torch.svd(torch.randn(core.shape[0], core.shape[2]))
            core.data[:, i, :] = U @ Vt.T

def initialize_tt_cores_uniform(cores, scale=0.1):
    """
    Initialise uniformément avec échelle adaptée
    """
    for i, core in enumerate(cores):
        # Échelle adaptative selon la position
        adaptive_scale = scale / np.sqrt(core.shape[1])
        core.data = torch.randn_like(core) * adaptive_scale
```

---

## Monitoring et Debugging

### Tracking des Propriétés TT

```python
def monitor_tt_properties(model, writer=None):
    """
    Surveille les propriétés du modèle TT pendant l'entraînement
    """
    metrics = {
        'param_count': sum(p.numel() for p in model.parameters()),
        'tt_ranks': [c.shape[2] for c in model.cores[:-1]],
        'core_norms': [torch.norm(c).item() for c in model.cores],
        'core_max_vals': [torch.max(torch.abs(c)).item() for c in model.cores],
    }
    
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, list):
                for i, v in enumerate(value):
                    writer.add_scalar(f'TT/{key}_{i}', v, global_step)
            else:
                writer.add_scalar(f'TT/{key}', value, global_step)
    
    return metrics
```

---

## Exercices

### Exercice 7.4.1
Implémentez un entraînement avec reconditionnement automatique des cores TT toutes les N itérations.

### Exercice 7.4.2
Comparez les performances d'entraînement direct vs fine-tuning depuis un modèle dense.

### Exercice 7.4.3
Créez un système d'ajustement adaptatif des rangs TT basé sur l'importance des gradients.

---

## Points Clés à Retenir

> 📌 **PyTorch calcule automatiquement les gradients à travers les contractions TT**

> 📌 **Le gradient clipping et la normalisation sont essentiels pour la stabilité**

> 📌 **Le fine-tuning depuis un modèle dense donne souvent de meilleurs résultats**

> 📌 **L'entraînement progressif avec rangs croissants peut être efficace**

> 📌 **Le monitoring des propriétés TT aide à diagnostiquer les problèmes**

---

*Section suivante : [7.5 Analyse de la Perte d'Expressivité](./07_05_Expressivite.md)*

