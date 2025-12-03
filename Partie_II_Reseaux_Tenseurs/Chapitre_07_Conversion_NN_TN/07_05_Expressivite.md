# 7.5 Analyse de la Perte d'Expressivité

---

## Introduction

La compression tensorielle réduit le nombre de paramètres mais peut également réduire l'expressivité du modèle. Cette section analyse comment quantifier et minimiser cette perte d'expressivité.

---

## Notions d'Expressivité

### Définition

L'**expressivité** d'un modèle mesure sa capacité à représenter des fonctions complexes. Pour les réseaux de tenseurs :
- **Expressivité maximale** : modèle dense non contraint
- **Expressivité réduite** : modèle avec contraintes tensorielles

---

## Métriques de Perte d'Expressivité

### 1. Erreur de Reconstruction

```python
import torch
import numpy as np

def reconstruction_error(original_model, tensorized_model, test_inputs):
    """
    Mesure l'erreur de reconstruction des sorties
    
    Args:
        original_model: modèle dense original
        tensorized_model: modèle tensorisé
        test_inputs: batch d'entrées de test
    """
    original_model.eval()
    tensorized_model.eval()
    
    with torch.no_grad():
        outputs_original = original_model(test_inputs)
        outputs_tensorized = tensorized_model(test_inputs)
        
        # Erreur relative
        error = torch.norm(outputs_original - outputs_tensorized, p='fro') / \
                torch.norm(outputs_original, p='fro')
        
        # Erreur par échantillon
        per_sample_errors = torch.norm(
            outputs_original - outputs_tensorized, 
            p=2, dim=1
        ) / torch.norm(outputs_original, p=2, dim=1)
    
    return {
        'relative_error': error.item(),
        'mean_per_sample_error': per_sample_errors.mean().item(),
        'std_per_sample_error': per_sample_errors.std().item(),
        'max_per_sample_error': per_sample_errors.max().item()
    }

# Exemple
test_data = torch.randn(100, 784)
# errors = reconstruction_error(original_model, tt_model, test_data)
# print(f"Erreur relative: {errors['relative_error']:.4f}")
```

### 2. Perte de Performance

```python
def performance_degradation(original_model, tensorized_model, test_loader):
    """
    Mesure la perte de performance (accuracy, F1, etc.)
    """
    def evaluate(model):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return 100.0 * correct / total
    
    acc_original = evaluate(original_model)
    acc_tensorized = evaluate(tensorized_model)
    
    degradation = acc_original - acc_tensorized
    relative_degradation = degradation / acc_original * 100
    
    return {
        'original_accuracy': acc_original,
        'tensorized_accuracy': acc_tensorized,
        'absolute_degradation': degradation,
        'relative_degradation': relative_degradation
    }
```

---

## Analyse Théorique

### Bornes sur l'Expressivité

```python
def theoretical_expressivity_bounds(tt_ranks, input_dims, output_dims):
    """
    Calcule des bornes théoriques sur l'expressivité
    
    L'expressivité est limitée par le rang TT
    """
    # Nombre de paramètres TT
    tt_params = compute_tt_parameters(tt_ranks, input_dims, output_dims)
    
    # Nombre de paramètres denses
    dense_params = np.prod(input_dims) * np.prod(output_dims)
    
    # Ratio de compression
    compression_ratio = dense_params / tt_params
    
    # Estime la dimension de l'espace de fonctions représentables
    # (Simplifié - nécessite théorie plus approfondie)
    expressivity_ratio = np.min(tt_ranks) / np.sqrt(dense_params)
    
    return {
        'compression_ratio': compression_ratio,
        'expressivity_ratio': expressivity_ratio,
        'tt_params': tt_params,
        'dense_params': dense_params
    }

def compute_tt_parameters(tt_ranks, input_dims, output_dims):
    """Calcule le nombre de paramètres TT"""
    params = 0
    
    # Cores d'entrée
    prev_rank = 1
    for i, dim in enumerate(input_dims):
        next_rank = tt_ranks[i]
        params += prev_rank * dim * next_rank
        prev_rank = next_rank
    
    # Cores de sortie
    for i, dim in enumerate(output_dims):
        if i < len(output_dims) - 1:
            next_rank = tt_ranks[len(input_dims) + i]
        else:
            next_rank = 1
        params += prev_rank * dim * next_rank
        prev_rank = next_rank
    
    return params

# Exemple
bounds = theoretical_expressivity_bounds(
    tt_ranks=[8, 8, 8, 8],
    input_dims=(16, 16, 4),
    output_dims=(16, 16, 2)
)
print(f"Bornes théoriques:")
print(f"  Compression: {bounds['compression_ratio']:.2f}x")
print(f"  Expressivité relative: {bounds['expressivity_ratio']:.4f}")
```

---

## Analyse Empirique

### Courbe Compression vs Performance

```python
def compression_performance_curve(original_model, test_loader, 
                                 input_dims, output_dims,
                                 rank_range=(2, 4, 8, 16, 32)):
    """
    Trace la courbe compression vs performance
    """
    results = []
    
    for rank in rank_range:
        # Crée modèle tensorisé avec ce rang
        tt_model = TensorizedLinear(input_dims, output_dims, tt_rank=rank)
        
        # Initialise depuis le modèle original
        initialize_tt_from_dense(tt_model, original_model)
        
        # Mesure la performance
        perf = performance_degradation(original_model, tt_model, test_loader)
        
        # Calcule la compression
        compression = tt_model.compression_ratio()
        
        results.append({
            'rank': rank,
            'compression': compression,
            'accuracy': perf['tensorized_accuracy'],
            'degradation': perf['absolute_degradation']
        })
        
        print(f"Rank {rank}: Compression {compression:.2f}x, "
              f"Accuracy {perf['tensorized_accuracy']:.2f}%, "
              f"Degradation {perf['absolute_degradation']:.2f}%")
    
    return results

# Exemple
# curve_data = compression_performance_curve(
#     original_model, test_loader,
#     input_dims=(16, 16, 4),
#     output_dims=(16, 16, 2),
#     rank_range=(4, 8, 16, 32)
# )
```

---

## Facteurs Affectant l'Expressivité

### 1. Rang TT

```python
def analyze_rank_impact(original_model, test_loader, 
                       input_dims, output_dims):
    """
    Analyse l'impact du rang TT sur l'expressivité
    """
    ranks = [2, 4, 8, 16, 32, 64]
    errors = []
    
    for rank in ranks:
        tt_model = TensorizedLinear(input_dims, output_dims, tt_rank=rank)
        initialize_tt_from_dense(tt_model, original_model)
        
        test_data = torch.randn(100, np.prod(input_dims))
        error_metrics = reconstruction_error(original_model, tt_model, test_data)
        
        errors.append({
            'rank': rank,
            'error': error_metrics['relative_error'],
            'compression': tt_model.compression_ratio()
        })
    
    return errors
```

### 2. Factorisation des Dimensions

```python
def analyze_factorization_impact(W, factorizations):
    """
    Analyse l'impact de différentes factorisations
    
    factorizations: liste de tuples (input_dims, output_dims)
    """
    results = []
    
    for input_dims, output_dims in factorizations:
        # Convertit en TT
        tt_W = matrix_to_tt_svd(W, input_dims, output_dims, max_rank=8)
        
        # Reconstruction error
        W_reconstructed = tt_W.to_matrix()
        error = np.linalg.norm(W - W_reconstructed) / np.linalg.norm(W)
        
        # Compression
        compression = W.size / tt_W.count_parameters()
        
        results.append({
            'input_dims': input_dims,
            'output_dims': output_dims,
            'error': error,
            'compression': compression
        })
    
    return results

# Test différentes factorisations
W_test = np.random.randn(1024, 512)
factorizations = [
    ((16, 16, 4), (16, 16, 2)),
    ((8, 8, 16), (8, 8, 8)),
    ((32, 32), (16, 32)),
]

# results = analyze_factorization_impact(W_test, factorizations)
```

---

## Minimisation de la Perte

### 1. Fine-tuning

```python
def minimize_loss_with_finetuning(tensorized_model, train_loader, 
                                  val_loader, epochs=50):
    """
    Minimise la perte d'expressivité via fine-tuning
    """
    optimizer = optim.Adam(tensorized_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Entraînement
        tensorized_model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = tensorized_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(tensorized_model.parameters(), 1.0)
            optimizer.step()
        
        # Validation
        tensorized_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = tensorized_model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(tensorized_model.state_dict(), 'best_model.pt')
    
    return tensorized_model
```

### 2. Rank Selection Optimale

```python
def find_optimal_rank(original_model, test_loader,
                     input_dims, output_dims,
                     rank_candidates, tolerance=0.01):
    """
    Trouve le rang optimal pour une tolérance de perte donnée
    """
    for rank in sorted(rank_candidates):
        tt_model = TensorizedLinear(input_dims, output_dims, tt_rank=rank)
        initialize_tt_from_dense(tt_model, original_model)
        
        # Fine-tune
        tt_model = minimize_loss_with_finetuning(
            tt_model, train_loader, test_loader, epochs=20
        )
        
        # Mesure la performance
        perf = performance_degradation(original_model, tt_model, test_loader)
        
        if perf['relative_degradation'] <= tolerance * 100:
            print(f"Rang optimal trouvé: {rank}")
            return rank, tt_model
    
    print("Aucun rang ne satisfait la tolérance")
    return None, None
```

---

## Analyse de Sensibilité

### Sensibilité aux Perturbations

```python
def sensitivity_analysis(model, test_inputs, noise_levels):
    """
    Analyse la sensibilité aux perturbations
    """
    model.eval()
    sensitivities = []
    
    with torch.no_grad():
        output_clean = model(test_inputs)
        
        for noise_level in noise_levels:
            # Ajoute du bruit aux poids
            perturbed_model = add_noise_to_weights(model, noise_level)
            
            output_perturbed = perturbed_model(test_inputs)
            
            sensitivity = torch.norm(output_clean - output_perturbed) / \
                         torch.norm(output_clean) / noise_level
            
            sensitivities.append({
                'noise_level': noise_level,
                'sensitivity': sensitivity.item()
            })
    
    return sensitivities

def add_noise_to_weights(model, noise_level):
    """Ajoute du bruit aux poids du modèle"""
    perturbed = copy.deepcopy(model)
    
    for param in perturbed.parameters():
        noise = torch.randn_like(param) * noise_level
        param.data = param.data + noise
    
    return perturbed
```

---

## Exercices

### Exercice 7.5.1
Créez une fonction qui trace automatiquement la courbe compression vs performance pour différentes architectures.

### Exercice 7.5.2
Analysez comment la perte d'expressivité varie avec la profondeur du réseau tensorisé.

### Exercice 7.5.3
Implémentez une méthode automatique pour trouver le rang TT optimal selon un critère de performance.

---

## Points Clés à Retenir

> 📌 **La perte d'expressivité est inévitable mais peut être minimisée**

> 📌 **Le rang TT est le facteur principal affectant l'expressivité**

> 📌 **Le fine-tuning restaure souvent une grande partie de la performance**

> 📌 **Il existe un compromis optimal compression/performance**

> 📌 **L'analyse empirique est essentielle pour comprendre les limites pratiques**

---

*Chapitre suivant : [Chapitre 8 - Techniques de Pruning](../Partie_III_Compression/Chapitre_08_Pruning/08_introduction.md)*

