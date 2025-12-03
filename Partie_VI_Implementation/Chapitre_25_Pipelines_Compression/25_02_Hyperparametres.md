# 25.2 Sélection Automatique des Hyperparamètres

---

## Introduction

La **sélection automatique des hyperparamètres** est cruciale pour optimiser le compromis entre compression et performance. Cette section présente les méthodes pour automatiser la recherche d'hyperparamètres optimaux, incluant grid search, random search, et méthodes avancées comme Bayesian optimization.

---

## Grid Search vs Random Search

### Méthodes Basiques

```python
import itertools
import random
from typing import Dict, List, Tuple
import numpy as np

class HyperparameterSearch:
    """
    Recherche d'hyperparamètres pour compression
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.results = []
    
    def grid_search(self, param_grid: Dict[str, List]):
        """
        Grid search exhaustif
        
        Args:
            param_grid: Dict {param_name: [values]}
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_score = -float('inf')
        best_params = None
        
        # Générer toutes combinaisons
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # Évaluer cette combinaison
            score = self.evaluate_params(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def random_search(self, param_distributions: Dict, n_iter=100):
        """
        Random search
        
        Args:
            param_distributions: Dict {param_name: distribution}
            n_iter: Nombre d'itérations
        """
        best_score = -float('inf')
        best_params = None
        
        for _ in range(n_iter):
            # Échantillonner hyperparamètres
            params = {}
            for param_name, distribution in param_distributions.items():
                params[param_name] = self.sample_from_distribution(distribution)
            
            # Évaluer
            score = self.evaluate_params(params)
            
            self.results.append({
                'params': params,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    def sample_from_distribution(self, distribution):
        """Échantillonne depuis distribution"""
        dist_type = distribution['type']
        
        if dist_type == 'uniform':
            return np.random.uniform(distribution['low'], distribution['high'])
        elif dist_type == 'loguniform':
            log_low = np.log(distribution['low'])
            log_high = np.log(distribution['high'])
            return np.exp(np.random.uniform(log_low, log_high))
        elif dist_type == 'choice':
            return random.choice(distribution['choices'])
        elif dist_type == 'int':
            return np.random.randint(distribution['low'], distribution['high'] + 1)
        
        return None
    
    def evaluate_params(self, params: Dict):
        """
        Évalue ensemble d'hyperparamètres
        
        Returns:
            score: Score à maximiser (ex: accuracy)
        """
        # Appliquer compression avec ces params
        compressed_model = self.apply_compression_with_params(params)
        
        # Fine-tuning rapide
        self.quick_finetune(compressed_model, epochs=5)
        
        # Évaluer sur validation set
        score = self.evaluate_model(compressed_model, self.val_loader)
        
        return score
    
    def apply_compression_with_params(self, params: Dict):
        """Applique compression avec hyperparamètres donnés"""
        # Implémentation dépend de méthode de compression
        # Exemple pour pruning
        if 'sparsity' in params:
            return self.apply_pruning(params['sparsity'])
        # ...
        return self.model
    
    def quick_finetune(self, model, epochs=5):
        """Fine-tuning rapide pour évaluation"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def evaluate_model(self, model, dataloader):
        """Évalue modèle et retourne score"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100 * correct / total  # Accuracy
```

---

## Bayesian Optimization

### Optimisation Bayésienne

```python
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

class BayesianHyperparameterSearch:
    """
    Recherche hyperparamètres avec Bayesian Optimization
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def optimize(self, search_space: Dict, n_calls=50):
        """
        Optimise avec Bayesian Optimization
        
        Args:
            search_space: Dict définissant espace de recherche
            n_calls: Nombre d'évaluations
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize not available")
        
        # Convertir espace de recherche en format skopt
        dimensions = []
        param_names = []
        
        for param_name, space_def in search_space.items():
            param_names.append(param_name)
            
            if space_def['type'] == 'real':
                dimensions.append(
                    Real(space_def['low'], space_def['high'], 
                         prior='uniform' if space_def.get('prior') != 'log' else 'log-uniform')
                )
            elif space_def['type'] == 'int':
                dimensions.append(
                    Integer(space_def['low'], space_def['high'])
                )
        
        # Fonction objectif
        @use_named_args(dimensions=dimensions)
        def objective(**params):
            return -self.evaluate_params(params)  # Négatif car on minimise
        
        # Optimisation
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=10,
            random_state=42
        )
        
        # Convertir résultats
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        return best_params, best_score
    
    def evaluate_params(self, params: Dict):
        """Évalue hyperparamètres (même méthode que précédemment)"""
        compressed_model = self.apply_compression_with_params(params)
        self.quick_finetune(compressed_model, epochs=5)
        return self.evaluate_model(compressed_model, self.val_loader)
```

---

## Optuna

### Framework d'Optimisation Moderne

```python
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class OptunaHyperparameterSearch:
    """
    Recherche hyperparamètres avec Optuna
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def optimize(self, n_trials=100, direction='maximize'):
        """
        Optimise avec Optuna
        
        Args:
            n_trials: Nombre de trials
            direction: 'maximize' ou 'minimize'
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        
        study = optuna.create_study(direction=direction)
        
        def objective(trial):
            # Suggérer hyperparamètres
            sparsity = trial.suggest_float('sparsity', 0.1, 0.9)
            learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
            epochs = trial.suggest_int('epochs', 5, 20)
            
            params = {
                'sparsity': sparsity,
                'lr': learning_rate,
                'epochs': epochs
            }
            
            # Évaluer
            score = self.evaluate_params(params)
            
            # Pruning (arrêt précoce si performance faible)
            trial.report(score, epoch=epochs)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
```

---

## Multi-Objective Optimization

### Optimisation Multi-Objectif

```python
class MultiObjectiveSearch:
    """
    Optimisation multi-objectif (compression vs performance)
    """
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def pareto_optimize(self, n_iter=100):
        """
        Trouve front de Pareto (trade-off compression/performance)
        """
        pareto_front = []
        
        for _ in range(n_iter):
            # Échantillonner hyperparamètres
            params = self.sample_params()
            
            # Évaluer objectifs
            accuracy = self.evaluate_accuracy(params)
            compression = self.evaluate_compression(params)
            
            # Vérifier si sur front de Pareto
            is_pareto = True
            for existing in pareto_front:
                if (existing['accuracy'] >= accuracy and 
                    existing['compression'] >= compression and
                    (existing['accuracy'] > accuracy or existing['compression'] > compression)):
                    is_pareto = False
                    break
            
            if is_pareto:
                # Retirer points dominés
                pareto_front = [
                    p for p in pareto_front 
                    if not (accuracy >= p['accuracy'] and compression >= p['compression'] and
                           (accuracy > p['accuracy'] or compression > p['compression']))
                ]
                pareto_front.append({
                    'params': params,
                    'accuracy': accuracy,
                    'compression': compression
                })
        
        return pareto_front
    
    def evaluate_accuracy(self, params):
        """Évalue accuracy"""
        compressed_model = self.apply_compression_with_params(params)
        self.quick_finetune(compressed_model, epochs=5)
        return self.evaluate_model(compressed_model, self.val_loader)
    
    def evaluate_compression(self, params):
        """Évalue ratio de compression"""
        compressed_model = self.apply_compression_with_params(params)
        original_params = sum(p.numel() for p in self.model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        return original_params / compressed_params
```

---

## Exercices

### Exercice 25.2.1
Implémentez grid search pour trouver meilleurs hyperparamètres de pruning.

### Exercice 25.2.2
Comparez grid search vs random search pour même budget d'évaluations.

### Exercice 25.2.3
Utilisez Optuna pour optimiser hyperparamètres de compression.

### Exercice 25.2.4
Créez optimisation multi-objectif qui trouve front de Pareto compression/performance.

---

## Points Clés à Retenir

> 📌 **Grid search est exhaustif mais coûteux**

> 📌 **Random search souvent meilleur pour budgets limités**

> 📌 **Bayesian optimization est efficace pour espaces continus**

> 📌 **Optuna offre framework moderne avec pruning automatique**

> 📌 **Optimisation multi-objectif trouve trade-offs optimaux**

> 📌 **Pruning automatique arrête évaluations prometteuses**

---

*Section précédente : [25.1 Workflow](./25_01_Workflow.md) | Section suivante : [25.3 Fine-tuning](./25_03_Finetuning.md)*

