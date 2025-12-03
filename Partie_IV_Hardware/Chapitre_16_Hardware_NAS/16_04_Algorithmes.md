# 16.4 Algorithmes de Recherche Efficaces

---

## Introduction

Les **algorithmes de recherche efficaces** sont cruciaux pour le Hardware-Aware NAS, car l'espace de recherche est souvent très large et chaque évaluation d'architecture est coûteuse (entraînement partiel, simulation hardware).

Cette section présente différents algorithmes de recherche optimisés pour Hardware-Aware NAS, depuis les méthodes basiques jusqu'aux approches avancées qui minimisent le nombre d'évaluations nécessaires.

---

## Classification des Algorithmes

### Vue d'Ensemble

```python
class SearchAlgorithmTaxonomy:
    """
    Taxonomie des algorithmes de recherche NAS
    """
    
    def __init__(self):
        self.taxonomy = {
            'black_box': {
                'description': 'Algorithmes qui traitent l\'architecture comme boîte noire',
                'examples': ['Random Search', 'Grid Search', 'Evolutionary', 'Bayesian Optimization'],
                'pros': ['Général', 'Pas besoin de gradients', 'Robuste'],
                'cons': ['Peut nécessiter beaucoup d\'évaluations', 'Pas de guidance directe'],
                'complexity': 'O(n_evaluations × eval_cost)'
            },
            'gradient_based': {
                'description': 'Algorithmes différentiables',
                'examples': ['DARTS', 'ProxylessNAS', 'SNAS'],
                'pros': ['Rapide', 'Efficace', 'Gradient-based optimization'],
                'cons': ['Limité à certains espaces', 'Approximation continue'],
                'complexity': 'O(training_epochs × forward_pass)'
            },
            'reinforcement_learning': {
                'description': 'RL pour guider la recherche',
                'examples': ['NASNet', 'ENAS', 'PNAS'],
                'pros': ['Apprentissage de stratégies', 'Peut être efficace'],
                'cons': ['Complexe', 'Beaucoup d\'évaluations', 'Instable'],
                'complexity': 'O(episodes × eval_cost)'
            },
            'performance_predictor': {
                'description': 'Utilise prédicteur ML pour guider la recherche',
                'examples': ['BANANAS', 'NPENAS'],
                'pros': ['Très rapide une fois prédicteur entraîné', 'Peut guider efficacement'],
                'cons': ['Nécessite données d\'entraînement', 'Dépend de la qualité du prédicteur'],
                'complexity': 'O(n_predictions + n_train_evaluations)'
            },
            'weight_sharing': {
                'description': 'Partage des poids entre architectures',
                'examples': ['ENAS', 'One-Shot NAS', 'SPOS'],
                'pros': ['Efficace', 'Une seule fois l\'entraînement'],
                'cons': ['Approximation', 'Biais potentiel'],
                'complexity': 'O(supernet_training + n_architectures)'
            }
        }
    
    def display_taxonomy(self):
        """Affiche la taxonomie"""
        print("\n" + "="*70)
        print("Taxonomie des Algorithmes de Recherche")
        print("="*70)
        
        for category, info in self.taxonomy.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Exemples: {', '.join(info['examples'])}")
            print(f"  Pros:")
            for pro in info['pros']:
                print(f"    + {pro}")
            print(f"  Cons:")
            for con in info['cons']:
                print(f"    - {con}")
            print(f"  Complexity: {info['complexity']}")

taxonomy = SearchAlgorithmTaxonomy()
taxonomy.display_taxonomy()
```

---

## Random Search avec Filtrage Hardware

### Baseline Amélioré

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

class FilteredRandomSearch:
    """
    Random Search avec filtrage hardware pour efficacité
    """
    
    def __init__(self, search_space, hardware_evaluator, constraints: Dict):
        """
        Args:
            search_space: ConstrainedSearchSpace
            hardware_evaluator: HardwareEvaluator
            constraints: Contraintes hardware
        """
        self.search_space = search_space
        self.hardware_eval = hardware_evaluator
        self.constraints = constraints
        
        # Historique de recherche
        self.evaluation_history = []
        self.valid_architectures = []
    
    def search(self, n_iterations: int = 100, input_shape: Tuple = (1, 784)) -> Dict:
        """
        Recherche random avec filtrage
        
        Returns:
            Meilleure architecture trouvée avec métriques
        """
        best_arch = None
        best_score = float('-inf')
        
        for iteration in range(n_iterations):
            # Générer configuration
            config = self.search_space.sample_valid_config(max_attempts=50, input_shape=input_shape)
            
            if config is None:
                continue  # Aucune config valide trouvée
            
            # Créer modèle
            model = self.search_space.create_model_from_config(config, input_shape[1])
            
            # Vérifier contraintes
            is_valid, metrics = self.search_space.check_hardware_constraints(model, input_shape)
            
            if not is_valid:
                continue
            
            # Évaluer performance (proxy rapide)
            performance = self._evaluate_performance_proxy(model, input_shape)
            
            # Score combiné (performance - pénalités hardware)
            score = self._compute_score(performance, metrics)
            
            # Enregistrer
            self.evaluation_history.append({
                'config': config,
                'performance': performance,
                'metrics': metrics,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_arch = {
                    'config': config,
                    'model': model,
                    'performance': performance,
                    'metrics': metrics,
                    'score': score
                }
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{n_iterations}: Best score = {best_score:.4f}")
        
        return best_arch
    
    def _evaluate_performance_proxy(self, model: nn.Module, input_shape: Tuple) -> float:
        """
        Proxy rapide pour estimer la performance
        (en pratique, utiliser entraînement partiel ou prédicteur)
        """
        # Proxy simplifié: corrélation avec nombre de paramètres
        n_params = sum(p.numel() for p in model.parameters())
        
        # Approximation: plus de paramètres = meilleure performance (jusqu'à un point)
        proxy = min(0.7 + (n_params / 1e6) * 0.2, 0.95)
        
        return proxy
    
    def _compute_score(self, performance: float, metrics: Dict) -> float:
        """
        Score combiné: performance - pénalités hardware
        """
        # Pénalités normalisées
        latency_penalty = (metrics['latency_us'] / 100.0) * 0.2  # référence: 100 μs
        energy_penalty = (metrics['energy_nj'] / 1000.0) * 0.1  # référence: 1000 nJ
        size_penalty = (metrics['model_size_mb'] / 10.0) * 0.1  # référence: 10 MB
        
        score = performance - latency_penalty - energy_penalty - size_penalty
        return score
```

---

## Evolutionary Search avec Contraintes Hardware

### Algorithme Évolutionnaire Optimisé

```python
class ConstrainedEvolutionarySearch:
    """
    Recherche évolutionnaire avec contraintes hardware
    """
    
    def __init__(self, search_space, hardware_evaluator, constraints: Dict):
        self.search_space = search_space
        self.hardware_eval = hardware_evaluator
        self.constraints = constraints
        
        self.population = []
        self.evaluation_history = []
    
    def search(self, population_size: int = 20, n_generations: int = 50,
               mutation_rate: float = 0.3, crossover_rate: float = 0.5,
               input_shape: Tuple = (1, 784)) -> Dict:
        """
        Recherche évolutionnaire
        
        Args:
            population_size: Taille de la population
            n_generations: Nombre de générations
            mutation_rate: Probabilité de mutation
            crossover_rate: Probabilité de croisement
        """
        # Initialisation: population aléatoire valide
        print("Initialisation de la population...")
        self.population = []
        for i in range(population_size):
            config = self.search_space.sample_valid_config(max_attempts=100, input_shape=input_shape)
            if config:
                model = self.search_space.create_model_from_config(config, input_shape[1])
                _, metrics = self.search_space.check_hardware_constraints(model, input_shape)
                performance = self._evaluate_performance_proxy(model, input_shape)
                score = self._compute_score(performance, metrics)
                
                self.population.append({
                    'config': config,
                    'score': score,
                    'metrics': metrics,
                    'performance': performance
                })
        
        print(f"Population initiale: {len(self.population)} individus valides")
        
        # Évolution
        for generation in range(n_generations):
            # Évaluation
            scores = [ind['score'] for ind in self.population]
            
            # Sélection (top 50% + quelques random)
            sorted_pop = sorted(self.population, key=lambda x: x['score'], reverse=True)
            elite_size = population_size // 2
            elite = sorted_pop[:elite_size]
            
            # Nouvelle génération
            new_population = elite.copy()
            
            # Générer enfants
            while len(new_population) < population_size:
                # Sélection de parents (tournament selection)
                parent1 = self._tournament_selection(elite, tournament_size=3)
                parent2 = self._tournament_selection(elite, tournament_size=3)
                
                # Croisement
                if random.random() < crossover_rate:
                    child_config = self._crossover(parent1['config'], parent2['config'])
                else:
                    child_config = parent1['config'].copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child_config = self._mutate(child_config)
                
                # Validation et évaluation
                model = self.search_space.create_model_from_config(child_config, input_shape[1])
                is_valid, metrics = self.search_space.check_hardware_constraints(model, input_shape)
                
                if is_valid:
                    performance = self._evaluate_performance_proxy(model, input_shape)
                    score = self._compute_score(performance, metrics)
                    
                    new_population.append({
                        'config': child_config,
                        'score': score,
                        'metrics': metrics,
                        'performance': performance
                    })
            
            self.population = new_population
            
            best = max(self.population, key=lambda x: x['score'])
            print(f"Generation {generation+1}/{n_generations}: Best score = {best['score']:.4f}, "
                  f"Latency = {best['metrics']['latency_us']:.2f} μs")
        
        # Retourner meilleure solution
        best = max(self.population, key=lambda x: x['score'])
        model = self.search_space.create_model_from_config(best['config'], input_shape[1])
        return {
            'config': best['config'],
            'model': model,
            'score': best['score'],
            'metrics': best['metrics'],
            'performance': best['performance']
        }
    
    def _tournament_selection(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """Sélection par tournoi"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x['score'])
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Croisement de deux configurations"""
        child = parent1.copy()
        
        # Croisement: moyenne pour valeurs numériques, choix pour catégorielles
        if 'num_layers' in child:
            child['num_layers'] = (parent1['num_layers'] + parent2['num_layers']) // 2
        
        if 'layer_widths' in child and 'layer_widths' in parent2:
            # Prendre largeurs alternées
            widths1 = parent1.get('layer_widths', [])
            widths2 = parent2.get('layer_widths', [])
            max_len = max(len(widths1), len(widths2))
            child_widths = []
            for i in range(max_len):
                if i % 2 == 0 and i < len(widths1):
                    child_widths.append(widths1[i])
                elif i < len(widths2):
                    child_widths.append(widths2[i])
            child['layer_widths'] = child_widths[:child['num_layers']]
        
        if 'activation' in child:
            child['activation'] = random.choice([parent1['activation'], parent2['activation']])
        
        return child
    
    def _mutate(self, config: Dict) -> Dict:
        """Mutation d'une configuration"""
        mutated = config.copy()
        
        # Mutation aléatoire d'un paramètre
        param_to_mutate = random.choice(list(mutated.keys()))
        
        if param_to_mutate == 'num_layers':
            mutated['num_layers'] = random.choice(self.search_space.base_space['num_layers'])
        elif param_to_mutate == 'layer_widths':
            mutated['layer_widths'] = [random.choice(self.search_space.base_space['layer_widths'])]
        elif param_to_mutate == 'activation':
            mutated['activation'] = random.choice(self.search_space.base_space['activation'])
        elif param_to_mutate == 'use_batch_norm':
            mutated['use_batch_norm'] = not mutated['use_batch_norm']
        
        return mutated
    
    def _evaluate_performance_proxy(self, model: nn.Module, input_shape: Tuple) -> float:
        """Proxy de performance"""
        n_params = sum(p.numel() for p in model.parameters())
        return min(0.7 + (n_params / 1e6) * 0.2, 0.95)
    
    def _compute_score(self, performance: float, metrics: Dict) -> float:
        """Score combiné"""
        latency_penalty = (metrics['latency_us'] / 100.0) * 0.2
        energy_penalty = (metrics['energy_nj'] / 1000.0) * 0.1
        return performance - latency_penalty - energy_penalty
```

---

## Bayesian Optimization avec Prédicteur Hardware

### Optimisation Bayésienne Efficace

```python
class BayesianOptimizationNAS:
    """
    Bayesian Optimization pour Hardware-Aware NAS
    
    Utilise un modèle probabiliste (Gaussian Process) pour guider la recherche
    """
    
    def __init__(self, search_space, hardware_evaluator, constraints: Dict):
        self.search_space = search_space
        self.hardware_eval = hardware_evaluator
        self.constraints = constraints
        
        # Historique d'observations
        self.X_observed = []  # Configurations
        self.y_observed = []  # Scores
        
        # En pratique, utiliser sklearn.gaussian_process ou GPyTorch
        self.gp_model = None
    
    def search(self, n_iterations: int = 50, n_initial: int = 10,
               input_shape: Tuple = (1, 784)) -> Dict:
        """
        Recherche par optimisation bayésienne
        
        Args:
            n_iterations: Nombre d'itérations
            n_initial: Nombre d'évaluations initiales (random)
        """
        # Phase d'exploration initiale
        print(f"Phase d'exploration initiale ({n_initial} évaluations)...")
        for i in range(n_initial):
            config = self.search_space.sample_valid_config(max_attempts=50, input_shape=input_shape)
            if config:
                score = self._evaluate_config(config, input_shape)
                self.X_observed.append(self._config_to_vector(config))
                self.y_observed.append(score)
        
        # Phase d'optimisation bayésienne
        print(f"Phase d'optimisation bayésienne ({n_iterations} itérations)...")
        for iteration in range(n_iterations):
            # Entraîner modèle GP (simplifié ici)
            # En pratique: self.gp_model.fit(self.X_observed, self.y_observed)
            
            # Acquisition function: choisir prochain point à évaluer
            # En pratique: utiliser Expected Improvement (EI) ou Upper Confidence Bound (UCB)
            next_config = self._acquisition_function_maximization()
            
            # Évaluer
            score = self._evaluate_config(next_config, input_shape)
            self.X_observed.append(self._config_to_vector(next_config))
            self.y_observed.append(score)
            
            best_idx = np.argmax(self.y_observed)
            best_score = self.y_observed[best_idx]
            
            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration+1}/{n_iterations}: Best score = {best_score:.4f}")
        
        # Retourner meilleure solution
        best_idx = np.argmax(self.y_observed)
        best_vector = self.X_observed[best_idx]
        best_config = self._vector_to_config(best_vector)
        model = self.search_space.create_model_from_config(best_config, input_shape[1])
        _, metrics = self.search_space.check_hardware_constraints(model, input_shape)
        
        return {
            'config': best_config,
            'model': model,
            'score': self.y_observed[best_idx],
            'metrics': metrics
        }
    
    def _evaluate_config(self, config: Dict, input_shape: Tuple) -> float:
        """Évalue une configuration"""
        model = self.search_space.create_model_from_config(config, input_shape[1])
        is_valid, metrics = self.search_space.check_hardware_constraints(model, input_shape)
        
        if not is_valid:
            return -float('inf')  # Pénalité forte
        
        performance = self._evaluate_performance_proxy(model, input_shape)
        score = self._compute_score(performance, metrics)
        return score
    
    def _acquisition_function_maximization(self) -> Dict:
        """
        Maximise la fonction d'acquisition pour choisir le prochain point
        
        En pratique, utiliser Expected Improvement avec GP
        Ici: approximation simple avec exploration/exploitation
        """
        # Simplification: combinaison exploration aléatoire et exploitation
        if len(self.y_observed) < 5:
            # Plus d'exploration au début
            return self.search_space.sample_valid_config(max_attempts=50, input_shape=(1, 784))
        else:
            # Exploitation: chercher autour des bonnes solutions
            best_idx = np.argmax(self.y_observed)
            best_vector = self.X_observed[best_idx]
            best_config = self._vector_to_config(best_vector)
            
            # Mutation locale
            mutated = self._mutate_local(best_config)
            return mutated
    
    def _mutate_local(self, config: Dict) -> Dict:
        """Mutation locale pour exploitation"""
        mutated = config.copy()
        
        # Petite mutation
        if random.random() < 0.5 and 'num_layers' in mutated:
            current = mutated['num_layers']
            options = [max(3, current-1), current, min(8, current+1)]
            mutated['num_layers'] = random.choice(options)
        
        return mutated
    
    def _config_to_vector(self, config: Dict) -> np.ndarray:
        """Convertit config en vecteur numérique"""
        # Encodage simple (en pratique, utiliser one-hot ou embedding)
        vector = [
            config.get('num_layers', 5) / 10.0,  # normalisé
            len(config.get('layer_widths', [])) / 10.0,
            1.0 if config.get('use_batch_norm', False) else 0.0,
            1.0 if config.get('use_dropout', False) else 0.0
        ]
        return np.array(vector)
    
    def _vector_to_config(self, vector: np.ndarray) -> Dict:
        """Convertit vecteur en config"""
        # Décodage (approximation)
        config = {
            'num_layers': int(vector[0] * 10),
            'layer_widths': [128] * int(vector[1] * 10),
            'use_batch_norm': bool(vector[2] > 0.5),
            'use_dropout': bool(vector[3] > 0.5),
            'activation': 'relu',
            'dropout_rate': 0.2
        }
        return config
    
    def _evaluate_performance_proxy(self, model: nn.Module, input_shape: Tuple) -> float:
        """Proxy de performance"""
        n_params = sum(p.numel() for p in model.parameters())
        return min(0.7 + (n_params / 1e6) * 0.2, 0.95)
    
    def _compute_score(self, performance: float, metrics: Dict) -> float:
        """Score combiné"""
        latency_penalty = (metrics['latency_us'] / 100.0) * 0.2
        return performance - latency_penalty
```

---

## Differentiable Architecture Search (DARTS) avec Hardware

### NAS Différentiable Adapté

```python
class HardwareAwareDARTS:
    """
    DARTS (Differentiable Architecture Search) adapté pour hardware
    
    Optimise simultanément les poids du modèle et l'architecture
    """
    
    def __init__(self, search_space, hardware_evaluator, constraints: Dict):
        self.search_space = search_space
        self.hardware_eval = hardware_evaluator
        self.constraints = constraints
        
        # Paramètres d'architecture alpha (à optimiser)
        self.alpha = None
    
    def search(self, train_loader, val_loader, epochs: int = 50,
               w_lr: float = 3e-4, alpha_lr: float = 3e-4) -> Dict:
        """
        Recherche différentiable
        
        Args:
            train_loader: DataLoader pour entraînement
            val_loader: DataLoader pour validation
            epochs: Nombre d'epochs
            w_lr: Learning rate pour poids
            alpha_lr: Learning rate pour alpha
        """
        import torch.optim as optim
        import torch.nn.functional as F
        
        # Initialiser alpha (paramètres d'architecture)
        # En pratique, alpha définit les poids des opérations dans le super-net
        # Simplifié ici
        self.alpha = nn.Parameter(torch.randn(8, 4))  # 8 opérations, 4 edges
        
        # Créer super-net (modèle avec toutes les opérations)
        supernet = self._create_supernet()
        
        # Optimiseurs
        w_optimizer = optim.Adam(supernet.parameters(), lr=w_lr)
        alpha_optimizer = optim.Adam([self.alpha], lr=alpha_lr)
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Phase 1: Entraîner les poids du modèle
            supernet.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                w_optimizer.zero_grad()
                
                # Forward avec architecture actuelle
                output = self._forward_with_alpha(supernet, data, self.alpha)
                
                loss = criterion(output, target)
                loss.backward()
                w_optimizer.step()
            
            # Phase 2: Optimiser alpha sur validation set
            supernet.eval()
            val_loss = 0
            for data, target in val_loader:
                alpha_optimizer.zero_grad()
                
                output = self._forward_with_alpha(supernet, data, self.alpha)
                loss = criterion(output, target)
                
                # Ajouter pénalité hardware
                if epoch > 10:  # Commencer après quelques epochs
                    hardware_penalty = self._compute_hardware_penalty(supernet)
                    loss = loss + 0.1 * hardware_penalty
                
                loss.backward()
                alpha_optimizer.step()
                
                val_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Val loss = {val_loss/len(val_loader):.4f}")
        
        # Dériver architecture finale depuis alpha
        final_arch = self._derive_architecture()
        
        return {
            'architecture': final_arch,
            'alpha': self.alpha.detach().numpy()
        }
    
    def _create_supernet(self) -> nn.Module:
        """
        Crée un super-net avec toutes les opérations possibles
        """
        # Simplifié: en pratique, créer un DAG avec toutes les opérations
        return nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def _forward_with_alpha(self, model: nn.Module, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Forward pass avec architecture pondérée par alpha
        """
        # Simplifié: en pratique, utiliser softmax sur alpha et mélanger opérations
        # Ici: forward standard
        return model(x)
    
    def _compute_hardware_penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Calcule pénalité hardware pour guider la recherche
        """
        # Évaluer métriques hardware (approximation)
        metrics = self.hardware_eval.evaluate_architecture(model, (1, 784))
        
        # Pénalité normalisée
        latency_penalty = torch.tensor(metrics['latency_us'] / 100.0)
        energy_penalty = torch.tensor(metrics['energy_nj'] / 1000.0)
        
        return latency_penalty + energy_penalty
    
    def _derive_architecture(self) -> Dict:
        """
        Dérive l'architecture discrète depuis alpha
        """
        # Prendre opérations avec alpha le plus élevé
        # Simplifié ici
        return {'num_layers': 4, 'width': 128}
```

---

## Comparaison des Algorithmes

### Benchmark

```python
class AlgorithmComparison:
    """
    Compare différents algorithmes de recherche
    """
    
    def __init__(self, search_space, hardware_evaluator, constraints: Dict):
        self.search_space = search_space
        self.hardware_eval = hardware_evaluator
        self.constraints = constraints
    
    def compare_algorithms(self, n_evaluations: int = 100, input_shape: Tuple = (1, 784)) -> Dict:
        """
        Compare plusieurs algorithmes
        """
        results = {}
        
        # Random Search
        print("\n" + "="*70)
        print("Random Search")
        print("="*70)
        random_search = FilteredRandomSearch(self.search_space, self.hardware_eval, self.constraints)
        random_result = random_search.search(n_iterations=n_evaluations, input_shape=input_shape)
        results['random'] = {
            'best_score': random_result['score'] if random_result else -float('inf'),
            'n_evaluations': n_evaluations
        }
        
        # Evolutionary
        print("\n" + "="*70)
        print("Evolutionary Search")
        print("="*70)
        evo_search = ConstrainedEvolutionarySearch(self.search_space, self.hardware_eval, self.constraints)
        evo_result = evo_search.search(population_size=20, n_generations=n_evaluations//20, input_shape=input_shape)
        results['evolutionary'] = {
            'best_score': evo_result['score'],
            'n_evaluations': n_evaluations
        }
        
        # Bayesian Optimization
        print("\n" + "="*70)
        print("Bayesian Optimization")
        print("="*70)
        bo_search = BayesianOptimizationNAS(self.search_space, self.hardware_eval, self.constraints)
        bo_result = bo_search.search(n_iterations=n_evaluations//2, n_initial=10, input_shape=input_shape)
        results['bayesian'] = {
            'best_score': bo_result['score'],
            'n_evaluations': n_evaluations//2 + 10
        }
        
        # Afficher comparaison
        print("\n" + "="*70)
        print("Comparaison des Algorithmes")
        print("="*70)
        
        for algo_name, result in results.items():
            print(f"\n{algo_name.upper()}:")
            print(f"  Best score: {result['best_score']:.4f}")
            print(f"  Evaluations: {result['n_evaluations']}")
            print(f"  Efficiency (score/eval): {result['best_score']/result['n_evaluations']:.6f}")
        
        return results
```

---

## Exercices

### Exercice 16.4.1
Implémentez une version améliorée de Random Search qui utilise un prédicteur de performance pour filtrer les architectures prometteuses.

### Exercice 16.4.2
Comparez les performances de Random Search, Evolutionary Search et Bayesian Optimization sur un espace de recherche donné avec contraintes hardware.

### Exercice 16.4.3
Adaptez DARTS pour inclure des pénalités hardware dans la fonction objectif et comparez avec version sans pénalités.

### Exercice 16.4.4
Implémentez un algorithme hybride qui combine Random Search (exploration) et Evolutionary Search (exploitation).

---

## Points Clés à Retenir

> 📌 **Random Search est une baseline simple mais peut être efficace avec filtrage hardware**

> 📌 **Evolutionary Search est robuste et bien adapté aux espaces discrets avec contraintes**

> 📌 **Bayesian Optimization est efficace pour espaces continus et nécessite peu d'évaluations**

> 📌 **DARTS est très rapide mais limité à certains espaces et nécessite adaptation pour hardware**

> 📌 **Le choix de l'algorithme dépend du budget d'évaluations et du type d'espace de recherche**

> 📌 **Les algorithmes hybrides peuvent combiner les avantages de différentes approches**

---

*Section précédente : [16.3 Espaces de Recherche Contraints](./16_03_Espaces.md) | Section suivante : [16.5 Co-design Modèle-Hardware](./16_05_CoDesign.md)*

