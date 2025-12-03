# Chapitre 16 : Hardware-Aware Neural Architecture Search

---

## Introduction

Le **Neural Architecture Search (NAS)** cherche automatiquement les meilleures architectures de réseaux. Le **Hardware-Aware NAS** prend en compte les contraintes hardware (latence, énergie, surface) pour trouver des architectures optimisées pour le déploiement.

---

## Plan du Chapitre

1. [Principes du NAS](./16_01_Principes_NAS.md)
2. [Métriques Hardware](./16_02_Metriques.md)
3. [Espaces de Recherche Contraints](./16_03_Espaces.md)
4. [Algorithmes de Recherche Efficaces](./16_04_Algorithmes.md)
5. [Co-design Modèle-Hardware](./16_05_CoDesign.md)

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│              Hardware-Aware NAS Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Search     │───▶│  Hardware   │───▶│  Evaluate   │        │
│  │  Space      │    │  Simulator  │    │  Accuracy   │        │
│  └─────────────┘    └─────────────┘    └──────┬──────┘        │
│                                                 │                │
│                                                 ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Update     │◀───│  Multi-Obj  │◀───│  Metrics    │        │
│  │  Strategy   │    │  Optimizer  │    │  (Acc+Lat)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Problème d'Optimisation Multi-Objectif

```python
import torch
import torch.nn as nn
import numpy as np

class HardwareAwareNAS:
    """
    NAS avec contraintes hardware
    """
    
    def __init__(self, search_space, hardware_model):
        self.search_space = search_space
        self.hardware_model = hardware_model  # Simulateur hardware
        
    def evaluate_architecture(self, arch_config):
        """
        Évalue une architecture sur plusieurs objectifs
        
        Returns:
            dict avec 'accuracy', 'latency', 'energy', 'model_size'
        """
        # Crée le modèle
        model = self.search_space.create_model(arch_config)
        
        # Évalue la précision (nécessite entraînement partiel ou proxy)
        accuracy = self._evaluate_accuracy_proxy(model)
        
        # Évalue les métriques hardware
        latency = self.hardware_model.estimate_latency(model)
        energy = self.hardware_model.estimate_energy(model)
        model_size = sum(p.numel() for p in model.parameters())
        
        return {
            'accuracy': accuracy,
            'latency': latency,
            'energy': energy,
            'model_size': model_size,
            'arch_config': arch_config
        }
    
    def _evaluate_accuracy_proxy(self, model):
        """
        Utilise un proxy rapide pour estimer la performance
        (ex: gradient magnitude, expressivité, etc.)
        """
        # Proxy simplifié: nombre de paramètres (corrélé avec performance)
        n_params = sum(p.numel() for p in model.parameters())
        # Normalisé (exemple)
        proxy = min(n_params / 1e6, 1.0) * 0.9  # Max 90% de proxy
        return proxy

# Exemple de space de recherche
class SearchSpace:
    """
    Définit l'espace de recherche d'architectures
    """
    
    def __init__(self):
        self.config_space = {
            'n_layers': [3, 4, 5, 6],
            'width': [64, 128, 256, 512],
            'activation': ['relu', 'gelu'],
            'use_batch_norm': [True, False]
        }
    
    def create_model(self, config):
        """Crée un modèle selon la configuration"""
        layers = []
        
        input_dim = 784
        for i in range(config['n_layers']):
            output_dim = config['width']
            
            layers.append(nn.Linear(input_dim, output_dim))
            
            if config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(output_dim))
            
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
            
            input_dim = output_dim
        
        # Dernière couche
        layers.append(nn.Linear(input_dim, 10))
        
        return nn.Sequential(*layers)
```

---

## Métriques Hardware

### Latence

```python
class LatencyEstimator:
    """
    Estime la latence d'un modèle sur différents hardware
    """
    
    def __init__(self, hardware_type='fpga'):
        self.hardware_type = hardware_type
        # Modèles de latence pré-entraînés ou analytiques
        self.latency_model = self._load_latency_model()
    
    def estimate_latency(self, model, input_shape=(1, 3, 224, 224)):
        """
        Estime la latence en nanosecondes
        """
        if self.hardware_type == 'fpga':
            return self._estimate_fpga_latency(model, input_shape)
        elif self.hardware_type == 'gpu':
            return self._estimate_gpu_latency(model, input_shape)
        else:
            return self._estimate_cpu_latency(model, input_shape)
    
    def _estimate_fpga_latency(self, model, input_shape):
        """
        Estimation pour FPGA
        
        Approximation: somme des latences des couches
        """
        total_cycles = 0
        clock_period_ns = 5  # 200 MHz
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Cycles pour une couche linéaire (approximation)
                cycles = module.in_features * module.out_features
                total_cycles += cycles
            elif isinstance(module, nn.Conv2d):
                # Plus complexe: dépend de la taille de l'input
                cycles = (module.out_channels * module.in_channels * 
                         module.kernel_size[0] * module.kernel_size[1])
                total_cycles += cycles
        
        latency_ns = total_cycles * clock_period_ns
        return latency_ns
    
    def _load_latency_model(self):
        """
        Charge un modèle ML pour prédire la latence
        (peut être entraîné sur des mesures réelles)
        """
        # Placeholder
        return None

latency_est = LatencyEstimator('fpga')
model_test = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 10))
latency = latency_est.estimate_latency(model_test)
print(f"Latence estimée: {latency:.0f} ns")
```

### Énergie

```python
class EnergyEstimator:
    """
    Estime la consommation énergétique
    """
    
    def estimate_energy(self, model, input_shape):
        """
        Estime l'énergie en pJ (picojoules)
        """
        total_energy = 0
        
        # Énergie par opération (valeurs typiques)
        energy_per_mult = 4.6  # pJ pour une multiplication 8-bit sur FPGA
        energy_per_add = 0.9   # pJ pour une addition
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                n_mults = module.in_features * module.out_features
                n_adds = module.out_features
                energy = n_mults * energy_per_mult + n_adds * energy_per_add
                total_energy += energy
        
        return total_energy

energy_est = EnergyEstimator()
energy = energy_est.estimate_energy(model_test, (1, 256))
print(f"Énergie estimée: {energy:.2e} pJ")
```

---

## Algorithmes de Recherche

### Random Search (Baseline)

```python
def random_search(search_space, n_samples=100, hardware_model=None):
    """
    Recherche aléatoire (baseline)
    """
    best_arch = None
    best_score = float('-inf')
    
    for _ in range(n_samples):
        # Génère configuration aléatoire
        config = search_space.sample_random_config()
        
        # Évalue
        model = search_space.create_model(config)
        
        if hardware_model:
            metrics = hardware_model.evaluate_architecture(config)
            score = metrics['accuracy'] - 0.1 * (metrics['latency'] / 1e6)  # Trade-off
        else:
            score = evaluate_model(model)
        
        if score > best_score:
            best_score = score
            best_arch = config
    
    return best_arch, best_score
```

### Evolutionary Search

```python
class EvolutionaryNAS:
    """
    NAS basé sur algorithme évolutionnaire
    """
    
    def __init__(self, search_space, population_size=20, n_generations=50):
        self.search_space = search_space
        self.population_size = population_size
        self.n_generations = n_generations
    
    def search(self, fitness_fn):
        """
        Recherche évolutionnaire
        
        fitness_fn: fonction qui évalue une architecture
        """
        # Population initiale
        population = [
            self.search_space.sample_random_config()
            for _ in range(self.population_size)
        ]
        
        for generation in range(self.n_generations):
            # Évalue la population
            fitness_scores = [fitness_fn(arch) for arch in population]
            
            # Sélection (top 50%)
            sorted_pop = sorted(
                zip(population, fitness_scores),
                key=lambda x: x[1],
                reverse=True
            )
            elite = [arch for arch, _ in sorted_pop[:self.population_size//2]]
            
            # Crossover et mutation
            new_population = elite.copy()
            while len(new_population) < self.population_size:
                # Sélection de parents
                parent1 = np.random.choice(elite)
                parent2 = np.random.choice(elite)
                
                # Crossover
                child = self._crossover(parent1, parent2)
                
                # Mutation
                child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            best_fitness = max(fitness_scores)
            print(f"Generation {generation+1}: Best fitness = {best_fitness:.4f}")
        
        return max(population, key=fitness_fn)
    
    def _crossover(self, parent1, parent2):
        """Croisement de deux architectures"""
        # Exemple: prend la moyenne des largeurs
        child = parent1.copy()
        if 'width' in child and 'width' in parent2:
            child['width'] = (parent1['width'] + parent2['width']) // 2
        return child
    
    def _mutate(self, arch):
        """Mutation d'une architecture"""
        mutated = arch.copy()
        
        # Mutation aléatoire d'un paramètre
        param = np.random.choice(list(mutated.keys()))
        if param == 'width':
            mutated[param] = np.random.choice([64, 128, 256, 512])
        elif param == 'n_layers':
            mutated[param] = np.random.choice([3, 4, 5, 6])
        
        return mutated
```

### Differentiable Architecture Search (DARTS)

```python
class DifferentiableNAS:
    """
    NAS différentiable: optimise les poids d'architecture via gradient
    """
    
    def __init__(self, search_space):
        self.search_space = search_space
        
        # Poids d'architecture (paramètres à optimiser)
        self.alpha = nn.ParameterDict({
            'op_weights': nn.Parameter(torch.randn(8, 4))  # 8 opérations, 4 edges
        })
    
    def search(self, train_loader, val_loader, epochs=50):
        """
        Recherche différentiable
        """
        # Optimiseurs: un pour les poids, un pour alpha
        w_optimizer = torch.optim.Adam(self.search_space.parameters(), lr=3e-4)
        alpha_optimizer = torch.optim.Adam(self.alpha.parameters(), lr=3e-4)
        
        for epoch in range(epochs):
            # Phase 1: Entraîne les poids du modèle
            for x, y in train_loader:
                w_optimizer.zero_grad()
                output = self.search_space.forward_with_alpha(x, self.alpha)
                loss = F.cross_entropy(output, y)
                loss.backward()
                w_optimizer.step()
            
            # Phase 2: Optimise alpha sur validation set
            for x, y in val_loader:
                alpha_optimizer.zero_grad()
                output = self.search_space.forward_with_alpha(x, self.alpha)
                loss = F.cross_entropy(output, y)
                loss.backward()
                alpha_optimizer.step()
        
        # Dérive l'architecture finale depuis alpha
        final_arch = self._derive_architecture()
        return final_arch
    
    def _derive_architecture(self):
        """
        Dérive l'architecture discrète depuis alpha
        """
        # Prend l'opération avec le poids alpha le plus élevé
        arch = {}
        op_weights = self.alpha['op_weights']
        best_ops = op_weights.argmax(dim=0)
        # Convertit en configuration
        return arch
```

---

## Co-design Modèle-Hardware

```python
class HardwareModelCoDesign:
    """
    Co-design: optimise simultanément l'architecture et le mapping hardware
    """
    
    def __init__(self, search_space, hardware_config_space):
        self.search_space = search_space
        self.hardware_config_space = hardware_config_space
    
    def joint_optimization(self, target_latency_ns):
        """
        Optimise architecture et configuration hardware simultanément
        """
        best_arch = None
        best_hw_config = None
        best_accuracy = 0
        
        # Recherche dans l'espace combiné
        for arch_config in self.search_space.enumerate():
            for hw_config in self.hardware_config_space.enumerate():
                # Crée le modèle
                model = self.search_space.create_model(arch_config)
                
                # Simule sur hardware
                latency = simulate_on_hardware(model, hw_config)
                
                if latency <= target_latency_ns:
                    # Évalue la précision
                    accuracy = evaluate_accuracy(model)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_arch = arch_config
                        best_hw_config = hw_config
        
        return best_arch, best_hw_config
```

---

## Applications pour FPGA

```python
class FPGA_AwareNAS:
    """
    NAS spécifiquement optimisé pour FPGA
    """
    
    def __init__(self, fpga_constraints):
        self.constraints = fpga_constraints
        # {
        #     'max_lut': 100000,
        #     'max_dsp': 2000,
        #     'max_bram': 1000,
        #     'target_latency_ns': 100
        # }
    
    def evaluate_fpga_feasibility(self, model):
        """
        Vérifie si un modèle tient dans les contraintes FPGA
        """
        resources = estimate_fpga_resources(model)
        
        feasible = (
            resources['lut'] <= self.constraints['max_lut'] and
            resources['dsp'] <= self.constraints['max_dsp'] and
            resources['bram'] <= self.constraints['max_bram']
        )
        
        return feasible, resources
    
    def search_fpga_optimal(self, n_iterations=100):
        """
        Cherche l'architecture optimale pour FPGA
        """
        best_arch = None
        best_score = 0
        
        for _ in range(n_iterations):
            arch = sample_architecture()
            model = create_model(arch)
            
            feasible, resources = self.evaluate_fpga_feasibility(model)
            
            if feasible:
                accuracy = evaluate_accuracy(model)
                latency = estimate_latency(model)
                
                # Score combinant précision et latence
                score = accuracy - 0.1 * (latency / self.constraints['target_latency_ns'])
                
                if score > best_score:
                    best_score = score
                    best_arch = arch
        
        return best_arch
```

---

## Exercices

### Exercice 16.1
Implémentez un estimateur de latence basé sur ML qui prédit la latence d'une architecture depuis ses caractéristiques.

### Exercice 16.2
Créez un espace de recherche contraint pour des modèles de trigger avec latence < 100 ns.

### Exercice 16.3
Comparez random search, evolutionary search et DARTS sur un petit problème de recherche d'architecture.

---

## Points Clés à Retenir

> 📌 **Le Hardware-Aware NAS trouve des architectures optimisées pour le déploiement**

> 📌 **Les métriques hardware (latence, énergie) sont cruciales pour le NAS**

> 📌 **Le co-design modèle-hardware peut améliorer les performances globales**

> 📌 **Pour FPGA, les contraintes de ressources sont souvent le facteur limitant**

---

*Section suivante : [16.1 Principes du NAS](./16_01_Principes_NAS.md)*

