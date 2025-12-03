# 16.1 Principes du Neural Architecture Search

---

## Introduction

Le **Neural Architecture Search (NAS)** automatise la recherche des meilleures architectures de réseaux de neurones. Cette section présente les principes fondamentaux du NAS et leur extension au **Hardware-Aware NAS**.

---

## Problème du NAS

### Définition

```python
class NASProblem:
    """
    Formulation du problème NAS
    """
    
    def __init__(self):
        self.problem_formulation = """
        Trouver l'architecture optimale A* qui maximise la performance
        sur une tâche donnée, dans un espace de recherche défini.
        
        A* = argmax_{A in SearchSpace} Performance(A, D)
        
        où:
        - A: Architecture
        - SearchSpace: Espace de recherche d'architectures
        - D: Dataset
        - Performance: Métrique (accuracy, F1-score, etc.)
        """
    
    def display_problem(self):
        """Affiche la formulation du problème"""
        print("\n" + "="*60)
        print("NAS Problem Formulation")
        print("="*60)
        print(self.problem_formulation)

nas_problem = NASProblem()
nas_problem.display_problem()
```

---

## Composants du NAS

### Structure Générale

```
┌─────────────────────────────────────────────────────────────────┐
│                    Composants du NAS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Search Space                                                │
│     └─ Définit les architectures possibles                     │
│                                                                 │
│  2. Search Strategy                                             │
│     └─ Algorithme de recherche (random, evolutionary, etc.)    │
│                                                                 │
│  3. Performance Estimator                                       │
│     └─ Estime la performance d'une architecture                │
│                                                                 │
│  4. Evaluation Function                                         │
│     └─ Évalue réellement une architecture (entraînement)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Search Space

### Types d'Espaces de Recherche

```python
class SearchSpace:
    """
    Espaces de recherche d'architectures
    """
    
    def __init__(self):
        self.space_types = {
            'micro_architecture': {
                'description': 'Recherche dans les cellules/blocs',
                'example': 'Cellules pour CNN (conv blocks)',
                'size': 'Relativement petit',
                'use_case': 'Recherche efficace de cellules réutilisables'
            },
            'macro_architecture': {
                'description': 'Recherche dans la structure globale',
                'example': "Nombre de couches, largeur de l'architecture",
                'size': 'Plus grand',
                'use_case': 'Recherche complète de l\'architecture'
            },
            'hierarchical': {
                'description': 'Recherche à plusieurs niveaux',
                'example': 'Cellules + composition de cellules',
                'size': 'Très grand mais structuré',
                'use_case': 'Meilleur compromis expressivité/efficacité'
            }
        }
    
    def create_micro_space(self):
        """
        Crée un espace de recherche micro-architecture
        
        Exemple: Recherche de cellules CNN
        """
        space = {
            'operations': [
                'conv_3x3',
                'conv_5x5',
                'depthwise_conv',
                'max_pool',
                'avg_pool',
                'identity',
                'skip_connection'
            ],
            'number_of_ops_per_cell': [2, 3, 4],
            'normalization': ['batch_norm', 'layer_norm', 'none'],
            'activation': ['relu', 'swish', 'gelu']
        }
        return space
    
    def create_macro_space(self):
        """
        Crée un espace de recherche macro-architecture
        
        Exemple: Architecture complète
        """
        space = {
            'num_layers': [3, 4, 5, 6, 7, 8],
            'layer_width': [64, 128, 256, 512],
            'layer_types': ['dense', 'conv2d'],
            'activation': ['relu', 'gelu', 'swish'],
            'use_dropout': [True, False],
            'dropout_rate': [0.1, 0.2, 0.3, 0.5]
        }
        return space
    
    def display_space_types(self):
        """Affiche les types d'espaces"""
        print("\n" + "="*60)
        print("Search Space Types")
        print("="*60)
        
        for space_type, info in self.space_types.items():
            print(f"\n{space_type.replace('_', ' ').title()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

search_space = SearchSpace()
search_space.display_space_types()

# Exemples
micro = search_space.create_micro_space()
macro = search_space.create_macro_space()

print("\n" + "="*60)
print("Example Search Spaces")
print("="*60)
print("\nMicro-architecture space:")
for key, value in micro.items():
    print(f"  {key}: {value}")

print("\nMacro-architecture space:")
for key, value in macro.items():
    print(f"  {key}: {value}")
```

---

## Search Strategy

### Algorithmes de Recherche

```python
class SearchStrategy:
    """
    Stratégies de recherche NAS
    """
    
    def __init__(self):
        self.strategies = {
            'random_search': {
                'description': 'Recherche aléatoire dans l\'espace',
                'pros': ['Simple', 'Pas de biais', 'Facile à paralléliser'],
                'cons': ['Peu efficace', 'Pas de guidance'],
                'complexity': 'O(n) pour n architectures'
            },
            'grid_search': {
                'description': 'Recherche exhaustive sur grille discrète',
                'pros': ['Complet sur la grille', 'Déterministe'],
                'cons': ['Combinatorial explosion', 'Impossible pour grands espaces'],
                'complexity': 'O(∏|dimensions|)'
            },
            'evolutionary': {
                'description': 'Algorithmes évolutionnaires (genetic algorithms)',
                'pros': ['Efficace', 'Peut explorer large espace'],
                'cons': ['Beaucoup d\'évaluations nécessaires'],
                'complexity': 'O(generations × population_size × eval_time)'
            },
            'reinforcement_learning': {
                'description': 'RL pour guider la recherche',
                'pros': ['Apprentissage de bonnes stratégies', 'Efficace à long terme'],
                'cons': ['Complexe', 'Besoin de beaucoup d\'évaluations'],
                'complexity': 'O(episodes × eval_time)'
            },
            'differentiable': {
                'description': 'NAS différentiable (DARTS, etc.)',
                'pros': ['Rapide (gradient-based)', 'Efficace'],
                'cons': ['Limité à certains espaces', 'Approximation continue'],
                'complexity': 'O(training_epochs × forward_pass)'
            },
            'bayesian_optimization': {
                'description': 'Optimisation bayésienne',
                'pros': ['Efficace pour espaces continus', 'Peu d\'évaluations'],
                'cons': ['Complexe', 'Nécessite modèle probabiliste'],
                'complexity': 'O(n²) pour n évaluations'
            }
        }
    
    def display_strategies(self):
        """Affiche les stratégies"""
        print("\n" + "="*60)
        print("Search Strategies")
        print("="*60)
        
        for strategy, info in self.strategies.items():
            print(f"\n{strategy.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Pros:")
            for pro in info['pros']:
                print(f"    + {pro}")
            print(f"  Cons:")
            for con in info['cons']:
                print(f"    - {con}")
            print(f"  Complexity: {info['complexity']}")

strategy = SearchStrategy()
strategy.display_strategies()
```

---

## Performance Estimation

### Méthodes d'Estimation

```python
class PerformanceEstimation:
    """
    Estimation de la performance d'une architecture
    """
    
    def __init__(self):
        self.estimation_methods = {
            'full_training': {
                'description': 'Entraînement complet du modèle',
                'accuracy': 'Très précise',
                'cost': 'Très élevé (heures/jours)',
                'use_case': 'Évaluation finale, petites recherches'
            },
            'partial_training': {
                'description': 'Entraînement partiel (quelques epochs)',
                'accuracy': 'Assez précise',
                'cost': 'Modéré',
                'use_case': 'Recherche NAS standard'
            },
            'weight_sharing': {
                'description': 'Partage des poids entre architectures',
                'accuracy': 'Modérée',
                'cost': 'Faible',
                'use_case': 'ENAS, One-shot NAS'
            },
            'performance_predictor': {
                'description': 'Modèle ML qui prédit la performance',
                'accuracy': 'Variable (dépend du predictor)',
                'cost': 'Très faible',
                'use_case': 'Recherche rapide, pré-filtrage'
            },
            'gradient_based_proxy': {
                'description': 'Métriques basées sur gradients',
                'accuracy': 'Faible-modérée',
                'cost': 'Très faible',
                'use_case': 'Proxies rapides, premières étapes'
            }
        }
    
    def train_partial(self, model, train_loader, epochs=5):
        """
        Entraînement partiel pour estimation rapide
        
        Args:
            model: Modèle à entraîner
            train_loader: DataLoader
            epochs: Nombre d'epochs (réduit)
        
        Returns:
            Accuracy estimée
        """
        import torch.nn as nn
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Évaluation rapide
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in train_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def performance_predictor(self, arch_features):
        """
        Prédicteur ML de performance
        
        Args:
            arch_features: Caractéristiques de l'architecture
                (nombre paramètres, profondeur, largeur, etc.)
        
        Returns:
            Performance prédite
        """
        # Exemple simplifié: corrélation avec nombre de paramètres
        # En pratique, utiliser un modèle ML entraîné
        n_params = arch_features.get('num_params', 0)
        depth = arch_features.get('depth', 0)
        
        # Approximation simplifiée
        predicted_perf = min(0.5 + (n_params / 1e6) * 0.3 + depth * 0.01, 0.95)
        return predicted_perf
    
    def display_methods(self):
        """Affiche les méthodes"""
        print("\n" + "="*60)
        print("Performance Estimation Methods")
        print("="*60)
        
        for method, info in self.estimation_methods.items():
            print(f"\n{method.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Accuracy: {info['accuracy']}")
            print(f"  Cost: {info['cost']}")
            print(f"  Use case: {info['use_case']}")

estimation = PerformanceEstimation()
estimation.display_methods()
```

---

## Hardware-Aware NAS

### Extension du NAS Standard

```python
class HardwareAwareNAS:
    """
    Extension du NAS pour prendre en compte le hardware
    """
    
    def __init__(self):
        self.extension = """
        Hardware-Aware NAS étend le problème NAS standard:
        
        NAS Standard:
        A* = argmax_{A} Accuracy(A, D)
        
        Hardware-Aware NAS:
        A* = argmax_{A} f(Accuracy(A, D), Hardware_Metrics(A, H))
        
        où:
        - Hardware_Metrics: Latence, énergie, surface, etc.
        - H: Configuration hardware cible
        - f: Fonction de compromis multi-objectif
        """
    
    def multi_objective_formulation(self):
        """
        Formulation multi-objectif
        """
        objectives = {
            'primary': 'Accuracy (ou autre métrique ML)',
            'secondary': [
                'Latency (ns)',
                'Energy (pJ)',
                'Model size (MB)',
                'Resource usage (LUT, DSP, etc.)'
            ]
        }
        
        methods = {
            'weighted_sum': {
                'formulation': 'score = w1*accuracy - w2*latency - w3*energy',
                'pros': ['Simple', 'Contrôle direct des poids'],
                'cons': ['Choix des poids arbitraire', 'Pareto front implicite']
            },
            'pareto_optimization': {
                'formulation': 'Trouve le front de Pareto',
                'pros': ['Pas besoin de poids', 'Toutes les solutions optimales'],
                'cons': ['Plusieurs solutions', 'Choix final nécessaire']
            },
            'constrained_optimization': {
                'formulation': 'Max accuracy s.t. latency < threshold',
                'pros': ['Contraintes explicites', 'Clair pour applications'],
                'cons': 'Choix du seuil critique'
            }
        }
        
        return objectives, methods
    
    def display_extension(self):
        """Affiche l'extension hardware-aware"""
        print("\n" + "="*60)
        print("Hardware-Aware NAS Extension")
        print("="*60)
        print(self.extension)
        
        objectives, methods = self.multi_objective_formulation()
        
        print("\nObjectives:")
        print(f"  Primary: {objectives['primary']}")
        print("  Secondary:")
        for obj in objectives['secondary']:
            print(f"    • {obj}")
        
        print("\nMulti-objective Methods:")
        for method, info in methods.items():
            print(f"\n  {method.replace('_', ' ').title()}:")
            print(f"    Formulation: {info['formulation']}")
            print(f"    Pros:")
            for pro in info['pros']:
                print(f"      + {pro}")
            if 'cons' in info:
                if isinstance(info['cons'], list):
                    for con in info['cons']:
                        print(f"      - {con}")
                else:
                    print(f"    Cons: {info['cons']}")

hardware_nas = HardwareAwareNAS()
hardware_nas.display_extension()
```

---

## Workflow Typique du Hardware-Aware NAS

```python
class NASWorkflow:
    """
    Workflow typique d'un Hardware-Aware NAS
    """
    
    def generate_workflow(self):
        """Génère le workflow"""
        workflow = """
Hardware-Aware NAS Workflow:

1. Problem Definition
   ├─ Définir tâche ML
   ├─ Identifier contraintes hardware
   └─ Définir métriques objectives

2. Search Space Design
   ├─ Définir espace d'architectures
   ├─ Inclure contraintes hardware dans l'espace
   └─ Valider espace de recherche

3. Hardware Model Setup
   ├─ Créer simulateur/estimateur hardware
   ├─ Valider prédictions hardware
   └─ Intégrer dans loop de recherche

4. Search Algorithm Selection
   ├─ Choisir stratégie de recherche
   ├─ Configurer algorithmes
   └─ Définir critères d'arrêt

5. Search Execution
   ├─ Itérer: générer architecture
   ├─ Évaluer: accuracy + hardware metrics
   ├─ Mettre à jour: stratégie de recherche
   └─ Répéter jusqu'à convergence

6. Architecture Selection
   ├─ Analyser résultats (Pareto front)
   ├─ Sélectionner architecture(s) finale(s)
   └─ Validation complète

7. Deployment
   ├─ Entraînement complet architecture sélectionnée
   ├─ Déploiement sur hardware cible
   └─ Validation en conditions réelles
"""
        return workflow
    
    def display_workflow(self):
        """Affiche le workflow"""
        print(self.generate_workflow())

workflow = NASWorkflow()
workflow.display_workflow()
```

---

## Exercices

### Exercice 16.1.1
Concevez un espace de recherche pour un MLP avec contraintes de latence < 1 μs sur FPGA.

### Exercice 16.1.2
Implémentez une fonction d'évaluation qui combine accuracy et latence avec des poids configurables.

---

## Points Clés à Retenir

> 📌 **NAS automatise la recherche d'architectures optimales**

> 📌 **Composants clés: Search Space, Search Strategy, Performance Estimation**

> 📌 **Hardware-Aware NAS étend NAS avec métriques hardware**

> 📌 **Problème multi-objectif: accuracy vs latence/énergie/ressources**

> 📌 **Workflow standard: définition → recherche → sélection → déploiement**

---

*Section suivante : [16.2 Métriques Hardware](./16_02_Metriques.md)*

