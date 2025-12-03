# 26.2 Design d'Expériences

---

## Introduction

Le **design d'expériences** est crucial pour obtenir des résultats valides et interprétables. Un bon design permet d'isoler les effets des variables étudiées, de contrôler les facteurs de confusion, et de garantir la reproductibilité. Cette section présente les principes de design d'expériences pour la recherche en IA et HEP.

---

## Principes Fondamentaux

### Variables et Contrôles

```python
"""
Concepts clés:

1. Variable indépendante (IV)
   - Variable manipulée par expérimentateur
   - Ex: taux de pruning, méthode de compression

2. Variable dépendante (DV)
   - Variable mesurée (outcome)
   - Ex: accuracy, latence, taille modèle

3. Variables de contrôle
   - Facteurs maintenus constants
   - Ex: dataset, architecture de base

4. Variables confondantes
   - Facteurs non contrôlés mais influents
   - À minimiser ou contrôler
"""

class ExperimentDesign:
    """
    Design d'expérience structuré
    """
    
    def __init__(self, research_question: str):
        self.research_question = research_question
        self.variables = {
            'independent': [],
            'dependent': [],
            'control': [],
            'confounding': []
        }
        self.experimental_conditions = []
    
    def define_variables(self):
        """Définit variables expérimentales"""
        # Exemple: étude impact pruning
        self.variables['independent'] = [
            {'name': 'pruning_rate', 'levels': [0.1, 0.3, 0.5, 0.7, 0.9]},
            {'name': 'pruning_method', 'levels': ['magnitude', 'gradient', 'random']}
        ]
        
        self.variables['dependent'] = [
            {'name': 'accuracy', 'metric': 'classification_accuracy'},
            {'name': 'compression_ratio', 'metric': 'size_reduction'},
            {'name': 'inference_time', 'metric': 'latency_ms'}
        ]
        
        self.variables['control'] = [
            {'name': 'dataset', 'value': 'CIFAR-10'},
            {'name': 'architecture', 'value': 'ResNet-18'},
            {'name': 'training_epochs', 'value': 50}
        ]
        
        self.variables['confounding'] = [
            {'name': 'random_seed', 'mitigation': 'multiple_seeds'},
            {'name': 'hardware', 'mitigation': 'same_environment'}
        ]
```

---

## Types de Designs

### Factorial et Factoriel Complet

```python
import itertools

class FactorialDesign:
    """
    Design factoriel complet
    """
    
    def __init__(self, factors: Dict):
        """
        Args:
            factors: Dict {factor_name: [levels]}
        """
        self.factors = factors
        self.design = None
    
    def create_full_factorial(self):
        """Crée design factoriel complet"""
        factor_names = list(self.factors.keys())
        factor_levels = [self.factors[name] for name in factor_names]
        
        # Toutes combinaisons
        self.design = []
        for combination in itertools.product(*factor_levels):
            condition = dict(zip(factor_names, combination))
            self.design.append(condition)
        
        return self.design
    
    def get_number_of_conditions(self):
        """Calcule nombre de conditions"""
        n_conditions = 1
        for levels in self.factors.values():
            n_conditions *= len(levels)
        return n_conditions

# Exemple
factors = {
    'pruning_rate': [0.3, 0.5, 0.7],
    'pruning_method': ['magnitude', 'gradient'],
    'learning_rate': [0.001, 0.0001]
}

factorial = FactorialDesign(factors)
design = factorial.create_full_factorial()
print(f"Nombre de conditions: {factorial.get_number_of_conditions()}")  # 3 * 2 * 2 = 12
```

---

## Randomisation et Réplication

### Contrôle des Biais

```python
import random
import numpy as np

class RandomizedDesign:
    """
    Design avec randomisation
    """
    
    def __init__(self, conditions: List[Dict]):
        self.conditions = conditions
        self.randomized_order = None
    
    def randomize_order(self, seed=None):
        """Randomise ordre d'exécution"""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.randomized_order = self.conditions.copy()
        random.shuffle(self.randomized_order)
        
        return self.randomized_order
    
    def create_blocks(self, block_size: int):
        """
        Crée blocs pour réduire variance
        
        Ex: si conditions prennent longtemps, grouper par batch
        """
        blocks = []
        for i in range(0, len(self.conditions), block_size):
            block = self.conditions[i:i+block_size]
            # Randomiser à l'intérieur du bloc
            random.shuffle(block)
            blocks.append(block)
        
        return blocks

# Exemple avec réplication
class ReplicatedDesign:
    """
    Design avec réplications
    """
    
    def __init__(self, base_conditions: List[Dict], n_replications: int):
        self.base_conditions = base_conditions
        self.n_replications = n_replications
    
    def create_replicated_design(self):
        """Crée design avec réplications"""
        replicated = []
        
        for condition in self.base_conditions:
            for rep in range(self.n_replications):
                replicated_condition = condition.copy()
                replicated_condition['replication'] = rep
                replicated_condition['random_seed'] = rep * 42  # Seeds différents
                replicated.append(replicated_condition)
        
        # Randomiser ordre complet
        random.shuffle(replicated)
        
        return replicated
```

---

## Contrôle Expérimental

### Stratégies de Contrôle

```python
class ExperimentalControl:
    """
    Stratégies de contrôle expérimental
    """
    
    def __init__(self):
        self.control_strategies = {
            'baseline': {
                'description': 'Condition de référence',
                'example': 'Modèle non compressé',
                'purpose': 'Comparaison'
            },
            'placebo': {
                'description': 'Condition contrôle',
                'example': 'Pseudo-compression sans effet réel',
                'purpose': 'Vérifier effet réel'
            },
            'negative_control': {
                'description': 'Contrôle négatif',
                'example': 'Méthode connue pour ne pas fonctionner',
                'purpose': 'Valider setup expérimental'
            }
        }
    
    def design_controlled_experiment(self):
        """Crée expérience contrôlée"""
        experiment = {
            'baseline': {
                'condition': 'original_model',
                'description': 'Modèle original non compressé'
            },
            'treatments': [
                {
                    'condition': 'pruned_30',
                    'description': 'Pruning 30%'
                },
                {
                    'condition': 'pruned_50',
                    'description': 'Pruning 50%'
                },
                {
                    'condition': 'quantized_int8',
                    'description': 'Quantization 8-bit'
                }
            ],
            'replications': 5,
            'randomization': True
        }
        
        return experiment
```

---

## Variables et Mesures

### Définition des Mesures

```python
class MeasurementProtocol:
    """
    Protocole de mesure standardisé
    """
    
    def define_measurements(self):
        """Définit mesures et protocoles"""
        measurements = {
            'accuracy': {
                'metric': 'classification_accuracy',
                'protocol': 'evaluate_on_test_set',
                'dataset': 'held_out_test',
                'n_samples': 'all',
                'reporting': 'mean_and_std'
            },
            'latency': {
                'metric': 'inference_time',
                'protocol': 'measure_n_times',
                'n_runs': 100,
                'warmup_runs': 10,
                'reporting': 'mean_p50_p95_p99'
            },
            'memory': {
                'metric': 'model_size',
                'protocol': 'count_parameters',
                'reporting': 'total_and_ratio'
            },
            'compression': {
                'metric': 'compression_ratio',
                'protocol': 'compare_to_baseline',
                'baseline': 'original_model',
                'reporting': 'ratio_and_absolute'
            }
        }
        
        return measurements
    
    def create_measurement_script(self, measurement_config):
        """Crée script de mesure standardisé"""
        script = f"""
# Protocol: {measurement_config['protocol']}

def measure_accuracy(model, test_loader):
    \"\"\"Measure accuracy according to protocol\"\"\"
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Reporting: {measurement_config['reporting']}
"""
        return script
```

---

## Plan d'Expérience Complet

### Exemple Structuré

```python
class CompleteExperimentPlan:
    """
    Plan d'expérience complet
    """
    
    def create_experiment_plan(self, research_question: str):
        """Crée plan d'expérience structuré"""
        plan = {
            'research_question': research_question,
            'hypothesis': self.formulate_hypothesis(research_question),
            'design': {
                'type': 'factorial',
                'factors': self.define_factors(),
                'replications': 5,
                'randomization': True
            },
            'conditions': self.create_conditions(),
            'measurements': self.define_measurements(),
            'analysis_plan': self.define_analysis(),
            'timeline': self.create_timeline()
        }
        
        return plan
    
    def formulate_hypothesis(self, question: str):
        """Formule hypothèse"""
        # Exemple
        return {
            'null_hypothesis': 'Pruning n\'affecte pas significativement l\'accuracy',
            'alternative_hypothesis': 'Pruning affecte l\'accuracy (au-delà seuil acceptable)',
            'significance_level': 0.05
        }
    
    def define_analysis(self):
        """Définit plan d'analyse"""
        return {
            'descriptive': ['mean', 'std', 'confidence_intervals'],
            'inferential': ['t_test', 'anova'],
            'post_hoc': ['multiple_comparisons_correction'],
            'visualization': ['boxplots', 'line_plots', 'heatmaps']
        }
```

---

## Exercices

### Exercice 26.2.1
Concevez un design factoriel complet pour étudier impact de 3 facteurs sur compression.

### Exercice 26.2.2
Créez plan d'expérience avec randomisation et réplications pour comparer méthodes.

### Exercice 26.2.3
Définissez protocole de mesure standardisé pour métriques de performance.

### Exercice 26.2.4
Identifiez variables confondantes potentielles et stratégies de contrôle.

---

## Points Clés à Retenir

> 📌 **Variables indépendantes/dépendantes doivent être clairement définies**

> 📌 **Randomisation réduit biais et variables confondantes**

> 📌 **Réplications augmentent robustesse résultats**

> 📌 **Contrôles (baseline, placebo) permettent comparaisons valides**

> 📌 **Protocoles de mesure standardisés garantissent comparabilité**

> 📌 **Plan d'analyse doit être défini avant expérimentation**

---

*Section précédente : [26.1 Littérature](./26_01_Litterature.md) | Section suivante : [26.3 Analyse Statistique](./26_03_Analyse_Statistique.md)*

