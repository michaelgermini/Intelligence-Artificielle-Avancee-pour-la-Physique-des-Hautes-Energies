# 16.3 Espaces de Recherche Contraints

---

## Introduction

Les **espaces de recherche contraints** sont cruciaux pour le Hardware-Aware NAS. Au lieu de chercher dans un espace illimité d'architectures, on contraint la recherche pour garantir que les architectures trouvées respectent les limites hardware (latence, énergie, ressources).

Cette section présente différentes méthodes pour créer et gérer des espaces de recherche contraints, depuis les contraintes simples jusqu'aux contraintes complexes multi-objectifs.

---

## Types de Contraintes

### Contraintes Hard vs Soft

```python
class ConstraintTypes:
    """
    Types de contraintes pour NAS
    """
    
    def __init__(self):
        self.constraint_types = {
            'hard_constraints': {
                'description': 'Contraintes strictes qui doivent être respectées',
                'examples': [
                    'Latence < 100 ns (trigger HEP)',
                    'Ressources FPGA < limites du chip',
                    'Mémoire < capacité disponible'
                ],
                'handling': 'Rejet des architectures invalides',
                'impact': 'Réduit l\'espace de recherche'
            },
            'soft_constraints': {
                'description': 'Contraintes préférentielles (pénalités)',
                'examples': [
                    'Préférer latence basse (mais tolérer dépassements)',
                    'Minimiser énergie (objectif, pas limite)',
                    'Préférer modèles compacts'
                ],
                'handling': 'Pénalités dans la fonction objectif',
                'impact': 'Guide la recherche sans exclure'
            },
            'conditional_constraints': {
                'description': 'Contraintes dépendant d\'autres choix',
                'examples': [
                    'Si quantifié 8-bit, alors latence < X',
                    'Si parallélisme > Y, alors ressources < Z',
                    'Si profondeur > 10, alors utiliser skip connections'
                ],
                'handling': 'Règles conditionnelles dans la génération',
                'impact': 'Complexifie l\'espace mais plus réaliste'
            }
        }
    
    def display_constraints(self):
        """Affiche les types de contraintes"""
        print("\n" + "="*70)
        print("Types de Contraintes")
        print("="*70)
        
        for ctype, info in self.constraint_types.items():
            print(f"\n{ctype.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Exemples:")
            for ex in info['examples']:
                print(f"    • {ex}")
            print(f"  Handling: {info['handling']}")
            print(f"  Impact: {info['impact']}")

constraints = ConstraintTypes()
constraints.display_constraints()
```

---

## Espaces de Recherche avec Contraintes Hard

### Génération d'Architectures Valides

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class ConstrainedSearchSpace:
    """
    Espace de recherche avec contraintes hardware
    """
    
    def __init__(self, hardware_constraints: Dict, hardware_evaluator):
        """
        Args:
            hardware_constraints: dict avec contraintes (latency_ns, energy_nj, etc.)
            hardware_evaluator: HardwareEvaluator pour vérifier contraintes
        """
        self.constraints = hardware_constraints
        self.hardware_eval = hardware_evaluator
        
        # Espace de recherche de base (non contraint)
        self.base_space = {
            'num_layers': [3, 4, 5, 6, 7, 8],
            'layer_widths': [32, 64, 128, 256, 512, 1024],
            'activation': ['relu', 'gelu', 'swish'],
            'use_batch_norm': [True, False],
            'use_dropout': [True, False],
            'dropout_rate': [0.1, 0.2, 0.3]
        }
        
        # Cache pour architectures validées
        self.valid_arch_cache = {}
        self.invalid_arch_cache = set()
    
    def sample_random_config(self) -> Dict:
        """
        Génère une configuration aléatoire dans l'espace de base
        """
        config = {
            'num_layers': np.random.choice(self.base_space['num_layers']),
            'layer_widths': np.random.choice(self.base_space['layer_widths'], 
                                            size=np.random.choice([1, 2, 3]), 
                                            replace=False).tolist(),
            'activation': np.random.choice(self.base_space['activation']),
            'use_batch_norm': np.random.choice(self.base_space['use_batch_norm']),
            'use_dropout': np.random.choice(self.base_space['use_dropout']),
            'dropout_rate': np.random.choice(self.base_space['dropout_rate']) if True else 0.0
        }
        
        # Compléter layer_widths si nécessaire
        if len(config['layer_widths']) < config['num_layers']:
            # Répéter ou interpoler
            while len(config['layer_widths']) < config['num_layers']:
                config['layer_widths'].append(np.random.choice(self.base_space['layer_widths']))
        
        return config
    
    def create_model_from_config(self, config: Dict, input_dim: int = 784, output_dim: int = 10) -> nn.Module:
        """
        Crée un modèle PyTorch depuis une configuration
        """
        layers = []
        
        # Déterminer largeurs de couches
        if len(config['layer_widths']) == 1:
            widths = [config['layer_widths'][0]] * config['num_layers']
        else:
            # Interpolation si plusieurs largeurs fournies
            widths = np.linspace(config['layer_widths'][0], 
                               config['layer_widths'][-1], 
                               config['num_layers'], 
                               dtype=int).tolist()
        
        current_dim = input_dim
        for i, width in enumerate(widths):
            # Couche linéaire
            layers.append(nn.Linear(current_dim, width))
            
            # BatchNorm
            if config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(width))
            
            # Activation
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())
            elif config['activation'] == 'swish':
                layers.append(nn.SiLU())
            
            # Dropout
            if config['use_dropout'] and i < len(widths) - 1:  # Pas sur dernière couche
                layers.append(nn.Dropout(config['dropout_rate']))
            
            current_dim = width
        
        # Dernière couche
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def check_hardware_constraints(self, model: nn.Module, input_shape: Tuple) -> Tuple[bool, Dict]:
        """
        Vérifie si un modèle respecte les contraintes hardware
        
        Returns:
            (is_valid, metrics)
        """
        # Vérifier cache
        model_hash = self._model_hash(model)
        if model_hash in self.valid_arch_cache:
            return True, self.valid_arch_cache[model_hash]
        if model_hash in self.invalid_arch_cache:
            return False, {}
        
        # Évaluer hardware
        metrics = self.hardware_eval.evaluate_architecture(model, input_shape, 
                                                          self.constraints.get('fpga_constraints'))
        
        # Vérifier contraintes
        is_valid = True
        
        if 'latency_ns' in self.constraints:
            if metrics['latency_ns'] > self.constraints['latency_ns']:
                is_valid = False
        
        if 'energy_nj' in self.constraints:
            if metrics['energy_nj'] > self.constraints['energy_nj']:
                is_valid = False
        
        if 'model_size_mb' in self.constraints:
            if metrics['model_size_mb'] > self.constraints['model_size_mb']:
                is_valid = False
        
        if 'feasible' in metrics and not metrics['feasible']:
            is_valid = False
        
        # Mettre en cache
        if is_valid:
            self.valid_arch_cache[model_hash] = metrics
        else:
            self.invalid_arch_cache.add(model_hash)
        
        return is_valid, metrics
    
    def _model_hash(self, model: nn.Module) -> str:
        """Génère un hash simple pour le modèle (basé sur architecture)"""
        arch_str = str([(name, tuple(m.weight.shape) if hasattr(m, 'weight') else None) 
                       for name, m in model.named_modules() 
                       if isinstance(m, (nn.Linear, nn.Conv2d))])
        return str(hash(arch_str))
    
    def sample_valid_config(self, max_attempts: int = 100, input_shape: Tuple = (1, 784)) -> Optional[Dict]:
        """
        Génère une configuration qui respecte les contraintes
        
        Returns:
            Config valide ou None si échec
        """
        for attempt in range(max_attempts):
            config = self.sample_random_config()
            model = self.create_model_from_config(config, input_dim=input_shape[1])
            is_valid, _ = self.check_hardware_constraints(model, input_shape)
            
            if is_valid:
                return config
        
        return None
    
    def filter_valid_configs(self, configs: List[Dict], input_shape: Tuple = (1, 784)) -> List[Tuple[Dict, Dict]]:
        """
        Filtre une liste de configurations pour garder seulement les valides
        
        Returns:
            List de (config, metrics) pour configs valides
        """
        valid_configs = []
        
        for config in configs:
            model = self.create_model_from_config(config, input_dim=input_shape[1])
            is_valid, metrics = self.check_hardware_constraints(model, input_shape)
            
            if is_valid:
                valid_configs.append((config, metrics))
        
        return valid_configs

# Exemple d'utilisation
from Livre_IA_HEP.Partie_IV_Hardware.Chapitre_16_Hardware_NAS import HardwareEvaluator  # Import fictif

# Créer évaluateur hardware
hardware_eval = HardwareEvaluator('fpga', bitwidth=8)

# Contraintes strictes (trigger HEP)
constraints = {
    'latency_ns': 100000,  # 100 μs max
    'energy_nj': 1000,     # 1000 nJ max
    'model_size_mb': 5.0,  # 5 MB max
    'fpga_constraints': {
        'max_lut': 548160,
        'max_dsp': 2520,
        'max_bram': 912
    }
}

# Créer espace contraint
constrained_space = ConstrainedSearchSpace(constraints, hardware_eval)

# Essayer de générer des configs valides
print("\n" + "="*70)
print("Génération d'Architectures Valides")
print("="*70)

valid_configs = []
for i in range(10):
    config = constrained_space.sample_valid_config(max_attempts=50, input_shape=(1, 784))
    if config:
        valid_configs.append(config)
        model = constrained_space.create_model_from_config(config)
        is_valid, metrics = constrained_space.check_hardware_constraints(model, (1, 784))
        print(f"\nConfig {i+1} valide:")
        print(f"  Layers: {config['num_layers']}, Widths: {config['layer_widths']}")
        print(f"  Latence: {metrics['latency_us']:.2f} μs")
        print(f"  Énergie: {metrics['energy_nj']:.2f} nJ")
        print(f"  Taille: {metrics['model_size_mb']:.2f} MB")

print(f"\n{len(valid_configs)} configurations valides sur 10 tentatives")
```

---

## Espaces de Recherche avec Contraintes Progressives

### Contraintes Adaptatives

```python
class AdaptiveConstrainedSpace:
    """
    Espace de recherche avec contraintes qui s'adaptent pendant la recherche
    """
    
    def __init__(self, initial_constraints: Dict, hardware_evaluator):
        self.initial_constraints = initial_constraints.copy()
        self.current_constraints = initial_constraints.copy()
        self.hardware_eval = hardware_evaluator
        
        # Statistiques sur les architectures évaluées
        self.evaluation_history = []
        self.best_metrics = {}
    
    def update_constraints(self, exploration_phase: str = 'initial'):
        """
        Met à jour les contraintes selon la phase d'exploration
        
        Strategies:
        - initial: Contraintes strictes pour exploration large
        - refinement: Contraintes plus strictes autour des bonnes solutions
        - exploitation: Contraintes très strictes pour optimisation fine
        """
        if exploration_phase == 'initial':
            # Phase initiale: contraintes relativement permissives
            self.current_constraints = self.initial_constraints.copy()
        
        elif exploration_phase == 'refinement':
            # Phase de raffinement: resserrer autour des bonnes solutions
            if self.evaluation_history:
                # Calculer percentiles des métriques
                latencies = [m.get('latency_ns', 0) for m in self.evaluation_history if 'latency_ns' in m]
                energies = [m.get('energy_nj', 0) for m in self.evaluation_history if 'energy_nj' in m]
                
                if latencies:
                    # Resserrer à 75e percentile
                    new_latency = np.percentile(latencies, 75)
                    self.current_constraints['latency_ns'] = min(
                        new_latency,
                        self.initial_constraints['latency_ns']
                    )
                
                if energies:
                    new_energy = np.percentile(energies, 75)
                    self.current_constraints['energy_nj'] = min(
                        new_energy,
                        self.initial_constraints['energy_nj']
                    )
        
        elif exploration_phase == 'exploitation':
            # Phase d'exploitation: contraintes très strictes
            if self.best_metrics:
                # Contraintes basées sur meilleures solutions
                self.current_constraints['latency_ns'] = self.best_metrics.get('latency_ns', 0) * 1.2
                self.current_constraints['energy_nj'] = self.best_metrics.get('energy_nj', 0) * 1.2
    
    def evaluate_and_record(self, model: nn.Module, input_shape: Tuple) -> Tuple[bool, Dict]:
        """
        Évalue et enregistre dans l'historique
        """
        is_valid, metrics = self._check_constraints(model, input_shape)
        
        # Enregistrer dans l'historique
        self.evaluation_history.append(metrics.copy())
        
        # Mettre à jour meilleures métriques
        if is_valid:
            if not self.best_metrics:
                self.best_metrics = metrics.copy()
            else:
                # Mettre à jour si meilleur (selon score combiné)
                current_score = self._compute_score(metrics)
                best_score = self._compute_score(self.best_metrics)
                if current_score < best_score:  # plus bas = mieux
                    self.best_metrics = metrics.copy()
        
        return is_valid, metrics
    
    def _check_constraints(self, model: nn.Module, input_shape: Tuple) -> Tuple[bool, Dict]:
        """Vérifie les contraintes actuelles"""
        metrics = self.hardware_eval.evaluate_architecture(model, input_shape)
        
        is_valid = True
        if metrics['latency_ns'] > self.current_constraints.get('latency_ns', float('inf')):
            is_valid = False
        if metrics['energy_nj'] > self.current_constraints.get('energy_nj', float('inf')):
            is_valid = False
        
        return is_valid, metrics
    
    def _compute_score(self, metrics: Dict) -> float:
        """Score combiné (plus bas = mieux)"""
        latency_score = metrics.get('latency_ns', 0) / 1e6  # normaliser
        energy_score = metrics.get('energy_nj', 0) / 1e3
        return latency_score + energy_score
```

---

## Espaces de Recherche Hiérarchiques

### Recherche Multi-Niveau

```python
class HierarchicalConstrainedSpace:
    """
    Espace de recherche hiérarchique avec contraintes à différents niveaux
    """
    
    def __init__(self, hardware_evaluator):
        self.hardware_eval = hardware_evaluator
        
        # Niveau 1: Macro-architecture (nombre de blocs, types)
        self.macro_space = {
            'num_blocks': [2, 3, 4, 5, 6],
            'block_types': ['residual', 'dense', 'inverted_residual'],
            'overall_width_factor': [0.5, 0.75, 1.0, 1.25, 1.5]
        }
        
        # Niveau 2: Micro-architecture (détails des blocs)
        self.micro_space = {
            'block_width': [64, 128, 256, 512],
            'block_depth': [1, 2, 3, 4],
            'kernel_size': [3, 5, 7],
            'expansion_ratio': [1, 2, 4, 6]
        }
        
        # Niveau 3: Hyperparamètres (activation, normalization)
        self.hyper_space = {
            'activation': ['relu', 'gelu', 'swish'],
            'normalization': ['batch_norm', 'layer_norm', 'group_norm'],
            'use_se': [True, False]  # Squeeze-and-Excitation
        }
    
    def sample_hierarchical_config(self) -> Dict:
        """
        Génère une configuration hiérarchique
        """
        # Niveau 1: Macro
        macro = {
            'num_blocks': np.random.choice(self.macro_space['num_blocks']),
            'block_type': np.random.choice(self.macro_space['block_types']),
            'width_factor': np.random.choice(self.macro_space['overall_width_factor'])
        }
        
        # Niveau 2: Micro (par bloc)
        blocks = []
        for i in range(macro['num_blocks']):
            block = {
                'width': int(np.random.choice(self.micro_space['block_width']) * macro['width_factor']),
                'depth': np.random.choice(self.micro_space['block_depth']),
                'kernel_size': np.random.choice(self.micro_space['kernel_size']),
                'expansion_ratio': np.random.choice(self.micro_space['expansion_ratio'])
            }
            blocks.append(block)
        
        # Niveau 3: Hyperparamètres
        hyper = {
            'activation': np.random.choice(self.hyper_space['activation']),
            'normalization': np.random.choice(self.hyper_space['normalization']),
            'use_se': np.random.choice(self.hyper_space['use_se'])
        }
        
        return {
            'macro': macro,
            'blocks': blocks,
            'hyper': hyper
        }
    
    def apply_constraints_level_by_level(self, config: Dict, constraints: Dict) -> Tuple[bool, Dict]:
        """
        Applique les contraintes niveau par niveau pour efficacité
        """
        # Niveau 1: Vérifier contraintes macro (rapide)
        if not self._check_macro_constraints(config['macro'], constraints):
            return False, {}
        
        # Niveau 2: Créer modèle partiel et vérifier
        # (simplifié ici, en pratique créer modèle partiel)
        
        # Niveau 3: Vérification complète
        model = self._config_to_model(config)
        metrics = self.hardware_eval.evaluate_architecture(model, (1, 3, 224, 224))
        
        is_valid = self._check_all_constraints(metrics, constraints)
        return is_valid, metrics
    
    def _check_macro_constraints(self, macro_config: Dict, constraints: Dict) -> bool:
        """Vérifications rapides au niveau macro"""
        # Exemple: nombre de blocs peut donner estimation grossière de taille
        estimated_size_mb = macro_config['num_blocks'] * 0.5  # approximation
        if estimated_size_mb > constraints.get('model_size_mb', float('inf')):
            return False
        return True
    
    def _config_to_model(self, config: Dict) -> nn.Module:
        """Convertit config hiérarchique en modèle (simplifié)"""
        # Placeholder - en pratique implémenter selon block_types
        layers = []
        for block in config['blocks']:
            for _ in range(block['depth']):
                layers.append(nn.Linear(block['width'], block['width']))
                if config['hyper']['activation'] == 'relu':
                    layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def _check_all_constraints(self, metrics: Dict, constraints: Dict) -> bool:
        """Vérifie toutes les contraintes"""
        if metrics['latency_ns'] > constraints.get('latency_ns', float('inf')):
            return False
        if metrics['energy_nj'] > constraints.get('energy_nj', float('inf')):
            return False
        if metrics['model_size_mb'] > constraints.get('model_size_mb', float('inf')):
            return False
        return True
```

---

## Espaces de Recherche avec Pruning Intégré

### Recherche avec Architecture Sparse

```python
class SparseConstrainedSpace:
    """
    Espace de recherche qui inclut le sparsity comme paramètre
    """
    
    def __init__(self, hardware_evaluator):
        self.hardware_eval = hardware_evaluator
        
        # Espace incluant taux de sparsity
        self.space = {
            'base_architecture': {
                'num_layers': [3, 4, 5, 6],
                'layer_widths': [128, 256, 512]
            },
            'sparsity': {
                'global_sparsity': [0.0, 0.25, 0.5, 0.75, 0.9],  # Taux global
                'layer_wise_sparsity': 'uniform',  # ou 'adaptive'
                'pruning_method': ['magnitude', 'gradient', 'lottery_ticket']
            }
        }
    
    def sample_sparse_config(self) -> Dict:
        """Génère une config avec sparsity"""
        config = {
            'num_layers': np.random.choice(self.space['base_architecture']['num_layers']),
            'layer_widths': [np.random.choice(self.space['base_architecture']['layer_widths'])] * 3,
            'sparsity': np.random.choice(self.space['sparsity']['global_sparsity']),
            'pruning_method': np.random.choice(self.space['sparsity']['pruning_method'])
        }
        return config
    
    def create_sparse_model(self, config: Dict) -> nn.Module:
        """
        Crée un modèle avec sparsity appliquée
        """
        # Créer modèle de base
        model = self._create_base_model(config)
        
        # Appliquer sparsity (simulé avec masques)
        if config['sparsity'] > 0:
            self._apply_sparsity_mask(model, config['sparsity'])
        
        return model
    
    def _create_base_model(self, config: Dict) -> nn.Module:
        """Crée modèle de base"""
        layers = []
        widths = config['layer_widths'][:config['num_layers']]
        
        current_dim = 784
        for width in widths:
            layers.append(nn.Linear(current_dim, width))
            layers.append(nn.ReLU())
            current_dim = width
        layers.append(nn.Linear(current_dim, 10))
        
        return nn.Sequential(*layers)
    
    def _apply_sparsity_mask(self, model: nn.Module, sparsity: float):
        """
        Applique un masque de sparsity (simulation)
        En pratique, utiliser torch.nn.utils.prune
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Simuler sparsity en mettant certains poids à zéro
                mask = torch.rand_like(module.weight) > sparsity
                module.weight.data *= mask.float()
```

---

## Exercices

### Exercice 16.3.1
Implémentez un espace de recherche contraint pour des modèles de trigger avec latence < 100 ns et énergie < 100 nJ.

### Exercice 16.3.2
Créez un système de contraintes adaptatives qui resserre progressivement les limites pendant la recherche NAS.

### Exercice 16.3.3
Développez un espace hiérarchique avec validation à plusieurs niveaux pour efficacité.

### Exercice 16.3.4
Comparez le taux de succès (architectures valides) avec et sans contraintes hard pour un espace de recherche donné.

---

## Points Clés à Retenir

> 📌 **Les contraintes hard rejettent les architectures invalides, réduisant l'espace de recherche**

> 📌 **Les contraintes soft guident la recherche via pénalités dans la fonction objectif**

> 📌 **Les espaces hiérarchiques permettent une validation efficace niveau par niveau**

> 📌 **Les contraintes adaptatives peuvent améliorer l'efficacité de la recherche**

> 📌 **L'intégration du sparsity dans l'espace de recherche permet d'explorer architectures compressées**

> 📌 **Le caching des architectures validées/invalides améliore significativement la vitesse**

---

*Section précédente : [16.2 Métriques Hardware](./16_02_Metriques.md) | Section suivante : [16.4 Algorithmes de Recherche Efficaces](./16_04_Algorithmes.md)*

