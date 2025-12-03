# 15.4 Stratégies de Parallélisation

---

## Introduction

Les **stratégies de parallélisation** dans hls4ml contrôlent comment les opérations sont réparties entre ressources matérielles. Cette section explore les différentes approches et leurs trade-offs.

---

## Concepts de Parallélisation

### Types de Parallélisme

```
┌─────────────────────────────────────────────────────────────────┐
│          Types de Parallélisme dans hls4ml                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Parallélisme Spatial (ReuseFactor = 1)                     │
│     ┌───┐ ┌───┐ ┌───┐ ┌───┐                                  │
│     │ PE│ │ PE│ │ PE│ │ PE│  Tous calculent simultanément     │
│     └───┘ └───┘ └───┘ └───┘                                  │
│                                                                 │
│  2. Partage de Ressources (ReuseFactor > 1)                    │
│     ┌───┐                                                      │
│     │ PE│ ──► Réutilisé sur plusieurs cycles                  │
│     └───┘                                                      │
│                                                                 │
│  3. Pipeline                                                    │
│     Stage1 → Stage2 → Stage3 → ...                             │
│     Traitement simultané de plusieurs données                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## ReuseFactor: Concept Clé

### Définition et Impact

```python
class ReuseFactorAnalysis:
    """
    Analyse du ReuseFactor
    """
    
    def __init__(self):
        self.concept = {
            'definition': 'Nombre de fois qu\'une ressource est réutilisée',
            'rf_1': 'Fully parallel: chaque opération a sa ressource dédiée',
            'rf_n': 'Resource sharing: 1 ressource pour N opérations',
            'tradeoff': 'RF↓ = Latence↓, Ressources↑ | RF↑ = Latence↑, Ressources↓'
        }
    
    def analyze_impact(self, layer_size, reuse_factor):
        """
        Analyse l'impact du ReuseFactor
        
        Args:
            layer_size: Nombre d'opérations (ex: in×out pour Dense)
            reuse_factor: Facteur de réutilisation
        """
        # Ressources nécessaires
        resources_needed = layer_size // reuse_factor
        
        # Latence (cycles)
        latency_cycles = reuse_factor
        
        # Throughput (après pipeline fill)
        throughput = 1.0 / reuse_factor  # Opérations par cycle
        
        return {
            'layer_size': layer_size,
            'reuse_factor': reuse_factor,
            'resources_needed': resources_needed,
            'latency_cycles': latency_cycles,
            'throughput_ops_per_cycle': throughput,
            'resource_reduction': layer_size / resources_needed
        }
    
    def display_concept(self):
        """Affiche le concept"""
        print("\n" + "="*60)
        print("ReuseFactor Concept")
        print("="*60)
        
        for key, value in self.concept.items():
            print(f"  {key}: {value}")

reuse_analysis = ReuseFactorAnalysis()
reuse_analysis.display_concept()

# Analyse comparative
print("\n" + "="*60)
print("ReuseFactor Impact Analysis (Layer: 256×128 = 32768 operations)")
print("="*60)

for rf in [1, 4, 16, 64]:
    analysis = reuse_analysis.analyze_impact(32768, rf)
    print(f"\nReuseFactor = {rf}:")
    print(f"  Resources needed: {analysis['resources_needed']}")
    print(f"  Latency: {analysis['latency_cycles']} cycles")
    print(f"  Throughput: {analysis['throughput_ops_per_cycle']:.3f} ops/cycle")
    print(f"  Resource reduction: {analysis['resource_reduction']:.1f}x")
```

---

## Stratégie 1: Fully Parallel (ReuseFactor = 1)

### Caractéristiques

```python
class FullyParallelStrategy:
    """
    Stratégie fully parallel
    """
    
    def __init__(self):
        self.characteristics = {
            'reuse_factor': 1,
            'latency': 'Minimal (1 cycle par opération théoriquement)',
            'resources': 'Maximal (1 DSP/PE par opération)',
            'throughput': 'Maximal (1 opération/cycle)',
            'use_case': 'Applications critiques en latence',
            'example': 'Trigger L1 où latence < 100ns'
        }
    
    def configure_for_fully_parallel(self, config):
        """Configure pour fully parallel"""
        config['Model']['ReuseFactor'] = 1
        config['Model']['Strategy'] = 'Latency'
        
        for layer_name in config['LayerName'].keys():
            config['LayerName'][layer_name]['ReuseFactor'] = 1
        
        return config
    
    def estimate_resources_fully_parallel(self, model):
        """Estime les ressources pour fully parallel"""
        total_ops = 0
        
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]
                total_ops += weights.size
        
        # Approximation: 1 DSP par multiplication
        dsp_needed = total_ops
        
        return {
            'total_operations': total_ops,
            'dsp_needed_estimate': dsp_needed,
            'latency_cycles_estimate': 1,  # Minimal
            'feasibility': 'Dépend des ressources FPGA disponibles'
        }

fully_parallel = FullyParallelStrategy()

print("\n" + "="*60)
print("Fully Parallel Strategy")
print("="*60)

for key, value in fully_parallel.characteristics.items():
    print(f"  {key}: {value}")

# Estimation
resource_est = fully_parallel.estimate_resources_fully_parallel(model)
print("\n" + "="*60)
print("Resource Estimation for Fully Parallel")
print("="*60)
print(f"  Total operations: {resource_est['total_operations']:,}")
print(f"  DSP needed (estimate): {resource_est['dsp_needed_estimate']:,}")
print(f"  Feasibility: {resource_est['feasibility']}")
```

---

## Stratégie 2: Resource Sharing (ReuseFactor > 1)

### Caractéristiques

```python
class ResourceSharingStrategy:
    """
    Stratégie de partage de ressources
    """
    
    def __init__(self):
        self.characteristics = {
            'reuse_factor': '> 1 (4, 8, 16, 32, 64, ...)',
            'latency': 'Augmente proportionnellement',
            'resources': 'Réduit proportionnellement',
            'throughput': 'Réduit proportionnellement',
            'use_case': 'Ressources limitées',
            'example': 'Grands modèles sur FPGA moyen'
        }
    
    def calculate_optimal_reuse_factor(self, layer_size, available_dsp, max_latency_cycles=None):
        """
        Calcule le ReuseFactor optimal
        
        Args:
            layer_size: Taille de la couche (opérations)
            available_dsp: DSP disponibles
            max_latency_cycles: Latence maximale acceptable (optionnel)
        """
        # Minimum nécessaire pour tenir dans ressources
        min_rf_for_resources = layer_size // available_dsp
        
        if max_latency_cycles:
            # Maximum permis par contrainte de latence
            max_rf_for_latency = layer_size // max_latency_cycles
            optimal_rf = min(min_rf_for_resources, max_rf_for_latency)
        else:
            optimal_rf = min_rf_for_resources
        
        return {
            'optimal_reuse_factor': max(1, optimal_rf),
            'dsp_utilized': layer_size // max(1, optimal_rf),
            'latency_cycles': max(1, optimal_rf),
            'utilization': (layer_size // max(1, optimal_rf)) / available_dsp
        }
    
    def configure_resource_sharing(self, config, reuse_factor):
        """Configure pour resource sharing"""
        config['Model']['ReuseFactor'] = reuse_factor
        config['Model']['Strategy'] = 'Resource'
        
        for layer_name in config['LayerName'].keys():
            config['LayerName'][layer_name]['ReuseFactor'] = reuse_factor
        
        return config

resource_sharing = ResourceSharingStrategy()

print("\n" + "="*60)
print("Resource Sharing Strategy")
print("="*60)

for key, value in resource_sharing.characteristics.items():
    print(f"  {key}: {value}")

# Exemple de calcul optimal
optimal = resource_sharing.calculate_optimal_reuse_factor(
    layer_size=32768,
    available_dsp=500,
    max_latency_cycles=100
)

print("\n" + "="*60)
print("Optimal ReuseFactor Calculation")
print("="*60)
print(f"  Layer size: 32768 operations")
print(f"  Available DSP: 500")
print(f"  Max latency: 100 cycles")
print(f"\n  Optimal ReuseFactor: {optimal['optimal_reuse_factor']}")
print(f"  DSP utilized: {optimal['dsp_utilized']}")
print(f"  Latency: {optimal['latency_cycles']} cycles")
print(f"  Utilization: {optimal['utilization']*100:.1f}%")
```

---

## Stratégie 3: Parallélisation Hétérogène

### Parallélisation par Couche

```python
class HeterogeneousParallelization:
    """
    Parallélisation différente par couche
    """
    
    def __init__(self):
        self.strategy = """
        Différentes couches peuvent avoir différents ReuseFactors
        selon leurs besoins et contraintes.
        
        Exemples:
        - Couches critiques: RF=1 (fully parallel)
        - Couches non-critiques: RF>1 (resource sharing)
        """
    
    def configure_heterogeneous(self, config, layer_reuse_factors):
        """
        Configure des ReuseFactors différents par couche
        
        Args:
            config: Configuration dict
            layer_reuse_factors: Dict {layer_name: reuse_factor}
        """
        for layer_name, reuse_factor in layer_reuse_factors.items():
            if layer_name in config['LayerName']:
                config['LayerName'][layer_name]['ReuseFactor'] = reuse_factor
        
        return config
    
    def optimize_per_layer(self, model, available_dsp, latency_budget):
        """
        Optimise le ReuseFactor par couche
        
        Args:
            model: Modèle Keras
            available_dsp: DSP disponibles
            latency_budget: Budget de latence total (cycles)
        """
        layer_configs = {}
        remaining_dsp = available_dsp
        remaining_latency = latency_budget
        
        # Tri par importance (ex: ordre de traitement)
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]
                layer_size = weights.size
                
                # Calcul optimal pour cette couche
                layer_dsp_budget = remaining_dsp // (len(model.layers) - i)
                layer_latency_budget = remaining_latency // (len(model.layers) - i)
                
                optimal_rf = max(1, layer_size // layer_dsp_budget)
                if layer_latency_budget:
                    optimal_rf = min(optimal_rf, layer_size // layer_latency_budget)
                
                layer_configs[layer.name] = optimal_rf
                remaining_dsp -= layer_size // optimal_rf
                remaining_latency -= optimal_rf
        
        return layer_configs

heterogeneous = HeterogeneousParallelization()

print("\n" + "="*60)
print("Heterogeneous Parallelization Strategy")
print("="*60)
print(heterogeneous.strategy)

# Exemple
layer_rfs = {
    'dense_1': 1,   # Couche critique: fully parallel
    'dense_2': 4,   # Couche intermédiaire: resource sharing
    'output': 2     # Couche de sortie: modéré
}

config_hetero = HLS4MLConfiguration.create_base_config(model)
config_hetero = heterogeneous.configure_heterogeneous(config_hetero, layer_rfs)

print("\n" + "="*60)
print("Heterogeneous Configuration Example")
print("="*60)
for layer, rf in layer_rfs.items():
    print(f"  {layer}: ReuseFactor = {rf}")
```

---

## Pipeline et Parallélisme Temporel

### Initiation Interval

```python
class PipelineParallelization:
    """
    Parallélisme temporel via pipeline
    """
    
    def __init__(self):
        self.concept = {
            'initiation_interval': 'Temps entre deux initiations (cycles)',
            'ii_1': 'Optimal: nouvelle donnée chaque cycle',
            'ii_n': 'Nouvelle donnée tous les N cycles',
            'throughput': '1/II samples per cycle'
        }
    
    def analyze_pipeline(self, num_stages, initiation_interval, clock_period_ns=5):
        """
        Analyse un pipeline
        
        Args:
            num_stages: Nombre de stages
            initiation_interval: Initiation interval (cycles)
            clock_period_ns: Période d'horloge
        """
        # Latence du pipeline
        latency_cycles = num_stages
        latency_ns = latency_cycles * clock_period_ns
        
        # Throughput
        throughput_samples_per_cycle = 1.0 / initiation_interval
        throughput_mhz = (1000.0 / clock_period_ns) / initiation_interval
        
        return {
            'num_stages': num_stages,
            'initiation_interval': initiation_interval,
            'latency_cycles': latency_cycles,
            'latency_ns': latency_ns,
            'throughput_samples_per_cycle': throughput_samples_per_cycle,
            'throughput_mhz': throughput_mhz
        }
    
    def display_concept(self):
        """Affiche le concept"""
        print("\n" + "="*60)
        print("Pipeline and Temporal Parallelism")
        print("="*60)
        
        for key, value in self.concept.items():
            print(f"  {key}: {value}")

pipeline = PipelineParallelization()
pipeline.display_concept()

# Analyse
analysis = pipeline.analyze_pipeline(
    num_stages=10,
    initiation_interval=1,
    clock_period_ns=5
)

print("\n" + "="*60)
print("Pipeline Analysis Example")
print("="*60)
print(f"  Stages: {analysis['num_stages']}")
print(f"  Initiation Interval: {analysis['initiation_interval']}")
print(f"  Latency: {analysis['latency_ns']:.1f} ns ({analysis['latency_cycles']} cycles)")
print(f"  Throughput: {analysis['throughput_mhz']:.1f} MSamples/s")
```

---

## Comparaison des Stratégies

```python
class StrategyComparison:
    """
    Comparaison des stratégies de parallélisation
    """
    
    @staticmethod
    def compare_strategies(layer_size=32768, available_dsp=500):
        """Compare les stratégies"""
        strategies = {
            'Fully Parallel (RF=1)': {
                'reuse_factor': 1,
                'dsp_used': layer_size,
                'latency_cycles': 1,
                'feasible': layer_size <= available_dsp
            },
            'Moderate Sharing (RF=4)': {
                'reuse_factor': 4,
                'dsp_used': layer_size // 4,
                'latency_cycles': 4,
                'feasible': (layer_size // 4) <= available_dsp
            },
            'Aggressive Sharing (RF=16)': {
                'reuse_factor': 16,
                'dsp_used': layer_size // 16,
                'latency_cycles': 16,
                'feasible': (layer_size // 16) <= available_dsp
            },
            'Maximal Sharing (RF=64)': {
                'reuse_factor': 64,
                'dsp_used': layer_size // 64,
                'latency_cycles': 64,
                'feasible': (layer_size // 64) <= available_dsp
            }
        }
        
        print("\n" + "="*60)
        print(f"Strategy Comparison (Layer size: {layer_size}, Available DSP: {available_dsp})")
        print("="*60)
        
        print(f"\n{'Strategy':<30} | {'RF':<6} | {'DSP Used':<10} | {'Latency':<10} | {'Feasible'}")
        print("-" * 70)
        
        for name, metrics in strategies.items():
            print(f"{name:<30} | {metrics['reuse_factor']:<6} | "
                  f"{metrics['dsp_used']:<10} | {metrics['latency_cycles']:<10} | "
                  f"{'✓' if metrics['feasible'] else '✗'}")

StrategyComparison.compare_strategies()
```

---

## Exercices

### Exercice 15.4.1
Optimisez le ReuseFactor pour un modèle avec contrainte de latence de 200 cycles et 300 DSP disponibles.

### Exercice 15.4.2
Créez une configuration hétérogène pour un modèle où la première couche doit être fully parallel et les autres peuvent partager ressources.

---

## Points Clés à Retenir

> 📌 **ReuseFactor = 1: Fully parallel, latence min, ressources max**

> 📌 **ReuseFactor > 1: Resource sharing, latence ↑, ressources ↓**

> 📌 **Parallélisation hétérogène: différent RF par couche**

> 📌 **Pipeline: parallélisme temporel, throughput après fill**

> 📌 **Initiation Interval contrôle le throughput**

> 📌 **Choix dépend contraintes: latence, ressources, throughput**

---

*Section suivante : [15.5 Intégration avec les Workflows de Physique](./15_05_Integration.md)*

