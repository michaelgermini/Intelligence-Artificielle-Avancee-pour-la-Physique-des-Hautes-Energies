# 14.3 Parallélisme Spatial vs Temporel

---

## Introduction

Le **parallélisme spatial** (plus d'unités en parallèle) et le **parallélisme temporel** (pipeline) sont deux stratégies complémentaires pour améliorer les performances sur FPGA.

---

## Concepts Fondamentaux

```
┌─────────────────────────────────────────────────────────────────┐
│          Parallélisme Spatial vs Temporel                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Parallélisme Spatial:                                          │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐                                      │
│  │ PE│ │ PE│ │ PE│ │ PE│  ──► Tous calculent en même temps    │
│  └───┘ └───┘ └───┘ └───┘                                      │
│                                                                 │
│  Parallélisme Temporel (Pipeline):                              │
│  ┌───┐                                                          │
│  │ S1│ ──► ┌───┐                                               │
│  └───┘     │ S2│ ──► ┌───┐                                    │
│            └───┘     │ S3│                                     │
│                      └───┘                                     │
│  ──► Traitement simultané de plusieurs données                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallélisme Spatial

### Principe

```python
class SpatialParallelism:
    """
    Parallélisme spatial: plusieurs unités en parallèle
    """
    
    def __init__(self):
        self.description = """
        Utilise plusieurs unités de calcul (PEs, DSPs) en parallèle
        pour traiter différentes données simultanément.
        
        Avantages:
        - Throughput proportionnel au nombre d'unités
        - Latence réduite (1 cycle si assez d'unités)
        
        Inconvénients:
        - Consommation ressources élevée
        - Communication complexe
        """
    
    def visualize(self, num_units=4):
        """Visualise le parallélisme spatial"""
        diagram = f"""
Spatial Parallelism ({num_units} units):

Input Data:
[D0] ──► [PE0] ──► [O0]
[D1] ──► [PE1] ──► [O1]
[D2] ──► [PE2] ──► [O2]
[D3] ──► [PE3] ──► [O3]

All PEs compute simultaneously in the same cycle.

Timeline:
Cycle:  0    1    2    3
      ┌────┐
      │D0-3│  All 4 processed in cycle 0
      └────┘
      
Throughput = {num_units} samples/cycle
Latency = 1 cycle
"""
        return diagram
    
    def estimate_throughput(self, num_units, clock_mhz):
        """
        Estime le throughput
        
        Args:
            num_units: Nombre d'unités parallèles
            clock_mhz: Fréquence d'horloge
        """
        throughput_msamples_per_s = num_units * clock_mhz
        
        return {
            'num_units': num_units,
            'clock_mhz': clock_mhz,
            'throughput_msamples_per_s': throughput_msamples_per_s,
            'samples_per_cycle': num_units,
            'latency_cycles': 1
        }
    
    def estimate_resources(self, num_units, resources_per_unit):
        """
        Estime les ressources nécessaires
        
        Args:
            num_units: Nombre d'unités
            resources_per_unit: Dict avec ressources par unité
        """
        total_resources = {}
        for resource, value in resources_per_unit.items():
            total_resources[resource] = value * num_units
        
        return total_resources

spatial = SpatialParallelism()
print(spatial.visualize(4))

throughput = spatial.estimate_throughput(num_units=8, clock_mhz=200)
print("\nThroughput Estimation:")
print(f"  {throughput['num_units']} units @ {throughput['clock_mhz']} MHz")
print(f"  Throughput: {throughput['throughput_msamples_per_s']:.0f} MSamples/s")
print(f"  Latency: {throughput['latency_cycles']} cycle")

# Estimation ressources
resources_per_unit = {'dsp': 2, 'lut': 500, 'bram': 1}
total = spatial.estimate_resources(8, resources_per_unit)
print("\nResource Estimation:")
for resource, value in total.items():
    print(f"  {resource.upper()}: {value}")
```

---

## Parallélisme Temporel (Pipeline)

### Principe

```python
class TemporalParallelism:
    """
    Parallélisme temporel: pipeline
    """
    
    def __init__(self):
        self.description = """
        Divise le calcul en stages séquentiels.
        Plusieurs données traversent le pipeline simultanément.
        
        Avantages:
        - Utilise efficacement les ressources
        - Throughput élevé après remplissage
        - Latence acceptable
        
        Inconvénients:
        - Latence initiale (pipeline fill)
        - Gestion de dépendances complexe
        """
    
    def visualize(self, num_stages=5):
        """Visualise le pipeline"""
        diagram = f"""
Temporal Parallelism (Pipeline, {num_stages} stages):

Stage 1 → Stage 2 → Stage 3 → ... → Stage {num_stages} → Output

Timeline (after pipeline fill):
Cycle  ───────────────────────────────────────────────────────►
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
        │ D0  │ │ D1  │ │ D2  │ │ D3  │ │ D4  │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           │       │       │       │       │
           ▼       ▼       ▼       ▼       ▼
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
        │ D1  │ │ D2  │ │ D3  │ │ D4  │ │ D5  │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           │       │       │       │       │
           ▼       ▼       ▼       ▼       ▼
        ... continues through all stages ...
        
Throughput = 1 sample/cycle (after fill)
Latency = {num_stages} cycles
Pipeline Fill Time = {num_stages} cycles
"""
        return diagram
    
    def calculate_throughput(self, clock_mhz, ii=1):
        """
        Calcule le throughput
        
        Args:
            clock_mhz: Fréquence d'horloge
            ii: Initiation Interval
        """
        throughput_msamples_per_s = clock_mhz / ii
        
        return {
            'clock_mhz': clock_mhz,
            'ii': ii,
            'throughput_msamples_per_s': throughput_msamples_per_s,
            'samples_per_cycle': 1.0 / ii
        }
    
    def calculate_latency(self, num_stages, clock_period_ns):
        """Calcule la latence"""
        latency_ns = num_stages * clock_period_ns
        
        return {
            'num_stages': num_stages,
            'clock_period_ns': clock_period_ns,
            'latency_ns': latency_ns,
            'latency_us': latency_ns / 1000,
            'latency_cycles': num_stages
        }

temporal = TemporalParallelism()
print(temporal.visualize(5))

throughput = temporal.calculate_throughput(200, ii=1)
print("\nPipeline Throughput:")
print(f"  {throughput['clock_mhz']} MHz, II={throughput['ii']}")
print(f"  Throughput: {throughput['throughput_msamples_per_s']:.0f} MSamples/s")

latency = temporal.calculate_latency(10, clock_period_ns=5)
print("\nPipeline Latency:")
print(f"  {latency['num_stages']} stages @ 5ns")
print(f"  Latency: {latency['latency_us']:.2f} μs")
```

---

## Combinaison: Parallélisme Spatial + Temporel

```python
class CombinedParallelism:
    """
    Combinaison de parallélisme spatial et temporel
    """
    
    def visualize_combined(self, spatial_units=4, temporal_stages=3):
        """Visualise la combinaison"""
        diagram = f"""
Combined Parallelism:
- Spatial: {spatial_units} units per stage
- Temporal: {temporal_stages} stages

Stage 1:              Stage 2:              Stage 3:
┌───┐ ┌───┐ ┌───┐    ┌───┐ ┌───┐ ┌───┐    ┌───┐ ┌───┐ ┌───┐
│PE0│ │PE1│ │PE2│    │PE0│ │PE1│ │PE2│    │PE0│ │PE1│ │PE2│
└───┘ └───┘ └───┘    └───┘ └───┘ └───┘    └───┘ └───┘ └───┘
  │     │     │         │     │     │         │     │     │
  └─────┴─────┘         └─────┴─────┘         └─────┴─────┘

Timeline (after pipeline fill):
Cycle ──────────────────────────────────────────────────────►
      Stage 1: [D0,D1,D2] [D3,D4,D5] [D6,D7,D8] ...
      Stage 2:            [D0,D1,D2] [D3,D4,D5] [D6,D7,D8] ...
      Stage 3:                         [D0,D1,D2] [D3,D4,D5] ...

Throughput = {spatial_units} samples/cycle
Latency = {temporal_stages} cycles
Total PEs = {spatial_units * temporal_stages}
"""
        return diagram
    
    def estimate_performance(self, spatial_units, temporal_stages, clock_mhz):
        """
        Estime les performances combinées
        
        Args:
            spatial_units: Unités par stage
            temporal_stages: Nombre de stages
            clock_mhz: Fréquence
        """
        throughput = spatial_units * clock_mhz
        latency_cycles = temporal_stages
        latency_us = (temporal_stages * 1000) / clock_mhz
        
        total_units = spatial_units * temporal_stages
        
        return {
            'spatial_units': spatial_units,
            'temporal_stages': temporal_stages,
            'total_units': total_units,
            'throughput_msamples_per_s': throughput,
            'latency_cycles': latency_cycles,
            'latency_us': latency_us,
            'efficiency': 'High (both parallelisms utilized)'
        }
    
    def estimate_resources(self, spatial_units, temporal_stages, resources_per_unit):
        """Estime les ressources"""
        total_units = spatial_units * temporal_stages
        
        total_resources = {}
        for resource, value in resources_per_unit.items():
            total_resources[resource] = value * total_units
        
        return total_resources

combined = CombinedParallelism()
print(combined.visualize_combined(4, 3))

perf = combined.estimate_performance(spatial_units=8, temporal_stages=5, clock_mhz=200)
print("\nCombined Performance:")
print(f"  Spatial: {perf['spatial_units']} units/stage")
print(f"  Temporal: {perf['temporal_stages']} stages")
print(f"  Total units: {perf['total_units']}")
print(f"  Throughput: {perf['throughput_msamples_per_s']:.0f} MSamples/s")
print(f"  Latency: {perf['latency_us']:.2f} μs")
```

---

## Exemple: Multiplication Matrice-Vecteur

```python
class MatrixVectorMultiply:
    """
    Exemple: Multiplication matrice-vecteur avec différents parallélismes
    """
    
    def __init__(self):
        self.operation = "Y = W × X"
        self.w_shape = (256, 128)  # 256 outputs, 128 inputs
        self.x_shape = (128,)
    
    def spatial_approach(self, parallel_outputs=32):
        """
        Approche spatiale: parallélise les outputs
        """
        num_pe = parallel_outputs
        cycles_needed = self.w_shape[1]  # 128 cycles pour accumuler
        
        throughput = num_pe / cycles_needed  # outputs par cycle
        
        return {
            'approach': 'Spatial (parallel outputs)',
            'parallel_outputs': parallel_outputs,
            'cycles_per_output': cycles_needed,
            'throughput_outputs_per_cycle': throughput,
            'latency_cycles': cycles_needed,
            'resources_pe': num_pe
        }
    
    def temporal_approach(self, pipeline_stages=8):
        """
        Approche temporelle: pipeline les dot products
        """
        dot_product_length = self.w_shape[1]  # 128
        
        # Divise en stages
        operations_per_stage = dot_product_length // pipeline_stages
        
        throughput = 1.0  # 1 output par cycle (après fill)
        latency_cycles = pipeline_stages
        
        return {
            'approach': 'Temporal (pipeline)',
            'pipeline_stages': pipeline_stages,
            'ops_per_stage': operations_per_stage,
            'throughput_outputs_per_cycle': throughput,
            'latency_cycles': latency_cycles,
            'resources_pe': 1  # 1 PE réutilisé
        }
    
    def combined_approach(self, parallel_outputs=16, pipeline_stages=4):
        """
        Approche combinée
        """
        dot_product_length = self.w_shape[1]
        operations_per_stage = dot_product_length // pipeline_stages
        
        throughput = parallel_outputs  # outputs par cycle (après fill)
        latency_cycles = pipeline_stages
        
        return {
            'approach': 'Combined',
            'parallel_outputs': parallel_outputs,
            'pipeline_stages': pipeline_stages,
            'throughput_outputs_per_cycle': throughput,
            'latency_cycles': latency_cycles,
            'resources_pe': parallel_outputs * pipeline_stages
        }
    
    def compare_approaches(self):
        """Compare les approches"""
        spatial = self.spatial_approach(parallel_outputs=32)
        temporal = self.temporal_approach(pipeline_stages=8)
        combined = self.combined_approach(parallel_outputs=16, pipeline_stages=4)
        
        approaches = [spatial, temporal, combined]
        
        print("\n" + "="*60)
        print("Comparison: Matrix-Vector Multiply (256×128)")
        print("="*60)
        
        for approach in approaches:
            print(f"\n{approach['approach']}:")
            print(f"  Throughput: {approach['throughput_outputs_per_cycle']:.2f} outputs/cycle")
            print(f"  Latency: {approach['latency_cycles']} cycles")
            print(f"  Resources: {approach['resources_pe']} PEs")

matvec = MatrixVectorMultiply()
matvec.compare_approaches()
```

---

## Trade-offs

```python
class ParallelismTradeoffs:
    """
    Trade-offs entre parallélisme spatial et temporel
    """
    
    tradeoffs = {
        'spatial': {
            'throughput': 'High (proportional to units)',
            'latency': 'Low (1 cycle if enough units)',
            'resources': 'High (linear with units)',
            'power': 'High (all units active)',
            'scalability': 'Limited by resources',
            'best_for': 'Low latency critical, resources available'
        },
        'temporal': {
            'throughput': 'High (1 sample/cycle after fill)',
            'latency': 'Medium (number of stages)',
            'resources': 'Low (reuse same units)',
            'power': 'Medium (stages active)',
            'scalability': 'Good (can add stages)',
            'best_for': 'High throughput, resource constrained'
        },
        'combined': {
            'throughput': 'Very High',
            'latency': 'Medium',
            'resources': 'Very High',
            'power': 'Very High',
            'scalability': 'Limited',
            'best_for': 'Maximum performance, resources available'
        }
    }
    
    @staticmethod
    def display_tradeoffs():
        """Affiche les trade-offs"""
        print("\n" + "="*60)
        print("Parallelism Trade-offs")
        print("="*60)
        
        for parallelism, metrics in ParallelismTradeoffs.tradeoffs.items():
            print(f"\n{parallelism.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

ParallelismTradeoffs.display_tradeoffs()
```

---

## Exercices

### Exercice 14.3.1
Concevez une architecture combinant parallélisme spatial et temporel pour une couche Conv2D avec des contraintes de ressources.

### Exercice 14.3.2
Comparez le throughput et les ressources nécessaires pour un réseau avec 10 couches en utilisant:
- Approche spatiale pure (10 PEs)
- Approche temporelle pure (pipeline 10 stages)
- Approche combinée (2 PEs/stage, 5 stages)

---

## Points Clés à Retenir

> 📌 **Spatial: plus d'unités = plus de throughput, mais plus de ressources**

> 📌 **Temporal: pipeline = réutilisation efficace, latence modérée**

> 📌 **Combiné: meilleures performances mais ressources élevées**

> 📌 **Trade-offs: resources vs latency vs throughput**

> 📌 **Choix dépend des contraintes spécifiques**

---

*Section suivante : [14.4 Optimisation des Accès Mémoire](./14_04_Memoire.md)*

