# 14.1 Défis Spécifiques aux FPGA

---

## Introduction

Le déploiement de réseaux de neurones sur FPGA présente des **défis uniques** comparé aux CPU/GPU. Cette section détaille ces défis et leurs implications pratiques.

---

## Vue d'Ensemble des Défis

```
┌─────────────────────────────────────────────────────────────────┐
│              Défis du Déploiement ML sur FPGA                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Mémoire Limitée                                            │
│     └─ BRAM: quelques MB seulement                             │
│     └─ Nécessite compression agressive                         │
│                                                                 │
│  2. Latence Critique                                           │
│     └─ Trigger L1: < 4 μs                                      │
│     └─ Pipeline complexe nécessaire                            │
│                                                                 │
│  3. Throughput Requis                                          │
│     └─ 40 MHz (1 événement toutes les 25 ns)                   │
│     └─ Initiation Interval = 1 idéal                           │
│                                                                 │
│  4. Ressources Finies                                          │
│     └─ LUT, DSP, BRAM fixes                                    │
│     └─ Trade-offs complexes                                    │
│                                                                 │
│  5. Consommation Énergétique                                   │
│     └─ Densité calcul vs puissance                             │
│     └─ Refroidissement limité                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Défi 1: Contraintes Mémoire

### Limites de Mémoire

```python
class MemoryConstraints:
    """
    Analyse des contraintes mémoire FPGA
    """
    
    def __init__(self):
        # Exemple: Xilinx Zynq-7000
        self.fpga_resources = {
            'bram_18k': 560,           # blocs BRAM
            'bram_size_kb': 18,        # KB par bloc
            'total_bram_mb': (560 * 18) / 1024,  # ~9.8 MB
            'lut': 53200,
            'dsp': 220
        }
    
    def analyze_model_memory(self, model, bits=8):
        """
        Analyse la mémoire nécessaire pour un modèle
        
        Args:
            model: Modèle PyTorch
            bits: Précision (8 pour int8, 32 pour float32)
        """
        total_params = sum(p.numel() for p in model.parameters())
        
        # Mémoire nécessaire pour les poids
        weight_memory_bits = total_params * bits
        weight_memory_mb = weight_memory_bits / (8 * 1024 * 1024)
        
        # Mémoire pour activations (estimation)
        # Approximatif: taille du batch × features
        activation_memory_estimate_mb = 10  # Estimation
        
        total_memory_mb = weight_memory_mb + activation_memory_estimate_mb
        
        # Conversion en BRAM
        bram_18k_needed = (weight_memory_bits) / (18 * 1024)
        
        return {
            'total_params': total_params,
            'weight_memory_mb': weight_memory_mb,
            'total_memory_mb': total_memory_mb,
            'bram_18k_needed': bram_18k_needed,
            'fits_in_fpga': bram_18k_needed <= self.fpga_resources['bram_18k'],
            'compression_needed': weight_memory_mb > self.fpga_resources['total_bram_mb'] * 0.8
        }
    
    def display_constraints(self):
        """Affiche les contraintes mémoire"""
        print("\n" + "="*60)
        print("FPGA Memory Constraints")
        print("="*60)
        print(f"\nFPGA Resources:")
        print(f"  BRAM 18K blocks: {self.fpga_resources['bram_18k']}")
        print(f"  Total BRAM: {self.fpga_resources['total_bram_mb']:.2f} MB")
        
        print("\nMemory Breakdown:")
        print("  Weight storage: Most critical")
        print("  Activation buffers: Intermediate results")
        print("  Input/output buffers: Data streaming")

# Exemple d'analyse
memory_constraints = MemoryConstraints()
memory_constraints.display_constraints()

# Analyse d'un modèle
import torch.nn as nn

model_example = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

analysis = memory_constraints.analyze_model_memory(model_example, bits=8)

print("\n" + "="*60)
print("Model Memory Analysis")
print("="*60)
print(f"  Total parameters: {analysis['total_params']:,}")
print(f"  Weight memory: {analysis['weight_memory_mb']:.2f} MB")
print(f"  BRAM 18K needed: {analysis['bram_18k_needed']:.1f}")
print(f"  Fits in FPGA: {analysis['fits_in_fpga']}")
print(f"  Compression needed: {analysis['compression_needed']}")
```

### Stratégies pour Réduire la Mémoire

```python
class MemoryOptimizationStrategies:
    """
    Stratégies pour réduire l'utilisation mémoire
    """
    
    strategies = {
        'quantization': {
            'description': 'Réduire précision (32→8 bits)',
            'reduction': '4x reduction',
            'tradeoff': 'Petite perte de précision'
        },
        'weight_sharing': {
            'description': 'Partager poids entre couches',
            'reduction': 'Variable',
            'tradeoff': 'Réduit expressivité'
        },
        'compression_tensor': {
            'description': 'Tensor decomposition (TT, Tucker)',
            'reduction': '5-10x typical',
            'tradeoff': 'Complexité calcul'
        },
        'pruning': {
            'description': 'Supprimer poids peu importants',
            'reduction': '2-5x typical',
            'tradeoff': 'Risque perte performance'
        },
        'weight_streaming': {
            'description': 'Charger poids depuis DDR au lieu de BRAM',
            'reduction': 'Libère BRAM',
            'tradeoff': 'Latence accrue'
        }
    }
    
    @staticmethod
    def display_strategies():
        """Affiche les stratégies"""
        print("\n" + "="*60)
        print("Memory Optimization Strategies")
        print("="*60)
        
        for strategy, info in MemoryOptimizationStrategies.strategies.items():
            print(f"\n{strategy.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Reduction: {info['reduction']}")
            print(f"  Tradeoff: {info['tradeoff']}")

MemoryOptimizationStrategies.display_strategies()
```

---

## Défi 2: Latence et Timing

### Contraintes de Latence

```python
class LatencyConstraints:
    """
    Analyse des contraintes de latence
    """
    
    def __init__(self):
        self.application_latencies = {
            'trigger_l1': {
                'max_latency_ns': 4000,  # 4 μs
                'description': 'LHC Level-1 Trigger',
                'critical': True
            },
            'trigger_hlt': {
                'max_latency_ms': 100,  # 100 ms
                'description': 'High-Level Trigger',
                'critical': False
            },
            'inference_edge': {
                'max_latency_ms': 10,  # 10 ms
                'description': 'Edge inference',
                'critical': False
            }
        }
    
    def analyze_pipeline_latency(self, num_stages, clock_period_ns=5):
        """
        Analyse la latence d'un pipeline
        
        Args:
            num_stages: Nombre de stages du pipeline
            clock_period_ns: Période d'horloge en ns
        """
        # Latence = nombre de stages × période d'horloge
        latency_ns = num_stages * clock_period_ns
        
        # Throughput = 1 / période (si II=1)
        throughput_mhz = 1000.0 / clock_period_ns
        
        # Vérifie si ça respecte la contrainte L1
        meets_l1 = latency_ns <= self.application_latencies['trigger_l1']['max_latency_ns']
        
        return {
            'latency_ns': latency_ns,
            'latency_us': latency_ns / 1000,
            'throughput_mhz': throughput_mhz,
            'meets_l1_constraint': meets_l1,
            'max_stages_for_l1': int(self.application_latencies['trigger_l1']['max_latency_ns'] / clock_period_ns)
        }
    
    def display_constraints(self):
        """Affiche les contraintes de latence"""
        print("\n" + "="*60)
        print("Latency Constraints by Application")
        print("="*60)
        
        for app, constraints in self.application_latencies.items():
            print(f"\n{app.replace('_', ' ').title()}:")
            print(f"  {constraints['description']}")
            for key, value in constraints.items():
                if key != 'description':
                    print(f"  {key}: {value}")

latency_constraints = LatencyConstraints()
latency_constraints.display_constraints()

# Analyse de pipeline
print("\n" + "="*60)
print("Pipeline Latency Analysis")
print("="*60)

for stages in [100, 200, 500, 800]:
    analysis = latency_constraints.analyze_pipeline_latency(stages, clock_period_ns=5)
    print(f"\n{stages} stages @ 200 MHz (5ns period):")
    print(f"  Latency: {analysis['latency_us']:.2f} μs")
    print(f"  Meets L1: {analysis['meets_l1_constraint']}")
    if not analysis['meets_l1_constraint']:
        print(f"  ⚠️  Exceeds L1 limit by {analysis['latency_ns'] - 4000:.0f} ns")
```

### Initiation Interval (II)

```python
class InitiationInterval:
    """
    Concept d'Initiation Interval
    """
    
    def __init__(self):
        self.concept = {
            'definition': 'Temps entre deux initiations successives d\'une opération',
            'ii_1': 'Nouvelle donnée chaque cycle (optimal)',
            'ii_n': 'Nouvelle donnée tous les N cycles',
            'impact': 'Détermine le throughput maximum'
        }
    
    def calculate_throughput(self, clock_mhz, ii=1):
        """
        Calcule le throughput basé sur II
        
        Args:
            clock_mhz: Fréquence d'horloge en MHz
            ii: Initiation Interval
        """
        throughput = clock_mhz / ii  # MSamples/s
        
        return {
            'clock_mhz': clock_mhz,
            'ii': ii,
            'throughput_msamples_per_s': throughput,
            'samples_per_event_period': throughput / 40,  # Pour LHC @ 40MHz
            'meets_lhc_requirement': throughput >= 40
        }
    
    def display_concept(self):
        """Affiche le concept"""
        print("\n" + "="*60)
        print("Initiation Interval (II) Concept")
        print("="*60)
        
        for key, value in self.concept.items():
            print(f"  {key}: {value}")
        
        print("\nThroughput Examples:")
        for clock in [100, 200, 300]:
            for ii in [1, 2, 4]:
                result = self.calculate_throughput(clock, ii)
                print(f"  {clock} MHz, II={ii}: {result['throughput_msamples_per_s']:.1f} MSamples/s")

ii_concept = InitiationInterval()
ii_concept.display_concept()
```

---

## Défi 3: Ressources Limitées

### Analyse des Ressources

```python
class ResourceConstraints:
    """
    Contraintes de ressources FPGA
    """
    
    def __init__(self):
        # Exemple: Zynq-7000
        self.resources = {
            'lut': 53200,
            'ff': 106400,
            'bram_18k': 560,
            'dsp': 220,
            'io': 200
        }
    
    def estimate_layer_resources(self, layer_type, config, reuse_factor=1):
        """
        Estime les ressources pour une couche
        
        Args:
            layer_type: 'linear', 'conv2d', etc.
            config: Configuration de la couche
            reuse_factor: Facteur de réutilisation
        """
        if layer_type == 'linear':
            in_features = config['in_features']
            out_features = config['out_features']
            
            # DSP: multiplications
            mults = in_features * out_features
            dsps = mults // reuse_factor
            
            # BRAM: stockage des poids (int8)
            weight_bits = in_features * out_features * 8
            brams = weight_bits / (18 * 1024)
            
            # LUT: logique additionnelle
            luts_estimate = out_features * 100  # Approximation
            
            return {
                'dsp': int(dsps),
                'bram_18k': int(brams),
                'lut': int(luts_estimate)
            }
        
        elif layer_type == 'conv2d':
            # Plus complexe, simplification
            return {
                'dsp': config.get('dsp_estimate', 100),
                'bram_18k': config.get('bram_estimate', 10),
                'lut': config.get('lut_estimate', 5000)
            }
        
        return {'dsp': 0, 'bram_18k': 0, 'lut': 0}
    
    def check_fits(self, model_estimate):
        """
        Vérifie si l'estimation rentre dans les ressources
        
        Args:
            model_estimate: Dict avec ressources estimées
        """
        fits = {}
        for resource, value in model_estimate.items():
            available = self.resources.get(resource, 0)
            fits[resource] = {
                'used': value,
                'available': available,
                'utilization': (value / available * 100) if available > 0 else 0,
                'fits': value <= available
            }
        
        return fits

resources = ResourceConstraints()

print("\n" + "="*60)
print("FPGA Resource Constraints")
print("="*60)
print("\nAvailable Resources:")
for resource, value in resources.resources.items():
    print(f"  {resource.upper()}: {value:,}")

# Estimation pour une couche
linear_config = {'in_features': 256, 'out_features': 128}
estimate = resources.estimate_layer_resources('linear', linear_config, reuse_factor=4)

print("\n" + "="*60)
print("Layer Resource Estimate (Linear 256→128, reuse=4)")
print("="*60)
for resource, value in estimate.items():
    print(f"  {resource.upper()}: {value:,}")

# Vérification
fits = resources.check_fits(estimate)
print("\nFit Check:")
for resource, info in fits.items():
    print(f"  {resource.upper()}: {info['utilization']:.1f}% used "
          f"({'✓ Fits' if info['fits'] else '✗ Exceeds'})")
```

---

## Défi 4: Consommation Énergétique

### Analyse de Puissance

```python
class PowerConsumption:
    """
    Consommation énergétique FPGA
    """
    
    def __init__(self):
        self.power_components = {
            'static_power': {
                'description': 'Puissance statique (fuites)',
                'typical_w': 1.0,
                'depends_on': 'Process, température'
            },
            'dynamic_power': {
                'description': 'Puissance dynamique (commutations)',
                'typical_w': 1.5,
                'depends_on': 'Fréquence, switching activity'
            },
            'io_power': {
                'description': 'Puissance I/O',
                'typical_w': 0.5,
                'depends_on': 'Standards I/O, charge'
            }
        }
        
        self.total_typical_w = sum(
            comp['typical_w'] for comp in self.power_components.values()
        )
    
    def estimate_power(self, clock_mhz, utilization_lut=0.5, utilization_dsp=0.5):
        """
        Estime la consommation
        
        Args:
            clock_mhz: Fréquence d'horloge
            utilization_lut: Utilisation des LUT (0-1)
            utilization_dsp: Utilisation des DSP (0-1)
        """
        # Modèle simplifié
        static = self.power_components['static_power']['typical_w']
        
        # Dynamique proportionnelle à fréquence et utilisation
        dynamic_base = self.power_components['dynamic_power']['typical_w']
        dynamic = dynamic_base * (clock_mhz / 200) * (
            utilization_lut * 0.6 + utilization_dsp * 0.4
        )
        
        io = self.power_components['io_power']['typical_w']
        
        total = static + dynamic + io
        
        return {
            'static_w': static,
            'dynamic_w': dynamic,
            'io_w': io,
            'total_w': total,
            'efficiency_gops_per_w': self._estimate_ops_per_watt(total)
        }
    
    def _estimate_ops_per_watt(self, power_w):
        """Estime GOPS/W (simplifié)"""
        # Approximation: ~10 GOPS/W typique pour FPGA
        return 10.0 / power_w if power_w > 0 else 0
    
    def display_components(self):
        """Affiche les composants"""
        print("\n" + "="*60)
        print("Power Consumption Components")
        print("="*60)
        
        for component, info in self.power_components.items():
            print(f"\n{component.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Typical: {info['typical_w']} W")
            print(f"  Depends on: {info['depends_on']}")

power = PowerConsumption()
power.display_components()

print("\n" + "="*60)
print("Power Estimation Examples")
print("="*60)

for clock in [100, 200, 300]:
    for util in [0.3, 0.6, 0.9]:
        estimate = power.estimate_power(clock, utilization_lut=util, utilization_dsp=util)
        print(f"\n{clock} MHz, {util*100:.0f}% utilization:")
        print(f"  Total: {estimate['total_w']:.2f} W")
        print(f"  Efficiency: {estimate['efficiency_gops_per_w']:.1f} GOPS/W")
```

---

## Exercices

### Exercice 14.1.1
Analysez la mémoire nécessaire pour un ResNet-18 quantifié en int8 et déterminez s'il peut tenir dans un FPGA avec 9 MB de BRAM.

### Exercice 14.1.2
Calculez le nombre maximum de stages de pipeline possibles pour respecter une contrainte de latence de 4 μs à 200 MHz.

---

## Points Clés à Retenir

> 📌 **Mémoire BRAM très limitée → compression nécessaire**

> 📌 **Latence critique pour triggers → pipeline optimisé**

> 📌 **II=1 idéal pour throughput maximum**

> 📌 **Ressources fixes → trade-offs complexes**

> 📌 **Puissance limitée → optimisation densité calcul/W**

---

*Section suivante : [14.2 Architectures de Dataflow](./14_02_Dataflow.md)*

