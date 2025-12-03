# 14.4 Optimisation des Accès Mémoire

---

## Introduction

L'**optimisation des accès mémoire** est critique sur FPGA où la mémoire est limitée et les accès peuvent être des goulots d'étranglement. Cette section couvre les techniques pour optimiser ces accès.

---

## Hiérarchie Mémoire FPGA

```
┌─────────────────────────────────────────────────────────────────┐
│              Hiérarchie Mémoire FPGA                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Registres (FF)                                                │
│  └─ Plus rapide, limité (~100k)                                │
│                                                                 │
│  BRAM (Block RAM)                                              │
│  └─ Rapide, taille moyenne (~10 MB)                            │
│                                                                 │
│  UltraRAM (URAM) - certains FPGA                               │
│  └─ Plus grand, un peu plus lent                               │
│                                                                 │
│  DDR / HBM (High Bandwidth Memory)                             │
│  └─ Grande capacité, plus lent                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Techniques d'Optimisation

### 1. Streaming et Buffering

```python
class MemoryStreaming:
    """
    Streaming et buffering pour optimiser accès mémoire
    """
    
    def __init__(self):
        self.techniques = {
            'streaming': {
                'description': 'Traite données au fur et à mesure',
                'advantage': 'Réduit buffers intermédiaires',
                'use_case': 'Large datasets, pipeline'
            },
            'double_buffering': {
                'description': 'Deux buffers alternent lecture/écriture',
                'advantage': 'Masque latence, permet overlap',
                'use_case': 'Pipelines avec accès DDR'
            },
            'circular_buffering': {
                'description': 'Buffer circulaire pour données continues',
                'advantage': 'Efficace pour streams',
                'use_case': 'Traitement continu'
            }
        }
    
    def visualize_double_buffering(self):
        """Visualise le double buffering"""
        diagram = """
Double Buffering:

Time ───────────────────────────────────────────────────►

Buffer A: [Read] ──► [Process] ──► [Write]
                │
Buffer B:       └──► [Read] ──► [Process] ──► [Write]
                │
                ▼ (alternate)

Cycle 0: Buffer A read from DDR
Cycle 1: Buffer A processing, Buffer B read from DDR
Cycle 2: Buffer A write to DDR, Buffer B processing
Cycle 3: Repeat...

Advantage: Read and process overlap
"""
        return diagram
    
    def estimate_buffer_size(self, data_size, num_buffers=2):
        """Estime la taille de buffer nécessaire"""
        return {
            'single_buffer_size': data_size,
            'double_buffer_size': data_size * num_buffers,
            'memory_overhead': data_size * (num_buffers - 1),
            'benefit': 'Overlaps computation with I/O'
        }

streaming = MemoryStreaming()
print(streaming.visualize_double_buffering())

for tech, info in streaming.techniques.items():
    print(f"\n{tech.replace('_', ' ').title()}:")
    print(f"  {info['description']}")
    print(f"  Advantage: {info['advantage']}")
```

---

### 2. Partitionnement de Tableaux

```python
class ArrayPartitioning:
    """
    Partitionnement de tableaux pour accès parallèle
    """
    
    def __init__(self):
        self.partition_types = {
            'complete': {
                'description': 'Chaque élément dans BRAM séparé',
                'parallel_access': 'Maximum',
                'resource_cost': 'High',
                'use_case': 'Small arrays, high throughput needed'
            },
            'cyclic': {
                'description': 'Éléments distribués cycliquement',
                'parallel_access': 'Good',
                'resource_cost': 'Medium',
                'use_case': 'Sequential access patterns'
            },
            'block': {
                'description': 'Éléments groupés en blocs',
                'parallel_access': 'Moderate',
                'resource_cost': 'Medium',
                'use_case': 'Block-based processing'
            }
        }
    
    def visualize_partitioning(self, array_size=16, partitions=4):
        """Visualise les types de partitionnement"""
        diagram = f"""
Array Partitioning ({array_size} elements, {partitions} partitions):

Original Array:
[A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15]

Complete Partitioning:
BRAM0: [A0]
BRAM1: [A1]
BRAM2: [A2]
BRAM3: [A3]
... (each element separate)

Cyclic Partitioning:
BRAM0: [A0 A4 A8  A12]
BRAM1: [A1 A5 A9  A13]
BRAM2: [A2 A6 A10 A14]
BRAM3: [A3 A7 A11 A15]

Block Partitioning:
BRAM0: [A0 A1 A2  A3 ]
BRAM1: [A4 A5 A6  A7 ]
BRAM2: [A8 A9 A10 A11]
BRAM3: [A12 A13 A14 A15]

Parallel Access:
Complete:  All elements simultaneously
Cyclic:    {partitions} elements simultaneously (cyclic pattern)
Block:     {partitions} elements simultaneously (block pattern)
"""
        return diagram
    
    def estimate_resources(self, array_size, partition_type, num_partitions):
        """Estime les ressources BRAM"""
        if partition_type == 'complete':
            bram_needed = array_size
        elif partition_type == 'cyclic' or partition_type == 'block':
            bram_needed = num_partitions
        else:
            bram_needed = 1
        
        return {
            'array_size': array_size,
            'partition_type': partition_type,
            'partitions': num_partitions,
            'bram_18k_needed': bram_needed,
            'parallel_access_width': num_partitions
        }

partitioning = ArrayPartitioning()
print(partitioning.visualize_partitioning())

for ptype, info in partitioning.partition_types.items():
    print(f"\n{ptype.replace('_', ' ').title()}:")
    print(f"  {info['description']}")
    print(f"  Parallel access: {info['parallel_access']}")
    print(f"  Resource cost: {info['resource_cost']}")

# Exemple
resources = partitioning.estimate_resources(256, 'cyclic', 8)
print(f"\nExample: 256 elements, cyclic, 8 partitions")
print(f"  BRAM needed: {resources['bram_18k_needed']}")
print(f"  Parallel access: {resources['parallel_access_width']} elements")
```

---

### 3. Burst Transfers

```python
class BurstTransfers:
    """
    Transfers en rafale pour optimiser DDR
    """
    
    def __init__(self):
        self.concept = """
        Transférer plusieurs données consécutives en une transaction
        au lieu de transactions individuelles.
        
        Avantages:
        - Réduit overhead de transaction
        - Utilise efficacement la bande passante DDR
        - Améliore throughput
        """
    
    def visualize_burst(self, burst_length=4):
        """Visualise les transfers en rafale"""
        diagram = f"""
Burst Transfer (Length {burst_length}):

Without Burst (Individual Transfers):
Time ──────────────────────────────────────────────────────►
      [T] [D0] [T] [D1] [T] [D2] [T] [D3]
      └─┘      └─┘      └─┘      └─┘
      Transaction overhead

With Burst:
Time ──────────────────────────────────────────────────────►
      [T] [D0][D1][D2][D3]
      └─┘ └─────────────┘
      Single transaction, {burst_length} data

Efficiency Gain:
- Without: 4 transactions, 4× overhead
- With:    1 transaction, 1× overhead
- Speedup: ~{burst_length}x for overhead
"""
        return diagram
    
    def calculate_bandwidth(self, data_width_bits, burst_length, clock_mhz):
        """
        Calcule la bande passante effective
        
        Args:
            data_width_bits: Largeur du bus (ex: 512 bits)
            burst_length: Longueur de rafale
            clock_mhz: Fréquence
        """
        bytes_per_transfer = (data_width_bits * burst_length) / 8
        transfers_per_second = clock_mhz * 1e6
        
        bandwidth_gbps = (bytes_per_transfer * transfers_per_second) / 1e9
        
        return {
            'data_width_bits': data_width_bits,
            'burst_length': burst_length,
            'clock_mhz': clock_mhz,
            'bytes_per_transfer': bytes_per_transfer,
            'bandwidth_gbps': bandwidth_gbps,
            'efficiency': 'High with burst, low without'
        }

burst = BurstTransfers()
print(burst.visualize_burst(8))

bandwidth = burst.calculate_bandwidth(data_width_bits=512, burst_length=16, clock_mhz=200)
print("\nBandwidth Calculation:")
print(f"  512-bit bus, 16-length burst @ 200 MHz")
print(f"  Bandwidth: {bandwidth['bandwidth_gbps']:.2f} GB/s")
```

---

### 4. Data Reuse et Caching

```python
class DataReuse:
    """
    Réutilisation de données pour réduire accès mémoire
    """
    
    def __init__(self):
        self.reuse_strategies = {
            'weight_stationary': {
                'description': 'Poids chargés une fois, réutilisés',
                'access_reduction': '1 read per weight',
                'memory_type': 'BRAM for weights'
            },
            'activation_reuse': {
                'description': 'Réutilise activations entre couches',
                'access_reduction': 'Variable',
                'memory_type': 'On-chip buffers'
            },
            'kernel_reuse_conv': {
                'description': 'Réutilise kernel dans convolution',
                'access_reduction': '1 read per kernel',
                'memory_type': 'BRAM or registers'
            }
        }
    
    def analyze_conv2d_reuse(self, image_size, kernel_size, num_kernels):
        """
        Analyse la réutilisation dans Conv2D
        
        Args:
            image_size: (H, W, C)
            kernel_size: (Kh, Kw, C)
            num_kernels: Nombre de kernels
        """
        H, W, C = image_size
        Kh, Kw, _ = kernel_size
        
        # Accès sans réutilisation
        accesses_no_reuse = H * W * Kh * Kw * C * num_kernels
        
        # Avec réutilisation kernel
        kernel_elements = Kh * Kw * C
        accesses_with_reuse = H * W * num_kernels * kernel_elements
        
        # Avec réutilisation complète (kernel + activation)
        # Approximatif
        accesses_full_reuse = H * W * num_kernels
        
        reduction_factor = accesses_no_reuse / accesses_with_reuse
        
        return {
            'accesses_no_reuse': accesses_no_reuse,
            'accesses_kernel_reuse': accesses_with_reuse,
            'accesses_full_reuse': accesses_full_reuse,
            'reduction_factor': reduction_factor,
            'memory_savings': 'Significant with proper reuse'
        }

reuse = DataReuse()

print("\n" + "="*60)
print("Data Reuse Strategies")
print("="*60)

for strategy, info in reuse.reuse_strategies.items():
    print(f"\n{strategy.replace('_', ' ').title()}:")
    print(f"  {info['description']}")
    print(f"  Access reduction: {info['access_reduction']}")

# Analyse Conv2D
analysis = reuse.analyze_conv2d_reuse(
    image_size=(224, 224, 3),
    kernel_size=(3, 3, 3),
    num_kernels=64
)

print("\n" + "="*60)
print("Conv2D Reuse Analysis (224×224×3, 3×3×3 kernel, 64 kernels)")
print("="*60)
print(f"  Accesses without reuse: {analysis['accesses_no_reuse']:,}")
print(f"  Accesses with kernel reuse: {analysis['accesses_kernel_reuse']:,}")
print(f"  Reduction factor: {analysis['reduction_factor']:.1f}x")
```

---

### 5. Prefetching

```python
class MemoryPrefetching:
    """
    Prefetching: charger données avant utilisation
    """
    
    def __init__(self):
        self.concept = """
        Charger les données nécessaires en avance pour masquer
        la latence des accès mémoire (DDR).
        
        Avantages:
        - Masque latence DDR
        - Améliore utilisation pipeline
        - Overlaps I/O avec computation
        
        Défis:
        - Prédiction des accès futurs
        - Gestion de buffer
        """
    
    def visualize_prefetching(self):
        """Visualise le prefetching"""
        diagram = """
Memory Prefetching:

Normal (No Prefetch):
Time ───────────────────────────────────────────────►
      [Compute D0] [Read D1] [Compute D1] [Read D2] ...
      
With Prefetch:
Time ───────────────────────────────────────────────►
      [Compute D0] [Compute D1] [Compute D2] ...
      [Read D1]    [Read D2]    [Read D3]    ...
      
      I/O overlaps with computation

Prefetch Buffer:
      ┌─────────┐
      │   D1    │ ──► Next to use
      │   D2    │ ──► Prefetched
      │   D3    │ ──► Prefetching...
      └─────────┘
      
Latency Hidden = Prefetch Time
"""
        return diagram
    
    def estimate_prefetch_benefit(self, computation_time, memory_latency, prefetch_overlap):
        """
        Estime le bénéfice du prefetching
        
        Args:
            computation_time: Temps de calcul par donnée
            memory_latency: Latence mémoire
            prefetch_overlap: Fraction de latence masquée (0-1)
        """
        time_without_prefetch = computation_time + memory_latency
        time_with_prefetch = computation_time + (memory_latency * (1 - prefetch_overlap))
        
        speedup = time_without_prefetch / time_with_prefetch
        
        return {
            'time_without_prefetch_ns': time_without_prefetch,
            'time_with_prefetch_ns': time_with_prefetch,
            'speedup': speedup,
            'latency_hidden_ns': memory_latency * prefetch_overlap
        }

prefetch = MemoryPrefetching()
print(prefetch.visualize_prefetching())

benefit = prefetch.estimate_prefetch_benefit(
    computation_time=100,  # ns
    memory_latency=200,    # ns
    prefetch_overlap=0.8   # 80% masqué
)

print("\nPrefetch Benefit Estimation:")
print(f"  Computation: 100 ns")
print(f"  Memory latency: 200 ns")
print(f"  Time without prefetch: {benefit['time_without_prefetch_ns']} ns")
print(f"  Time with prefetch: {benefit['time_with_prefetch_ns']:.1f} ns")
print(f"  Speedup: {benefit['speedup']:.2f}x")
```

---

## Exemple Complet: Optimisation Multi-Niveaux

```python
class ComprehensiveMemoryOptimization:
    """
    Exemple complet d'optimisation mémoire multi-niveaux
    """
    
    def optimize_layer(self, layer_type, config):
        """
        Optimise mémoire pour une couche
        
        Args:
            layer_type: 'linear', 'conv2d'
            config: Configuration
        """
        optimizations = []
        
        if layer_type == 'linear':
            # Partition weights for parallel access
            optimizations.append({
                'technique': 'Weight array partitioning (block)',
                'reason': 'Parallel weight access',
                'resource_cost': 'BRAM'
            })
            
            # Stream activations
            optimizations.append({
                'technique': 'Activation streaming',
                'reason': 'Reduce buffer size',
                'resource_cost': 'Minimal'
            })
            
            # Reuse weights
            optimizations.append({
                'technique': 'Weight stationary',
                'reason': 'Load once, reuse many times',
                'resource_cost': 'BRAM for weights'
            })
        
        elif layer_type == 'conv2d':
            # Double buffering for input
            optimizations.append({
                'technique': 'Double buffering input',
                'reason': 'Overlap I/O with computation',
                'resource_cost': '2× input buffer'
            })
            
            # Kernel caching
            optimizations.append({
                'technique': 'Kernel caching in BRAM',
                'reason': 'Reuse kernels',
                'resource_cost': 'BRAM'
            })
            
            # Output streaming
            optimizations.append({
                'technique': 'Output streaming',
                'reason': 'No full output buffer needed',
                'resource_cost': 'Minimal'
            })
        
        return optimizations

optimizer = ComprehensiveMemoryOptimization()

print("\n" + "="*60)
print("Comprehensive Memory Optimization Example")
print("="*60)

# Optimiser une couche dense
linear_opts = optimizer.optimize_layer('linear', {'in': 256, 'out': 128})
print("\nLinear Layer (256→128) Optimizations:")
for opt in linear_opts:
    print(f"  • {opt['technique']}")
    print(f"    Reason: {opt['reason']}")

# Optimiser une couche convolution
conv_opts = optimizer.optimize_layer('conv2d', {'size': (224, 224, 3)})
print("\nConv2D Layer Optimizations:")
for opt in conv_opts:
    print(f"  • {opt['technique']}")
    print(f"    Reason: {opt['reason']}")
```

---

## Exercices

### Exercice 14.4.1
Concevez un système de buffering pour une couche Conv2D avec double buffering et partitionnement de kernels.

### Exercice 14.4.2
Calculez la bande passante nécessaire et optimisez les accès DDR pour un réseau avec 10 couches.

---

## Points Clés à Retenir

> 📌 **Streaming réduit buffers intermédiaires**

> 📌 **Double buffering masque latence I/O**

> 📌 **Partitionnement permet accès parallèle**

> 📌 **Burst transfers optimisent DDR**

> 📌 **Data reuse réduit accès mémoire**

> 📌 **Prefetching masque latence**

---

*Section suivante : [14.5 Frameworks de Déploiement](./14_05_Frameworks.md)*

