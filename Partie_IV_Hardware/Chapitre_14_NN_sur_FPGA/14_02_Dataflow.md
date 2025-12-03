# 14.2 Architectures de Dataflow

---

## Introduction

Les **architectures de dataflow** organisent le calcul pour maximiser le pipeline et le parallélisme. Cette section présente les différents styles de dataflow adaptés aux réseaux de neurones sur FPGA.

---

## Styles de Dataflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataflow Architectures                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Layer-by-Layer                                              │
│     └─ Traite une couche complète avant la suivante            │
│                                                                 │
│  2. Row-Stationary                                               │
│     └─ Traite par lignes d'activations                          │
│                                                                 │
│  3. Output-Stationary                                            │
│     └─ Accumule résultats au même endroit                       │
│                                                                 │
│  4. Weight-Stationary                                            │
│     └─ Poids restent en mémoire, données bougent                │
│                                                                 │
│  5. Systolic Array                                              │
│     └─ Array de processeurs interconnectés                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Dataflow

### Principe

```python
class LayerByLayerDataflow:
    """
    Dataflow layer-by-layer: traite couche par couche
    """
    
    def __init__(self):
        self.description = """
        Architecture simple où chaque couche est complètement
        traitée avant de passer à la suivante.
        
        Avantages:
        - Simple à implémenter
        - Facile à déboguer
        - Réutilise efficacement les ressources
        
        Inconvénients:
        - Latence élevée
        - Buffer intermédiaires nécessaires
        """
    
    def visualize(self, num_layers=3):
        """Visualise le dataflow"""
        diagram = f"""
Layer-by-Layer Dataflow (3 layers):

Input  ──► [Layer 1] ──► Buffer ──► [Layer 2] ──► Buffer ──► [Layer 3] ──► Output
          └─Process─┘   └─Store─┘   └─Process─┘   └─Store─┘   └─Process─┘
          
Timeline:
Time ───────────────────────────────────────────────────────────────►
      ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
      │   Process L1    │ │   Process L2    │ │   Process L3    │
      └─────────────────┘ └─────────────────┘ └─────────────────┘
      
      Latency = Sum(all layer latencies)
      Throughput = Limited by slowest layer
"""
        return diagram
    
    def estimate_resources(self, layers_config):
        """
        Estime les ressources nécessaires
        
        Args:
            layers_config: List de configs de couches
        """
        # Ressources = max de toutes les couches (réutilisation)
        max_dsp = 0
        max_bram = 0
        
        for layer in layers_config:
            layer_dsp = layer.get('dsp_estimate', 0)
            layer_bram = layer.get('bram_estimate', 0)
            max_dsp = max(max_dsp, layer_dsp)
            max_bram = max(max_bram, layer_bram)
        
        # Buffers intermédiaires
        buffer_bram = sum(layer.get('activation_bram', 10) for layer in layers_config)
        
        return {
            'compute_dsp': max_dsp,
            'compute_bram': max_bram,
            'buffer_bram': buffer_bram,
            'total_bram': max_bram + buffer_bram,
            'resource_reuse': True  # Réutilisation entre couches
        }

layer_by_layer = LayerByLayerDataflow()
print(layer_by_layer.visualize())

# Exemple d'estimation
layers = [
    {'dsp_estimate': 50, 'bram_estimate': 5, 'activation_bram': 10},
    {'dsp_estimate': 100, 'bram_estimate': 8, 'activation_bram': 8},
    {'dsp_estimate': 30, 'bram_estimate': 3, 'activation_bram': 5}
]

resources = layer_by_layer.estimate_resources(layers)
print("\nResource Estimation:")
print(f"  Compute DSP: {resources['compute_dsp']}")
print(f"  Total BRAM: {resources['total_bram']}")
```

---

## Row-Stationary Dataflow

### Principe

```python
class RowStationaryDataflow:
    """
    Row-Stationary: traite par lignes
    """
    
    def __init__(self):
        self.description = """
        Traite les activations ligne par ligne.
        Poids peuvent être streamed ou stationary.
        
        Avantages:
        - Bon pour convolutions
        - Réduit buffer intermédiaires
        - Permet overlap processing
        
        Inconvénients:
        - Plus complexe à implémenter
        - Gestion mémoire plus sophistiquée
        """
    
    def visualize_conv2d(self):
        """Visualise pour convolution 2D"""
        diagram = """
Row-Stationary for Conv2D:

Input Image (3x3 example):
┌─────┬─────┬─────┐
│ I00 │ I01 │ I02 │
├─────┼─────┼─────┤
│ I10 │ I11 │ I12 │  ──► Process row by row
├─────┼─────┼─────┤
│ I20 │ I21 │ I22 │
└─────┴─────┴─────┘

Kernel (2x2):
┌─────┬─────┐
│ K00 │ K01 │
├─────┼─────┤
│ K10 │ K11 │
└─────┴─────┘

Processing:
Row 0: Process I00,I01 with K → O00
       Process I01,I02 with K → O01
Row 1: Process I10,I11 with K → O10
       Process I11,I12 with K → O11
       
Pipeline:
Time ───────────────────────────────────────►
      ┌─────────┐ ┌─────────┐ ┌─────────┐
      │ Row 0   │ │ Row 1   │ │ Row 2   │
      └─────────┘ └─────────┘ └─────────┘
      ┌─────────┐ ┌─────────┐
      │ Output  │ │ Output  │
      └─────────┘ └─────────┘
      
Overlap possible for better throughput
"""
        return diagram
    
    def estimate_memory_access(self, image_height, kernel_size, channels):
        """
        Estime les accès mémoire
        
        Args:
            image_height: Hauteur de l'image
            kernel_size: Taille du noyau
            channels: Nombre de canaux
        """
        # Lignes nécessaires en buffer (kernel_size - 1)
        buffer_rows = kernel_size - 1
        
        # Mémoire pour buffer
        buffer_size = buffer_rows * image_height * channels
        
        return {
            'buffer_rows': buffer_rows,
            'buffer_size_elements': buffer_size,
            'memory_efficiency': 'Better than full image buffer'
        }

row_stationary = RowStationaryDataflow()
print(row_stationary.visualize_conv2d())

memory = row_stationary.estimate_memory_access(224, 3, 3)
print("\nMemory Access Estimation:")
print(f"  Buffer rows: {memory['buffer_rows']}")
print(f"  Buffer size: {memory['buffer_size_elements']:,} elements")
```

---

## Output-Stationary Dataflow

### Principe

```python
class OutputStationaryDataflow:
    """
    Output-Stationary: accumule résultats au même endroit
    """
    
    def __init__(self):
        self.description = """
        Les résultats d'output restent stationnaires pendant
        que les inputs et poids sont streamed.
        
        Avantages:
        - Bon pour réduction accumulation
        - Minimise écritures mémoire
        - Efficace pour dot products
        
        Inconvénients:
        - Plus complexe pour certaines opérations
        """
    
    def visualize_dot_product(self, vector_size=4):
        """Visualise pour produit scalaire"""
        diagram = f"""
Output-Stationary for Dot Product:

Input A:  [A0] ──┐
        [A1] ──┼──► [Accumulator] ──► Output (stationary)
        [A2] ──┤
        [A3] ──┘
                ▲
                │
Input B:  [B0] ─┼── Multiply
        [B1] ──┤
        [B2] ──┤
        [B3] ──┘
        
Processing:
Time ──────────────────────────────────────────►
      ┌────┐ ┌────┐ ┌────┐ ┌────┐
      │A0*B0│ │A1*B1│ │A2*B2│ │A3*B3│  (multiplies stream)
      └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
         └───────┴───────┴───────┘
                    │
                    ▼
              ┌──────────┐
              │ Accumulate│  (accumulator stays)
              └──────┬───┘
                     ▼
                 Output
"""
        return diagram
    
    def estimate_accumulator_size(self, input_precision=8, output_precision=16, vector_size=256):
        """
        Estime la taille de l'accumulateur nécessaire
        
        Args:
            input_precision: Bits d'input (8 pour int8)
            output_precision: Bits d'accumulation (16 pour int16)
            vector_size: Taille du vecteur
        """
        # Bits nécessaires pour éviter overflow
        # Worst case: tous les éléments sont max
        max_value = (2**(input_precision-1) - 1)  # Max int8 = 127
        max_accum = max_value * max_value * vector_size
        
        # Bits nécessaires
        bits_needed = max_accum.bit_length() + 1  # +1 pour signe
        
        return {
            'accumulator_bits': bits_needed,
            'accumulator_bytes': (bits_needed + 7) // 8,
            'overflow_risk': bits_needed > output_precision,
            'recommended_bits': max(bits_needed, output_precision)
        }

output_stationary = OutputStationaryDataflow()
print(output_stationary.visualize_dot_product())

accum = output_stationary.estimate_accumulator_size()
print("\nAccumulator Size Estimation:")
print(f"  Bits needed: {accum['accumulator_bits']}")
print(f"  Overflow risk: {accum['overflow_risk']}")
print(f"  Recommended: {accum['recommended_bits']} bits")
```

---

## Weight-Stationary Dataflow

### Principe

```python
class WeightStationaryDataflow:
    """
    Weight-Stationary: poids restent, données streament
    """
    
    def __init__(self):
        self.description = """
        Les poids sont chargés une fois et restent en mémoire.
        Les données d'activation streament à travers.
        
        Avantages:
        - Réduit accès poids (1x lecture)
        - Bon quand poids < activations
        - Efficace pour inference
        
        Inconvénients:
        - Nécessite BRAM pour poids
        - Moins adapté si poids changent souvent
        """
    
    def visualize_matrix_multiply(self):
        """Visualise pour multiplication matrice"""
        diagram = """
Weight-Stationary for Matrix Multiply (Y = W × X):

Weights W (stationary in BRAM):
┌─────────┐
│ W00 W01 │  ──► Load once, keep in memory
│ W10 W11 │
│ W20 W21 │
└─────────┘

Input X (streaming):
[X0] ──┐
[X1] ──┼──► Stream through compute units
       │
       ▼
   ┌───────┐
   │  MAC  │ ──► Multiply-Accumulate
   └───┬───┘
       │
       ▼
    ┌─────┐
    │ Y0  │ ──► Output Y (streams out)
    │ Y1  │
    │ Y2  │
    └─────┘

Timeline:
Time ──────────────────────────────────────────────►
      ┌─────────────┐
      │ Load W      │  (once)
      └─────────────┘
      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
      │ X0  │ │ X1  │ │ X2  │ │ X3  │  (stream)
      └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
         └───────┴───────┴───────┘
                   │
                   ▼
            Process with W
            (W stays in place)
"""
        return diagram
    
    def estimate_weight_memory(self, weight_size, precision_bits=8):
        """
        Estime la mémoire pour les poids
        
        Args:
            weight_size: Nombre de poids
            precision_bits: Précision (8 pour int8)
        """
        weight_bits = weight_size * precision_bits
        weight_mb = weight_bits / (8 * 1024 * 1024)
        
        # BRAM nécessaire (18K bits par BRAM)
        bram_18k_needed = weight_bits / (18 * 1024)
        
        return {
            'weight_bits': weight_bits,
            'weight_mb': weight_mb,
            'bram_18k_needed': bram_18k_needed,
            'fits_in_bram': bram_18k_needed <= 560  # Zynq-7000 example
        }

weight_stationary = WeightStationaryDataflow()
print(weight_stationary.visualize_matrix_multiply())

# Exemple: couche dense 256→128
memory = weight_stationary.estimate_weight_memory(256 * 128, precision_bits=8)
print("\nWeight Memory Estimation (256→128 layer, int8):")
print(f"  Weight size: {256 * 128:,} parameters")
print(f"  Memory: {memory['weight_mb']:.3f} MB")
print(f"  BRAM 18K: {memory['bram_18k_needed']:.1f}")
print(f"  Fits in BRAM: {memory['fits_in_bram']}")
```

---

## Systolic Array

### Principe

```python
class SystolicArray:
    """
    Systolic Array: array de processeurs interconnectés
    """
    
    def __init__(self):
        self.description = """
        Array régulier de processeurs simples interconnectés.
        Données et poids "pulsent" à travers l'array.
        
        Avantages:
        - Parallélisme massif
        - Throughput très élevé
        - Échelle bien
        
        Inconvénients:
        - Complexe à concevoir
        - Utilisation ressources élevée
        """
    
    def visualize_array(self, rows=4, cols=4):
        """Visualise un systolic array"""
        diagram = f"""
Systolic Array ({rows}x{cols}):

Weights flow down:           Data flows right:
        
  W0  W1  W2  W3              D0  D1  D2  D3
   │   │   │   │               │   │   │   │
   ▼   ▼   ▼   ▼               ▼   ▼   ▼   ▼
┌────┬────┬────┬────┐      ┌────┬────┬────┬────┐
│ PE │ PE │ PE │ PE │      │ PE │ PE │ PE │ PE │  Row 0
├────┼────┼────┼────┤      ├────┼────┼────┼────┤
│ PE │ PE │ PE │ PE │      │ PE │ PE │ PE │ PE │  Row 1
├────┼────┼────┼────┤      ├────┼────┼────┼────┤
│ PE │ PE │ PE │ PE │      │ PE │ PE │ PE │ PE │  Row 2
├────┼────┼────┼────┤      ├────┼────┼────┼────┤
│ PE │ PE │ PE │ PE │      │ PE │ PE │ PE │ PE │  Row 3
└────┴────┴────┴────┘      └────┴────┴────┴────┘
   │   │   │   │               │   │   │   │
   ▼   ▼   ▼   ▼               ▼   ▼   ▼   ▼
  O0  O1  O2  O3              ... ... ... ...

PE = Processing Element (MAC unit)

Timeline (pipelined):
Cycle 0: Load W0, D0 into PE[0,0]
Cycle 1: W0→PE[1,0], D0→PE[0,1], Compute PE[0,0]
Cycle 2: Cascade continues...
"""
        return diagram
    
    def estimate_resources(self, array_size, mac_per_pe=1):
        """
        Estime les ressources pour un systolic array
        
        Args:
            array_size: (rows, cols)
            mac_per_pe: MAC units par PE
        """
        rows, cols = array_size
        total_pe = rows * cols
        total_mac = total_pe * mac_per_pe
        
        # Chaque MAC ≈ 1 DSP
        dsp_needed = total_mac
        
        # Logique de contrôle
        lut_per_pe = 500  # Approximation
        luts_needed = total_pe * lut_per_pe
        
        return {
            'total_pe': total_pe,
            'total_mac': total_mac,
            'dsp_needed': dsp_needed,
            'lut_needed': luts_needed,
            'throughput_ops_per_cycle': total_mac  # En théorie
        }

systolic = SystolicArray()
print(systolic.visualize_array(4, 4))

resources = systolic.estimate_resources((8, 8), mac_per_pe=1)
print("\nSystolic Array Resource Estimation (8x8):")
print(f"  Total PEs: {resources['total_pe']}")
print(f"  Total MACs: {resources['total_mac']}")
print(f"  DSP needed: {resources['dsp_needed']}")
print(f"  Throughput: {resources['throughput_ops_per_cycle']} ops/cycle")
```

---

## Comparaison des Architectures

```python
class DataflowComparison:
    """
    Comparaison des architectures dataflow
    """
    
    comparison = {
        'layer_by_layer': {
            'latency': 'High',
            'throughput': 'Medium',
            'resources': 'Low (reuse)',
            'complexity': 'Low',
            'best_for': 'Simple networks, prototyping'
        },
        'row_stationary': {
            'latency': 'Medium',
            'throughput': 'High',
            'resources': 'Medium',
            'complexity': 'Medium',
            'best_for': 'Convolutions'
        },
        'output_stationary': {
            'latency': 'Low',
            'throughput': 'High',
            'resources': 'Medium',
            'complexity': 'Medium',
            'best_for': 'Dense layers, dot products'
        },
        'weight_stationary': {
            'latency': 'Medium',
            'throughput': 'High',
            'resources': 'High (BRAM for weights)',
            'complexity': 'Medium',
            'best_for': 'Inference, fixed weights'
        },
        'systolic_array': {
            'latency': 'Low',
            'throughput': 'Very High',
            'resources': 'Very High',
            'complexity': 'High',
            'best_for': 'Large-scale, high-throughput'
        }
    }
    
    @staticmethod
    def display_comparison():
        """Affiche la comparaison"""
        print("\n" + "="*60)
        print("Dataflow Architecture Comparison")
        print("="*60)
        
        for arch, metrics in DataflowComparison.comparison.items():
            print(f"\n{arch.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

DataflowComparison.display_comparison()
```

---

## Exercices

### Exercice 14.2.1
Concevez une architecture dataflow pour un réseau avec 3 couches: Conv2D → Dense → Dense. Choisissez le style approprié pour chaque couche.

### Exercice 14.2.2
Calculez les ressources nécessaires pour un systolic array 16x16 sur un FPGA Zynq-7000.

---

## Points Clés à Retenir

> 📌 **Layer-by-layer: simple mais latence élevée**

> 📌 **Row-stationary: bon pour convolutions**

> 📌 **Output-stationary: efficace pour accumulation**

> 📌 **Weight-stationary: réduit accès poids**

> 📌 **Systolic array: parallélisme maximal mais ressources élevées**

> 📌 **Choix dépend de contraintes: latence, throughput, ressources**

---

*Section suivante : [14.3 Parallélisme Spatial vs Temporel](./14_03_Parallelisme.md)*

