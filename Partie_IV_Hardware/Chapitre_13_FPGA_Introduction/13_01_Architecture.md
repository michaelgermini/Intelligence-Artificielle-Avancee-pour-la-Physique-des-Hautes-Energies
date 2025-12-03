# 13.1 Architecture des FPGA

---

## Introduction

Cette section détaille l'**architecture interne des FPGA**, leurs composants fondamentaux, et comment ils permettent la réalisation de circuits logiques complexes et parallèles.

---

## Vue d'Ensemble de l'Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Architecture Générale FPGA                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              I/O Blocks (IOB)                            │  │
│  │  Pins d'entrée/sortie configurés                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Configurable Logic Blocks (CLB)                  │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │  │
│  │  │ CLB  │ │ CLB  │ │ CLB  │ │ CLB  │  ...              │  │
│  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                  │  │
│  └─────┼────────┼────────┼────────┼─────────────────────────┘  │
│        │        │        │        │                            │
│  ┌─────▼────────▼────────▼────────▼─────────────────────────┐  │
│  │              Routing Resources                            │  │
│  │  Interconnexions programmables (switches, wires)         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Block RAM (BRAM)                                 │  │
│  │  Mémoires distribuées pour stockage                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         DSP Slices                                        │  │
│  │  Multiplicateurs-accumulateurs hardwired                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configurable Logic Blocks (CLB)

### Structure d'un CLB

```python
class CLBArchitecture:
    """
    Architecture détaillée d'un Configurable Logic Block
    """
    
    def __init__(self):
        self.components = {
            'LUT': {
                'name': 'Look-Up Table',
                'inputs': '4 ou 6 inputs typiquement',
                'outputs': '1 output',
                'function': 'Implémente toute fonction booléenne de N variables',
                'size': '2^N entrées de mémoire'
            },
            'FF': {
                'name': 'Flip-Flop',
                'type': 'D-type register',
                'function': 'Stockage synchrone de données',
                'clock': 'Synchronisé par horloge globale'
            },
            'MUX': {
                'name': 'Multiplexer',
                'function': 'Sélection de signaux',
                'configurable': 'Vrai'
            },
            'Carry_Chain': {
                'name': 'Carry Logic',
                'function': 'Propagation rapide de retenue pour addition',
                'optimization': 'Arithmétique haute performance'
            }
        }
    
    def display_architecture(self):
        """Affiche l'architecture d'un CLB"""
        print("="*60)
        print("Configurable Logic Block (CLB) Architecture")
        print("="*60)
        
        for comp_name, comp_info in self.components.items():
            print(f"\n{comp_info['name']} ({comp_name}):")
            for key, value in comp_info.items():
                if key != 'name':
                    print(f"  {key}: {value}")

# Illustration ASCII
CLB_ASCII = """
┌─────────────────────────────────────┐
│            CLB Structure            │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │   LUT (6in)  │──│  MUX 2:1    │ │
│  │              │  │             │──┼─ Output
│  │  2^6 = 64    │  │             │ │
│  │   entries    │  └─────────────┘ │
│  └──────┬───────┘                  │
│         │                          │
│         ▼                          │
│  ┌──────────────┐                  │
│  │  Flip-Flop   │                  │
│  │    (FF)      │                  │
│  │   Clock ──►  │                  │
│  └──────────────┘                  │
│         │                          │
│         └──────────────────────────┘
│                                     │
│  Carry Chain (horizontal)          │
│                                     │
└─────────────────────────────────────┘
"""

print(CLB_ASCII)
clb = CLBArchitecture()
clb.display_architecture()
```

---

## Look-Up Tables (LUT)

### Fonctionnement d'une LUT

```python
import numpy as np

class LUT:
    """
    Implémentation pédagogique d'une Look-Up Table
    """
    
    def __init__(self, num_inputs=4):
        """
        Args:
            num_inputs: Nombre d'entrées de la LUT (4 ou 6 typiquement)
        """
        self.num_inputs = num_inputs
        self.size = 2 ** num_inputs
        # Table de vérité: chaque entrée peut être 0 ou 1
        self.truth_table = np.zeros(self.size, dtype=int)
    
    def configure(self, function):
        """
        Configure la LUT pour implémenter une fonction booléenne
        
        Args:
            function: Fonction booléenne ou table de vérité
        """
        if callable(function):
            # Si c'est une fonction, génère la table de vérité
            for i in range(self.size):
                # Convertit i en binaire pour les inputs
                inputs = [(i >> j) & 1 for j in range(self.num_inputs)]
                self.truth_table[i] = int(function(*inputs))
        else:
            # Si c'est directement une table de vérité
            self.truth_table = np.array(function)
    
    def evaluate(self, *inputs):
        """
        Évalue la LUT pour des inputs donnés
        
        Args:
            *inputs: Valeurs d'entrée (0 ou 1)
        
        Returns:
            Valeur de sortie (0 ou 1)
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")
        
        # Convertit les inputs en index
        index = 0
        for i, inp in enumerate(inputs):
            index |= (int(inp) << i)
        
        return self.truth_table[index]
    
    def implement_and(self):
        """Configure la LUT pour implémenter une porte AND"""
        def and_function(*inputs):
            return all(inputs)
        self.configure(and_function)
    
    def implement_xor(self):
        """Configure la LUT pour implémenter XOR"""
        def xor_function(*inputs):
            result = inputs[0]
            for inp in inputs[1:]:
                result ^= inp
            return result
        self.configure(xor_function)

# Exemple d'utilisation
print("\n" + "="*60)
print("Exemple: LUT 4-input implémentant AND")
print("="*60)

lut = LUT(num_inputs=4)
lut.implement_and()

print("\nTable de vérité:")
for i in range(lut.size):
    inputs = [(i >> j) & 1 for j in range(4)]
    output = lut.evaluate(*inputs)
    inputs_str = "".join(str(x) for x in inputs)
    print(f"  {inputs_str} → {output}")

# Test
print(f"\nTest: lut(1, 1, 1, 1) = {lut.evaluate(1, 1, 1, 1)}")
print(f"Test: lut(1, 0, 1, 1) = {lut.evaluate(1, 0, 1, 1)}")
```

---

## Routing Resources

### Architecture de Routage

```python
class RoutingArchitecture:
    """
    Architecture des ressources de routage dans un FPGA
    """
    
    def __init__(self):
        self.routing_types = {
            'local': {
                'length': 'Court',
                'purpose': 'Connexions entre CLB adjacents',
                'delay': 'Faible (~100ps)',
                'example': 'CLB → CLB voisin'
            },
            'intermediate': {
                'length': 'Moyen',
                'purpose': 'Connexions moyennes distances',
                'delay': 'Moyen (~500ps)',
                'example': 'CLB → CLB distant (même tile)'
            },
            'long': {
                'length': 'Long',
                'purpose': 'Connexions globales',
                'delay': 'Élevé (~2ns)',
                'example': 'CLB → CLB opposé du chip'
            },
            'clock': {
                'length': 'Global',
                'purpose': 'Distribution d\'horloge',
                'delay': 'Contrôlé (skew minimal)',
                'example': 'Clock network → tous les CLB'
            }
        }
        
        self.switch_boxes = {
            'function': 'Connexion programmable de fils',
            'configurable': True,
            'types': ['6-way', '8-way', 'complex']
        }
    
    def display_routing(self):
        """Affiche l'architecture de routage"""
        print("\n" + "="*60)
        print("Routing Resources")
        print("="*60)
        
        for rtype, info in self.routing_types.items():
            print(f"\n{rtype.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

# Diagramme ASCII du routage
ROUTING_ASCII = """
┌──────────────────────────────────────────────────────┐
│              Routing Architecture                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────┐  Local    ┌──────┐                       │
│  │ CLB1 │ ←──────→  │ CLB2 │                       │
│  └──┬───┘           └──┬───┘                       │
│     │                  │                            │
│     │ Intermediate     │                            │
│     │                  │                            │
│     │                  │                            │
│  ┌──▼───┐          ┌──▼───┐                        │
│  │ CLB3 │          │ CLB4 │                        │
│  └──────┘          └──────┘                        │
│     │                  │                            │
│     └────── Long ──────┘                            │
│                                                      │
│  Switch Box:                                         │
│  ┌─────────────────────────────────────────┐        │
│  │  Wire1 ──┐                              │        │
│  │  Wire2 ──┼──→ Configurable Switch ──►   │        │
│  │  Wire3 ──┘                              │        │
│  └─────────────────────────────────────────┘        │
│                                                      │
└──────────────────────────────────────────────────────┘
"""

print(ROUTING_ASCII)
routing = RoutingArchitecture()
routing.display_routing()
```

---

## Block RAM (BRAM)

### Architecture BRAM

```python
class BlockRAM:
    """
    Block RAM dans un FPGA
    """
    
    def __init__(self, depth=1024, width=36):
        """
        Args:
            depth: Profondeur de la mémoire (nombre d'emplacements)
            width: Largeur en bits (18 ou 36 typiquement)
        """
        self.depth = depth
        self.width = width
        self.size_kb = (depth * width) / (8 * 1024)
        self.memory = np.zeros((depth, width), dtype=int)
    
    def write(self, address, data):
        """
        Écrit des données à une adresse
        
        Args:
            address: Adresse (0 à depth-1)
            data: Données à écrire (longueur width bits)
        """
        if address >= self.depth:
            raise ValueError(f"Address {address} >= depth {self.depth}")
        
        self.memory[address] = data
    
    def read(self, address):
        """
        Lit des données à une adresse
        
        Args:
            address: Adresse à lire
        
        Returns:
            Données lues
        """
        if address >= self.depth:
            raise ValueError(f"Address {address} >= depth {self.depth}")
        
        return self.memory[address].copy()
    
    def get_capacity(self):
        """Retourne la capacité en bits et KB"""
        total_bits = self.depth * self.width
        total_kb = total_bits / (8 * 1024)
        return {
            'total_bits': total_bits,
            'total_kb': total_kb,
            'depth': self.depth,
            'width': self.width
        }

# Exemple BRAM
print("\n" + "="*60)
print("Block RAM Example")
print("="*60)

bram = BlockRAM(depth=2048, width=18)
print(f"\nBRAM Configuration:")
print(f"  Depth: {bram.depth}")
print(f"  Width: {bram.width} bits")
print(f"  Total: {bram.size_kb:.2f} KB")

# Opérations
bram.write(0, [1]*18)
bram.write(1, [0, 1]*9)
print(f"\nRead address 0: {bram.read(0)}")
print(f"Read address 1: {bram.read(1)}")
```

---

## DSP Slices

### Architecture DSP

```python
class DSPSlice:
    """
    DSP Slice pour opérations arithmétiques
    """
    
    def __init__(self):
        self.capabilities = {
            'multiply': 'Multiplicateur 18x18 ou 25x18',
            'multiply_accumulate': 'MAC operations',
            'pipeline': 'Registres pipeline pour haute fréquence',
            'precision': 'Précision configurable'
        }
        
        self.registers = {
            'A': 'Input register A',
            'B': 'Input register B',
            'C': 'Input register C (accumulator)',
            'P': 'Output register P'
        }
    
    def multiply(self, a, b):
        """Opération de multiplication"""
        return a * b
    
    def mac(self, a, b, c):
        """
        Multiply-Accumulate
        
        Returns: a * b + c
        """
        return a * b + c

# Diagramme DSP
DSP_ASCII = """
┌─────────────────────────────────────────┐
│           DSP Slice                     │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────┐         ┌──────┐            │
│  │  A   │ ──► ┌───┤      │            │
│  └──────┘     │   │  ×   │            │
│               │   │      │            │
│  ┌──────┐ ────┘   └───┬──┘            │
│  │  B   │             │                │
│  └──────┘             ▼                │
│                  ┌─────────┐           │
│  ┌──────┐        │    +    │           │
│  │  C   │ ──────►│ (accum) │           │
│  └──────┘        └────┬────┘           │
│                       │                │
│                       ▼                │
│                  ┌─────────┐           │
│                  │    P    │           │
│                  └─────────┘           │
│                                         │
└─────────────────────────────────────────┘
"""

print(DSP_ASCII)

dsp = DSPSlice()
print("\nDSP Capabilities:")
for capability, desc in dsp.capabilities.items():
    print(f"  {capability}: {desc}")

# Exemple d'utilisation
result_mult = dsp.multiply(25, 18)
result_mac = dsp.mac(10, 20, 5)

print(f"\nExamples:")
print(f"  Multiply(25, 18) = {result_mult}")
print(f"  MAC(10, 20, 5) = {result_mac}")
```

---

## Architecture Globale: Exemple Xilinx/Zynq

```python
class XilinxFPGAArchitecture:
    """
    Architecture typique d'un FPGA Xilinx (ex: Zynq-7000, UltraScale+)
    """
    
    def __init__(self, model='Zynq-7000'):
        self.model = model
        self.resources = {
            'CLB': {
                'name': 'Configurable Logic Blocks',
                'count': '53,200 (varie selon modèle)',
                'luts': '6-input LUTs',
                'ffs': '2 FFs par LUT'
            },
            'BRAM': {
                'name': 'Block RAM',
                'count': '560 blocks',
                'size_per_block': '36 KB',
                'total': '~20 MB'
            },
            'DSP': {
                'name': 'DSP Slices',
                'count': '1,200 slices',
                'capability': '48-bit MAC operations'
            },
            'IO': {
                'name': 'I/O Pins',
                'count': '500+ pins',
                'standards': 'LVDS, LVCMOS, etc.'
            }
        }
    
    def display_resources(self):
        """Affiche les ressources du FPGA"""
        print(f"\n{'='*60}")
        print(f"{self.model} FPGA Resources")
        print(f"{'='*60}")
        
        for resource, info in self.resources.items():
            print(f"\n{info['name']} ({resource}):")
            for key, value in info.items():
                if key != 'name':
                    print(f"  {key}: {value}")

fpga = XilinxFPGAArchitecture()
fpga.display_resources()
```

---

## Exercices

### Exercice 13.1.1
Implémentez une LUT 6-input et configurez-la pour implémenter une fonction personnalisée (ex: addition binaire 3+3 bits).

### Exercice 13.1.2
Calculez la capacité totale en bits d'un FPGA ayant 1000 CLB, chacun contenant 8 LUTs 6-input.

---

## Points Clés à Retenir

> 📌 **CLB = LUT + Flip-Flops + Routage local**

> 📌 **LUT permet d'implémenter toute fonction booléenne**

> 📌 **Routage programmable connecte les CLB entre eux**

> 📌 **BRAM fournit mémoire distribuée sur le chip**

> 📌 **DSP Slices optimisés pour arithmétique**

---

*Section suivante : [13.2 Flux de Conception FPGA](./13_02_Flux.md)*

