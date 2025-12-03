# Chapitre 13 : Introduction aux FPGA

---

## Introduction

Les **FPGA** (Field-Programmable Gate Arrays) sont des circuits logiques programmables qui offrent un excellent compromis entre flexibilité et performance. Ils sont devenus essentiels pour le déploiement de modèles ML dans les systèmes temps réel comme les triggers du LHC.

---

## Plan du Chapitre

1. [Architecture des FPGA](./13_01_Architecture.md)
2. [Flux de Conception FPGA](./13_02_Flux.md)
3. [Langages HDL (Verilog, VHDL)](./13_03_HDL.md)
4. [High-Level Synthesis (HLS)](./13_04_HLS.md)
5. [Outils de Développement](./13_05_Outils.md)

---

## Qu'est-ce qu'un FPGA ?

```
┌─────────────────────────────────────────────────────────────────┐
│                    FPGA vs CPU vs ASIC                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPU:                                                           │
│  ✓ Flexibilité maximale                                        │
│  ✗ Performance limitée par architecture générique              │
│                                                                 │
│  ASIC:                                                          │
│  ✓ Performance maximale                                        │
│  ✗ Pas flexible, coûteux à développer                          │
│                                                                 │
│  FPGA:                                                          │
│  ✓ Bon compromis flexibilité/performance                       │
│  ✓ Reprogrammable                                              │
│  ✓ Parallélisme massif                                         │
│  ✗ Plus lent que ASIC, moins flexible que CPU                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Composants Principaux

### Logic Blocks (CLB)

```python
class CLBDescription:
    """
    Description pédagogique d'un Configurable Logic Block
    """
    def __init__(self):
        self.components = {
            'LUT': 'Look-Up Table - implémente toute fonction booléenne',
            'Flip-Flops': 'Éléments de mémoire pour registres',
            'Multiplexers': 'Sélection de signaux',
            'Carry Logic': 'Arithmétique rapide'
        }
        
        print("Composants d'un CLB:")
        for comp, desc in self.components.items():
            print(f"  {comp}: {desc}")

CLBDescription()
```

---

## Applications au CERN

### Trigger L1 avec FPGA

```python
class FPGAInTrigger:
    """
    Utilisation des FPGA dans le système de trigger
    """
    
    requirements = {
        'latency': '4 μs maximum',
        'throughput': '40 MHz (un événement toutes les 25 ns)',
        'power': 'Limité (refroidissement)',
        'reliability': 'Très haute (pas d'erreurs tolérées)'
    }
    
    advantages = [
        'Parallélisme massif pour traitement simultané',
        'Latence déterministe et faible',
        'Reconfigurabilité pour mises à jour algorithmes'
    ]
    
    print("Requirements FPGA pour Trigger L1:")
    for key, value in requirements.items():
        print(f"  {key}: {value}")
    
    print("\nAvantages:")
    for adv in advantages:
        print(f"  • {adv}")

FPGAInTrigger()
```

---

## Points Clés à Retenir

> 📌 **Les FPGA offrent un parallélisme massif grâce à leur architecture configurable**

> 📌 **La latence déterministe est cruciale pour les applications temps réel**

> 📌 **HLS simplifie le développement par rapport aux HDL traditionnels**

> 📌 **Les FPGA sont essentiels pour le déploiement ML dans les triggers**

---

*Section suivante : [13.1 Architecture des FPGA](./13_01_Architecture.md)*

