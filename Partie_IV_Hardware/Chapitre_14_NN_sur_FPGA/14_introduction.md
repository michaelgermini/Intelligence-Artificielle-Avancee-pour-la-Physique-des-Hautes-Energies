# Chapitre 14 : Déploiement de Réseaux de Neurones sur FPGA

---

## Introduction

Le déploiement de réseaux de neurones sur FPGA présente des défis uniques liés aux contraintes mémoire, de latence et d'énergie. Ce chapitre couvre les stratégies et techniques pour optimiser ce déploiement.

---

## Plan du Chapitre

1. [Défis Spécifiques aux FPGA](./14_01_Defis.md)
2. [Architectures de Dataflow](./14_02_Dataflow.md)
3. [Parallélisme Spatial vs Temporel](./14_03_Parallelisme.md)
4. [Optimisation des Accès Mémoire](./14_04_Memoire.md)
5. [Frameworks de Déploiement](./14_05_Frameworks.md)

---

## Défis Principaux

```
┌─────────────────────────────────────────────────────────────────┐
│              Défis du Déploiement ML sur FPGA                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Contraintes Mémoire                                        │
│     • BRAM limité (quelques MB)                                │
│     • Nécessite compression/quantification                     │
│                                                                 │
│  2. Latence et Throughput                                      │
│     • Pipeline nécessaire pour haute fréquence                 │
│     • Initiation Interval = 1 pour throughput max              │
│                                                                 │
│  3. Consommation Énergétique                                   │
│     • Densité de calcul vs puissance                           │
│     • Optimisation des chemins critiques                       │
│                                                                 │
│  4. Ressources Limitées                                        │
│     • LUT, DSP, BRAM doivent être utilisés efficacement        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stratégies d'Optimisation

### Parallélisation

```python
class FPGAOptimization:
    """
    Techniques d'optimisation pour FPGA
    """
    
    @staticmethod
    def compute_resource_usage(model, input_shape, reuse_factor=1):
        """
        Estime l'utilisation des ressources FPGA
        """
        total_mults = 0
        total_adds = 0
        
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                # Multiplications: in_features × out_features
                mults = layer.in_features * layer.out_features
                adds = layer.out_features  # Additions pour biais
                total_mults += mults
                total_adds += adds
            
            elif isinstance(layer, nn.Conv2d):
                # Plus complexe: dépend de la taille de l'image
                mults = (layer.out_channels * layer.in_channels * 
                        layer.kernel_size[0] * layer.kernel_size[1])
                total_mults += mults
        
        # Ressources nécessaires (avec reuse)
        dsp_needed = (total_mults + total_adds) // reuse_factor
        
        # Estimation BRAM pour les poids
        weight_bits = sum(p.numel() for p in model.parameters()) * 8  # int8
        bram_18k_needed = weight_bits / (18 * 1024)  # 18k bits par BRAM
        
        return {
            'dsp_estimate': dsp_needed,
            'bram_18k_estimate': bram_18k_needed,
            'total_multiplications': total_mults
        }

# Exemple
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

resources = FPGAOptimization.compute_resource_usage(model, (1, 256))

print("Estimation des ressources FPGA:")
print(f"  DSP slices: {resources['dsp_estimate']:.0f}")
print(f"  BRAM 18K: {resources['bram_18k_estimate']:.1f}")
```

---

## Points Clés à Retenir

> 📌 **Le pipelining est essentiel pour atteindre un throughput élevé**

> 📌 **Le reuse factor contrôle le compromis ressources/latence**

> 📌 **Les accès mémoire doivent être optimisés (streaming, burst)**

> 📌 **La quantification est souvent nécessaire pour tenir dans les ressources**

---

*Section suivante : [14.1 Défis Spécifiques aux FPGA](./14_01_Defis.md)*

