# 15.1 Introduction à hls4ml

---

## Introduction

**hls4ml** (High-Level Synthesis for Machine Learning) est une bibliothèque open-source qui traduit des modèles de machine learning entraînés dans des frameworks comme Keras, TensorFlow, PyTorch (via ONNX) en code C++ optimisé pour High-Level Synthesis (HLS), permettant ainsi leur déploiement sur FPGA.

---

## Historique et Contexte

### Origines

```python
class HLS4MLHistory:
    """
    Historique et contexte de développement de hls4ml
    """
    
    def __init__(self):
        self.timeline = {
            '2018': {
                'event': 'Publication initiale',
                'context': 'Développé au CERN et Fermilab',
                'motivation': 'Besoin de ML dans les triggers LHC',
                'paper': 'Duarte et al., JINST 13 (2018)'
            },
            '2019': {
                'event': 'Support CNN amélioré',
                'context': 'Extension pour convolutions 2D',
                'adoption': 'Adoption croissante dans la communauté HEP'
            },
            '2020-2021': {
                'event': 'Support PyTorch, ONNX',
                'context': 'Interopérabilité accrue',
                'features': 'Quantization-aware training, pruning'
            },
            '2022+': {
                'event': 'Production deployment',
                'context': 'Déploiement dans CMS, ATLAS',
                'future': 'Support GPU, optimizations avancées'
            }
        }
    
    def display_history(self):
        """Affiche l'historique"""
        print("\n" + "="*60)
        print("hls4ml History and Context")
        print("="*60)
        
        for year, info in self.timeline.items():
            print(f"\n{year}:")
            print(f"  Event: {info['event']}")
            print(f"  Context: {info['context']}")
            if 'motivation' in info:
                print(f"  Motivation: {info['motivation']}")
            if 'paper' in info:
                print(f"  Paper: {info['paper']}")

history = HLS4MLHistory()
history.display_history()
```

---

## Pourquoi hls4ml ?

### Problématique Initiale

```
┌─────────────────────────────────────────────────────────────────┐
│              Problématique: ML dans les Triggers                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Contraintes LHC:                                               │
│  • Latence < 4 μs (Level-1 Trigger)                            │
│  • Throughput: 40 MHz (1 événement / 25 ns)                     │
│  • Ressources limitées                                          │
│  • Consommation énergétique contrôlée                           │
│                                                                 │
│  Solutions existantes insuffisantes:                            │
│  ✗ CPU: Trop lent                                               │
│  ✗ GPU: Latence trop élevée                                     │
│  ✗ ASIC: Pas flexible, coûteux                                  │
│                                                                 │
│  ✓ FPGA: Compromis idéal                                        │
│    - Latence déterministe et faible                             │
│    - Reprogrammable                                             │
│    - Parallélisme massif                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Avantages de hls4ml

```python
class HLS4MLAdvantages:
    """
    Avantages de hls4ml pour le déploiement ML sur FPGA
    """
    
    advantages = {
        'automatic_translation': {
            'description': 'Conversion automatique Keras/TF → HLS C++',
            'benefit': 'Pas besoin d\'écrire du HDL manuellement',
            'impact': 'Réduit temps de développement de 10×'
        },
        'framework_agnostic': {
            'description': 'Support multiple frameworks',
            'frameworks': ['Keras', 'TensorFlow', 'PyTorch (via ONNX)', 'ONNX'],
            'benefit': 'Flexibilité dans le choix du framework d\'entraînement'
        },
        'optimization_aware': {
            'description': 'Intégration avec compression/quantization',
            'features': ['Quantization', 'Pruning', 'Knowledge distillation'],
            'benefit': 'Modèles optimisés avant déploiement'
        },
        'configurable': {
            'description': 'Contrôle fin sur ressources/latence',
            'parameters': ['ReuseFactor', 'Precision', 'Strategy'],
            'benefit': 'Trade-offs adaptés aux contraintes'
        },
        'validated': {
            'description': 'Validé dans expériences HEP',
            'applications': ['CMS trigger', 'ATLAS trigger', 'Production use'],
            'benefit': 'Outils éprouvés en conditions réelles'
        }
    }
    
    @staticmethod
    def display_advantages():
        """Affiche les avantages"""
        print("\n" + "="*60)
        print("hls4ml Advantages")
        print("="*60)
        
        for adv, info in HLS4MLAdvantages.advantages.items():
            print(f"\n{adv.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            if isinstance(info.get('frameworks'), list):
                print(f"  Frameworks: {', '.join(info['frameworks'])}")
            if isinstance(info.get('features'), list):
                print(f"  Features: {', '.join(info['features'])}")
            if isinstance(info.get('parameters'), list):
                print(f"  Parameters: {', '.join(info['parameters'])}")
            if isinstance(info.get('applications'), list):
                print(f"  Applications: {', '.join(info['applications'])}")
            print(f"  Benefit: {info['benefit']}")

HLS4MLAdvantages.display_advantages()
```

---

## Architecture de hls4ml

### Pipeline Complet

```python
class HLS4MLPipeline:
    """
    Pipeline complet de hls4ml
    """
    
    def visualize_pipeline(self):
        """Visualise le pipeline"""
        diagram = """
┌─────────────────────────────────────────────────────────────────┐
│                    hls4ml Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Model Input                                                 │
│     ┌──────────────┐                                            │
│     │  Keras/TF/   │                                            │
│     │  PyTorch/    │                                            │
│     │  ONNX Model  │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  2. Model Parsing                                               │
│     ┌──────────────┐                                            │
│     │  Parse Graph │                                            │
│     │  Extract     │                                            │
│     │  Weights     │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  3. Configuration                                               │
│     ┌──────────────┐                                            │
│     │  Precision   │                                            │
│     │  ReuseFactor │                                            │
│     │  Strategy    │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  4. HLS Code Generation                                         │
│     ┌──────────────┐                                            │
│     │  Generate    │                                            │
│     │  C++ Code    │                                            │
│     │  (HLS)       │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  5. C Simulation                                                │
│     ┌──────────────┐                                            │
│     │  Validate    │                                            │
│     │  Algorithm   │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  6. HLS Synthesis (Vivado HLS)                                  │
│     ┌──────────────┐                                            │
│     │  Generate    │                                            │
│     │  RTL         │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  7. C/RTL Co-simulation                                         │
│     ┌──────────────┐                                            │
│     │  Verify RTL  │                                            │
│     │  Correctness │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  8. Vivado Implementation                                       │
│     ┌──────────────┐                                            │
│     │  Place &     │                                            │
│     │  Route       │                                            │
│     └──────┬───────┘                                            │
│            │                                                    │
│            ▼                                                    │
│  9. Bitstream                                                   │
│     ┌──────────────┐                                            │
│     │  FPGA        │                                            │
│     │  Bitstream   │                                            │
│     └──────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
        return diagram
    
    def explain_stages(self):
        """Explique les étapes"""
        stages = {
            '1': {
                'name': 'Model Input',
                'description': 'Modèle entraîné depuis framework ML',
                'formats': 'HDF5 (Keras), SavedModel (TF), ONNX'
            },
            '2': {
                'name': 'Model Parsing',
                'description': 'Extraction de la structure et des poids',
                'output': 'Graph representation, weights arrays'
            },
            '3': {
                'name': 'Configuration',
                'description': 'Paramètres de déploiement FPGA',
                'parameters': 'Precision, ReuseFactor, Strategy, etc.'
            },
            '4': {
                'name': 'HLS Code Generation',
                'description': 'Génération code C++ pour HLS',
                'output': 'Project files, C++ headers, implementation'
            },
            '5': {
                'name': 'C Simulation',
                'description': 'Validation algorithmique',
                'purpose': 'Vérifier que le code C++ est correct'
            },
            '6': {
                'name': 'HLS Synthesis',
                'description': 'Conversion C++ → RTL via Vivado HLS',
                'output': 'Verilog/VHDL RTL'
            },
            '7': {
                'name': 'C/RTL Co-simulation',
                'description': 'Validation RTL vs C',
                'purpose': 'S\'assurer que RTL correspond au C++'
            },
            '8': {
                'name': 'Vivado Implementation',
                'description': 'Place & Route sur FPGA',
                'output': 'Placed and routed design'
            },
            '9': {
                'name': 'Bitstream',
                'description': 'Génération fichier de configuration',
                'output': '.bit file pour programmer FPGA'
            }
        }
        
        print("\n" + "="*60)
        print("Pipeline Stages Explanation")
        print("="*60)
        
        for stage, info in stages.items():
            print(f"\nStage {stage}: {info['name']}")
            print(f"  Description: {info['description']}")
            if 'formats' in info:
                print(f"  Formats: {info['formats']}")
            if 'output' in info:
                print(f"  Output: {info['output']}")
            if 'parameters' in info:
                print(f"  Parameters: {info['parameters']}")
            if 'purpose' in info:
                print(f"  Purpose: {info['purpose']}")

pipeline = HLS4MLPipeline()
print(pipeline.visualize_pipeline())
pipeline.explain_stages()
```

---

## Installation et Premiers Pas

### Installation

```python
# Installation de hls4ml
"""
# Via pip
pip install hls4ml

# Ou depuis source
git clone https://github.com/fastmachinelearning/hls4ml.git
cd hls4ml
pip install -e .

# Dépendances
# - TensorFlow ou PyTorch (pour modèles)
# - Vivado HLS (pour synthèse, optionnel pour C sim)
# - NumPy, h5py, pyyaml
"""

import hls4ml
import numpy as np
from tensorflow import keras

# Vérification de l'installation
print(f"hls4ml version: {hls4ml.__version__}")

# Exemple minimal
def minimal_example():
    """Exemple minimal d'utilisation"""
    
    # 1. Créer un modèle simple
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(8,)),
        keras.layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # 2. Configuration de base
    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    
    # 3. Conversion
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir='my_hls_project',
        part='xc7z020clg400-1'  # Zynq-7000
    )
    
    # 4. Compilation (génère code C++)
    hls_model.compile()
    
    return hls_model

# hls_model = minimal_example()
```

---

## Concepts Clés

### Précision (Precision)

```python
class PrecisionConcepts:
    """
    Concepts de précision dans hls4ml
    """
    
    def __init__(self):
        self.precision_types = {
            'ap_fixed<W,I>': {
                'description': 'Fixed-point signed',
                'W': 'Total width in bits',
                'I': 'Integer bits',
                'F': 'Fractional bits = W - I',
                'range': '[-2^(I-1), 2^(I-1) - 2^-F]',
                'example': 'ap_fixed<16,6> = 16 bits total, 6 integer, 10 fractional'
            },
            'ap_ufixed<W,I>': {
                'description': 'Fixed-point unsigned',
                'range': '[0, 2^I - 2^-F]',
                'example': 'ap_ufixed<8,4> = 8 bits, unsigned, 4 integer'
            },
            'ap_int<W>': {
                'description': 'Integer signed',
                'range': '[-2^(W-1), 2^(W-1)-1]',
                'example': 'ap_int<8> = 8-bit signed integer'
            },
            'ap_uint<W>': {
                'description': 'Integer unsigned',
                'range': '[0, 2^W-1]',
                'example': 'ap_uint<8> = 8-bit unsigned integer'
            }
        }
    
    def display_precisions(self):
        """Affiche les types de précision"""
        print("\n" + "="*60)
        print("Precision Types in hls4ml")
        print("="*60)
        
        for ptype, info in self.precision_types.items():
            print(f"\n{ptype}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    def calculate_precision_requirements(self, data_range, error_tolerance):
        """
        Calcule la précision nécessaire
        
        Args:
            data_range: (min, max)
            error_tolerance: Maximum acceptable error
        """
        min_val, max_val = data_range
        range_val = max_val - min_val
        
        # Bits entiers nécessaires
        integer_bits = int(np.ceil(np.log2(max(abs(min_val), abs(max_val)) + 1))) + 1
        
        # Bits fractionnaires pour l'erreur
        fractional_bits = int(np.ceil(-np.log2(error_tolerance)))
        
        total_bits = integer_bits + fractional_bits
        
        return {
            'integer_bits': integer_bits,
            'fractional_bits': fractional_bits,
            'total_bits': total_bits,
            'recommended': f'ap_fixed<{total_bits},{integer_bits}>'
        }

precision = PrecisionConcepts()
precision.display_precisions()

# Exemple de calcul
prec_req = precision.calculate_precision_requirements(
    data_range=(-10.0, 10.0),
    error_tolerance=0.01
)
print("\n" + "="*60)
print("Precision Requirements Example")
print("="*60)
print(f"  Data range: [-10, 10]")
print(f"  Error tolerance: 0.01")
print(f"  Recommended: {prec_req['recommended']}")
print(f"  Integer bits: {prec_req['integer_bits']}")
print(f"  Fractional bits: {prec_req['fractional_bits']}")
```

---

## Exercices

### Exercice 15.1.1
Installez hls4ml et convertissez un modèle Keras simple de 2 couches dense avec différentes précisions.

### Exercice 15.1.2
Calculez la précision optimale pour des données dans la plage [-5, 5] avec une tolérance d'erreur de 0.001.

---

## Points Clés à Retenir

> 📌 **hls4ml traduit automatiquement ML → HLS → FPGA**

> 📌 **Support multiple frameworks: Keras, TensorFlow, PyTorch, ONNX**

> 📌 **Pipeline: Model → Parsing → Config → HLS Code → Synthesis → Bitstream**

> 📌 **Précision configurable: ap_fixed, ap_int, etc.**

> 📌 **Validé en production dans les expériences HEP**

---

*Section suivante : [15.2 Modèles Supportés et Limitations](./15_02_Modeles.md)*

