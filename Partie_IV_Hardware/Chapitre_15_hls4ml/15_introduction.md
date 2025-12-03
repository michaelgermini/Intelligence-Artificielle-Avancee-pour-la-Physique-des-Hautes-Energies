# Chapitre 15 : hls4ml - Machine Learning pour FPGA

---

## Introduction

**hls4ml** est une bibliothèque open-source qui traduit des modèles de machine learning en firmware FPGA via High-Level Synthesis (HLS). Développée au CERN et Fermilab, elle est devenue l'outil de référence pour le déploiement de ML dans les systèmes de trigger des expériences de physique des particules.

---

## Plan du Chapitre

1. [Introduction à hls4ml](./15_01_Introduction.md)
2. [Modèles Supportés et Limitations](./15_02_Modeles.md)
3. [Configuration et Optimisation](./15_03_Configuration.md)
4. [Stratégies de Parallélisation](./15_04_Parallelisation.md)
5. [Intégration avec les Workflows de Physique](./15_05_Integration.md)
6. [Études de Cas au CERN](./15_06_Cas_CERN.md)

---

## Vue d'Ensemble de hls4ml

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline hls4ml                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Keras/    │    │   hls4ml    │    │    HLS      │         │
│  │  PyTorch    │───▶│  Converter  │───▶│   (Vivado)  │         │
│  │   Model     │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│                                               ▼                 │
│                     ┌─────────────┐    ┌─────────────┐         │
│                     │   Bitstream │◀───│  Synthesis  │         │
│                     │    (FPGA)   │    │   & P&R     │         │
│                     └─────────────┘    └─────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation et Configuration

```python
# Installation
# pip install hls4ml

import hls4ml
import numpy as np
from tensorflow import keras

# Vérification de l'installation
print(f"hls4ml version: {hls4ml.__version__}")

# Configuration du backend
hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
```

---

## Conversion d'un Modèle Keras

```python
def create_simple_classifier():
    """Crée un classificateur simple pour démonstration"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(16,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def convert_to_hls4ml(keras_model, output_dir='my_hls_project'):
    """
    Convertit un modèle Keras en projet HLS
    """
    # Configuration de base
    config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    
    # Affiche la configuration par défaut
    print("Configuration par défaut:")
    print(f"  Précision: {config['Model']['Precision']}")
    print(f"  ReuseFactor: {config['Model']['ReuseFactor']}")
    
    # Personnalisation de la configuration
    config['Model']['Precision'] = 'ap_fixed<16,6>'
    config['Model']['ReuseFactor'] = 1  # Fully parallel
    
    # Configuration par couche
    for layer in config['LayerName'].keys():
        config['LayerName'][layer]['Precision'] = {
            'weight': 'ap_fixed<8,2>',
            'bias': 'ap_fixed<8,2>',
            'result': 'ap_fixed<16,6>'
        }
    
    # Conversion
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=config,
        output_dir=output_dir,
        part='xcu250-figd2104-2L-e'  # FPGA Alveo U250
    )
    
    return hls_model, config

# Exemple
keras_model = create_simple_classifier()
hls_model, config = convert_to_hls4ml(keras_model)

# Compilation (génère le code HLS)
hls_model.compile()

# Synthèse (nécessite Vivado HLS)
# hls_model.build(csim=True, synth=True, cosim=True)
```

---

## Configuration Avancée

```python
class HLS4MLConfigurator:
    """
    Utilitaire pour configurer hls4ml de manière optimale
    """
    
    @staticmethod
    def create_config(model, 
                     precision='ap_fixed<16,6>',
                     reuse_factor=1,
                     strategy='Latency'):
        """
        Crée une configuration personnalisée
        
        Args:
            precision: Précision par défaut
            reuse_factor: 1 = fully parallel, >1 = resource sharing
            strategy: 'Latency' ou 'Resource'
        """
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        
        config['Model']['Precision'] = precision
        config['Model']['ReuseFactor'] = reuse_factor
        config['Model']['Strategy'] = strategy
        
        return config
    
    @staticmethod
    def optimize_for_latency(config, target_latency_ns=100):
        """
        Optimise pour minimiser la latence
        """
        # Fully parallel
        config['Model']['ReuseFactor'] = 1
        config['Model']['Strategy'] = 'Latency'
        
        # Pipeline initiation interval = 1
        for layer in config['LayerName'].keys():
            config['LayerName'][layer]['ReuseFactor'] = 1
        
        return config
    
    @staticmethod
    def optimize_for_resources(config, target_utilization=0.8):
        """
        Optimise pour minimiser l'utilisation de ressources
        """
        # Augmente le reuse factor
        config['Model']['Strategy'] = 'Resource'
        
        # Estime le reuse factor nécessaire
        # (basé sur la taille du modèle et les ressources cibles)
        
        return config
    
    @staticmethod
    def layer_specific_precision(config, layer_precisions):
        """
        Configure la précision par couche
        
        layer_precisions: dict {layer_name: {'weight': ..., 'bias': ..., 'result': ...}}
        """
        for layer_name, precisions in layer_precisions.items():
            if layer_name in config['LayerName']:
                config['LayerName'][layer_name]['Precision'] = precisions
        
        return config

# Exemple d'utilisation
config = HLS4MLConfigurator.create_config(keras_model)
config = HLS4MLConfigurator.optimize_for_latency(config)

# Précisions spécifiques
layer_prec = {
    'dense': {'weight': 'ap_fixed<6,2>', 'bias': 'ap_fixed<6,2>', 'result': 'ap_fixed<12,4>'},
    'dense_1': {'weight': 'ap_fixed<8,2>', 'bias': 'ap_fixed<8,2>', 'result': 'ap_fixed<14,5>'},
}
config = HLS4MLConfigurator.layer_specific_precision(config, layer_prec)
```

---

## Stratégies de Parallélisation

```python
class ParallelizationStrategies:
    """
    Différentes stratégies de parallélisation pour hls4ml
    """
    
    @staticmethod
    def fully_parallel(config):
        """
        Parallélisation complète: latence minimale, ressources maximales
        
        Chaque multiplication est implémentée en hardware dédié
        """
        config['Model']['ReuseFactor'] = 1
        config['Model']['Strategy'] = 'Latency'
        
        for layer in config['LayerName'].keys():
            config['LayerName'][layer]['ReuseFactor'] = 1
        
        return config
    
    @staticmethod
    def resource_sharing(config, reuse_factor):
        """
        Partage de ressources: ressources réduites, latence augmentée
        
        Les multiplieurs sont réutilisés sur plusieurs cycles
        """
        config['Model']['ReuseFactor'] = reuse_factor
        config['Model']['Strategy'] = 'Resource'
        
        return config
    
    @staticmethod
    def layer_fusion(model):
        """
        Fusion de couches pour réduire les accès mémoire
        
        Ex: Conv + BatchNorm + ReLU → une seule opération
        """
        # hls4ml fait automatiquement certaines fusions
        # Batch normalization est fusionnée avec la couche précédente
        pass
    
    @staticmethod
    def estimate_resources(config, model):
        """
        Estime l'utilisation de ressources
        """
        # Compte les multiplications
        total_mults = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]
                total_mults += weights.size
        
        reuse_factor = config['Model']['ReuseFactor']
        
        # DSP slices nécessaires (approximation)
        dsp_needed = total_mults // reuse_factor
        
        # Latence estimée (cycles)
        latency_cycles = total_mults // (total_mults // reuse_factor)
        
        return {
            'total_multiplications': total_mults,
            'dsp_estimate': dsp_needed,
            'latency_cycles': latency_cycles
        }

# Comparaison des stratégies
print("Comparaison des stratégies de parallélisation:")

for rf in [1, 4, 16, 64]:
    config = HLS4MLConfigurator.create_config(keras_model)
    config['Model']['ReuseFactor'] = rf
    
    resources = ParallelizationStrategies.estimate_resources(config, keras_model)
    print(f"\nReuseFactor={rf}:")
    print(f"  DSP estimés: {resources['dsp_estimate']}")
    print(f"  Latence (cycles): {resources['latency_cycles']}")
```

---

## Validation et Test

```python
class HLS4MLValidator:
    """
    Outils de validation pour les modèles hls4ml
    """
    
    @staticmethod
    def compare_predictions(keras_model, hls_model, test_data, tolerance=1e-3):
        """
        Compare les prédictions Keras vs HLS
        """
        # Prédictions Keras
        keras_pred = keras_model.predict(test_data)
        
        # Prédictions HLS (simulation C)
        hls_pred = hls_model.predict(test_data)
        
        # Comparaison
        max_diff = np.abs(keras_pred - hls_pred).max()
        mean_diff = np.abs(keras_pred - hls_pred).mean()
        
        # Vérification de la classification
        keras_class = np.argmax(keras_pred, axis=1)
        hls_class = np.argmax(hls_pred, axis=1)
        accuracy_match = (keras_class == hls_class).mean()
        
        return {
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'classification_match': accuracy_match,
            'within_tolerance': max_diff < tolerance
        }
    
    @staticmethod
    def profile_latency(hls_model, n_samples=1000):
        """
        Profile la latence d'inférence
        """
        import time
        
        test_input = np.random.randn(1, hls_model.config.get_input_shape()[0])
        
        # Warmup
        for _ in range(10):
            _ = hls_model.predict(test_input)
        
        # Mesure
        start = time.time()
        for _ in range(n_samples):
            _ = hls_model.predict(test_input)
        elapsed = time.time() - start
        
        return {
            'total_time_ms': elapsed * 1000,
            'avg_latency_us': (elapsed / n_samples) * 1e6,
            'throughput_khz': n_samples / elapsed / 1000
        }
    
    @staticmethod
    def analyze_synthesis_report(report_path):
        """
        Parse le rapport de synthèse Vivado
        """
        # Le rapport contient:
        # - Utilisation des ressources (LUT, FF, BRAM, DSP)
        # - Timing (fréquence max, latence)
        # - Warnings et erreurs
        
        # Parsing simplifié
        resources = {
            'LUT': 0,
            'FF': 0,
            'BRAM': 0,
            'DSP': 0
        }
        
        timing = {
            'clock_period_ns': 0,
            'latency_cycles': 0,
            'initiation_interval': 0
        }
        
        return {'resources': resources, 'timing': timing}

# Validation
"""
test_data = np.random.randn(100, 16).astype(np.float32)
validation = HLS4MLValidator.compare_predictions(keras_model, hls_model, test_data)
print(f"Validation: {validation}")
"""
```

---

## Applications au CERN

```python
class CERNTriggerModel:
    """
    Exemple de modèle pour le trigger L1 du CMS
    """
    
    @staticmethod
    def create_jet_tagger(n_features=16, n_classes=5):
        """
        Crée un tagger de jets pour le trigger
        
        Contraintes:
        - Latence < 100 ns
        - Ressources limitées
        """
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', 
                             input_shape=(n_features,),
                             kernel_initializer='glorot_uniform'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def get_trigger_config():
        """
        Configuration optimisée pour le trigger
        """
        config = {
            'Model': {
                'Precision': 'ap_fixed<10,4>',
                'ReuseFactor': 1,
                'Strategy': 'Latency'
            },
            'LayerName': {
                'dense': {
                    'Precision': {
                        'weight': 'ap_fixed<6,1>',
                        'bias': 'ap_fixed<6,1>',
                        'result': 'ap_fixed<10,4>'
                    },
                    'ReuseFactor': 1
                },
                'dense_1': {
                    'Precision': {
                        'weight': 'ap_fixed<6,1>',
                        'bias': 'ap_fixed<6,1>',
                        'result': 'ap_fixed<10,4>'
                    },
                    'ReuseFactor': 1
                },
                'dense_2': {
                    'Precision': {
                        'weight': 'ap_fixed<8,2>',
                        'bias': 'ap_fixed<8,2>',
                        'result': 'ap_fixed<12,6>'
                    },
                    'ReuseFactor': 1
                }
            }
        }
        return config
    
    @staticmethod
    def validate_for_trigger(hls_model, target_latency_ns=100, clock_freq_mhz=200):
        """
        Valide que le modèle respecte les contraintes du trigger
        """
        # Latence en cycles
        target_cycles = target_latency_ns * clock_freq_mhz / 1000
        
        # Parse le rapport de synthèse
        # actual_latency = ...
        
        # Vérifie les ressources disponibles
        # ...
        
        return {
            'target_cycles': target_cycles,
            'meets_latency': True,  # À vérifier
            'meets_resources': True  # À vérifier
        }

# Création et conversion
jet_model = CERNTriggerModel.create_jet_tagger()
trigger_config = CERNTriggerModel.get_trigger_config()

print("Modèle de jet tagger pour trigger:")
jet_model.summary()
```

---

## Exercices

### Exercice 15.1
Convertissez un CNN simple en hls4ml et mesurez l'impact de différentes précisions sur la précision de classification.

### Exercice 15.2
Optimisez un modèle pour atteindre une latence de 50 ns sur un FPGA Xilinx.

### Exercice 15.3
Comparez les ressources utilisées pour différentes stratégies de reuse factor.

---

## Points Clés à Retenir

> 📌 **hls4ml traduit automatiquement les modèles ML en firmware FPGA**

> 📌 **Le compromis latence/ressources est contrôlé par le ReuseFactor**

> 📌 **La quantification est essentielle pour tenir dans les ressources FPGA**

> 📌 **La validation C-simulation vs RTL est cruciale avant déploiement**

---

## Références

1. Duarte, J. et al. "Fast inference of deep neural networks in FPGAs for particle physics." JINST 13 (2018)
2. hls4ml Documentation: https://fastmachinelearning.org/hls4ml/
3. Summers, S. et al. "Fast inference of Boosted Decision Trees in FPGAs for particle physics." JINST 15 (2020)

---

*Chapitre suivant : [Chapitre 16 - Hardware-Aware Neural Architecture Search](../Chapitre_16_Hardware_NAS/16_introduction.md)*

