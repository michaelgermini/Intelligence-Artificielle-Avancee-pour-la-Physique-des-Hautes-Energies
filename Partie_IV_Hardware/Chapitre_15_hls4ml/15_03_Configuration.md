# 15.3 Configuration et Optimisation

---

## Introduction

La **configuration** de hls4ml est cruciale pour optimiser le compromis entre **latence, ressources et précision**. Cette section détaille les paramètres de configuration et les stratégies d'optimisation.

---

## Structure de Configuration

### Configuration de Base

```python
import hls4ml
from tensorflow import keras

class HLS4MLConfiguration:
    """
    Configuration complète de hls4ml
    """
    
    @staticmethod
    def create_base_config(model):
        """
        Crée une configuration de base
        
        Args:
            model: Modèle Keras
        """
        config = hls4ml.utils.config_from_keras_model(
            model,
            granularity='name'  # ou 'type' pour config par type de couche
        )
        
        return config
    
    @staticmethod
    def display_config_structure(config):
        """Affiche la structure d'une configuration"""
        print("\n" + "="*60)
        print("Configuration Structure")
        print("="*60)
        
        print("\nTop-level keys:")
        for key in config.keys():
            print(f"  • {key}")
        
        print("\nModel-level config:")
        if 'Model' in config:
            for key, value in config['Model'].items():
                print(f"  {key}: {value}")
        
        print("\nLayer-specific config:")
        if 'LayerName' in config:
            for layer_name in list(config['LayerName'].keys())[:3]:  # Affiche les 3 premiers
                print(f"  {layer_name}:")
                layer_config = config['LayerName'][layer_name]
                for key, value in layer_config.items():
                    print(f"    {key}: {value}")

# Exemple
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(16,), name='dense_1'),
    keras.layers.Dense(16, activation='relu', name='dense_2'),
    keras.layers.Dense(5, activation='softmax', name='output')
])

config = HLS4MLConfiguration.create_base_config(model)
HLS4MLConfiguration.display_config_structure(config)
```

---

## Paramètres de Configuration Principaux

### Precision

```python
class PrecisionConfiguration:
    """
    Configuration de la précision
    """
    
    @staticmethod
    def set_global_precision(config, precision):
        """
        Définit la précision globale
        
        Args:
            config: Configuration dict
            precision: 'ap_fixed<W,I>' ou 'ap_int<W>'
        """
        config['Model']['Precision'] = precision
        return config
    
    @staticmethod
    def set_layer_precision(config, layer_name, precision_dict):
        """
        Définit la précision pour une couche spécifique
        
        Args:
            config: Configuration dict
            layer_name: Nom de la couche
            precision_dict: {
                'weight': 'ap_fixed<W,I>',
                'bias': 'ap_fixed<W,I>',
                'result': 'ap_fixed<W,I>'
            }
        """
        if layer_name in config['LayerName']:
            config['LayerName'][layer_name]['Precision'] = precision_dict
        return config
    
    @staticmethod
    def set_mixed_precision(config, precision_strategy):
        """
        Définit une stratégie de précision mixte
        
        Args:
            config: Configuration dict
            precision_strategy: Dict {layer_name: precision_dict}
        """
        for layer_name, precision_dict in precision_strategy.items():
            PrecisionConfiguration.set_layer_precision(
                config, layer_name, precision_dict
            )
        return config
    
    @staticmethod
    def recommend_precision(data_range, error_tolerance=0.01):
        """
        Recommande une précision basée sur la plage de données
        
        Args:
            data_range: (min, max) des données
            error_tolerance: Tolérance d'erreur acceptable
        """
        import numpy as np
        
        min_val, max_val = data_range
        max_abs = max(abs(min_val), abs(max_val))
        
        # Bits entiers
        integer_bits = int(np.ceil(np.log2(max_abs + 1))) + 1  # +1 pour signe
        
        # Bits fractionnaires
        fractional_bits = int(np.ceil(-np.log2(error_tolerance)))
        
        total_bits = integer_bits + fractional_bits
        
        return {
            'weight': f'ap_fixed<{total_bits},{integer_bits}>',
            'bias': f'ap_fixed<{total_bits},{integer_bits}>',
            'result': f'ap_fixed<{total_bits+2},{integer_bits+1}>'  # Plus large pour accumulation
        }

# Exemple
config = HLS4MLConfiguration.create_base_config(model)

# Précision globale
config = PrecisionConfiguration.set_global_precision(
    config, 'ap_fixed<16,6>'
)

# Précision par couche
layer_prec = PrecisionConfiguration.recommend_precision(
    data_range=(-5.0, 5.0),
    error_tolerance=0.01
)
config = PrecisionConfiguration.set_layer_precision(
    config, 'dense_1', layer_prec
)

print("\n" + "="*60)
print("Precision Configuration Example")
print("="*60)
print(f"Recommended precision for range [-5, 5]:")
print(f"  {layer_prec}")
```

---

### ReuseFactor

```python
class ReuseFactorConfiguration:
    """
    Configuration du ReuseFactor (facteur de réutilisation)
    """
    
    @staticmethod
    def set_global_reuse_factor(config, reuse_factor):
        """
        Définit le ReuseFactor global
        
        Args:
            config: Configuration dict
            reuse_factor: 1 (fully parallel) à N (resource sharing)
        """
        config['Model']['ReuseFactor'] = reuse_factor
        return config
    
    @staticmethod
    def set_layer_reuse_factor(config, layer_name, reuse_factor):
        """Définit le ReuseFactor pour une couche spécifique"""
        if layer_name in config['LayerName']:
            config['LayerName'][layer_name]['ReuseFactor'] = reuse_factor
        return config
    
    @staticmethod
    def calculate_optimal_reuse_factor(layer_params, available_dsp, target_latency=None):
        """
        Calcule un ReuseFactor optimal
        
        Args:
            layer_params: Nombre de paramètres de la couche
            available_dsp: DSP slices disponibles
            target_latency: Latence cible (cycles), optionnel
        """
        # Approximation: 1 DSP par multiplication (simplifié)
        min_dsp_needed = layer_params
        reuse_factor = max(1, min_dsp_needed // available_dsp)
        
        # Si latence cible spécifiée, ajuster
        if target_latency:
            # Simplification: latency ≈ params / reuse_factor
            reuse_factor = max(reuse_factor, layer_params // target_latency)
        
        return {
            'recommended_reuse_factor': reuse_factor,
            'dsp_needed': min_dsp_needed // reuse_factor,
            'latency_estimate': layer_params // reuse_factor
        }

# Exemple
config = HLS4MLConfiguration.create_base_config(model)

# ReuseFactor global
config = ReuseFactorConfiguration.set_global_reuse_factor(config, 4)

# ReuseFactor par couche
reuse_info = ReuseFactorConfiguration.calculate_optimal_reuse_factor(
    layer_params=32 * 16,  # dense layer 32→16
    available_dsp=100,
    target_latency=64
)
config = ReuseFactorConfiguration.set_layer_reuse_factor(
    config, 'dense_1', reuse_info['recommended_reuse_factor']
)

print("\n" + "="*60)
print("ReuseFactor Configuration Example")
print("="*60)
print(f"Optimal ReuseFactor calculation:")
print(f"  Recommended: {reuse_info['recommended_reuse_factor']}")
print(f"  DSP needed: {reuse_info['dsp_needed']}")
print(f"  Latency estimate: {reuse_info['latency_estimate']} cycles")
```

---

### Strategy

```python
class StrategyConfiguration:
    """
    Configuration de la stratégie d'optimisation
    """
    
    strategies = {
        'Latency': {
            'description': 'Optimise pour minimiser la latence',
            'characteristics': [
                'ReuseFactor = 1 (fully parallel)',
                'Pipeline agressif',
                'Maximum ressources',
                'Latence minimale'
            ],
            'use_case': 'Applications temps réel critiques'
        },
        'Resource': {
            'description': 'Optimise pour minimiser les ressources',
            'characteristics': [
                'ReuseFactor élevé',
                'Partage ressources',
                'Ressources minimales',
                'Latence accrue'
            ],
            'use_case': 'Ressources limitées'
        },
        'Stable': {
            'description': 'Compromis stable',
            'characteristics': [
                'Balance latence/ressources',
                'Configuration modérée'
            ],
            'use_case': 'Cas général'
        }
    }
    
    @staticmethod
    def set_strategy(config, strategy):
        """
        Définit la stratégie d'optimisation
        
        Args:
            config: Configuration dict
            strategy: 'Latency', 'Resource', ou 'Stable'
        """
        if strategy not in StrategyConfiguration.strategies:
            raise ValueError(f"Strategy must be one of {list(StrategyConfiguration.strategies.keys())}")
        
        config['Model']['Strategy'] = strategy
        
        # Ajuste ReuseFactor selon stratégie
        if strategy == 'Latency':
            config['Model']['ReuseFactor'] = 1
        elif strategy == 'Resource':
            # Laisse ReuseFactor configuré ou utilise valeur par défaut
            if 'ReuseFactor' not in config['Model']:
                config['Model']['ReuseFactor'] = 4
        
        return config
    
    @staticmethod
    def display_strategies():
        """Affiche les stratégies disponibles"""
        print("\n" + "="*60)
        print("Optimization Strategies")
        print("="*60)
        
        for strategy, info in StrategyConfiguration.strategies.items():
            print(f"\n{strategy}:")
            print(f"  Description: {info['description']}")
            print(f"  Characteristics:")
            for char in info['characteristics']:
                print(f"    • {char}")
            print(f"  Use case: {info['use_case']}")

StrategyConfiguration.display_strategies()

# Exemple
config = HLS4MLConfiguration.create_base_config(model)
config = StrategyConfiguration.set_strategy(config, 'Latency')

print("\n" + "="*60)
print("Strategy Configuration Example")
print("="*60)
print(f"Strategy: {config['Model']['Strategy']}")
print(f"ReuseFactor: {config['Model'].get('ReuseFactor', 'Not set')}")
```

---

## Optimisations Avancées

### Configuration par Type de Couche

```python
class LayerTypeConfiguration:
    """
    Configuration par type de couche
    """
    
    @staticmethod
    def configure_by_type(config, layer_type_configs):
        """
        Configure par type de couche
        
        Args:
            config: Configuration dict
            layer_type_configs: Dict {
                'Dense': {...},
                'Conv2D': {...},
                'Activation': {...}
            }
        """
        # Si granularity='type' était utilisé, modifier ici
        # Sinon, appliquer à toutes les couches du type
        for layer_name, layer_config in config['LayerName'].items():
            # Détecter type (simplifié)
            layer_type = layer_config.get('Type', 'Unknown')
            
            if layer_type in layer_type_configs:
                layer_config.update(layer_type_configs[layer_type])
        
        return config

# Exemple
layer_type_configs = {
    'Dense': {
        'Precision': {
            'weight': 'ap_fixed<8,2>',
            'bias': 'ap_fixed<8,2>',
            'result': 'ap_fixed<16,6>'
        },
        'ReuseFactor': 4
    },
    'Activation': {
        'Precision': {
            'result': 'ap_fixed<16,6>'
        }
    }
}

config = LayerTypeConfiguration.configure_by_type(config, layer_type_configs)
```

---

### Optimisation pour Latence Minimale

```python
class LatencyOptimization:
    """
    Optimisation spécifique pour minimiser la latence
    """
    
    @staticmethod
    def optimize_for_latency(config, target_latency_ns=None, clock_freq_mhz=200):
        """
        Optimise la configuration pour minimiser la latence
        
        Args:
            config: Configuration dict
            target_latency_ns: Latence cible en ns (optionnel)
            clock_freq_mhz: Fréquence d'horloge en MHz
        """
        # Strategy
        config['Model']['Strategy'] = 'Latency'
        config['Model']['ReuseFactor'] = 1
        
        # Pipeline initiation interval = 1
        for layer_name in config['LayerName'].keys():
            config['LayerName'][layer_name]['ReuseFactor'] = 1
            # Assure pipeline optimal
            config['LayerName'][layer_name]['PipelineStyle'] = 'pipeline'
        
        # Précision minimale acceptable (réduit latence)
        # Peut nécessiter ajustement selon application
        
        return config
    
    @staticmethod
    def estimate_latency(config, model):
        """
        Estime la latence basée sur la configuration
        
        Args:
            config: Configuration dict
            model: Modèle Keras
        """
        total_latency_cycles = 0
        
        for layer in model.layers:
            layer_name = layer.name
            if layer_name in config['LayerName']:
                layer_config = config['LayerName'][layer_name]
                reuse_factor = layer_config.get('ReuseFactor', 
                                                config['Model'].get('ReuseFactor', 1))
                
                # Estimation simplifiée
                if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
                    # Approximatif: nombre d'opérations / reuse_factor
                    layer_ops = layer.input_shape[-1] * layer.output_shape[-1]
                    layer_latency = layer_ops // reuse_factor
                    total_latency_cycles += layer_latency
        
        return {
            'latency_cycles': total_latency_cycles,
            'latency_ns_200mhz': total_latency_cycles * 5,  # 5ns @ 200MHz
            'meets_target': True  # À vérifier avec target
        }

config_latency = HLS4MLConfiguration.create_base_config(model)
config_latency = LatencyOptimization.optimize_for_latency(config_latency)

latency_est = LatencyOptimization.estimate_latency(config_latency, model)
print("\n" + "="*60)
print("Latency Optimization Example")
print("="*60)
print(f"Estimated latency: {latency_est['latency_cycles']} cycles")
print(f"  = {latency_est['latency_ns_200mhz']:.1f} ns @ 200 MHz")
```

---

### Optimisation pour Ressources Minimales

```python
class ResourceOptimization:
    """
    Optimisation pour minimiser les ressources
    """
    
    @staticmethod
    def optimize_for_resources(config, available_dsp, available_bram):
        """
        Optimise pour minimiser l'utilisation de ressources
        
        Args:
            config: Configuration dict
            available_dsp: DSP slices disponibles
            available_bram: BRAM disponibles (18K blocks)
        """
        config['Model']['Strategy'] = 'Resource'
        
        # Calcule ReuseFactor pour chaque couche
        for layer_name, layer_config in config['LayerName'].items():
            # Estimation simplifiée
            # En pratique, utiliser informations du modèle
            layer_config['ReuseFactor'] = 8  # Exemple
        
        # Précision réduite si acceptable
        config['Model']['Precision'] = 'ap_fixed<10,4>'  # Plus petite
        
        return config
    
    @staticmethod
    def estimate_resources(config, model):
        """Estime l'utilisation de ressources"""
        total_dsp = 0
        total_bram = 0
        
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]
                num_params = weights.size
                
                layer_name = layer.name
                if layer_name in config['LayerName']:
                    reuse_factor = config['LayerName'][layer_name].get(
                        'ReuseFactor', config['Model'].get('ReuseFactor', 1)
                    )
                    
                    # DSP: multiplications
                    layer_dsp = num_params // reuse_factor
                    total_dsp += layer_dsp
                    
                    # BRAM: poids (int8 = 1 byte)
                    weight_bits = num_params * 8
                    layer_bram = weight_bits / (18 * 1024)
                    total_bram += layer_bram
        
        return {
            'dsp_estimate': int(total_dsp),
            'bram_18k_estimate': int(total_bram)
        }

config_resources = HLS4MLConfiguration.create_base_config(model)
config_resources = ResourceOptimization.optimize_for_resources(
    config_resources, available_dsp=100, available_bram=50
)

resource_est = ResourceOptimization.estimate_resources(config_resources, model)
print("\n" + "="*60)
print("Resource Optimization Example")
print("="*60)
print(f"DSP estimate: {resource_est['dsp_estimate']}")
print(f"BRAM 18K estimate: {resource_est['bram_18k_estimate']}")
```

---

## Exercices

### Exercice 15.3.1
Créez une configuration optimisée pour un modèle avec contrainte de latence de 100 ns à 200 MHz.

### Exercice 15.3.2
Optimisez un modèle pour utiliser moins de 50 DSP slices tout en minimisant la perte de précision.

---

## Points Clés à Retenir

> 📌 **Precision: contrôle la précision numérique (ap_fixed, ap_int)**

> 📌 **ReuseFactor: trade-off ressources/latence (1=parallel, >1=sharing)**

> 📌 **Strategy: Latency (min latency), Resource (min resources), Stable (balance)**

> 📌 **Configuration par couche possible pour fine-tuning**

> 📌 **Optimisation dépend des contraintes: latence, ressources, précision**

---

*Section suivante : [15.4 Stratégies de Parallélisation](./15_04_Parallelisation.md)*

