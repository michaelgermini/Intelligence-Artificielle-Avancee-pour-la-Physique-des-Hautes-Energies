# 12.1 Architecture et Conception

---

## Introduction

Cette section décrit l'architecture générale de la bibliothèque **pQuant**, ses principes de conception, et comment les différents composants interagissent pour fournir un système unifié de compression de modèles.

---

## Principes de Conception

### Objectifs

1. **Modularité** : Chaque technique de compression est un module indépendant
2. **Flexibilité** : Combinaison facile de différentes méthodes
3. **Extensibilité** : Facile d'ajouter de nouvelles méthodes
4. **Performance** : Optimisé pour la production
5. **Compatibilité** : Support de PyTorch, TensorFlow, JAX

---

## Architecture Générale

```
┌─────────────────────────────────────────────────────────────────┐
│                    Architecture pQuant                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │           Interface Utilisateur (High-Level)          │    │
│  │  compress_model(), CompressionPipeline                │    │
│  └───────────────────────┬────────────────────────────────┘    │
│                          │                                      │
│  ┌───────────────────────▼────────────────────────────────┐    │
│  │          Compression Strategies                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │    │
│  │  │Low-Rank  │ │Tensor NN │ │Quantiz.  │ │ Pruning  │ │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │    │
│  └───────────────────────┬────────────────────────────────┘    │
│                          │                                      │
│  ┌───────────────────────▼────────────────────────────────┐    │
│  │         Core Abstractions                              │    │
│  │  CompressionMethod, LayerAdapter, Optimizer           │    │
│  └───────────────────────┬────────────────────────────────┘    │
│                          │                                      │
│  ┌───────────────────────▼────────────────────────────────┐    │
│  │         Backend Adapters                               │    │
│  │  PyTorchBackend, TensorFlowBackend, JAXBackend        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Structure des Modules

```python
# Structure de pQuant
"""
pquant/
├── __init__.py
├── core/
│   ├── compression_method.py      # Interface de base
│   ├── layer_adapter.py           # Adaptation de couches
│   └── optimizer.py               # Optimiseurs spécialisés
├── methods/
│   ├── low_rank/
│   │   ├── svd.py                 # SVD compression
│   │   ├── lora.py                # LoRA
│   │   └── factorization.py       # Factorisation générique
│   ├── tensor_networks/
│   │   ├── tensor_train.py        # TT compression
│   │   ├── tucker.py              # Tucker compression
│   │   └── cp.py                  # CP decomposition
│   ├── quantization/
│   │   ├── ptq.py                 # Post-training quantization
│   │   ├── qat.py                 # Quantization-aware training
│   │   └── mixed_precision.py     # Mixed precision
│   └── pruning/
│       ├── unstructured.py        # Unstructured pruning
│       └── structured.py          # Structured pruning
├── pipelines/
│   ├── compression_pipeline.py    # Pipeline principal
│   └── evaluation_pipeline.py     # Évaluation
├── utils/
│   ├── model_analysis.py          # Analyse de modèles
│   ├── benchmarking.py            # Benchmarks
│   └── visualization.py           # Visualisation
└── backends/
    ├── pytorch.py                 # PyTorch backend
    ├── tensorflow.py              # TensorFlow backend
    └── jax.py                     # JAX backend
"""
```

---

## Abstractions de Base

### Interface CompressionMethod

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class CompressionMethod(ABC):
    """
    Interface de base pour toutes les méthodes de compression
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration spécifique à la méthode
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compress(self, model, train_loader=None, val_loader=None):
        """
        Compresse un modèle
        
        Args:
            model: Modèle à compresser
            train_loader: DataLoader pour entraînement/calibration
            val_loader: DataLoader pour validation
        
        Returns:
            Modèle compressé
        """
        pass
    
    @abstractmethod
    def get_compression_info(self, original_model, compressed_model):
        """
        Retourne des informations sur la compression
        
        Returns:
            Dict avec compression_ratio, param_count, etc.
        """
        pass
    
    def validate_config(self):
        """Valide la configuration"""
        # Vérifie les paramètres requis
        pass

# Exemple d'implémentation
class LowRankCompression(CompressionMethod):
    """
    Compression par rang faible (SVD)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.get('rank', 64)
        self.method = config.get('method', 'svd')  # 'svd', 'factorization'
    
    def compress(self, model, train_loader=None, val_loader=None):
        """Implémente la compression low-rank"""
        compressed_model = model  # Copie
        
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Compresse la couche
                compressed_layer = self._compress_linear(module)
                # Remplace dans le modèle
                # (nécessite logique de remplacement)
        
        return compressed_model
    
    def _compress_linear(self, linear_layer):
        """Compresse une couche linéaire"""
        if self.method == 'svd':
            from .low_rank.svd import TruncatedSVDLinear
            return TruncatedSVDLinear.from_linear(linear_layer, rank=self.rank)
        # Autres méthodes...
    
    def get_compression_info(self, original_model, compressed_model):
        """Calcule les métriques de compression"""
        orig_params = sum(p.numel() for p in original_model.parameters())
        comp_params = sum(p.numel() for p in compressed_model.parameters())
        
        return {
            'compression_ratio': orig_params / comp_params,
            'original_params': orig_params,
            'compressed_params': comp_params,
            'method': 'low_rank',
            'rank': self.rank
        }
```

---

## Backend Abstraction

```python
class Backend(ABC):
    """
    Interface pour les différents backends (PyTorch, TensorFlow, etc.)
    """
    
    @abstractmethod
    def create_layer(self, layer_type, config):
        """Crée une couche du backend"""
        pass
    
    @abstractmethod
    def get_weights(self, layer):
        """Récupère les poids d'une couche"""
        pass
    
    @abstractmethod
    def set_weights(self, layer, weights):
        """Définit les poids d'une couche"""
        pass
    
    @abstractmethod
    def forward(self, layer, x):
        """Forward pass"""
        pass

class PyTorchBackend(Backend):
    """Backend PyTorch"""
    
    def create_layer(self, layer_type, config):
        if layer_type == 'linear':
            return nn.Linear(config['in_features'], config['out_features'])
        elif layer_type == 'conv2d':
            return nn.Conv2d(**config)
        # ...
    
    def get_weights(self, layer):
        return layer.weight.data
    
    def set_weights(self, layer, weights):
        layer.weight.data = weights
    
    def forward(self, layer, x):
        return layer(x)

# Factory pour sélectionner le backend
def get_backend(framework='pytorch'):
    """Retourne le backend approprié"""
    backends = {
        'pytorch': PyTorchBackend,
        'tensorflow': TensorFlowBackend,
        'jax': JAXBackend
    }
    
    return backends[framework]()
```

---

## Layer Adapters

```python
class LayerAdapter:
    """
    Adaptateur pour convertir/détecter les types de couches
    """
    
    def __init__(self, backend):
        self.backend = backend
    
    def is_compressible(self, layer):
        """
        Vérifie si une couche peut être compressée
        """
        compressible_types = (nn.Linear, nn.Conv2d, nn.Conv1d)
        return isinstance(layer, compressible_types)
    
    def get_layer_info(self, layer):
        """Retourne les informations d'une couche"""
        if isinstance(layer, nn.Linear):
            return {
                'type': 'linear',
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'shape': (layer.out_features, layer.in_features)
            }
        elif isinstance(layer, nn.Conv2d):
            return {
                'type': 'conv2d',
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'shape': (layer.out_channels, layer.in_channels, 
                         *layer.kernel_size)
            }
        return None
    
    def replace_layer(self, model, old_layer_name, new_layer):
        """
        Remplace une couche dans le modèle
        
        (Nécessite logique de navigation dans le graphe)
        """
        # Parse le nom pour trouver le parent
        parts = old_layer_name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Remplace
        setattr(parent, parts[-1], new_layer)
```

---

## Configuration System

```python
class ConfigManager:
    """
    Gestionnaire de configuration pour pQuant
    """
    
    @staticmethod
    def load_config(config_path=None):
        """
        Charge une configuration depuis un fichier ou utilise les défauts
        """
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = ConfigManager.default_config()
        
        return config
    
    @staticmethod
    def default_config():
        """Configuration par défaut"""
        return {
            'compression': {
                'methods': ['low_rank', 'quantization'],
                'low_rank': {
                    'rank': 64,
                    'method': 'svd'
                },
                'quantization': {
                    'bits': 8,
                    'method': 'ptq'
                }
            },
            'training': {
                'epochs': 10,
                'lr': 1e-4,
                'batch_size': 32
            },
            'evaluation': {
                'metrics': ['accuracy', 'compression_ratio', 'latency']
            }
        }
    
    @staticmethod
    def validate_config(config):
        """Valide une configuration"""
        required_keys = ['compression']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # Validation des méthodes
        methods = config['compression'].get('methods', [])
        valid_methods = ['low_rank', 'quantization', 'pruning', 'tensor_network']
        
        for method in methods:
            if method not in valid_methods:
                raise ValueError(f"Unknown compression method: {method}")

# Exemple d'utilisation
config = ConfigManager.load_config('pquant_config.yaml')
config_validated = ConfigManager.validate_config(config)
```

---

## Registre de Méthodes

```python
class CompressionMethodRegistry:
    """
    Registre central pour les méthodes de compression
    Permet l'extension facile de nouvelles méthodes
    """
    
    _methods = {}
    
    @classmethod
    def register(cls, name: str, method_class):
        """
        Enregistre une nouvelle méthode de compression
        """
        cls._methods[name] = method_class
    
    @classmethod
    def get(cls, name: str, config: Dict):
        """
        Instancie une méthode de compression
        """
        if name not in cls._methods:
            raise ValueError(f"Unknown compression method: {name}. "
                           f"Available: {list(cls._methods.keys())}")
        
        method_class = cls._methods[name]
        return method_class(config)
    
    @classmethod
    def list_available(cls):
        """Liste les méthodes disponibles"""
        return list(cls._methods.keys())

# Enregistrement des méthodes standard
CompressionMethodRegistry.register('low_rank', LowRankCompression)
CompressionMethodRegistry.register('quantization', QuantizationCompression)
CompressionMethodRegistry.register('tensor_train', TensorTrainCompression)
# ...

# Utilisation
method = CompressionMethodRegistry.get('low_rank', {'rank': 64})
```

---

## Exercices

### Exercice 12.1.1
Concevez une extension pour ajouter une nouvelle méthode de compression à pQuant.

### Exercice 12.1.2
Implémentez un adapter pour un nouveau type de couche (ex: Attention).

---

## Points Clés à Retenir

> 📌 **Architecture modulaire permet combinaison flexible de méthodes**

> 📌 **Backend abstraction supporte multiple frameworks**

> 📌 **Layer adapters facilitent la détection et conversion de couches**

> 📌 **Config system centralise la configuration**

> 📌 **Method registry permet extension facile**

---

*Section suivante : [12.2 API et Interfaces Principales](./12_02_API.md)*

