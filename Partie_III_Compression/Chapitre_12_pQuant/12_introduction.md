# Chapitre 12 : La Bibliothèque pQuant

---

## Introduction

**pQuant** est une bibliothèque open-source développée au CERN pour la compression de modèles deep learning utilisant des techniques de rang faible et de réseaux de tenseurs. Elle est conçue pour faciliter la recherche et le déploiement de modèles compressés dans les applications de physique des hautes énergies.

---

## Plan du Chapitre

1. [Architecture et Conception](./12_01_Architecture.md)
2. [API et Interfaces Principales](./12_02_API.md)
3. [Implémentation des Méthodes de Compression](./12_03_Implementation.md)
4. [Pipelines de Compression Automatisés](./12_04_Pipelines.md)
5. [Benchmarking et Évaluation](./12_05_Benchmarking.md)
6. [Contribution Open-Source et Bonnes Pratiques](./12_06_Contribution.md)

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    Architecture pQuant                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Interface Utilisateur                   │      │
│  │  (Compression Pipeline, Configuration)               │      │
│  └───────────────────────┬──────────────────────────────┘      │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────┐      │
│  │         Méthodes de Compression                      │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │      │
│  │  │  Low-Rank│ │Tensor NN │ │Quantiz.  │            │      │
│  │  │  (SVD,   │ │(TT, CP)  │ │(INT8,    │            │      │
│  │  │   LoRA)  │ │          │ │  FP16)   │            │      │
│  │  └──────────┘ └──────────┘ └──────────┘            │      │
│  └───────────────────────┬──────────────────────────────┘      │
│                          │                                      │
│  ┌───────────────────────▼──────────────────────────────┐      │
│  │      Backends (PyTorch, TensorFlow, JAX)             │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation et Configuration

```python
# Installation (exemple)
# pip install pquant
# ou depuis source:
# git clone https://github.com/cern/pquant.git
# cd pquant
# pip install -e .

import pquant
import torch
import torch.nn as nn

print(f"pQuant version: {pquant.__version__}")

# Vérification de l'installation
assert pquant is not None, "pQuant non installé"
```

---

## Utilisation Basique

### Compression Simple

```python
from pquant import compress_model
from pquant.compression import LowRankCompression, TensorTrainCompression

# Modèle à compresser
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Compression par rang faible
compressed_model = compress_model(
    model,
    method='low_rank',
    rank=64,
    target_sparsity=0.5
)

print(f"Modèle original: {sum(p.numel() for p in model.parameters()):,} paramètres")
print(f"Modèle compressé: {sum(p.numel() for p in compressed_model.parameters()):,} paramètres")
```

### Compression avec Tensor Train

```python
from pquant.compression import TensorTrainCompression

# Configuration
config = {
    'method': 'tensor_train',
    'rank': 32,
    'train_after_compression': True,
    'epochs': 10
}

# Compression
tt_compressor = TensorTrainCompression(config)
compressed_model = tt_compressor.compress(model)

# Évaluation
original_accuracy = evaluate(model, test_loader)
compressed_accuracy = evaluate(compressed_model, test_loader)

print(f"Accuracy originale: {original_accuracy:.2%}")
print(f"Accuracy compressée: {compressed_accuracy:.2%}")
print(f"Dégradation: {(original_accuracy - compressed_accuracy)*100:.2f}%")
```

---

## API Principale

### Classe de Compression

```python
class CompressionPipeline:
    """
    Pipeline de compression modulaire
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionnaire de configuration
                {
                    'methods': ['low_rank', 'quantization'],
                    'low_rank_rank': 64,
                    'quantization_bits': 8,
                    ...
                }
        """
        self.config = config
        self.methods = []
        
        # Initialise les méthodes selon la config
        if 'low_rank' in config.get('methods', []):
            self.methods.append(LowRankCompression(config))
        
        if 'quantization' in config.get('methods', []):
            self.methods.append(QuantizationCompression(config))
    
    def compress(self, model, train_loader=None):
        """
        Compresse le modèle en appliquant toutes les méthodes
        
        Args:
            model: Modèle PyTorch
            train_loader: DataLoader pour calibration/fine-tuning
        
        Returns:
            Modèle compressé
        """
        compressed = model
        
        for method in self.methods:
            compressed = method.compress(compressed, train_loader)
        
        return compressed
    
    def evaluate(self, original_model, compressed_model, test_loader):
        """
        Compare les performances original vs compressé
        """
        results = {
            'original': evaluate_model(original_model, test_loader),
            'compressed': evaluate_model(compressed_model, test_loader),
            'compression_ratio': self._compute_compression_ratio(
                original_model, compressed_model
            )
        }
        
        return results
```

---

## Intégration avec les Workflows HEP

```python
class HEPModelCompression:
    """
    Compression spécialisée pour les modèles de physique des particules
    """
    
    @staticmethod
    def compress_jet_tagger(model, train_loader, val_loader):
        """
        Compresse un modèle de classification de jets
        
        Optimisé pour préserver les performances sur les jets rares
        """
        config = {
            'methods': ['low_rank', 'quantization'],
            'low_rank_rank': 64,
            'quantization_bits': 8,
            'preserve_rare_classes': True,  # Important pour HEP
            'fine_tune_epochs': 20
        }
        
        pipeline = CompressionPipeline(config)
        compressed = pipeline.compress(model, train_loader)
        
        # Évaluation spéciale pour HEP
        results = pipeline.evaluate(model, compressed, val_loader)
        
        # Métriques additionnelles pour HEP
        results['b_tag_efficiency'] = evaluate_b_tagging(
            model, compressed, val_loader
        )
        
        return compressed, results
    
    @staticmethod
    def compress_trigger_model(model, target_latency_ns=100):
        """
        Compresse un modèle pour le trigger L1
        
        Contraintes strictes de latence
        """
        config = {
            'methods': ['aggressive_quantization', 'structured_pruning'],
            'quantization_bits': 6,  # Très agressif
            'pruning_sparsity': 0.9,
            'target_latency_ns': target_latency_ns
        }
        
        pipeline = CompressionPipeline(config)
        compressed = pipeline.compress(model)
        
        # Validation de la latence
        latency = measure_latency(compressed)
        assert latency < target_latency_ns, f"Latence {latency}ns > cible {target_latency_ns}ns"
        
        return compressed
```

---

## Benchmarks et Métriques

```python
from pquant.benchmarks import benchmark_compression

# Benchmark standard
results = benchmark_compression(
    model=model,
    dataset='CIFAR10',
    methods=['low_rank', 'quantization', 'pruning'],
    metrics=['accuracy', 'inference_time', 'model_size']
)

print("Résultats du benchmark:")
for method, metrics in results.items():
    print(f"\n{method}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
```

---

## Points Clés à Retenir

> 📌 **pQuant fournit une interface unifiée pour diverses techniques de compression**

> 📌 **La bibliothèque est optimisée pour les modèles utilisés en physique des particules**

> 📌 **L'intégration avec les workflows existants est facilitée**

> 📌 **Les contributions open-source sont encouragées pour améliorer la bibliothèque**

---

## Références

- Repository GitHub: https://github.com/cern/pquant
- Documentation: https://pquant.readthedocs.io/
- Exemples: https://github.com/cern/pquant/tree/main/examples

---

*Section suivante : [12.1 Architecture et Conception](./12_01_Architecture.md)*

