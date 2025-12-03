# 18.4 Intégration de l'IA dans les Triggers

---

## Introduction

L'intégration de l'**intelligence artificielle** dans les systèmes de trigger a révolutionné la sélection d'événements au LHC. Les modèles de machine learning permettent d'améliorer significativement l'efficacité de sélection tout en maintenant des taux de trigger acceptables. Cette intégration présente des défis uniques liés aux contraintes temporelles strictes (L1) et aux besoins de throughput élevé (HLT).

Cette section détaille les méthodes d'intégration de l'IA dans les triggers, les architectures de modèles utilisées, et les techniques d'optimisation pour respecter les contraintes hardware et temporelles.

---

## Stratégies d'Intégration

### Où Intégrer l'IA

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List

class IAIntegrationStrategy:
    """
    Stratégies d'intégration de l'IA dans les triggers
    """
    
    def __init__(self):
        self.integration_points = {
            'l1_object_classification': {
                'level': 'L1',
                'description': 'Classification d\'objets L1 (e/γ, μ, jets)',
                'latency_budget_ns': 100,
                'hardware': 'FPGA',
                'complexity': 'Faible (réseaux très petits)',
                'benefit': 'Améliore identification L1'
            },
            'l1_event_classification': {
                'level': 'L1',
                'description': 'Classification globale événement L1',
                'latency_budget_ns': 500,
                'hardware': 'FPGA',
                'complexity': 'Moyenne',
                'benefit': 'Décision L1 plus intelligente'
            },
            'hlt_feature_extraction': {
                'level': 'HLT',
                'description': 'Extraction de features avec ML',
                'latency_budget_ms': 50,
                'hardware': 'CPU/GPU',
                'complexity': 'Moyenne-élevée',
                'benefit': 'Features plus discriminantes'
            },
            'hlt_object_tagging': {
                'level': 'HLT',
                'description': 'Tagging d\'objets (b-tagging, etc.)',
                'latency_budget_ms': 30,
                'hardware': 'CPU/GPU',
                'complexity': 'Élevée',
                'benefit': 'Amélioration significative pureté'
            },
            'hlt_event_classification': {
                'level': 'HLT',
                'description': 'Classification globale événement HLT',
                'latency_budget_ms': 100,
                'hardware': 'CPU/GPU',
                'complexity': 'Très élevée',
                'benefit': 'Décision trigger optimale'
            }
        }
    
    def display_integration_points(self):
        """Affiche les points d'intégration"""
        print("\n" + "="*70)
        print("Points d'Intégration de l'IA dans les Triggers")
        print("="*70)
        
        for point, info in self.integration_points.items():
            print(f"\n{point.replace('_', ' ').title()}:")
            print(f"  Niveau: {info['level']}")
            print(f"  Description: {info['description']}")
            print(f"  Budget latence: {info['latency_budget_ns'] if 'ns' in info else info['latency_budget_ms']} {info['latency_budget_ns'] if 'ns' in info else 'ms'}")
            print(f"  Hardware: {info['hardware']}")
            print(f"  Complexité: {info['complexity']}")
            print(f"  Bénéfice: {info['benefit']}")

strategy = IAIntegrationStrategy()
strategy.display_integration_points()
```

---

## IA dans le Level-1 Trigger

### Architectures Optimisées pour FPGA

```python
class L1IAArchitectures:
    """
    Architectures IA pour L1 Trigger sur FPGA
    """
    
    def create_ultra_lightweight_model(self, input_dim=16, output_dim=1):
        """
        Modèle ultra-léger pour L1
        
        Contraintes:
        - < 500 paramètres
        - Latence < 100 ns
        - Ressources FPGA limitées
        """
        model = nn.Sequential(
            nn.Linear(input_dim, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, output_dim, bias=True),
            nn.Sigmoid()
        )
        
        return model
    
    def create_quantized_l1_model(self, input_dim=32, bitwidth=8):
        """
        Modèle quantifié pour L1
        
        Quantification 8-bit pour efficacité FPGA
        """
        # Créer modèle float32 d'abord
        base_model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Quantification (simulation)
        # En pratique: utiliser torch.quantization
        quantized_state = {}
        scales = {}
        
        for name, param in base_model.named_parameters():
            param_max = torch.max(torch.abs(param))
            scale = param_max / (2 ** (bitwidth - 1) - 1)
            
            quantized = torch.round(param / scale)
            quantized = torch.clamp(quantized, 
                                   -(2 ** (bitwidth - 1) - 1),
                                   2 ** (bitwidth - 1) - 1)
            
            quantized_state[name] = quantized.int()
            scales[name] = scale
        
        return {
            'model': base_model,
            'quantized_state': quantized_state,
            'scales': scales,
            'bitwidth': bitwidth
        }
    
    def create_tensor_network_l1(self, input_dim=16, bond_dim=4):
        """
        Réseau de tenseurs pour L1 (alternative aux réseaux classiques)
        
        Avantages:
        - Moins de paramètres
        - Structure exploitable sur FPGA
        """
        # Architecture Tensor Train simplifiée
        # En pratique: utiliser bibliothèque tensor networks
        
        # Approximation: modèle linéaire avec structure contrainte
        model = nn.Sequential(
            nn.Linear(input_dim, bond_dim, bias=False),
            nn.Linear(bond_dim, bond_dim, bias=False),
            nn.Linear(bond_dim, 1, bias=False)
        )
        
        # Constrainer pour simuler structure tensorielle
        # (Simplifié ici)
        
        return model

l1_ia = L1IAArchitectures()

# Créer modèles
ultra_light = l1_ia.create_ultra_lightweight_model()
quantized = l1_ia.create_quantized_l1_model()
tensor_net = l1_ia.create_tensor_network_l1()

print("\n" + "="*70)
print("Architectures IA pour L1")
print("="*70)

print(f"\nUltra-lightweight:")
print(f"  Paramètres: {sum(p.numel() for p in ultra_light.parameters())}")
print(f"  MACs: ~{ultra_light[0].in_features * ultra_light[0].out_features}")

print(f"\nQuantized (8-bit):")
print(f"  Paramètres: {sum(p.numel() for p in quantized['model'].parameters())}")
print(f"  Bits: {quantized['bitwidth']}")
```

---

## IA dans le High-Level Trigger

### Modèles Profonds pour Classification

```python
class HLTIAArchitectures:
    """
    Architectures IA pour HLT
    """
    
    def create_deep_classifier(self, input_dim=200, n_classes=10, depth=5):
        """
        Classificateur profond pour HLT
        
        Plus de temps disponible permet réseaux plus complexes
        """
        layers = []
        hidden_dim = 256
        
        # Couches d'encoder
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.3))
        
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Couche de sortie
        layers.append(nn.Linear(hidden_dim, n_classes))
        
        return nn.Sequential(*layers)
    
    def create_graph_neural_network(self, n_nodes=50, node_dim=10):
        """
        Graph Neural Network pour représentation événement
        
        Événement = graphe d'objets (jets, leptons, etc.)
        """
        # Simplification: représentation comme séquence
        # En pratique: utiliser bibliothèque GNN (PyTorch Geometric)
        
        model = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # Pooling global
            nn.Linear(32 * n_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Classification
        )
        
        return model
    
    def create_transformer_trigger(self, seq_len=100, d_model=64, n_heads=4):
        """
        Transformer pour séquence d'objets dans événement
        
        Permet attention sur objets importants
        """
        # Architecture simplifiée
        # En pratique: utiliser nn.TransformerEncoder
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1
        )
        
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Classifier final
        classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
        return {
            'transformer': transformer,
            'classifier': classifier,
            'input_projection': nn.Linear(10, d_model)  # Project input to d_model
        }
    
    def create_ensemble_model(self, models: List[nn.Module], weights: List[float] = None):
        """
        Ensemble de modèles pour amélioration robustesse
        """
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        def forward(x):
            outputs = []
            for model in models:
                with torch.no_grad():
                    output = model(x)
                    if output.dim() > 1:
                        output = torch.softmax(output, dim=1)
                    outputs.append(output)
            
            # Moyenne pondérée
            ensemble_output = sum(w * out for w, out in zip(weights, outputs))
            return ensemble_output
        
        return {'forward': forward, 'models': models, 'weights': weights}

hlt_ia = HLTIAArchitectures()

# Créer modèles
deep_classifier = hlt_ia.create_deep_classifier(depth=5)
gnn_model = hlt_ia.create_graph_neural_network()
transformer_model = hlt_ia.create_transformer_trigger()

print("\n" + "="*70)
print("Architectures IA pour HLT")
print("="*70)

print(f"\nDeep Classifier:")
print(f"  Paramètres: {sum(p.numel() for p in deep_classifier.parameters()):,}")

print(f"\nTransformer:")
transformer_params = (sum(p.numel() for p in transformer_model['transformer'].parameters()) +
                     sum(p.numel() for p in transformer_model['classifier'].parameters()) +
                     sum(p.numel() for p in transformer_model['input_projection'].parameters()))
print(f"  Paramètres: {transformer_params:,}")
```

---

## Techniques d'Optimisation

### Compression et Accélération

```python
class IAOptimization:
    """
    Techniques d'optimisation pour IA dans triggers
    """
    
    @staticmethod
    def knowledge_distillation(teacher_model: nn.Module, 
                              student_model: nn.Module,
                              temperature: float = 3.0):
        """
        Knowledge Distillation: comprimer modèle complexe en modèle simple
        
        Teacher: modèle complexe (HLT)
        Student: modèle simple (L1)
        """
        def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5):
            """Loss de distillation"""
            # Soft targets (teacher)
            soft_targets = torch.softmax(teacher_logits / temperature, dim=1)
            soft_prob = torch.log_softmax(student_logits / temperature, dim=1)
            
            soft_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size(0)
            
            # Hard targets (labels)
            hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
            
            # Combinaison
            return alpha * soft_loss * (temperature ** 2) + (1 - alpha) * hard_loss
        
        return distillation_loss
    
    @staticmethod
    def pruning_structured(model: nn.Module, sparsity: float = 0.5):
        """
        Pruning structuré: retire entières couches ou channels
        """
        # Pruning de channels (plus adapté pour hardware)
        pruned_model = model
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Calculer importance des channels/neurones
                if hasattr(module, 'weight'):
                    importance = torch.mean(torch.abs(module.weight), dim=1)
                    threshold = torch.quantile(importance, sparsity)
                    
                    # Mask
                    mask = importance > threshold
                    # En pratique: restructurer modèle avec seulement channels importants
        
        return pruned_model
    
    @staticmethod
    def quantization_aware_training(model: nn.Module):
        """
        Quantization-Aware Training (QAT)
        
        Entraîne modèle en simulant quantification pour meilleure adaptation
        """
        # En pratique: utiliser torch.quantization.quantize_qat
        # Ici: simulation
        
        def quantize_weights(weights, bitwidth=8):
            """Simule quantification"""
            scale = torch.max(torch.abs(weights)) / (2 ** (bitwidth - 1) - 1)
            quantized = torch.round(weights / scale)
            quantized = torch.clamp(quantized, 
                                   -(2 ** (bitwidth - 1) - 1),
                                   2 ** (bitwidth - 1) - 1)
            return quantized * scale
        
        # Appliquer à tous les poids
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.data = quantize_weights(param.data)
        
        return model
    
    @staticmethod
    def model_compression_pipeline(model: nn.Module, target_size_mb: float):
        """
        Pipeline complet de compression
        
        1. Pruning
        2. Quantization
        3. Distillation (si nécessaire)
        """
        # Étape 1: Pruning
        pruned = IAOptimization.pruning_structured(model, sparsity=0.5)
        
        # Étape 2: Quantization
        quantized = IAOptimization.quantization_aware_training(pruned)
        
        # Vérifier taille
        model_size_mb = sum(p.numel() * 4 for p in quantized.parameters()) / (1024 ** 2)
        
        return {
            'model': quantized,
            'original_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2),
            'compressed_size_mb': model_size_mb,
            'compression_ratio': model_size_mb / (sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2))
        }

opt = IAOptimization()
```

---

## Déploiement et Intégration

### Intégration dans le Flux de Traitement

```python
class IADeployment:
    """
    Déploiement de modèles IA dans le flux trigger
    """
    
    def __init__(self):
        self.deployment_config = {
            'l1_models': {
                'format': 'FPGA bitstream',
                'framework': 'hls4ml',
                'update_frequency': 'Rare (firmware update)'
            },
            'hlt_models': {
                'format': 'ONNX ou PyTorch',
                'framework': 'ONNX Runtime ou PyTorch',
                'update_frequency': 'Frequent (software update)'
            }
        }
    
    def deploy_l1_model(self, model: nn.Module, target_fpga: str = 'UltraScale+'):
        """
        Déploie modèle L1 sur FPGA
        
        Utilise hls4ml ou similar
        """
        # En pratique: utiliser hls4ml.convert_from_pytorch_model
        deployment = {
            'model': model,
            'target_fpga': target_fpga,
            'bitstream_generated': False,
            'resources': {
                'lut': 0,
                'dsp': 0,
                'bram': 0
            }
        }
        
        return deployment
    
    def deploy_hlt_model(self, model: nn.Module, backend: str = 'onnx'):
        """
        Déploie modèle HLT sur CPU/GPU farm
        """
        if backend == 'onnx':
            # Convertir en ONNX
            # torch.onnx.export(model, ...)
            deployment = {
                'format': 'ONNX',
                'optimized': True,
                'device': 'CPU/GPU'
            }
        else:
            deployment = {
                'format': 'PyTorch',
                'jit_compiled': False,
                'device': 'GPU'
            }
        
        return deployment
    
    def model_versioning(self):
        """
        Système de versioning pour modèles
        
        Permet rollback si nouveau modèle dégrade performances
        """
        return {
            'version_1': {'model_path': 'models/v1.pt', 'deployed': False},
            'version_2': {'model_path': 'models/v2.pt', 'deployed': True},
            'version_3': {'model_path': 'models/v3.pt', 'deployed': False}
        }
    
    def monitoring_performance(self, model_id: str):
        """
        Monitoring des performances du modèle en production
        """
        metrics = {
            'inference_time_ms': 25.3,
            'throughput_events_per_sec': 4000,
            'efficiency_signal': 0.96,
            'background_rejection': 0.98,
            'false_positive_rate': 0.02
        }
        
        return metrics

deployment = IADeployment()
```

---

## Exercices

### Exercice 18.4.1
Concevez un modèle IA pour identification d'électrons au L1 avec < 300 paramètres et latence < 80 ns.

### Exercice 18.4.2
Implémentez un pipeline de compression (pruning + quantization) qui réduit un modèle HLT de 50 MB à < 5 MB.

### Exercice 18.4.3
Créez un système de knowledge distillation qui transfère connaissances d'un modèle HLT complexe vers un modèle L1 simple.

### Exercice 18.4.4
Analysez le trade-off entre complexité du modèle et efficacité de sélection pour différents budgets de latence.

---

## Points Clés à Retenir

> 📌 **L'IA dans L1 nécessite modèles ultra-légers avec quantification agressive**

> 📌 **L'IA dans HLT permet modèles plus complexes (réseaux profonds, transformers)**

> 📌 **La compression (pruning, quantization, distillation) est essentielle pour L1**

> 📌 **Le déploiement L1 nécessite conversion vers firmware FPGA (hls4ml)**

> 📌 **Le déploiement HLT peut utiliser ONNX ou PyTorch directement**

> 📌 **Le monitoring et versioning sont cruciaux pour modèles en production**

---

*Section précédente : [18.3 High-Level Trigger](./18_03_HLT.md) | Section suivante : [18.5 Performance](./18_05_Performance.md)*

