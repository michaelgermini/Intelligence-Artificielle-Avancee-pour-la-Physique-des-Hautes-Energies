# 25.1 Workflow de Compression Typique

---

## Introduction

Un **workflow de compression** bien structuré est essentiel pour appliquer efficacement des techniques de compression tout en maintenant la qualité du modèle. Cette section présente un workflow typique de compression, de la préparation des données au déploiement.

---

## Structure du Workflow

### Pipeline Complet

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class CompressionWorkflow:
    """
    Workflow de compression typique
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.baseline_metrics = {}
        self.compressed_models = {}
        self.compression_configs = []
    
    def run_workflow(self, compression_methods: List[str], 
                     target_compression_ratios: List[float]):
        """
        Exécute workflow complet
        
        Args:
            compression_methods: Liste méthodes ['pruning', 'quantization', etc.]
            target_compression_ratios: Ratios cibles pour chaque méthode
        """
        # 1. Évaluer baseline
        print("Step 1: Evaluating baseline...")
        self.baseline_metrics = self.evaluate_model(self.model, self.test_loader)
        
        # 2. Compression
        print("Step 2: Applying compression...")
        for method, ratio in zip(compression_methods, target_compression_ratios):
            compressed_model = self.apply_compression(method, ratio)
            self.compressed_models[method] = compressed_model
        
        # 3. Fine-tuning
        print("Step 3: Fine-tuning compressed models...")
        for method, model in self.compressed_models.items():
            self.fine_tune_model(model, epochs=10)
        
        # 4. Évaluation
        print("Step 4: Evaluating compressed models...")
        results = {}
        for method, model in self.compressed_models.items():
            metrics = self.evaluate_model(model, self.test_loader)
            results[method] = metrics
        
        # 5. Comparaison
        print("Step 5: Comparing results...")
        self.compare_results(self.baseline_metrics, results)
        
        return results
    
    def evaluate_model(self, model, dataloader):
        """Évalue modèle et retourne métriques"""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'params': sum(p.numel() for p in model.parameters())
        }
    
    def apply_compression(self, method: str, ratio: float):
        """Applique méthode de compression"""
        compressed_model = self.model
        
        if method == 'pruning':
            compressed_model = self.apply_pruning(ratio)
        elif method == 'quantization':
            compressed_model = self.apply_quantization()
        elif method == 'distillation':
            compressed_model = self.apply_distillation(ratio)
        
        return compressed_model
    
    def apply_pruning(self, sparsity: float):
        """Applique pruning"""
        from torch.nn.utils import prune
        
        model = self.model
        
        # Global pruning
        parameters_to_prune = [
            (module, 'weight') for module in model.modules()
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d)
        ]
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
        
        # Retirer masques pour créer modèle permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def apply_quantization(self):
        """Applique quantization"""
        model = self.model
        
        # Quantification dynamique
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        return quantized_model
    
    def fine_tune_model(self, model, epochs=10, lr=0.001):
        """Fine-tune modèle compressé"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(epochs):
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
    
    def compare_results(self, baseline: Dict, results: Dict):
        """Compare résultats baseline vs compressés"""
        print("\n" + "="*70)
        print("Compression Results Comparison")
        print("="*70)
        print(f"\nBaseline:")
        print(f"  Accuracy: {baseline['accuracy']:.2f}%")
        print(f"  Parameters: {baseline['params']:,}")
        
        print(f"\nCompressed Models:")
        for method, metrics in results.items():
            accuracy_drop = baseline['accuracy'] - metrics['accuracy']
            compression_ratio = baseline['params'] / metrics['params']
            
            print(f"\n{method}:")
            print(f"  Accuracy: {metrics['accuracy']:.2f}% (drop: {accuracy_drop:.2f}%)")
            print(f"  Parameters: {metrics['params']:,}")
            print(f"  Compression: {compression_ratio:.2f}×")
```

---

## Étapes Détaillées

### 1. Préparation et Baseline

```python
class WorkflowStep1_Preparation:
    """
    Étape 1: Préparation et baseline
    """
    
    def prepare_data(self, dataset, batch_size=64, val_split=0.2, test_split=0.1):
        """Prépare données avec splits"""
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - test_size - val_size
        
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_baseline(self, model, train_loader, val_loader, epochs=50):
        """Entraîne modèle baseline"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            model.train()
            for data, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_baseline.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.2f}%")
        
        return best_val_acc
```

---

## Orchestration avec Configuration

### Workflow avec Config File

```python
import yaml

class ConfigurableWorkflow(CompressionWorkflow):
    """
    Workflow configurable via fichier YAML
    """
    
    def __init__(self, config_path: str):
        """Charge configuration depuis fichier"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialiser composants depuis config
        self.model = self.load_model(self.config['model'])
        self.train_loader, self.val_loader, self.test_loader = \
            self.load_data(self.config['data'])
    
    def load_model(self, model_config):
        """Charge modèle depuis config"""
        model_type = model_config['type']
        model_params = model_config['params']
        
        if model_type == 'resnet18':
            from torchvision.models import resnet18
            model = resnet18(**model_params)
        # ... autres modèles
        
        return model
    
    def load_data(self, data_config):
        """Charge données depuis config"""
        dataset_path = data_config['path']
        batch_size = data_config['batch_size']
        
        # Charger dataset
        # ...
        
        return train_loader, val_loader, test_loader
    
    def run_from_config(self):
        """Exécute workflow depuis configuration"""
        compression_configs = self.config['compression']
        
        results = {}
        for comp_config in compression_configs:
            method = comp_config['method']
            params = comp_config['params']
            
            compressed_model = self.apply_compression_with_params(method, params)
            self.fine_tune_with_config(compressed_model, comp_config.get('finetuning', {}))
            
            metrics = self.evaluate_model(compressed_model, self.test_loader)
            results[method] = metrics
        
        return results

# Exemple config.yaml:
"""
model:
  type: resnet18
  params:
    num_classes: 10

data:
  path: ./data
  batch_size: 64

compression:
  - method: pruning
    params:
      sparsity: 0.5
    finetuning:
      epochs: 10
      lr: 0.001
  
  - method: quantization
    params:
      dtype: qint8
"""
```

---

## Exercices

### Exercice 25.1.1
Créez un workflow complet qui applique pruning, quantization, et distillation séquentiellement.

### Exercice 25.1.2
Implémentez un système de configuration YAML pour workflow de compression.

### Exercice 25.1.3
Ajoutez logging détaillé au workflow pour tracker progression et métriques.

### Exercice 25.1.4
Créez un workflow qui teste différentes combinaisons de méthodes de compression.

---

## Points Clés à Retenir

> 📌 **Workflow structuré facilite reproduction et automatisation**

> 📌 **Baseline est essentiel pour comparer résultats**

> 📌 **Configuration externe permet expériences sans modifier code**

> 📌 **Fine-tuning post-compression est crucial**

> 📌 **Comparaison systématique permet choisir meilleure méthode**

---

*Section précédente : [25.0 Introduction](./25_introduction.md) | Section suivante : [25.2 Hyperparamètres](./25_02_Hyperparametres.md)*

