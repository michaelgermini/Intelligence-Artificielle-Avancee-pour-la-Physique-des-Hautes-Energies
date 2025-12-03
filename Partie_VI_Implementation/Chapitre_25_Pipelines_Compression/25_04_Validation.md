# 25.4 Validation et Tests de Régression

---

## Introduction

La **validation rigoureuse** est essentielle pour garantir que les modèles compressés maintiennent leur qualité avant déploiement. Cette section présente les méthodes de validation, tests de régression, et vérification de contraintes pour modèles compressés.

---

## Métriques de Validation

### Évaluation Complète

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report

class ModelValidator:
    """
    Validation complète de modèles compressés
    """
    
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def comprehensive_validation(self) -> Dict:
        """
        Validation complète avec multiples métriques
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = self.compute_accuracy()
        
        # Loss
        metrics['loss'] = self.compute_loss()
        
        # Per-class metrics
        metrics['per_class'] = self.compute_per_class_metrics()
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.compute_confusion_matrix()
        
        # Latency
        metrics['latency'] = self.measure_latency()
        
        # Memory
        metrics['memory'] = self.measure_memory()
        
        return metrics
    
    def compute_accuracy(self) -> float:
        """Calcule accuracy globale"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100 * correct / total
    
    def compute_loss(self) -> float:
        """Calcule loss moyenne"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def compute_per_class_metrics(self) -> Dict:
        """Calcule métriques par classe"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Classification report
        report = classification_report(all_targets, all_preds, output_dict=True)
        
        return report
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Calcule matrice de confusion"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return confusion_matrix(all_targets, all_preds)
    
    def measure_latency(self, n_runs=100) -> Dict:
        """Mesure latence inférence"""
        self.model.eval()
        
        # Warmup
        dummy_input = next(iter(self.test_loader))[0][:1]
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Mesure
        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start.record()
                    _ = self.model(dummy_input)
                    end.record()
                    torch.cuda.synchronize()
                    latency = start.elapsed_time(end)  # milliseconds
                else:
                    import time
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    latency = (time.time() - start_time) * 1000  # milliseconds
                
                latencies.append(latency)
        
        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99)
        }
    
    def measure_memory(self) -> Dict:
        """Mesure utilisation mémoire"""
        import sys
        
        # Paramètres
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        
        total_size = param_size + buffer_size
        
        return {
            'parameters_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2),
            'total_mb': total_size / (1024 ** 2),
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
```

---

## Tests de Régression

### Comparaison avec Baseline

```python
class RegressionTester:
    """
    Tests de régression pour modèles compressés
    """
    
    def __init__(self, baseline_model, compressed_model, test_loader):
        self.baseline = baseline_model
        self.compressed = compressed_model
        self.test_loader = test_loader
    
    def regression_test(self, tolerance: float = 0.05) -> Dict:
        """
        Test de régression
        
        Args:
            tolerance: Tolérance pour dégradation (5% par défaut)
        """
        baseline_validator = ModelValidator(self.baseline, self.test_loader)
        compressed_validator = ModelValidator(self.compressed, self.test_loader)
        
        baseline_metrics = baseline_validator.comprehensive_validation()
        compressed_metrics = compressed_validator.comprehensive_validation()
        
        # Comparer métriques
        results = {
            'baseline': baseline_metrics,
            'compressed': compressed_metrics,
            'regression': {}
        }
        
        # Accuracy regression
        acc_drop = baseline_metrics['accuracy'] - compressed_metrics['accuracy']
        acc_drop_pct = acc_drop / baseline_metrics['accuracy'] if baseline_metrics['accuracy'] > 0 else 0
        
        results['regression']['accuracy_drop'] = acc_drop
        results['regression']['accuracy_drop_pct'] = acc_drop_pct * 100
        results['regression']['accuracy_regression'] = acc_drop_pct > tolerance
        
        # Loss regression
        loss_increase = compressed_metrics['loss'] - baseline_metrics['loss']
        loss_increase_pct = loss_increase / baseline_metrics['loss'] if baseline_metrics['loss'] > 0 else 0
        
        results['regression']['loss_increase'] = loss_increase
        results['regression']['loss_increase_pct'] = loss_increase_pct * 100
        
        # Speedup
        speedup = baseline_metrics['latency']['mean'] / compressed_metrics['latency']['mean']
        results['regression']['speedup'] = speedup
        
        # Compression ratio
        compression_ratio = (baseline_metrics['memory']['num_parameters'] / 
                           compressed_metrics['memory']['num_parameters'])
        results['regression']['compression_ratio'] = compression_ratio
        
        # Overall pass/fail
        results['regression']['passed'] = (
            not results['regression']['accuracy_regression'] and
            speedup > 1.0 and
            compression_ratio > 1.0
        )
        
        return results
    
    def print_regression_report(self, results: Dict):
        """Affiche rapport de régression"""
        print("\n" + "="*70)
        print("Regression Test Report")
        print("="*70)
        
        print(f"\nAccuracy:")
        print(f"  Baseline: {results['baseline']['accuracy']:.2f}%")
        print(f"  Compressed: {results['compressed']['accuracy']:.2f}%")
        print(f"  Drop: {results['regression']['accuracy_drop']:.2f}% "
              f"({results['regression']['accuracy_drop_pct']:.2f}%)")
        
        print(f"\nLatency:")
        print(f"  Baseline: {results['baseline']['latency']['mean']:.2f} ms")
        print(f"  Compressed: {results['compressed']['latency']['mean']:.2f} ms")
        print(f"  Speedup: {results['regression']['speedup']:.2f}×")
        
        print(f"\nMemory:")
        print(f"  Baseline: {results['baseline']['memory']['total_mb']:.2f} MB")
        print(f"  Compressed: {results['compressed']['memory']['total_mb']:.2f} MB")
        print(f"  Compression: {results['regression']['compression_ratio']:.2f}×")
        
        print(f"\nOverall: {'PASSED' if results['regression']['passed'] else 'FAILED'}")
```

---

## Tests de Robustesse

### Validation sur Distributions

```python
class RobustnessTester:
    """
    Tests de robustesse pour modèles compressés
    """
    
    def __init__(self, model, test_loaders: Dict[str, torch.utils.data.DataLoader]):
        self.model = model
        self.test_loaders = test_loaders  # {'clean': loader, 'noisy': loader, etc.}
    
    def robustness_test(self) -> Dict:
        """Test robustesse sur différentes distributions"""
        results = {}
        
        for distribution_name, loader in self.test_loaders.items():
            validator = ModelValidator(self.model, loader)
            metrics = validator.comprehensive_validation()
            results[distribution_name] = metrics
        
        return results
    
    def compare_robustness(self, baseline_results: Dict, 
                          compressed_results: Dict) -> Dict:
        """Compare robustesse baseline vs compressé"""
        comparison = {}
        
        for dist_name in baseline_results.keys():
            baseline_acc = baseline_results[dist_name]['accuracy']
            compressed_acc = compressed_results[dist_name]['accuracy']
            
            drop = baseline_acc - compressed_acc
            drop_pct = (drop / baseline_acc * 100) if baseline_acc > 0 else 0
            
            comparison[dist_name] = {
                'baseline_accuracy': baseline_acc,
                'compressed_accuracy': compressed_acc,
                'drop': drop,
                'drop_pct': drop_pct
            }
        
        return comparison
```

---

## Exercices

### Exercice 25.4.1
Créez un système de validation complet qui calcule toutes métriques importantes.

### Exercice 25.4.2
Implémentez tests de régression qui comparent baseline vs modèles compressés.

### Exercice 25.4.3
Testez robustesse de modèles compressés sur données avec bruit ou distributions différentes.

### Exercice 25.4.4
Créez suite de tests automatisés qui vérifie contraintes avant déploiement.

---

## Points Clés à Retenir

> 📌 **Validation complète inclut métriques multiples (accuracy, latency, memory)**

> 📌 **Tests de régression garantissent pas de dégradation excessive**

> 📌 **Mesure de latence et mémoire est cruciale pour déploiement**

> 📌 **Tests de robustesse vérifient performance sur différentes distributions**

> 📌 **Automation des tests permet intégration continue**

---

*Section précédente : [25.3 Fine-tuning](./25_03_Finetuning.md) | Section suivante : [25.5 Déploiement](./25_05_Deploiement.md)*

