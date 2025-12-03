# 25.5 Déploiement et Monitoring

---

## Introduction

Le **déploiement** de modèles compressés en production nécessite des considérations spécifiques. Cette section présente les méthodes d'export, les formats de modèles, l'intégration dans systèmes, et le monitoring en production pour détecter dégradation et problèmes.

---

## Export de Modèles

### Formats d'Export

```python
import torch
import torch.onnx
import json

class ModelExporter:
    """
    Export de modèles compressés pour déploiement
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def export_pytorch(self, path: str):
        """Export modèle PyTorch standard"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': type(self.model).__name__
        }, path)
        print(f"Model saved to {path}")
    
    def export_onnx(self, dummy_input, path: str, opset_version=11):
        """
        Export vers ONNX
        
        Args:
            dummy_input: Exemple d'input pour tracer modèle
            path: Chemin fichier ONNX
            opset_version: Version opset ONNX
        """
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model exported to {path}")
    
    def export_torchscript(self, dummy_input, path: str, method='trace'):
        """
        Export vers TorchScript
        
        Args:
            dummy_input: Exemple input
            path: Chemin fichier
            method: 'trace' ou 'script'
        """
        if method == 'trace':
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(path)
        elif method == 'script':
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(path)
        
        print(f"TorchScript model exported to {path}")
    
    def export_metadata(self, path: str, metadata: Dict):
        """Export métadonnées du modèle"""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
```

---

## Quantification pour Déploiement

### Post-Training Quantization

```python
class QuantizedExporter:
    """
    Export avec quantification pour déploiement
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def export_quantized_pytorch(self, calibration_loader, path: str):
        """
        Quantifie et exporte modèle PyTorch
        
        Args:
            calibration_loader: DataLoader pour calibration
            path: Chemin fichier
        """
        # Calibration
        self.model.eval()
        with torch.no_grad():
            for data, _ in calibration_loader:
                _ = self.model(data)
        
        # Quantification dynamique
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        # Sauvegarder
        torch.save(quantized_model.state_dict(), path)
        print(f"Quantized model saved to {path}")
    
    def export_quantized_onnx(self, dummy_input, path: str):
        """Export ONNX avec quantification"""
        # Quantifier
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        
        # Export ONNX
        # Note: ONNX support quantification via QONNX extension
        torch.onnx.export(
            quantized_model,
            dummy_input,
            path,
            opset_version=13  # Support quantization
        )
```

---

## Monitoring en Production

### Système de Monitoring

```python
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List

class ProductionMonitor:
    """
    Monitoring de modèles en production
    """
    
    def __init__(self, model_name: str, window_size: int = 1000):
        self.model_name = model_name
        self.window_size = window_size
        
        # Métriques
        self.latency_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        self.accuracy_tracking = []  # Si ground truth disponible
        
        # Compteurs
        self.request_count = 0
        self.error_count = 0
        
        # Alerts
        self.alerts = []
        
        # Logging
        logging.basicConfig(
            filename=f'{model_name}_monitoring.log',
            level=logging.INFO
        )
        self.logger = logging.getLogger(model_name)
    
    def log_inference(self, latency_ms: float, batch_size: int = 1):
        """Log inférence"""
        self.request_count += 1
        
        # Latence
        self.latency_history.append(latency_ms)
        
        # Throughput (requêtes/seconde)
        throughput = batch_size / (latency_ms / 1000)
        self.throughput_history.append(throughput)
        
        # Vérifier seuils
        self._check_thresholds()
    
    def log_error(self, error_type: str, error_message: str):
        """Log erreur"""
        self.error_count += 1
        self.error_history.append({
            'type': error_type,
            'message': error_message,
            'timestamp': time.time()
        })
        
        self.logger.error(f"Error: {error_type} - {error_message}")
    
    def log_prediction(self, prediction, ground_truth=None):
        """Log prédiction (si ground truth disponible)"""
        if ground_truth is not None:
            correct = (prediction == ground_truth)
            self.accuracy_tracking.append(correct)
            
            # Garder seulement récent
            if len(self.accuracy_tracking) > self.window_size:
                self.accuracy_tracking = self.accuracy_tracking[-self.window_size:]
    
    def _check_thresholds(self):
        """Vérifie seuils d'alerte"""
        if len(self.latency_history) < 100:
            return
        
        # Latence moyenne récente
        recent_latency = np.mean(list(self.latency_history)[-100:])
        
        # Alerte si latence > 2x moyenne
        avg_latency = np.mean(self.latency_history)
        if recent_latency > 2 * avg_latency:
            self.alert("high_latency", 
                      f"Recent latency {recent_latency:.2f}ms > 2x average {avg_latency:.2f}ms")
        
        # Taux d'erreur
        error_rate = self.error_count / max(self.request_count, 1)
        if error_rate > 0.05:  # 5%
            self.alert("high_error_rate", f"Error rate: {error_rate:.2%}")
    
    def alert(self, alert_type: str, message: str):
        """Génère alerte"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time()
        }
        self.alerts.append(alert)
        self.logger.warning(f"ALERT: {alert_type} - {message}")
    
    def get_metrics(self) -> Dict:
        """Retourne métriques actuelles"""
        metrics = {
            'model_name': self.model_name,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
        }
        
        if self.latency_history:
            metrics['latency'] = {
                'mean': np.mean(self.latency_history),
                'p50': np.percentile(self.latency_history, 50),
                'p95': np.percentile(self.latency_history, 95),
                'p99': np.percentile(self.latency_history, 99),
                'max': np.max(self.latency_history)
            }
        
        if self.throughput_history:
            metrics['throughput'] = {
                'mean': np.mean(self.throughput_history),
                'current': self.throughput_history[-1] if self.throughput_history else 0
            }
        
        if self.accuracy_tracking:
            metrics['accuracy'] = np.mean(self.accuracy_tracking)
        
        return metrics
    
    def get_report(self) -> str:
        """Génère rapport"""
        metrics = self.get_metrics()
        
        report = f"\n{'='*70}\n"
        report += f"Monitoring Report: {self.model_name}\n"
        report += f"{'='*70}\n"
        report += f"Requests: {metrics['request_count']:,}\n"
        report += f"Errors: {metrics['error_count']} ({metrics['error_rate']:.2%})\n"
        
        if 'latency' in metrics:
            report += f"\nLatency (ms):\n"
            report += f"  Mean: {metrics['latency']['mean']:.2f}\n"
            report += f"  P95: {metrics['latency']['p95']:.2f}\n"
            report += f"  P99: {metrics['latency']['p99']:.2f}\n"
        
        if 'throughput' in metrics:
            report += f"\nThroughput: {metrics['throughput']['mean']:.2f} req/s\n"
        
        if 'accuracy' in metrics:
            report += f"Accuracy: {metrics['accuracy']:.2%}\n"
        
        if self.alerts:
            report += f"\nAlerts: {len(self.alerts)}\n"
            for alert in self.alerts[-5:]:  # 5 derniers
                report += f"  - {alert['type']}: {alert['message']}\n"
        
        return report
```

---

## Déploiement sur Serveur

### API REST

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

class ModelServer:
    """
    Serveur pour modèles compressés
    """
    
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        self.monitor = ProductionMonitor("production_model")
    
    def load_model(self, path: str):
        """Charge modèle"""
        model = torch.load(path, map_location=self.device)
        return model
    
    def predict(self, input_data):
        """Prédiction"""
        start_time = time.time()
        
        try:
            # Preprocessing
            tensor_input = torch.tensor(input_data).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(tensor_input)
                predictions = torch.softmax(output, dim=1)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Logging
            self.monitor.log_inference(latency_ms, batch_size=len(input_data))
            
            return {
                'predictions': predictions.cpu().numpy().tolist(),
                'latency_ms': latency_ms
            }
        
        except Exception as e:
            self.monitor.log_error("prediction_error", str(e))
            raise

# Initialiser serveur
server = ModelServer('compressed_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint prédiction"""
    data = request.json
    input_data = data['input']
    
    try:
        result = server.predict(input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint métriques"""
    return jsonify(server.monitor.get_metrics())

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Exercices

### Exercice 25.5.1
Exportez un modèle compressé vers ONNX et testez inférence.

### Exercice 25.5.2
Créez un système de monitoring qui track latence et erreurs.

### Exercice 25.5.3
Implémentez un serveur API REST pour déployer modèle compressé.

### Exercice 25.5.4
Ajoutez alertes automatiques pour détecter dégradation performance.

---

## Points Clés à Retenir

> 📌 **Export ONNX/TorchScript facilite déploiement cross-platform**

> 📌 **Quantification réduit taille et accélère inférence**

> 📌 **Monitoring est essentiel pour détecter problèmes production**

> 📌 **Alertes automatiques permettent réponse rapide**

> 📌 **Métriques (latency, throughput, accuracy) guident optimisation**

> 📌 **Health checks permettent vérifier disponibilité service**

---

*Section précédente : [25.4 Validation](./25_04_Validation.md)*

