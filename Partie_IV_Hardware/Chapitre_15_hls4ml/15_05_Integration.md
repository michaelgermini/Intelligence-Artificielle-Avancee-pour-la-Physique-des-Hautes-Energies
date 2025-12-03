# 15.5 Intégration avec les Workflows de Physique

---

## Introduction

L'**intégration de hls4ml** dans les workflows de physique des particules nécessite de prendre en compte les spécificités des expériences HEP : contraintes de trigger, formats de données, et chaînes de traitement.

---

## Architecture d'Intégration dans les Triggers

### Workflow Type

```
┌─────────────────────────────────────────────────────────────────┐
│          Intégration hls4ml dans Trigger L1                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Détecteur                                                      │
│     │                                                           │
│     ▼                                                           │
│  Frontend Electronics                                          │
│     │                                                           │
│     ▼                                                           │
│  ┌────────────────────┐                                        │
│  │  Preprocessing     │                                        │
│  │  (Feature Extr.)   │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  ML Model (FPGA)   │ ◀── hls4ml generated                  │
│  │  hls4ml inference  │                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  ┌────────────────────┐                                        │
│  │  Decision Logic    │                                        │
│  │  (Trigger Decision)│                                        │
│  └─────────┬──────────┘                                        │
│            │                                                    │
│            ▼                                                    │
│  Data Acquisition System                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Formats de Données HEP

### Conversion des Données

```python
class HEPDataIntegration:
    """
    Intégration avec formats de données HEP
    """
    
    def __init__(self):
        self.data_formats = {
            'raw_hits': {
                'description': 'Données brutes du détecteur',
                'format': 'Bits, ADC counts',
                'processing': 'Requiert preprocessing avant ML'
            },
            'clusters': {
                'description': 'Clusters de hits',
                'format': 'Positions, énergies',
                'processing': 'Features extraites'
            },
            'tracks': {
                'description': 'Traces de particules',
                'format': 'Paramètres de trajectoire',
                'processing': 'Features géométriques'
            },
            'jets': {
                'description': 'Jets reconstruits',
                'format': 'Variables de jet (pT, η, φ, etc.)',
                'processing': 'Features physiques'
            }
        }
    
    def convert_to_ml_features(self, hep_data, feature_extractor):
        """
        Convertit données HEP en features ML
        
        Args:
            hep_data: Données au format HEP
            feature_extractor: Fonction d'extraction
        """
        features = feature_extractor(hep_data)
        
        # Normalisation pour hls4ml (fixed-point)
        features_normalized = self.normalize_features(features)
        
        return features_normalized
    
    def normalize_features(self, features, target_range=(-1, 1)):
        """
        Normalise les features pour fixed-point
        
        Args:
            features: Array de features
            target_range: Plage cible (ex: (-1, 1))
        """
        import numpy as np
        
        # Normalisation min-max
        min_val, max_val = target_range
        features_min = features.min(axis=0, keepdims=True)
        features_max = features.max(axis=0, keepdims=True)
        
        features_norm = (features - features_min) / (features_max - features_min)
        features_norm = features_norm * (max_val - min_val) + min_val
        
        return features_norm

hep_integration = HEPDataIntegration()

print("\n" + "="*60)
print("HEP Data Formats")
print("="*60)

for fmt, info in hep_integration.data_formats.items():
    print(f"\n{fmt}:")
    print(f"  Description: {info['description']}")
    print(f"  Format: {info['format']}")
    print(f"  Processing: {info['processing']}")
```

---

## Interface avec les Systèmes de Trigger

### Interface FPGA ↔ Système Trigger

```python
class TriggerSystemInterface:
    """
    Interface avec système de trigger
    """
    
    def __init__(self):
        self.interface_requirements = {
            'latency': {
                'constraint': '< 4 μs (L1 trigger)',
                'measurement': 'End-to-end depuis détecteur',
                'breakdown': {
                    'preprocessing': '~1 μs',
                    'ml_inference': '~2 μs',
                    'decision_logic': '~0.5 μs',
                    'margin': '~0.5 μs'
                }
            },
            'throughput': {
                'constraint': '40 MHz (1 événement / 25 ns)',
                'requirement': 'Pipeline avec II=1 idéalement'
            },
            'data_format': {
                'input': 'Raw detector data ou features préprocessées',
                'output': 'Trigger decision (bit) ou score',
                'interface': 'AXI Stream typiquement'
            }
        }
    
    def design_interface(self, hls_model, input_format, output_format):
        """
        Conçoit l'interface FPGA
        
        Args:
            hls_model: Modèle hls4ml
            input_format: Format d'entrée
            output_format: Format de sortie
        """
        interface_config = {
            'input': {
                'protocol': 'AXI Stream',
                'width': self._calculate_width(input_format),
                'latency_mode': 'pipeline'
            },
            'output': {
                'protocol': 'AXI Stream',
                'width': self._calculate_width(output_format),
                'decision_threshold': 0.5  # Exemple
            },
            'control': {
                'reset': 'Active low',
                'clock': '200 MHz (exemple)'
            }
        }
        
        return interface_config
    
    def _calculate_width(self, data_format):
        """Calcule la largeur du bus"""
        # Simplifié: dépend du format
        if isinstance(data_format, int):
            return data_format * 8  # bits
        return 64  # Défaut

trigger_interface = TriggerSystemInterface()

print("\n" + "="*60)
print("Trigger System Interface Requirements")
print("="*60)

for req, details in trigger_interface.interface_requirements.items():
    print(f"\n{req.replace('_', ' ').title()}:")
    if isinstance(details, dict):
        for key, value in details.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {details}")
```

---

## Validation dans le Contexte HEP

### Métriques de Validation

```python
class HEPValidation:
    """
    Validation spécifique pour applications HEP
    """
    
    def __init__(self):
        self.validation_metrics = {
            'accuracy': {
                'description': 'Précision de classification',
                'target': '> 95% typiquement',
                'measurement': 'Sur dataset de validation HEP'
            },
            'efficiency': {
                'description': 'Efficacité de détection',
                'example': 'Efficacité de b-tagging',
                'target': 'Dépend de l\'application'
            },
            'background_rejection': {
                'description': 'Rejet de bruit de fond',
                'importance': 'Critique pour triggers',
                'target': 'Aussi élevé que possible'
            },
            'latency': {
                'description': 'Latence end-to-end',
                'constraint': '< 4 μs pour L1',
                'measurement': 'Sur FPGA réel'
            },
            'throughput': {
                'description': 'Throughput soutenu',
                'requirement': '40 MHz sans perte',
                'measurement': 'Test continu'
            }
        }
    
    def validate_ml_model(self, keras_model, hls_model, hep_test_data):
        """
        Valide le modèle dans contexte HEP
        
        Args:
            keras_model: Modèle Keras original
            hls_model: Modèle hls4ml
            hep_test_data: Données de test HEP
        """
        # Prédictions Keras
        keras_pred = keras_model.predict(hep_test_data['features'])
        
        # Prédictions hls4ml
        hls_pred = hls_model.predict(hep_test_data['features'])
        
        # Comparaison
        accuracy_diff = self._compare_predictions(keras_pred, hls_pred)
        
        # Métriques HEP spécifiques
        hep_metrics = self._calculate_hep_metrics(
            keras_pred, hls_pred, hep_test_data['labels']
        )
        
        return {
            'accuracy_difference': accuracy_diff,
            'hep_metrics': hep_metrics,
            'validation_passed': accuracy_diff < 0.01  # 1% tolerance
        }
    
    def _compare_predictions(self, keras_pred, hls_pred):
        """Compare les prédictions"""
        import numpy as np
        return np.abs(keras_pred - hls_pred).max()
    
    def _calculate_hep_metrics(self, keras_pred, hls_pred, labels):
        """Calcule métriques HEP"""
        # Simplifié: calculer efficacité, rejet, etc.
        return {
            'efficiency_match': True,
            'background_rejection_match': True
        }

validation = HEPValidation()

print("\n" + "="*60)
print("HEP Validation Metrics")
print("="*60)

for metric, info in validation.validation_metrics.items():
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Description: {info['description']}")
    if 'target' in info:
        print(f"  Target: {info['target']}")
    if 'constraint' in info:
        print(f"  Constraint: {info['constraint']}")
    if 'requirement' in info:
        print(f"  Requirement: {info['requirement']}")
```

---

## Workflow d'Intégration Complet

```python
class CompleteIntegrationWorkflow:
    """
    Workflow complet d'intégration
    """
    
    def generate_workflow(self):
        """Génère le workflow complet"""
        workflow = """
Complete hls4ml Integration Workflow:

1. Model Development
   ├─ Train model on HEP data
   ├─ Validate physics performance
   └─ Optimize for FPGA constraints

2. Model Conversion
   ├─ Convert to hls4ml
   ├─ Configure precision and parallelism
   └─ Validate conversion

3. FPGA Implementation
   ├─ Generate HLS code
   ├─ Synthesize with Vivado HLS
   ├─ Implement in Vivado
   └─ Generate bitstream

4. Hardware Validation
   ├─ Program FPGA
   ├─ Test with real detector data
   ├─ Measure latency and throughput
   └─ Validate physics performance

5. Integration
   ├─ Integrate with trigger system
   ├─ Connect data interfaces
   └─ Test end-to-end

6. Deployment
   ├─ Commission in detector
   ├─ Monitor performance
   └─ Iterate if needed
"""
        return workflow
    
    def create_integration_checklist(self):
        """Crée une checklist d'intégration"""
        checklist = {
            'model': [
                'Model trained and validated on HEP data',
                'Physics performance acceptable',
                'Model size fits FPGA constraints'
            ],
            'conversion': [
                'hls4ml conversion successful',
                'Accuracy preserved (< 1% degradation)',
                'Configuration optimized'
            ],
            'fpga': [
                'Synthesis successful',
                'Timing constraints met',
                'Resources within limits',
                'Bitstream generated'
            ],
            'validation': [
                'C simulation matches Keras',
                'RTL co-simulation passes',
                'Hardware test successful',
                'Latency requirements met',
                'Throughput requirements met'
            ],
            'integration': [
                'Interfaces connected',
                'End-to-end test passes',
                'Physics performance validated',
                'Stable operation confirmed'
            ]
        }
        
        return checklist

integration_workflow = CompleteIntegrationWorkflow()

print(integration_workflow.generate_workflow())

print("\n" + "="*60)
print("Integration Checklist")
print("="*60)

checklist = integration_workflow.create_integration_checklist()
for category, items in checklist.items():
    print(f"\n{category.upper()}:")
    for item in items:
        print(f"  ☐ {item}")
```

---

## Outils et Scripts d'Intégration

```python
class IntegrationTools:
    """
    Outils pour faciliter l'intégration
    """
    
    def generate_validation_script(self):
        """Génère un script de validation"""
        script = """
# Validation script for hls4ml integration

import numpy as np
import hls4ml
from tensorflow import keras

def validate_integration(keras_model, hls_model, test_data, hep_labels):
    '''
    Valide l'intégration complète
    '''
    # 1. Accuracy comparison
    keras_pred = keras_model.predict(test_data)
    hls_pred = hls_model.predict(test_data)
    
    accuracy_diff = np.abs(keras_pred - hls_pred).max()
    print(f"Max accuracy difference: {accuracy_diff}")
    
    # 2. HEP metrics
    efficiency_keras = calculate_efficiency(keras_pred, hep_labels)
    efficiency_hls = calculate_efficiency(hls_pred, hep_labels)
    
    print(f"Efficiency: Keras={efficiency_keras:.3f}, hls4ml={efficiency_hls:.3f}")
    
    # 3. Latency check (if hardware available)
    # latency = measure_hardware_latency(hls_model)
    # print(f"Hardware latency: {latency} ns")
    
    return {
        'accuracy_ok': accuracy_diff < 0.01,
        'efficiency_match': abs(efficiency_keras - efficiency_hls) < 0.01
    }

def calculate_efficiency(predictions, labels):
    '''
    Calcule l'efficacité de détection
    '''
    # Simplifié
    return np.mean(predictions.argmax(axis=1) == labels.argmax(axis=1))
"""
        return script

tools = IntegrationTools()
print(tools.generate_validation_script())
```

---

## Exercices

### Exercice 15.5.1
Concevez une interface complète entre un système de trigger L1 et un modèle hls4ml pour la classification de jets.

### Exercice 15.5.2
Créez un script de validation complet qui vérifie à la fois la précision ML et les métriques physiques.

---

## Points Clés à Retenir

> 📌 **Intégration nécessite considération des formats de données HEP**

> 📌 **Contraintes de trigger: latence < 4 μs, throughput 40 MHz**

> 📌 **Validation doit inclure métriques physiques (efficacité, rejet)**

> 📌 **Interface FPGA doit respecter protocoles (AXI Stream)**

> 📌 **Workflow complet: développement → conversion → FPGA → intégration → déploiement**

---

*Section suivante : [15.6 Études de Cas au CERN](./15_06_Cas_CERN.md)*

