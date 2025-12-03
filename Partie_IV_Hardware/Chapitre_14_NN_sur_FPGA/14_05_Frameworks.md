# 14.5 Frameworks de Déploiement

---

## Introduction

Les **frameworks de déploiement** facilitent la conversion de modèles ML en implémentations FPGA. Cette section présente les principaux frameworks, notamment **hls4ml** utilisé au CERN.

---

## Vue d'Ensemble des Frameworks

```
┌─────────────────────────────────────────────────────────────────┐
│              Frameworks ML → FPGA                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  hls4ml                                                         │
│  └─ CERN, Keras/TensorFlow → HLS C++                           │
│                                                                 │
│  FINN                                                           │
│  └─ Xilinx, Quantized NNs, Dataflow                            │
│                                                                 │
│  Vitis AI                                                       │
│  └─ Xilinx, Optimized IP cores                                 │
│                                                                 │
│  Intel OpenVINO                                                 │
│  └─ Intel, Optimized for Intel FPGAs                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## hls4ml

### Introduction

```python
class HLS4ML:
    """
    hls4ml: Framework de déploiement ML sur FPGA (CERN)
    """
    
    def __init__(self):
        self.description = """
        hls4ml convertit des modèles Keras/TensorFlow en code HLS C++
        pour déploiement sur FPGA.
        
        Développé au CERN pour applications HEP.
        """
        
        self.features = {
            'input_formats': ['Keras', 'TensorFlow', 'ONNX', 'PyTorch (via ONNX)'],
            'output': 'HLS C++ code',
            'targets': 'Xilinx FPGAs (via Vitis HLS)',
            'optimizations': [
                'Quantization',
                'Compression',
                'Reuse factors',
                'Pipeline optimization'
            ]
        }
    
    def display_overview(self):
        """Affiche la vue d'ensemble"""
        print("\n" + "="*60)
        print("hls4ml Overview")
        print("="*60)
        print(self.description)
        print("\nFeatures:")
        for feature, value in self.features.items():
            if isinstance(value, list):
                print(f"  {feature}:")
                for item in value:
                    print(f"    • {item}")
            else:
                print(f"  {feature}: {value}")

hls4ml = HLS4ML()
hls4ml.display_overview()
```

---

### Workflow hls4ml

```python
class HLS4MLWorkflow:
    """
    Workflow typique avec hls4ml
    """
    
    def generate_workflow(self):
        """Génère un exemple de workflow"""
        workflow = """
hls4ml Workflow:

1. Train Model (Keras/TensorFlow)
   └─ model = Sequential([...])
   └─ model.compile(...)
   └─ model.fit(...)

2. Convert to hls4ml
   └─ import hls4ml
   └─ config = hls4ml.utils.config_from_keras_model(model)
   └─ config['Model']['ReuseFactor'] = 4
   └─ hls_model = hls4ml.converters.convert_from_keras_model(...)

3. Compile HLS
   └─ hls_model.compile()

4. Create IP
   └─ hls_model.build(csim=False, synth=True, cosim=False)

5. Integrate in Vivado
   └─ Use generated IP core
"""
        return workflow
    
    def generate_example_code(self):
        """Génère un exemple de code"""
        code = """
# Example: Using hls4ml

import hls4ml
from tensorflow import keras

# 1. Load or create model
model = keras.models.load_model('model.h5')

# 2. Configure hls4ml
config = hls4ml.utils.config_from_keras_model(model)
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 4
config['LayerName']['dense']['ReuseFactor'] = 8

# 3. Convert
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my-hls-project',
    part='xc7z020clg400-1'
)

# 4. Compile
hls_model.compile()

# 5. Build
hls_model.build(
    csim=False,      # Skip C simulation
    synth=True,      # Run synthesis
    cosim=False      # Skip co-simulation
)
"""
        return code

workflow = HLS4MLWorkflow()
print(workflow.generate_workflow())

print("\n" + "="*60)
print("Example Code")
print("="*60)
print(workflow.generate_example_code())
```

---

### Configuration hls4ml

```python
class HLS4MLConfiguration:
    """
    Configuration hls4ml
    """
    
    def __init__(self):
        self.config_options = {
            'precision': {
                'description': 'Précision des nombres',
                'options': ['ap_fixed<W,I>', 'ap_int<W>', 'float'],
                'example': "ap_fixed<16,6>  # 16 bits total, 6 integer bits"
            },
            'reuse_factor': {
                'description': 'Facteur de réutilisation',
                'impact': 'Plus élevé = moins de ressources, plus de cycles',
                'example': 'ReuseFactor = 4 means 4× fewer resources, 4× more latency'
            },
            'strategy': {
                'description': 'Stratégie de pipeline',
                'options': ['Latency', 'Resource', 'Stable'],
                'example': "'Latency' minimizes latency, 'Resource' minimizes resources"
            },
            'compression': {
                'description': 'Compression des poids',
                'options': ['None', 'Sparse', 'Dense'],
                'example': 'Sparse compression for pruned models'
            }
        }
    
    def generate_config_example(self):
        """Génère un exemple de configuration"""
        config_example = """
Example hls4ml Configuration:

config = {
    'Model': {
        'Precision': 'ap_fixed<16,6>',
        'ReuseFactor': 4,
        'Strategy': 'Latency'
    },
    'LayerName': {
        'dense_1': {
            'ReuseFactor': 8,      # Override global
            'Precision': 'ap_fixed<12,4>'
        },
        'conv2d_1': {
            'Strategy': 'Resource'  # Override global
        }
    }
}
"""
        return config_example
    
    def display_options(self):
        """Affiche les options"""
        print("\n" + "="*60)
        print("hls4ml Configuration Options")
        print("="*60)
        
        for option, info in self.config_options.items():
            print(f"\n{option.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            if 'options' in info:
                print(f"  Options: {', '.join(info['options'])}")
            if 'impact' in info:
                print(f"  Impact: {info['impact']}")
            if 'example' in info:
                print(f"  Example: {info['example']}")

config = HLS4MLConfiguration()
config.display_options()
print(config.generate_config_example())
```

---

## FINN

### Introduction

```python
class FINN:
    """
    FINN: Framework pour quantized NNs sur FPGA (Xilinx)
    """
    
    def __init__(self):
        self.description = """
        FINN est un framework Xilinx pour déployer des réseaux
        quantifiés sur FPGA avec optimisations dataflow.
        
        Spécialisé pour réseaux binaires et ternaires.
        """
        
        self.features = {
            'focus': 'Quantized neural networks',
            'specialization': 'Binary/ternary networks',
            'architecture': 'Dataflow architecture',
            'optimization': 'Hardware-aware optimizations'
        }
    
    def display_overview(self):
        """Affiche la vue d'ensemble"""
        print("\n" + "="*60)
        print("FINN Overview")
        print("="*60)
        print(self.description)
        print("\nFeatures:")
        for feature, value in self.features.items():
            print(f"  {feature}: {value}")

finn = FINN()
finn.display_overview()
```

---

## Vitis AI

### Introduction

```python
class VitisAI:
    """
    Vitis AI: Framework Xilinx pour AI inference
    """
    
    def __init__(self):
        self.description = """
        Vitis AI fournit une stack complète pour déployer
        des modèles ML sur Xilinx FPGAs et SoCs.
        
        Inclut optimizations, IP cores, et runtime.
        """
        
        self.components = {
            'ai_optimizer': 'Quantization, pruning, compilation',
            'ai_quantizer': 'Quantization-aware training',
            'ai_compiler': 'Model to hardware compilation',
            'ai_profiler': 'Performance profiling',
            'runtime': 'DPU runtime for inference'
        }
    
    def display_overview(self):
        """Affiche la vue d'ensemble"""
        print("\n" + "="*60)
        print("Vitis AI Overview")
        print("="*60)
        print(self.description)
        print("\nComponents:")
        for component, description in self.components.items():
            print(f"  {component.replace('_', ' ').title()}: {description}")

vitis_ai = VitisAI()
vitis_ai.display_overview()
```

---

## Comparaison des Frameworks

```python
class FrameworkComparison:
    """
    Comparaison des frameworks ML → FPGA
    """
    
    comparison = {
        'hls4ml': {
            'developer': 'CERN',
            'input': 'Keras/TensorFlow/ONNX',
            'strengths': [
                'Optimized for HEP',
                'Good for custom architectures',
                'Active CERN community'
            ],
            'weaknesses': [
                'Primarily Xilinx',
                'Steeper learning curve'
            ],
            'best_for': 'HEP applications, research'
        },
        'finn': {
            'developer': 'Xilinx',
            'input': 'Quantized models',
            'strengths': [
                'Excellent for binary/ternary',
                'Dataflow optimization',
                'High throughput'
            ],
            'weaknesses': [
                'Limited to quantized models',
                'Xilinx only'
            ],
            'best_for': 'Quantized NNs, edge inference'
        },
        'vitis_ai': {
            'developer': 'Xilinx',
            'input': 'ONNX/TensorFlow/PyTorch',
            'strengths': [
                'Complete toolchain',
                'Optimized IP cores',
                'Production-ready'
            ],
            'weaknesses': [
                'Xilinx only',
                'Less flexible'
            ],
            'best_for': 'Production deployment, SoCs'
        }
    }
    
    @staticmethod
    def display_comparison():
        """Affiche la comparaison"""
        print("\n" + "="*60)
        print("Framework Comparison")
        print("="*60)
        
        for framework, info in FrameworkComparison.comparison.items():
            print(f"\n{framework.upper()}:")
            print(f"  Developer: {info['developer']}")
            print(f"  Input: {info['input']}")
            print(f"  Strengths:")
            for strength in info['strengths']:
                print(f"    + {strength}")
            print(f"  Weaknesses:")
            for weakness in info['weaknesses']:
                print(f"    - {weakness}")
            print(f"  Best for: {info['best_for']}")

FrameworkComparison.display_comparison()
```

---

## Exemple Complet: Déploiement avec hls4ml

```python
class CompleteDeploymentExample:
    """
    Exemple complet de déploiement avec hls4ml
    """
    
    def generate_complete_workflow(self):
        """Génère un workflow complet"""
        workflow = """
Complete Deployment Workflow:

1. Model Training (Python/Keras)
   ├─ Load HEP dataset
   ├─ Design model architecture
   ├─ Train model
   ├─ Evaluate accuracy
   └─ Save model.h5

2. Model Optimization
   ├─ Quantization (8-bit)
   ├─ Pruning (optional)
   └─ Validate accuracy after optimization

3. hls4ml Conversion
   ├─ Load model.h5
   ├─ Configure hls4ml
   ├─ Convert to HLS C++
   └─ Generate project

4. HLS Synthesis
   ├─ C simulation (validate)
   ├─ Synthesis (generate RTL)
   ├─ C/RTL co-simulation
   └─ Export IP

5. Vivado Integration
   ├─ Create Vivado project
   ├─ Add hls4ml IP core
   ├─ Add constraints
   ├─ Run implementation
   └─ Generate bitstream

6. Testing
   ├─ Program FPGA
   ├─ Run inference tests
   ├─ Measure latency
   └─ Validate accuracy
"""
        return workflow
    
    def generate_validation_code(self):
        """Génère du code de validation"""
        code = """
# Validation: Compare Keras vs hls4ml

import numpy as np
from tensorflow import keras
import hls4ml

# Load model
model = keras.models.load_model('model.h5')

# Load hls4ml model
hls_model = hls4ml.model.HLSModel.from_config('hls_config.json')

# Test data
X_test = np.random.rand(100, 784)  # Example

# Keras prediction
y_keras = model.predict(X_test)

# hls4ml prediction (C simulation)
y_hls = hls_model.predict(X_test)

# Compare
accuracy_diff = np.mean(np.abs(y_keras - y_hls))
print(f"Average difference: {accuracy_diff}")

# Validate
assert accuracy_diff < 1e-3, "Difference too large!"
"""
        return code

example = CompleteDeploymentExample()
print(example.generate_complete_workflow())

print("\n" + "="*60)
print("Validation Code Example")
print("="*60)
print(example.generate_validation_code())
```

---

## Exercices

### Exercice 14.5.1
Convertissez un modèle Keras simple en hls4ml et analysez les ressources générées.

### Exercice 14.5.2
Comparez les performances (latence, throughput) entre hls4ml et FINN pour un même modèle quantifié.

---

## Points Clés à Retenir

> 📌 **hls4ml: CERN, bon pour HEP, flexible**

> 📌 **FINN: Xilinx, optimisé pour quantized NNs**

> 📌 **Vitis AI: Stack complète Xilinx**

> 📌 **Choix dépend de l'application et du modèle**

> 📌 **Validation importante après conversion**

---

*Chapitre suivant : [Chapitre 15 - hls4ml en Détail](../Chapitre_15_hls4ml/15_introduction.md)*

