# 15.2 Modèles Supportés et Limitations

---

## Introduction

Cette section détaille les **types de modèles supportés** par hls4ml, leurs **limitations**, et comment adapter les modèles pour un déploiement optimal sur FPGA.

---

## Types de Couches Supportées

### Couches Complètement Supportées

```python
class SupportedLayers:
    """
    Couches supportées par hls4ml
    """
    
    def __init__(self):
        self.fully_supported = {
            'Dense': {
                'status': 'Fully supported',
                'description': 'Couche dense (fully connected)',
                'features': [
                    'Arbitrary input/output sizes',
                    'Customizable precision',
                    'Bias support'
                ],
                'optimizations': ['Matrix multiplication', 'Parallel dot products']
            },
            'Conv2D': {
                'status': 'Fully supported',
                'description': 'Convolution 2D',
                'features': [
                    'Arbitrary kernel sizes',
                    'Padding (same/valid)',
                    'Strides support'
                ],
                'optimizations': ['Sliding window', 'Parallel filters']
            },
            'Activation': {
                'status': 'Fully supported',
                'description': 'Fonctions d\'activation',
                'supported': ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softmax'],
                'implementation': 'LUT or polynomial approximation'
            },
            'BatchNormalization': {
                'status': 'Fully supported',
                'description': 'Normalisation par batch',
                'note': 'Fusionnée automatiquement avec couche précédente',
                'optimization': 'Réduit opérations, améliore performance'
            },
            'MaxPooling2D': {
                'status': 'Fully supported',
                'description': 'Pooling maximum',
                'features': ['Arbitrary pool size', 'Strides']
            },
            'AveragePooling2D': {
                'status': 'Fully supported',
                'description': 'Pooling moyen',
                'features': ['Arbitrary pool size', 'Strides']
            },
            'Flatten': {
                'status': 'Fully supported',
                'description': 'Remise à plat',
                'note': 'Réorganisation mémoire, pas de calcul'
            },
            'Reshape': {
                'status': 'Fully supported',
                'description': 'Remodelage',
                'note': 'Remap indices, pas de calcul'
            },
            'Concatenate': {
                'status': 'Fully supported',
                'description': 'Concaténation',
                'note': 'Along specified axis'
            }
        }
    
    def display_supported(self):
        """Affiche les couches supportées"""
        print("\n" + "="*60)
        print("Fully Supported Layers")
        print("="*60)
        
        for layer, info in self.fully_supported.items():
            print(f"\n{layer}:")
            print(f"  Status: {info['status']}")
            print(f"  Description: {info['description']}")
            if 'features' in info:
                print(f"  Features:")
                for feat in info['features']:
                    print(f"    • {feat}")
            if 'supported' in info:
                print(f"  Supported variants: {', '.join(info['supported'])}")
            if 'note' in info:
                print(f"  Note: {info['note']}")

supported = SupportedLayers()
supported.display_supported()
```

---

### Couches Partiellement Supportées / Limitations

```python
class LayerLimitations:
    """
    Limitations et couches partiellement supportées
    """
    
    def __init__(self):
        self.limitations = {
            'LSTM/GRU': {
                'status': 'Limited support',
                'issue': 'Recurrent structures complexes',
                'alternative': 'Unroll explicitement ou utiliser des couches séquentielles',
                'note': 'Support en développement'
            },
            'Transformer': {
                'status': 'Experimental',
                'issue': 'Attention mechanisms complexes',
                'alternative': 'Simplifications, approximations',
                'note': 'Recherche active'
            },
            'DepthwiseConv2D': {
                'status': 'Supported with restrictions',
                'issue': 'Optimisations spécifiques nécessaires',
                'note': 'Peut nécessiter configurations spéciales'
            },
            'GlobalPooling': {
                'status': 'Supported',
                'note': 'Peut être inefficace pour grandes images',
                'optimization': 'Considérer alternatives'
            },
            'Dropout': {
                'status': 'Removed in inference',
                'note': 'Supprimé automatiquement (pas nécessaire en inference)'
            },
            'Custom Layers': {
                'status': 'Not directly supported',
                'solution': 'Implémenter manuellement ou utiliser primitives',
                'note': 'Contact communauté pour support'
            }
        }
    
    def display_limitations(self):
        """Affiche les limitations"""
        print("\n" + "="*60)
        print("Layer Limitations and Partial Support")
        print("="*60)
        
        for layer, info in self.limitations.items():
            print(f"\n{layer}:")
            print(f"  Status: {info['status']}")
            if 'issue' in info:
                print(f"  Issue: {info['issue']}")
            if 'alternative' in info:
                print(f"  Alternative: {info['alternative']}")
            if 'solution' in info:
                print(f"  Solution: {info['solution']}")
            print(f"  Note: {info['note']}")

limitations = LayerLimitations()
limitations.display_limitations()
```

---

## Architectures Supportées

```python
class SupportedArchitectures:
    """
    Architectures de modèles supportées
    """
    
    def __init__(self):
        self.architectures = {
            'MLP': {
                'description': 'Multi-Layer Perceptron',
                'support': 'Excellent',
                'example': 'Dense layers avec activations',
                'use_case': 'Classification, regression simple'
            },
            'CNN': {
                'description': 'Convolutional Neural Networks',
                'support': 'Excellent',
                'example': 'Conv2D + Pooling + Dense',
                'use_case': 'Image classification, feature extraction',
                'note': 'Support pour architectures classiques (LeNet, VGG-like)'
            },
            'ResNet-like': {
                'description': 'Residual Networks',
                'support': 'Good',
                'example': 'Skip connections, residual blocks',
                'use_case': 'Deep image classification',
                'note': 'Skip connections supportées via Add/Merge layers'
            },
            'MobileNet-like': {
                'description': 'Mobile Networks',
                'support': 'Moderate',
                'example': 'Depthwise separable convolutions',
                'use_case': 'Efficient mobile/edge inference',
                'note': 'DepthwiseConv peut nécessiter configuration spéciale'
            },
            'RNN': {
                'description': 'Recurrent Neural Networks',
                'support': 'Limited',
                'example': 'LSTM, GRU',
                'use_case': 'Sequential data',
                'note': 'Support en développement'
            },
            'Transformer': {
                'description': 'Transformer Architecture',
                'support': 'Experimental',
                'example': 'Self-attention, encoder-decoder',
                'use_case': 'NLP, vision transformers',
                'note': 'Recherche active, support limité'
            }
        }
    
    def display_architectures(self):
        """Affiche les architectures"""
        print("\n" + "="*60)
        print("Supported Model Architectures")
        print("="*60)
        
        for arch, info in self.architectures.items():
            print(f"\n{arch}:")
            print(f"  Description: {info['description']}")
            print(f"  Support: {info['support']}")
            print(f"  Example: {info['example']}")
            print(f"  Use case: {info['use_case']}")
            if 'note' in info:
                print(f"  Note: {info['note']}")

architectures = SupportedArchitectures()
architectures.display_architectures()
```

---

## Contraintes et Limitations Générales

### Limitations de Taille

```python
class SizeLimitations:
    """
    Limitations de taille et ressources
    """
    
    def __init__(self):
        self.constraints = {
            'memory': {
                'description': 'Mémoire BRAM limitée',
                'impact': 'Grands modèles peuvent ne pas tenir',
                'solutions': [
                    'Quantization (8-bit, 4-bit)',
                    'Compression (pruning)',
                    'Weight streaming depuis DDR'
                ]
            },
            'dsp_slices': {
                'description': 'DSP slices limités',
                'impact': 'Parallélisme limité',
                'solutions': [
                    'Augmenter ReuseFactor',
                    'Optimiser architecture',
                    'Utiliser LUTs pour petites multiplications'
                ]
            },
            'lut': {
                'description': 'LUTs limitées',
                'impact': 'Logique complexe peut dépasser',
                'solutions': [
                    'Simplifier architecture',
                    'Pipeline plus agressif',
                    'Utiliser DSPs au lieu de LUTs'
                ]
            },
            'latency': {
                'description': 'Latence accumulée',
                'impact': 'Nombreuses couches = latence élevée',
                'solutions': [
                    'Pipelines parallèles',
                    'Réduire nombre de couches',
                    'Fusion de couches'
                ]
            }
        }
    
    def estimate_model_fit(self, model, fpga_resources):
        """
        Estime si un modèle peut tenir dans un FPGA
        
        Args:
            model: Modèle Keras
            fpga_resources: Dict avec ressources FPGA
        """
        # Estimation simplifiée
        total_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
        
        # Mémoire (int8)
        memory_bits = total_params * 8
        memory_mb = memory_bits / (8 * 1024 * 1024)
        
        # DSP estimation (approximatif)
        dsp_estimate = total_params // 10  # Approximation
        
        fits = {
            'memory': memory_mb < fpga_resources.get('bram_mb', 10),
            'dsp': dsp_estimate < fpga_resources.get('dsp', 220),
            'estimated_memory_mb': memory_mb,
            'estimated_dsp': dsp_estimate
        }
        
        return fits
    
    def display_constraints(self):
        """Affiche les contraintes"""
        print("\n" + "="*60)
        print("General Constraints and Limitations")
        print("="*60)
        
        for constraint, info in self.constraints.items():
            print(f"\n{constraint.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Impact: {info['impact']}")
            print(f"  Solutions:")
            for solution in info['solutions']:
                print(f"    • {solution}")

constraints = SizeLimitations()
constraints.display_constraints()

# Exemple d'estimation
fpga_resources = {
    'bram_mb': 10,
    'dsp': 220,
    'lut': 53000
}

print("\n" + "="*60)
print("Model Fit Estimation Example")
print("="*60)
print(f"FPGA Resources: {fpga_resources}")
print("\nNote: Estimation simplifiée, utiliser hls4ml pour analyse précise")
```

---

## Bonnes Pratiques pour Modèles Compatibles

```python
class ModelBestPractices:
    """
    Bonnes pratiques pour créer des modèles compatibles hls4ml
    """
    
    def __init__(self):
        self.practices = {
            'architecture': {
                'recommendations': [
                    'Utiliser couches standard (Dense, Conv2D)',
                    'Éviter opérations complexes non supportées',
                    'Simplifier architecture si possible',
                    'Tester compatibilité tôt dans le développement'
                ]
            },
            'training': {
                'recommendations': [
                    'Entraîner avec précision float32',
                    'Utiliser BatchNorm (fusionné automatiquement)',
                    'Éviter Dropout (supprimé en inference)',
                    'Valider avec données représentatives'
                ]
            },
            'optimization': {
                'recommendations': [
                    'Quantifier après entraînement ou QAT',
                    'Pruner pour réduire taille',
                    'Optimiser pour FPGA constraints',
                    'Valider accuracy après optimisation'
                ]
            },
            'testing': {
                'recommendations': [
                    'Tester avec hls4ml avant finalisation',
                    'Comparer Keras vs hls4ml predictions',
                    'Valider sur données réelles',
                    'Mesurer latency et ressources'
                ]
            }
        }
    
    def display_practices(self):
        """Affiche les bonnes pratiques"""
        print("\n" + "="*60)
        print("Best Practices for hls4ml-Compatible Models")
        print("="*60)
        
        for category, info in self.practices.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for rec in info['recommendations']:
                print(f"  • {rec}")

practices = ModelBestPractices()
practices.display_practices()
```

---

## Exemple: Modèle Compatible

```python
class CompatibleModelExample:
    """
    Exemple de modèle optimisé pour hls4ml
    """
    
    @staticmethod
    def create_compatible_model(input_shape, num_classes):
        """
        Crée un modèle compatible avec hls4ml
        
        Args:
            input_shape: Shape d'input (ex: (16,) pour features)
            num_classes: Nombre de classes
        """
        from tensorflow import keras
        
        model = keras.Sequential([
            # Couche d'entrée dense
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            
            # BatchNorm (fusionné automatiquement)
            keras.layers.BatchNormalization(),
            
            # Couche dense intermédiaire
            keras.layers.Dense(32, activation='relu'),
            
            # BatchNorm
            keras.layers.BatchNormalization(),
            
            # Couche de sortie
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def create_compatible_cnn(input_shape, num_classes):
        """
        Crée un CNN compatible
        
        Args:
            input_shape: (height, width, channels)
            num_classes: Nombre de classes
        """
        from tensorflow import keras
        
        model = keras.Sequential([
            # Conv block 1
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Conv block 2
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            
            # Flatten et dense
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model

# Exemples
print("\n" + "="*60)
print("Compatible Model Examples")
print("="*60)

mlp_model = CompatibleModelExample.create_compatible_model((16,), 5)
print("\nMLP Model:")
mlp_model.summary()

cnn_model = CompatibleModelExample.create_compatible_cnn((32, 32, 3), 10)
print("\nCNN Model:")
cnn_model.summary()
```

---

## Exercices

### Exercice 15.2.1
Créez un modèle avec des couches non supportées (ex: LSTM) et identifiez comment l'adapter pour hls4ml.

### Exercice 15.2.2
Estimez si un ResNet-18 peut tenir dans un FPGA Zynq-7000, et proposez des optimisations si nécessaire.

---

## Points Clés à Retenir

> 📌 **Couches standards bien supportées: Dense, Conv2D, Activations, Pooling**

> 📌 **BatchNorm fusionné automatiquement → bon pour performance**

> 📌 **LSTM/RNN support limité, Transformers expérimental**

> 📌 **Contraintes: mémoire BRAM, DSP, latence accumulée**

> 📌 **Quantization et compression souvent nécessaires pour grands modèles**

> 📌 **Tester compatibilité tôt dans le développement**

---

*Section suivante : [15.3 Configuration et Optimisation](./15_03_Configuration.md)*

