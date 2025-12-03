# 9.4 Quantification Mixte

---

## Introduction

La **quantification mixte** utilise différentes précisions pour différentes parties du modèle. Cela permet d'optimiser le compromis entre compression, vitesse et précision.

---

## Principe

### Motivation

- Certaines couches sont plus sensibles à la quantification
- Certaines parties nécessitent plus de précision
- Optimiser précision/cout par couche

---

## Stratégies de Mixed Precision

```python
import torch
import torch.nn as nn

class MixedPrecisionConfig:
    """
    Configuration pour quantification mixte
    """
    
    def __init__(self):
        self.layer_configs = {}  # name -> {'weight': bits, 'activation': bits}
    
    def set_layer_precision(self, layer_name, weight_bits, activation_bits):
        """Définit la précision pour une couche spécifique"""
        self.layer_configs[layer_name] = {
            'weight': weight_bits,
            'activation': activation_bits
        }
    
    def get_precision(self, layer_name):
        """Récupère la précision d'une couche"""
        return self.layer_configs.get(layer_name, {'weight': 8, 'activation': 8})

# Exemple: configuration mixte
config = MixedPrecisionConfig()
config.set_layer_precision('conv1', weight_bits=8, activation_bits=8)
config.set_layer_precision('conv2', weight_bits=4, activation_bits=8)  # Poids plus compressés
config.set_layer_precision('fc', weight_bits=8, activation_bits=16)    # Dernière couche précise
```

---

## Analyse de Sensibilité

### Identification des Couches Sensibles

```python
def sensitivity_analysis(model, test_loader, n_bits_candidates=[8, 6, 4]):
    """
    Analyse la sensibilité de chaque couche à la quantification
    """
    baseline_acc = evaluate_accuracy(model, test_loader)
    
    sensitivities = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            sensitivities[name] = []
            
            for n_bits in n_bits_candidates:
                # Quantifie uniquement cette couche
                quantized_model = quantize_single_layer(model, name, n_bits)
                acc = evaluate_accuracy(quantized_model, test_loader)
                
                degradation = baseline_acc - acc
                sensitivities[name].append({
                    'bits': n_bits,
                    'accuracy': acc,
                    'degradation': degradation
                })
    
    return sensitivities

def quantize_single_layer(model, layer_name, n_bits):
    """Quantifie une seule couche pour test"""
    quantized_model = copy.deepcopy(model)
    
    for name, module in quantized_model.named_modules():
        if name == layer_name:
            if isinstance(module, nn.Linear):
                # Quantifie les poids
                quantizer = UniformQuantizer(n_bits=n_bits)
                scale, zp = quantizer.compute_scale_zero_point(module.weight.data)
                q_weight = quantizer.quantize(module.weight.data, scale, zp)
                module.weight.data = quantizer.dequantize(q_weight, scale, zp)
    
    return quantized_model
```

---

## Sélection Automatique de Précision

```python
class AutomaticPrecisionSelection:
    """
    Sélection automatique de précision mixte
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 target_compression_ratio=0.5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_compression = target_compression_ratio
        
        self.baseline_acc = evaluate_accuracy(model, val_loader)
        self.target_acc = self.baseline_acc * 0.98  # Accepte 2% de dégradation
    
    def find_optimal_precision(self, layer_name, precision_candidates=[8, 6, 4]):
        """
        Trouve la précision optimale pour une couche
        """
        best_bits = 8
        best_acc = 0.0
        
        for bits in sorted(precision_candidates, reverse=True):
            # Teste avec cette précision
            test_model = quantize_single_layer(self.model, layer_name, bits)
            acc = evaluate_accuracy(test_model, self.val_loader)
            
            if acc >= self.target_acc and acc > best_acc:
                best_acc = acc
                best_bits = bits
        
        return best_bits, best_acc
    
    def optimize_all_layers(self):
        """
        Optimise la précision de toutes les couches
        """
        config = MixedPrecisionConfig()
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                optimal_bits, acc = self.find_optimal_precision(name)
                config.set_layer_precision(name, 
                                         weight_bits=optimal_bits,
                                         activation_bits=optimal_bits)
                print(f"{name}: {optimal_bits} bits, Acc: {acc:.2f}%")
        
        return config
```

---

## Implémentation Mixed Precision

```python
class MixedPrecisionQuantizer:
    """
    Quantificateur avec précision mixte par couche
    """
    
    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.quantizers = {}
    
    def quantize_model(self, model):
        """
        Quantifie un modèle avec précisions mixtes
        """
        quantized_model = copy.deepcopy(model)
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                precisions = self.config.get_precision(name)
                
                # Quantifie les poids
                weight_bits = precisions['weight']
                quantizer_w = UniformQuantizer(n_bits=weight_bits)
                scale_w, zp_w = quantizer_w.compute_scale_zero_point(module.weight.data)
                q_weight = quantizer_w.quantize(module.weight.data, scale_w, zp_w)
                module.weight.data = quantizer_w.dequantize(q_weight, scale_w, zp_w)
        
        return quantized_model
```

---

## Exercices

### Exercice 9.4.1
Implémentez un algorithme qui trouve automatiquement la configuration de précision mixte optimale sous contrainte de compression.

### Exercice 9.4.2
Comparez quantification uniforme vs mixte sur un réseau profond (ResNet, VGG).

---

## Points Clés à Retenir

> 📌 **Mixed precision optimise le compromis précision/compression**

> 📌 **L'analyse de sensibilité identifie les couches critiques**

> 📌 **La sélection automatique peut trouver des configurations optimales**

---

*Section suivante : [9.5 Quantification Binaire et Ternaire](./09_05_Binary_Ternary.md)*

