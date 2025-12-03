# Chapitre 9 : Quantification des Réseaux de Neurones

---

## Introduction

La **quantification** réduit la précision numérique des poids et activations d'un réseau de neurones. Passer de float32 à int8 divise par 4 la mémoire et accélère significativement l'inférence sur hardware adapté.

---

## Plan du Chapitre

1. [Fondements de la Quantification](./09_01_Fondements.md)
2. [Post-Training Quantization (PTQ)](./09_02_PTQ.md)
3. [Quantization-Aware Training (QAT)](./09_03_QAT.md)
4. [Quantification Mixte](./09_04_Mixed_Precision.md)
5. [Quantification Binaire et Ternaire](./09_05_Binary_Ternary.md)
6. [Quantification pour Réseaux de Tenseurs](./09_06_TN_Quantization.md)

---

## Types de Quantification

```
┌─────────────────────────────────────────────────────────────────┐
│                    Schémas de Quantification                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Par précision:                                                │
│  ├── FP32 → FP16 (half precision)                             │
│  ├── FP32 → INT8 (8-bit integer)                              │
│  ├── FP32 → INT4 (4-bit integer)                              │
│  └── FP32 → Binary/Ternary (1-2 bits)                         │
│                                                                 │
│  Par méthode:                                                   │
│  ├── Post-Training Quantization (PTQ)                          │
│  │   └── Rapide, légère perte de précision                    │
│  └── Quantization-Aware Training (QAT)                         │
│      └── Plus long, meilleure précision                        │
│                                                                 │
│  Par granularité:                                               │
│  ├── Per-tensor (un scale pour tout le tenseur)               │
│  ├── Per-channel (un scale par canal)                          │
│  └── Per-group (un scale par groupe d'éléments)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quantification Uniforme

```python
import torch
import torch.nn as nn
import numpy as np

class UniformQuantizer:
    """
    Quantification uniforme: map les valeurs sur une grille régulière
    
    q = round((x - zero_point) / scale)
    x_dequant = q * scale + zero_point
    """
    
    def __init__(self, n_bits=8, symmetric=True):
        self.n_bits = n_bits
        self.symmetric = symmetric
        
        if symmetric:
            self.q_min = -(2 ** (n_bits - 1))
            self.q_max = 2 ** (n_bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** n_bits - 1
    
    def compute_scale_zp(self, x):
        """
        Calcule scale et zero_point pour quantifier x
        """
        x_min, x_max = x.min().item(), x.max().item()
        
        if self.symmetric:
            # Symétrique autour de 0
            abs_max = max(abs(x_min), abs(x_max))
            scale = abs_max / self.q_max
            zero_point = 0
        else:
            # Asymétrique
            scale = (x_max - x_min) / (self.q_max - self.q_min)
            zero_point = self.q_min - round(x_min / scale)
        
        return scale, zero_point
    
    def quantize(self, x, scale, zero_point):
        """Quantifie x"""
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, self.q_min, self.q_max)
        return q.to(torch.int8 if self.n_bits == 8 else torch.int32)
    
    def dequantize(self, q, scale, zero_point):
        """Déquantifie q"""
        return (q.float() - zero_point) * scale
    
    def quantize_dequantize(self, x):
        """Simule la quantification (pour l'entraînement)"""
        scale, zp = self.compute_scale_zp(x)
        q = self.quantize(x, scale, zp)
        return self.dequantize(q, scale, zp)

# Démonstration
quantizer = UniformQuantizer(n_bits=8, symmetric=True)
x = torch.randn(1000)

scale, zp = quantizer.compute_scale_zp(x)
q = quantizer.quantize(x, scale, zp)
x_recon = quantizer.dequantize(q, scale, zp)

error = (x - x_recon).abs().mean()
print(f"Scale: {scale:.6f}, Zero Point: {zp}")
print(f"Erreur moyenne de quantification: {error:.6f}")
print(f"Erreur relative: {error / x.abs().mean():.2%}")
```

---

## Post-Training Quantization (PTQ)

```python
class PostTrainingQuantization:
    """
    Quantification après entraînement
    
    Rapide mais peut dégrader la précision pour les modèles sensibles
    """
    
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.quantizers = {}
        
    def calibrate(self):
        """
        Calibre les paramètres de quantification sur les données
        """
        self.model.eval()
        
        # Collecte les statistiques des activations
        activation_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {'min': [], 'max': []}
                activation_stats[name]['min'].append(output.min().item())
                activation_stats[name]['max'].append(output.max().item())
            return hook
        
        # Enregistre les hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass sur les données de calibration
        with torch.no_grad():
            for x in self.calibration_data:
                _ = self.model(x)
        
        # Supprime les hooks
        for hook in hooks:
            hook.remove()
        
        # Calcule les paramètres de quantification
        for name, stats in activation_stats.items():
            min_val = min(stats['min'])
            max_val = max(stats['max'])
            
            quantizer = UniformQuantizer(n_bits=8, symmetric=False)
            scale = (max_val - min_val) / 255
            zero_point = -round(min_val / scale)
            
            self.quantizers[name] = {'scale': scale, 'zero_point': zero_point}
        
        return self.quantizers
    
    def quantize_weights(self):
        """
        Quantifie les poids du modèle
        """
        quantized_state = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                quantizer = UniformQuantizer(n_bits=8, symmetric=True)
                scale, zp = quantizer.compute_scale_zp(param.data)
                q_weight = quantizer.quantize(param.data, scale, zp)
                
                quantized_state[name] = {
                    'quantized': q_weight,
                    'scale': scale,
                    'zero_point': zp
                }
        
        return quantized_state

# Exemple d'utilisation
"""
model = load_pretrained_model()
calibration_data = get_calibration_samples(n=100)

ptq = PostTrainingQuantization(model, calibration_data)
activation_params = ptq.calibrate()
weight_params = ptq.quantize_weights()
"""
```

---

## Quantization-Aware Training (QAT)

```python
class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization pour QAT
    
    Forward: quantize + dequantize (simule la quantification)
    Backward: Straight-Through Estimator (gradient passe tel quel)
    """
    
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        # Quantize
        q = torch.round(x / scale + zero_point)
        q = torch.clamp(q, q_min, q_max)
        
        # Dequantize
        x_q = (q - zero_point) * scale
        
        return x_q
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: gradient passe inchangé
        return grad_output, None, None, None, None


class QATLinear(nn.Module):
    """
    Couche linéaire avec quantification simulée
    """
    
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.n_bits = n_bits
        
        # Paramètres de quantification apprenables
        self.weight_scale = nn.Parameter(torch.tensor(1.0))
        self.activation_scale = nn.Parameter(torch.tensor(1.0))
        
        self.q_min = -(2 ** (n_bits - 1))
        self.q_max = 2 ** (n_bits - 1) - 1
        
    def forward(self, x):
        # Quantifie les poids
        w_q = FakeQuantize.apply(
            self.linear.weight,
            self.weight_scale,
            0,  # zero_point symétrique
            self.q_min,
            self.q_max
        )
        
        # Forward avec poids quantifiés
        output = nn.functional.linear(x, w_q, self.linear.bias)
        
        # Quantifie les activations
        output_q = FakeQuantize.apply(
            output,
            self.activation_scale,
            0,
            self.q_min,
            self.q_max
        )
        
        return output_q


class QuantizationAwareTraining:
    """
    Entraînement avec quantification simulée
    """
    
    def __init__(self, model, n_bits=8):
        self.model = model
        self.n_bits = n_bits
        self.qat_model = self._convert_to_qat(model)
        
    def _convert_to_qat(self, model):
        """Convertit le modèle pour QAT"""
        qat_model = model  # Copie
        
        # Remplace les couches par leurs versions QAT
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                qat_layer = QATLinear(
                    module.in_features,
                    module.out_features,
                    self.n_bits
                )
                qat_layer.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    qat_layer.linear.bias.data = module.bias.data.clone()
                
                # Remplace dans le modèle
                # (nécessite une logique de remplacement récursive)
        
        return qat_model
    
    def train(self, train_loader, epochs=10, lr=1e-4):
        """
        Entraîne avec quantification simulée
        """
        optimizer = torch.optim.Adam(self.qat_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.qat_model.train()
            
            for x, y in train_loader:
                optimizer.zero_grad()
                output = self.qat_model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                # Met à jour les scales de quantification
                self._update_scales()
            
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    def _update_scales(self):
        """Met à jour les paramètres de quantification"""
        for module in self.qat_model.modules():
            if isinstance(module, QATLinear):
                # Calcule le scale optimal pour les poids
                w_max = module.linear.weight.abs().max()
                module.weight_scale.data = w_max / self.q_max
```

---

## Quantification pour FPGA

```python
class FPGAQuantization:
    """
    Quantification optimisée pour déploiement FPGA
    
    Contraintes spécifiques:
    - Précisions fixes (puissances de 2 préférées)
    - Pas de virgule flottante
    - Multiplication par shift quand possible
    """
    
    @staticmethod
    def power_of_two_scale(x, n_bits=8):
        """
        Trouve un scale qui est une puissance de 2
        Permet de remplacer la multiplication par un shift
        """
        x_max = x.abs().max().item()
        q_max = 2 ** (n_bits - 1) - 1
        
        # Scale idéal
        ideal_scale = x_max / q_max
        
        # Arrondit à la puissance de 2 la plus proche
        log_scale = np.log2(ideal_scale)
        po2_scale = 2 ** np.ceil(log_scale)
        
        return po2_scale
    
    @staticmethod
    def fixed_point_representation(x, int_bits, frac_bits):
        """
        Représentation en virgule fixe
        
        int_bits: bits pour la partie entière
        frac_bits: bits pour la partie fractionnaire
        """
        scale = 2 ** frac_bits
        
        # Quantifie
        q = torch.round(x * scale)
        
        # Clamp
        max_val = 2 ** (int_bits + frac_bits - 1) - 1
        min_val = -2 ** (int_bits + frac_bits - 1)
        q = torch.clamp(q, min_val, max_val)
        
        return q.to(torch.int32), scale
    
    @staticmethod
    def generate_hls_config(model, bit_configs):
        """
        Génère la configuration pour hls4ml
        """
        config = {
            'Model': {
                'Precision': 'ap_fixed<16,6>',  # Défaut
                'ReuseFactor': 1
            },
            'LayerName': {}
        }
        
        for name, bits in bit_configs.items():
            config['LayerName'][name] = {
                'Precision': {
                    'weight': f'ap_fixed<{bits["weight"]},{bits["weight"]//2}>',
                    'bias': f'ap_fixed<{bits["bias"]},{bits["bias"]//2}>',
                    'result': f'ap_fixed<{bits["result"]},{bits["result"]//2}>'
                }
            }
        
        return config

# Exemple pour hls4ml
bit_config = {
    'layer1': {'weight': 8, 'bias': 8, 'result': 16},
    'layer2': {'weight': 6, 'bias': 6, 'result': 12},
    'output': {'weight': 8, 'bias': 8, 'result': 16}
}

hls_config = FPGAQuantization.generate_hls_config(None, bit_config)
print("Configuration hls4ml:")
for key, value in hls_config.items():
    print(f"  {key}: {value}")
```

---

## Exercices

### Exercice 9.1
Implémentez la quantification per-channel et comparez-la à la quantification per-tensor.

### Exercice 9.2
Entraînez un réseau avec QAT et comparez sa précision à un réseau quantifié avec PTQ.

### Exercice 9.3
Créez une fonction qui trouve automatiquement la précision minimale par couche pour maintenir une précision cible.

---

## Points Clés à Retenir

> 📌 **INT8 réduit la mémoire de 4x et accélère l'inférence de 2-4x**

> 📌 **PTQ est rapide mais QAT donne de meilleurs résultats pour les précisions faibles**

> 📌 **La calibration est cruciale pour PTQ**

> 📌 **Les FPGA préfèrent les scales en puissance de 2**

---

*Chapitre suivant : [Chapitre 10 - Knowledge Distillation](../Chapitre_10_Knowledge_Distillation/10_introduction.md)*

