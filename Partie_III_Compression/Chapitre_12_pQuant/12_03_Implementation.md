# 12.3 Implémentation des Méthodes de Compression

---

## Introduction

Cette section détaille l'implémentation concrète des différentes méthodes de compression dans pQuant, leurs optimisations, et les détails techniques importants.

---

## Implémentation Low-Rank

### SVD Compression

```python
# pquant/methods/low_rank/svd.py

import torch
import torch.nn as nn
from pquant.core.compression_method import CompressionMethod

class SVDCompression(CompressionMethod):
    """
    Implémentation de compression SVD
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.get('rank', 64)
        self.optimize_rank = config.get('optimize_rank', False)
        self.error_threshold = config.get('error_threshold', 0.05)
    
    def compress(self, model, train_loader=None, val_loader=None):
        """Compresse le modèle via SVD"""
        compressed_model = self._create_copy(model)
        
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Détermine le rang optimal si demandé
                if self.optimize_rank:
                    optimal_rank = self._find_optimal_rank(
                        module.weight.data, self.error_threshold
                    )
                else:
                    optimal_rank = self.rank
                
                # Compresse
                compressed_layer = self._compress_linear(module, optimal_rank)
                
                # Remplace dans le modèle
                self._replace_layer(compressed_model, name, compressed_layer)
        
        return compressed_model
    
    def _compress_linear(self, linear_layer, rank):
        """Compresse une couche linéaire via SVD"""
        from .svd_layer import TruncatedSVDLinear
        
        return TruncatedSVDLinear.from_linear(linear_layer, rank=rank)
    
    def _find_optimal_rank(self, weight_matrix, max_error):
        """Trouve le rang optimal pour une erreur maximale"""
        U, S, Vt = torch.svd(weight_matrix)
        
        total_energy = (S ** 2).sum()
        
        for rank in range(1, len(S) + 1):
            error_energy = (S[rank:] ** 2).sum()
            relative_error = torch.sqrt(error_energy / total_energy)
            
            if relative_error <= max_error:
                return rank
        
        return len(S)  # Utilise le rang complet si nécessaire
```

---

## Implémentation Tensor Train

```python
# pquant/methods/tensor_networks/tensor_train.py

class TensorTrainCompression(CompressionMethod):
    """
    Compression via Tensor Train
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.get('rank', 32)
        self.factorize_dims = config.get('factorize_dimensions', True)
        self.train_after = config.get('train_after_compression', False)
    
    def compress(self, model, train_loader=None, val_loader=None):
        """Compresse avec Tensor Train"""
        compressed_model = self._create_copy(model)
        
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Factorise les dimensions
                if self.factorize_dims:
                    input_dims, output_dims = self._factorize_dimensions(
                        module.in_features, module.out_features
                    )
                else:
                    # Utilise factorisation par défaut
                    input_dims = self._default_factorization(module.in_features)
                    output_dims = self._default_factorization(module.out_features)
                
                # Crée couche TT
                from .tt_layer import TTLinear
                tt_layer = TTLinear(input_dims, output_dims, self.rank)
                
                # Initialise depuis les poids originaux
                self._initialize_from_dense(tt_layer, module)
                
                # Remplace
                self._replace_layer(compressed_model, name, tt_layer)
        
        # Fine-tuning optionnel
        if self.train_after and train_loader is not None:
            compressed_model = self._fine_tune(compressed_model, train_loader)
        
        return compressed_model
    
    def _factorize_dimensions(self, in_features, out_features):
        """
        Factorise intelligemment les dimensions
        """
        # Trouve des facteurs proches de √n
        def factorize(n):
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return factors
        
        input_factors = factorize(in_features)
        output_factors = factorize(out_features)
        
        return tuple(input_factors), tuple(output_factors)
```

---

## Implémentation Quantization

```python
# pquant/methods/quantization/ptq.py

class PTQCompression(CompressionMethod):
    """
    Post-Training Quantization
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.bits = config.get('bits', 8)
        self.calibration_method = config.get('calibration_method', 'min_max')
        self.per_channel = config.get('per_channel', False)
    
    def compress(self, model, train_loader=None, val_loader=None):
        """Compresse via PTQ"""
        # Calibration
        if train_loader is None:
            raise ValueError("PTQ requires calibration data")
        
        calibration_data = self._get_calibration_samples(train_loader)
        quantization_params = self._calibrate(model, calibration_data)
        
        # Quantifie le modèle
        quantized_model = self._quantize_model(model, quantization_params)
        
        return quantized_model
    
    def _calibrate(self, model, calibration_data):
        """Calibre les paramètres de quantification"""
        model.eval()
        
        activation_stats = {}
        
        # Hooks pour capturer les activations
        def hook_fn(name):
            def hook(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append(output.detach())
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward sur données de calibration
        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)
        
        # Calcule les paramètres
        quantization_params = {}
        for name, activations in activation_stats.items():
            all_acts = torch.cat(activations, dim=0)
            
            if self.calibration_method == 'min_max':
                min_val, max_val = all_acts.min().item(), all_acts.max().item()
            elif self.calibration_method == 'percentile':
                min_val = torch.quantile(all_acts, 0.01).item()
                max_val = torch.quantile(all_acts, 0.99).item()
            # Autres méthodes...
            
            scale = (max_val - min_val) / (2 ** self.bits - 1)
            zero_point = 0  # Symétrique
            
            quantization_params[name] = {
                'scale': scale,
                'zero_point': zero_point
            }
        
        # Supprime les hooks
        for hook in hooks:
            hook.remove()
        
        return quantization_params
```

---

## Implémentation LoRA

```python
# pquant/methods/low_rank/lora.py

class LoRACompression(CompressionMethod):
    """
    Low-Rank Adaptation
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.rank = config.get('rank', 16)
        self.alpha = config.get('alpha', 1.0)
        self.target_modules = config.get('target_modules', ['linear', 'attention'])
    
    def compress(self, model, train_loader=None, val_loader=None):
        """Ajoute des adaptateurs LoRA"""
        lora_model = self._create_copy(model)
        
        for name, module in lora_model.named_modules():
            # Vérifie si le module doit être adapté
            should_adapt = any(
                target in name.lower() for target in self.target_modules
            )
            
            if should_adapt and isinstance(module, nn.Linear):
                # Crée LoRA layer
                from .lora_layer import LoRALayer
                lora_module = LoRALayer(module, rank=self.rank, alpha=self.alpha)
                
                # Remplace
                self._replace_layer(lora_model, name, lora_module)
        
        return lora_model
```

---

## Optimisations

### Mémoire Efficace

```python
class MemoryEfficientCompression:
    """
    Compression avec gestion mémoire optimisée
    """
    
    @staticmethod
    def compress_inplace(model, compression_method):
        """
        Compresse en place pour économiser la mémoire
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Compresse directement, remplace les poids
                compressed = compression_method.compress_layer(module)
                
                # Remplace les poids sans créer de nouveau module
                module.weight.data = compressed.weight.data
                if hasattr(compressed, 'bias') and compressed.bias is not None:
                    module.bias.data = compressed.bias.data
```

### Calcul Parallèle

```python
class ParallelCompression:
    """
    Compression parallèle pour plusieurs couches
    """
    
    def compress_parallel(self, model, compression_method, num_workers=4):
        """
        Compresse les couches en parallèle
        """
        from multiprocessing import Pool
        
        # Collecte les couches à compresser
        layers_to_compress = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers_to_compress.append((name, module))
        
        # Compression parallèle
        with Pool(num_workers) as pool:
            compressed_layers = pool.starmap(
                compression_method.compress_layer,
                [(module,) for _, module in layers_to_compress]
            )
        
        # Remplace dans le modèle
        for (name, _), compressed in zip(layers_to_compress, compressed_layers):
            self._replace_layer(model, name, compressed)
        
        return model
```

---

## Exercices

### Exercice 12.3.1
Implémentez une variante optimisée de SVD qui utilise la SVD randomisée pour très grandes matrices.

### Exercice 12.3.2
Créez un système de cache pour éviter de recalculer les compressions pour les mêmes configurations.

---

## Points Clés à Retenir

> 📌 **Implémentations modulaires facilitent la maintenance**

> 📌 **Optimisations mémoire et parallélisation pour grandes modèles**

> 📌 **Calibration robuste pour PTQ**

> 📌 **Initialisation optimale depuis modèles existants**

---

*Section suivante : [12.4 Pipelines de Compression Automatisés](./12_04_Pipelines.md)*

