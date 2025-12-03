# 11.4 Combinaison Rang Faible + Quantification

---

## Introduction

La **combinaison de rang faible et quantification** permet d'atteindre des compressions très importantes. En factorisant d'abord les poids, puis en quantifiant les facteurs, on obtient une compression multiplicative.

---

## Principe de Combinaison

### Compression Multiplicative

1. **Rang faible** : $W = U @ V^T$ → compression $\times r_1$
2. **Quantification** : INT8 → compression $\times 4$

**Total** : compression $\times (r_1 \times 4)$

---

## Implémentation

```python
import torch
import torch.nn as nn
import numpy as np

class QuantizedLowRankLinear(nn.Module):
    """
    Couche linéaire avec rang faible + quantification
    """
    
    def __init__(self, in_features, out_features, rank, n_bits=8):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_bits = n_bits
        
        # Facteurs de rang faible (seront quantifiés)
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(in_features, rank))
        
        # Paramètres de quantification
        self.register_buffer('U_scale', torch.tensor(1.0))
        self.register_buffer('V_scale', torch.tensor(1.0))
        self.register_buffer('U_zero_point', torch.tensor(0, dtype=torch.int32))
        self.register_buffer('V_zero_point', torch.tensor(0, dtype=torch.int32))
        
        # Poids quantifiés (buffers, non entraînables)
        self.register_buffer('U_q', None)
        self.register_buffer('V_q', None)
        
        self.quantized = False
    
    def quantize_weights(self):
        """
        Quantifie les facteurs U et V
        """
        # Quantifie U
        U_min = self.U.data.min().item()
        U_max = self.U.data.max().item()
        q_max = 2 ** (self.n_bits - 1) - 1
        
        U_scale = max(abs(U_min), abs(U_max)) / q_max
        U_q = torch.round(self.U.data / U_scale).clamp(-q_max, q_max).to(torch.int8)
        
        # Quantifie V
        V_min = self.V.data.min().item()
        V_max = self.V.data.max().item()
        V_scale = max(abs(V_min), abs(V_max)) / q_max
        V_q = torch.round(self.V.data / V_scale).clamp(-q_max, q_max).to(torch.int8)
        
        # Stocke
        self.U_scale = torch.tensor(U_scale)
        self.V_scale = torch.tensor(V_scale)
        self.U_q = U_q
        self.V_q = V_q
        self.quantized = True
    
    def dequantize(self):
        """Déquantifie pour calcul"""
        if not self.quantized:
            return self.U.data, self.V.data
        
        U_deq = (self.U_q.float() - self.U_zero_point.float()) * self.U_scale
        V_deq = (self.V_q.float() - self.V_zero_point.float()) * self.V_scale
        
        return U_deq, V_deq
    
    def forward(self, x):
        """
        Forward avec poids quantifiés
        """
        if self.quantized:
            U, V = self.dequantize()
        else:
            U, V = self.U, self.V
        
        # y = x @ V @ U^T
        output = x @ V @ U.T
        
        return output
    
    @classmethod
    def from_linear(cls, linear_layer, rank, n_bits=8):
        """
        Crée depuis une couche standard via SVD + quantification
        """
        # SVD
        U, S, Vt = torch.svd(linear_layer.weight.data)
        
        # Troncature
        sqrt_S = torch.sqrt(S[:rank])
        U_r = U[:, :rank]
        Vt_r = Vt[:rank, :]
        
        # Forme compacte
        U_factored = U_r @ torch.diag(sqrt_S)
        V_factored = (torch.diag(sqrt_S) @ Vt_r).T
        
        # Crée la couche
        quantized_lr = cls(linear_layer.in_features, 
                          linear_layer.out_features, 
                          rank, n_bits)
        
        quantized_lr.U.data = U_factored
        quantized_lr.V.data = V_factored
        
        # Quantifie
        quantized_lr.quantize_weights()
        
        return quantized_lr
    
    def compression_ratio(self):
        """Ratio de compression total"""
        # Original (FP32)
        original_size = self.in_features * self.out_features * 4  # bytes
        
        # Compressé (INT8 quantifié)
        compressed_size = (self.rank * (self.in_features + self.out_features)) * 1  # bytes
        
        # + paramètres de quantification (négligeable)
        compression = original_size / compressed_size
        
        return compression
    
    def count_parameters(self):
        """Paramètres (quantifiés)"""
        if self.quantized:
            return self.U_q.numel() + self.V_q.numel()
        else:
            return self.U.numel() + self.V.numel()

# Exemple
original = nn.Linear(1024, 512)
quantized_lr = QuantizedLowRankLinear.from_linear(original, rank=64, n_bits=8)

print("Quantifié + Rang Faible:")
print(f"  Original: {original.weight.numel() * 4 / 1024:.1f} KB")
print(f"  Compressé: {quantized_lr.count_parameters() * 1 / 1024:.1f} KB")
print(f"  Compression: {quantized_lr.compression_ratio():.1f}x")
```

---

## QAT avec Rang Faible

```python
class QATLowRankLinear(nn.Module):
    """
    Low-rank avec Quantization-Aware Training
    """
    
    def __init__(self, in_features, out_features, rank, n_bits=8):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.n_bits = n_bits
        
        # Facteurs (entraînables en FP32)
        self.U = nn.Parameter(torch.randn(out_features, rank))
        self.V = nn.Parameter(torch.randn(in_features, rank))
        
        # Fake quantization
        from quantization import FakeQuantizeModule
        self.U_fake_quant = FakeQuantizeModule(n_bits=n_bits, symmetric=True)
        self.V_fake_quant = FakeQuantizeModule(n_bits=n_bits, symmetric=True)
    
    def forward(self, x):
        """Forward avec fake quantization"""
        # Fake quantize les facteurs
        U_q = self.U_fake_quant(self.U)
        V_q = self.V_fake_quant(self.V)
        
        # Forward
        output = x @ V_q @ U_q.T
        
        return output

# Training
def train_qat_low_rank(model, train_loader, epochs=50):
    """
    Entraîne avec QAT + Low-rank
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}')
    
    return model
```

---

## Pipeline Complet

```python
class CombinedCompressionPipeline:
    """
    Pipeline combinant plusieurs techniques de compression
    """
    
    def __init__(self, model):
        self.model = model
    
    def apply_low_rank_then_quantize(self, rank=64, n_bits=8):
        """
        Étape 1: Rang faible
        Étape 2: Quantification
        """
        compressed_model = self.model
        
        # Remplace les couches par versions low-rank
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_lr = QuantizedLowRankLinear.from_linear(
                    module, rank=rank, n_bits=n_bits
                )
                # Remplace dans le modèle
        
        return compressed_model
    
    def apply_quantize_then_low_rank(self, rank=64, n_bits=8):
        """
        Étape 1: Quantification
        Étape 2: Rang faible sur les poids quantifiés
        """
        # Alternative: quantifier d'abord, puis factoriser
        pass

# Exemple
pipeline = CombinedCompressionPipeline(model)
compressed = pipeline.apply_low_rank_then_quantize(rank=64, n_bits=8)
```

---

## Analyse de Compression Totale

```python
def analyze_combined_compression(original_model, rank, n_bits):
    """
    Analyse la compression totale
    """
    total_original = sum(p.numel() * 4 for p in original_model.parameters())
    
    # Estime la compression
    total_compressed = 0
    for module in original_model.modules():
        if isinstance(module, nn.Linear):
            # Compression low-rank + quant
            compressed_size = rank * (module.in_features + module.out_features) * 1
            total_compressed += compressed_size
    
    compression_ratio = total_original / total_compressed
    
    print(f"Compression Combinée:")
    print(f"  Original: {total_original / (1024*1024):.2f} MB")
    print(f"  Compressé: {total_compressed / (1024*1024):.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}x")
    
    return compression_ratio
```

---

## Exercices

### Exercice 11.4.1
Comparez l'ordre d'application : low-rank→quantize vs quantize→low-rank.

### Exercice 11.4.2
Implémentez QAT pour un modèle low-rank et comparez avec PTQ.

---

## Points Clés à Retenir

> 📌 **La combinaison low-rank + quant donne compression multiplicative**

> 📌 **Low-rank puis quantifier est généralement meilleur que l'inverse**

> 📌 **QAT peut améliorer les résultats pour très faibles précisions**

> 📌 **Compression totale = compression_rank × compression_quant**

---

*Section suivante : [11.5 Analyse Théorique des Erreurs d'Approximation](./11_05_Erreurs.md)*

