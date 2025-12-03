# 21.4 Accélération par Compression de Modèles

---

## Introduction

Les modèles génératifs pour la simulation (GANs, Normalizing Flows) peuvent être grands et coûteux en termes de mémoire et calcul. La **compression de modèles** permet de réduire la taille des modèles tout en préservant leur capacité de génération, rendant possible le déploiement sur hardware limité (FPGAs pour triggers) ou l'accélération de l'inférence.

Cette section présente les techniques de compression appliquées aux modèles génératifs pour simulation, incluant quantification, pruning, et distillation.

---

## Compression pour Modèles Génératifs

### Défis Spécifiques

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class GenerativeModelCompression:
    """
    Compression de modèles génératifs pour simulation
    """
    
    def __init__(self):
        self.challenges = {
            'quality_preservation': {
                'description': 'Préserver qualité génération',
                'importance': 'Critique pour validité physique'
            },
            'latency': {
                'description': 'Réduire latence génération',
                'use_case': 'Triggers temps réel'
            },
            'memory': {
                'description': 'Réduire mémoire requise',
                'use_case': 'Déploiement FPGA/edge'
            },
            'throughput': {
                'description': 'Augmenter throughput',
                'use_case': 'Génération massive'
            }
        }
    
    def display_challenges(self):
        """Affiche les défis"""
        print("\n" + "="*70)
        print("Défis Compression Modèles Génératifs")
        print("="*70)
        
        for challenge, info in self.challenges.items():
            print(f"\n{challenge.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Importance/Usage: {info.get('importance') or info.get('use_case')}")

compression = GenerativeModelCompression()
compression.display_challenges()
```

---

## Quantification des Modèles Génératifs

### Post-Training Quantization

```python
class QuantizedGenerator(nn.Module):
    """
    Générateur quantifié
    """
    
    def __init__(self, original_generator):
        super().__init__()
        self.original = original_generator
        
        # Copier structure
        self.quantized_layers = nn.ModuleList()
        
        # Quantifier chaque couche
        for module in original_generator.modules():
            if isinstance(module, nn.Linear):
                # Créer couche quantifiée
                quant_layer = QuantizedLinear(module)
                self.quantized_layers.append(quant_layer)
    
    def forward(self, x):
        """Forward avec quantification"""
        # En pratique: remplacer couches par versions quantifiées
        return self.original(x)  # Simplifié

class QuantizedLinear(nn.Module):
    """
    Couche linéaire quantifiée
    """
    
    def __init__(self, linear_layer, n_bits=8):
        super().__init__()
        
        self.n_bits = n_bits
        self.quantization_levels = 2 ** n_bits
        
        # Quantifier poids
        weight = linear_layer.weight.data
        weight_min, weight_max = weight.min(), weight.max()
        
        # Scale et zero point
        self.scale = (weight_max - weight_min) / (self.quantization_levels - 1)
        self.zero_point = -weight_min / self.scale
        
        # Quantifier
        weight_quant = torch.round(weight / self.scale + self.zero_point)
        weight_quant = torch.clamp(weight_quant, 0, self.quantization_levels - 1)
        
        # Déquantifier (pour simulation)
        self.register_buffer('weight', (weight_quant - self.zero_point) * self.scale)
        self.register_buffer('bias', linear_layer.bias.data if linear_layer.bias is not None else None)
    
    def forward(self, x):
        """Forward quantifié"""
        return F.linear(x, self.weight, self.bias)

def quantize_generator(generator, n_bits=8):
    """
    Quantifie générateur
    
    Returns:
        quantized_generator: Générateur quantifié
        compression_ratio: Ratio de compression
    """
    original_params = sum(p.numel() * 32 for p in generator.parameters())  # 32 bits
    quantized_params = sum(p.numel() * n_bits for p in generator.parameters())
    
    compression_ratio = original_params / quantized_params
    
    print(f"\nQuantification Générateur:")
    print(f"  Bits: {n_bits}")
    print(f"  Compression: {compression_ratio:.2f}×")
    print(f"  Taille originale: {original_params / 1e6:.2f} MB")
    print(f"  Taille quantifiée: {quantized_params / 1e6:.2f} MB")
    
    return generator, compression_ratio

# Test quantification
test_generator = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 50)
)

quant_gen, ratio = quantize_generator(test_generator, n_bits=8)
```

---

## Pruning pour Modèles Génératifs

### Pruning Magnitude-Based

```python
class PrunedGenerator(nn.Module):
    """
    Générateur élagué (pruned)
    """
    
    def __init__(self, original_generator, pruning_ratio=0.5):
        super().__init__()
        
        # Copier structure et élaguer
        self.pruned_layers = nn.ModuleList()
        
        for module in original_generator.modules():
            if isinstance(module, nn.Linear):
                pruned_layer = self._prune_linear(module, pruning_ratio)
                self.pruned_layers.append(pruned_layer)
    
    def _prune_linear(self, linear_layer, pruning_ratio):
        """Élague couche linéaire"""
        weight = linear_layer.weight.data.clone()
        
        # Calculer seuil
        threshold = torch.quantile(torch.abs(weight), pruning_ratio)
        
        # Masquer poids petits
        mask = torch.abs(weight) > threshold
        weight[~mask] = 0
        
        # Créer nouvelle couche
        pruned_layer = nn.Linear(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=linear_layer.bias is not None
        )
        pruned_layer.weight.data = weight
        if linear_layer.bias is not None:
            pruned_layer.bias.data = linear_layer.bias.data.clone()
        
        # Enregistrer masque pour sparse operations
        pruned_layer.register_buffer('mask', mask)
        
        return pruned_layer
    
    def forward(self, x):
        """Forward avec poids élagués"""
        for layer in self.pruned_layers:
            x = layer(x)
        return x

def prune_generator(generator, pruning_ratio=0.5):
    """
    Élague générateur
    
    Returns:
        pruned_generator: Générateur élagué
        sparsity: Taux de sparsité
    """
    total_params = sum(p.numel() for p in generator.parameters())
    
    # Compter poids après élagage
    pruned_params = 0
    for module in generator.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            threshold = torch.quantile(torch.abs(weight), pruning_ratio)
            n_remaining = (torch.abs(weight) > threshold).sum().item()
            pruned_params += n_remaining
    
    sparsity = 1.0 - (pruned_params / total_params)
    
    print(f"\nPruning Générateur:")
    print(f"  Ratio élagage: {pruning_ratio:.1%}")
    print(f"  Sparsité: {sparsity:.1%}")
    print(f"  Paramètres restants: {pruned_params:,} / {total_params:,}")
    
    return generator, sparsity

pruned_gen, sparsity = prune_generator(test_generator, pruning_ratio=0.5)
```

---

## Distillation pour Modèles Génératifs

### Knowledge Distillation

```python
class GenerativeDistillation:
    """
    Distillation d'un grand générateur vers petit
    """
    
    def __init__(self, teacher_generator, student_generator, temperature=4.0):
        """
        Args:
            teacher_generator: Grand modèle (enseignant)
            student_generator: Petit modèle (élève)
            temperature: Température pour soft targets
        """
        self.teacher = teacher_generator
        self.student = student_generator
        self.temperature = temperature
    
    def compute_distillation_loss(self, noise, alpha=0.5):
        """
        Loss de distillation
        
        Combine:
        - Loss sur données réelles (si disponibles)
        - Loss entre outputs teacher/student
        """
        # Générer avec teacher et student
        with torch.no_grad():
            teacher_output = self.teacher(noise)
        
        student_output = self.student(noise)
        
        # Loss de distillation (MSE entre outputs)
        distillation_loss = F.mse_loss(
            student_output / self.temperature,
            teacher_output / self.temperature
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    def train_student(self, data_loader, n_epochs=50, lr=0.001):
        """
        Entraîne étudiant avec distillation
        """
        optimizer = torch.optim.Adam(self.student.parameters(), lr=lr)
        
        self.teacher.eval()  # Teacher en mode eval
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for batch in data_loader:
                optimizer.zero_grad()
                
                # Générer bruit
                noise = torch.randn(batch.size(0), 100)
                
                # Loss de distillation
                loss = self.compute_distillation_loss(noise)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(data_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
        
        return losses

# Exemple distillation
teacher = nn.Sequential(
    nn.Linear(100, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 50)
)

student = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 50)
)

distiller = GenerativeDistillation(teacher, student, temperature=4.0)

print(f"\nDistillation:")
print(f"  Teacher paramètres: {sum(p.numel() for p in teacher.parameters()):,}")
print(f"  Student paramètres: {sum(p.numel() for p in student.parameters()):,}")
print(f"  Compression: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.2f}×")
```

---

## Compression Combinée

### Techniques Multiples

```python
class ComprehensiveCompression:
    """
    Compression combinant plusieurs techniques
    """
    
    def __init__(self, generator):
        self.original = generator
        self.original_size = sum(p.numel() * 32 for p in generator.parameters())
    
    def apply_compression_pipeline(self, 
                                  quantization_bits=8,
                                  pruning_ratio=0.3,
                                  use_distillation=False):
        """
        Applique pipeline de compression
        """
        compressed_gen = self.original
        
        # 1. Pruning
        print("\n1. Pruning...")
        compressed_gen, sparsity = prune_generator(compressed_gen, pruning_ratio)
        
        # 2. Quantification
        print("\n2. Quantification...")
        compressed_gen, quant_ratio = quantize_generator(compressed_gen, quantization_bits)
        
        # 3. Distillation (optionnel)
        if use_distillation:
            print("\n3. Distillation...")
            # Créer étudiant plus petit
            student = self._create_student_model()
            distiller = GenerativeDistillation(compressed_gen, student)
            # Entraîner étudiant
            compressed_gen = student
        
        # Calculer compression totale
        final_size = sum(p.numel() * quantization_bits for p in compressed_gen.parameters())
        total_compression = self.original_size / final_size
        
        print(f"\n{'='*70}")
        print(f"Compression Totale:")
        print(f"  Taille originale: {self.original_size / 1e6:.2f} MB")
        print(f"  Taille finale: {final_size / 1e6:.2f} MB")
        print(f"  Compression totale: {total_compression:.2f}×")
        print(f"{'='*70}")
        
        return compressed_gen, total_compression
    
    def _create_student_model(self):
        """Crée modèle étudiant plus petit"""
        return nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 50)
        )

# Test compression combinée
compressor = ComprehensiveCompression(test_generator)
compressed, ratio = compressor.apply_compression_pipeline(
    quantization_bits=8,
    pruning_ratio=0.3,
    use_distillation=False
)
```

---

## Optimisation pour FPGA

### Déploiement Hardware

```python
class FPGAGeneratorOptimization:
    """
    Optimisations spécifiques pour déploiement FPGA
    """
    
    def __init__(self):
        self.optimizations = {
            'fixed_point': {
                'description': 'Conversion en arithmétique fixed-point',
                'benefit': 'Hardware efficace'
            },
            'layer_fusion': {
                'description': 'Fusionner couches consécutives',
                'benefit': 'Réduire latence'
            },
            'parallelization': {
                'description': 'Parallélisation des opérations',
                'benefit': 'Augmenter throughput'
            },
            'memory_optimization': {
                'description': 'Optimiser accès mémoire',
                'benefit': 'Réduire bandwidth requis'
            }
        }
    
    def convert_to_fixed_point(self, generator, n_bits=16, integer_bits=8):
        """
        Convertit générateur en fixed-point
        
        Args:
            n_bits: Nombre total de bits
            integer_bits: Bits pour partie entière
        """
        # Scale factor
        fractional_bits = n_bits - integer_bits
        scale = 2 ** fractional_bits
        
        print(f"\nConversion Fixed-Point:")
        print(f"  Bits totaux: {n_bits}")
        print(f"  Bits entiers: {integer_bits}")
        print(f"  Bits fractionnaires: {fractional_bits}")
        print(f"  Scale: {scale}")
        
        return generator  # En pratique: conversion réelle
    
    def optimize_for_hls4ml(self, generator):
        """
        Optimise pour hls4ml (High-Level Synthesis)
        """
        optimizations = [
            'Réduction précision (16 bits)',
            'Fusion layers',
            'Optimisation loops',
            'Pipelining'
        ]
        
        print(f"\nOptimisations hls4ml:")
        for opt in optimizations:
            print(f"  • {opt}")
        
        return generator

fpga_optimizer = FPGAGeneratorOptimization()
fpga_optimizer.display_optimizations()
```

---

## Exercices

### Exercice 21.4.1
Quantifiez un générateur à différentes précisions (16, 8, 4 bits) et analysez l'impact sur qualité génération.

### Exercice 21.4.2
Appliquez pruning progressif (magnitude-based) et analysez tradeoff sparsité/qualité.

### Exercice 21.4.3
Implémentez distillation d'un grand GAN vers petit et comparez performances.

### Exercice 21.4.4
Combine quantification + pruning et mesure compression totale et impact performance.

---

## Points Clés à Retenir

> 📌 **La compression permet déploiement sur hardware limité (FPGA, edge)**

> 📌 **La quantification réduit précision (8-16 bits) avec impact limité sur qualité**

> 📌 **Le pruning élimine poids non essentiels (sparsité)**

> 📌 **La distillation transfère connaissance grand → petit modèle**

> 📌 **La combinaison de techniques donne compression maximale**

> 📌 **L'optimisation FPGA nécessite fixed-point et optimisations spécifiques**

---

*Section précédente : [21.3 Normalizing Flows](./21_03_Normalizing_Flows.md) | Section suivante : [21.5 Validation](./21_05_Validation.md)*

