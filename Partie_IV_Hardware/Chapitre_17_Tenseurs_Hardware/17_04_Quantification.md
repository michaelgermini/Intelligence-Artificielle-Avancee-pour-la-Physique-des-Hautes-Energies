# 17.4 Quantification Hardware-Aware pour Tenseurs

---

## Introduction

La **quantification des réseaux de tenseurs** présente des défis uniques comparés aux réseaux de neurones classiques. Les tenseurs accumulent des erreurs lors de contractions séquentielles, et la structure des réseaux de tenseurs (bond dimensions, rang) affecte la propagation des erreurs de quantification.

Cette section présente les méthodes de quantification spécifiques aux tenseurs, incluant la quantification uniforme, non-uniforme, et adaptative, ainsi que les techniques pour gérer l'accumulation d'erreurs.

---

## Défis Spécifiques aux Tenseurs

### Accumulation d'Erreurs

```python
import numpy as np
import torch

class TensorQuantizationChallenges:
    """
    Défis spécifiques à la quantification de tenseurs
    """
    
    def __init__(self):
        self.challenges = {
            'error_accumulation': {
                'description': 'Erreurs s\'accumulent lors de contractions séquentielles',
                'example': 'MPS avec 100 contractions: erreurs se propagent',
                'impact': 'Nécessite quantification plus conservative ou correction'
            },
            'bond_dimension_sensitivity': {
                'description': 'Sensibilité aux bond dimensions',
                'example': 'Grandes bond dimensions amplifient erreurs',
                'impact': 'Peut nécessiter bond dimension adaptative'
            },
            'rank_preservation': {
                'description': 'Quantification peut changer le rang effectif',
                'example': 'Rang apparent augmente avec erreurs',
                'impact': 'Affecte efficacité compression'
            },
            'numerical_stability': {
                'description': 'Stabilité numérique dans accumulations longues',
                'example': 'Overflow/underflow dans sommes longues',
                'impact': 'Nécessite gestion dynamique de range'
            }
        }
    
    def demonstrate_error_accumulation(self, n_contractions=10):
        """
        Démontre accumulation d'erreurs dans contractions quantifiées
        """
        # Valeurs réelles (float32)
        values_real = np.random.rand(n_contractions).astype(np.float32)
        
        # Quantification 8-bit
        scale = 255.0 / np.max(np.abs(values_real))
        values_quantized = np.round(values_real * scale).astype(np.int8)
        values_dequantized = values_quantized.astype(np.float32) / scale
        
        # Erreur par étape
        error_per_step = np.abs(values_real - values_dequantized)
        
        # Accumulation (simulation: somme des valeurs)
        sum_real = np.sum(values_real)
        sum_quantized = np.sum(values_dequantized)
        total_error = np.abs(sum_real - sum_quantized)
        
        return {
            'error_per_step': error_per_step,
            'total_error': total_error,
            'error_amplification': total_error / np.mean(error_per_step)
        }

challenges = TensorQuantizationChallenges()
print("\n" + "="*70)
print("Défis de Quantification pour Tenseurs")
print("="*70)
for challenge, info in challenges.challenges.items():
    print(f"\n{challenge.replace('_', ' ').title()}:")
    print(f"  {info['description']}")
    print(f"  Impact: {info['impact']}")

# Démontration
error_demo = challenges.demonstrate_error_accumulation()
print(f"\nAmplification d'erreur: {error_demo['error_amplification']:.2f}x")
```

---

## Quantification Uniforme

### Quantification Standard Adaptée

```python
class UniformTensorQuantization:
    """
    Quantification uniforme pour tenseurs
    """
    
    def __init__(self, num_bits=8, symmetric=True):
        """
        Args:
            num_bits: Nombre de bits (4, 8, 16)
            symmetric: True pour [-max, max], False pour [min, max]
        """
        self.num_bits = num_bits
        self.symmetric = symmetric
        self.quant_max = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** num_bits - 1
        self.quant_min = -self.quant_max if symmetric else 0
    
    def quantize_tensor(self, tensor):
        """
        Quantifie un tenseur
        
        Returns:
            (quantized_tensor, scale, zero_point)
        """
        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.detach().cpu().numpy()
            return_torch = True
        else:
            tensor_np = tensor
            return_torch = False
        
        # Calculer scale et zero_point
        if self.symmetric:
            scale = np.max(np.abs(tensor_np)) / self.quant_max
            zero_point = 0
        else:
            tensor_min = np.min(tensor_np)
            tensor_max = np.max(tensor_np)
            scale = (tensor_max - tensor_min) / (self.quant_max - self.quant_min)
            zero_point = np.round(self.quant_min - tensor_min / scale)
        
        # Quantifier
        quantized = np.round(tensor_np / scale + zero_point)
        quantized = np.clip(quantized, self.quant_min, self.quant_max).astype(np.int8)
        
        # Déquantifier (pour vérification)
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        result = {
            'quantized': torch.from_numpy(quantized) if return_torch else quantized,
            'dequantized': torch.from_numpy(dequantized) if return_torch else dequantized,
            'scale': scale,
            'zero_point': zero_point
        }
        
        return result
    
    def quantize_mps_tensors(self, mps_tensors):
        """
        Quantifie tous les tenseurs d'un MPS
        
        Stratégie: Quantifier chaque tenseur indépendamment
        """
        quantized_mps = []
        scales = []
        
        for tensor in mps_tensors:
            quant_result = self.quantize_tensor(tensor)
            quantized_mps.append(quant_result['quantized'])
            scales.append(quant_result['scale'])
        
        return {
            'quantized_tensors': quantized_mps,
            'scales': scales,
            'num_bits': self.num_bits
        }
    
    def simulate_quantized_contraction(self, tensor_A, tensor_B):
        """
        Simule une contraction avec tenseurs quantifiés
        """
        # Quantifier
        quant_A = self.quantize_tensor(tensor_A)
        quant_B = self.quantize_tensor(tensor_B)
        
        # Contraction sur tenseurs quantifiés
        # A[i,j,k] * B[j,k,l] → C[i,l]
        A_q = quant_A['quantized'].float()
        B_q = quant_B['quantized'].float()
        scale_A = quant_A['scale']
        scale_B = quant_B['scale']
        
        # Contraction quantifiée
        if isinstance(A_q, torch.Tensor):
            C_q = torch.einsum('ijk,jkl->il', A_q, B_q)
        else:
            C_q = np.einsum('ijk,jkl->il', A_q, B_q)
        
        # Scale du résultat
        scale_C = scale_A * scale_B
        
        # Quantifier le résultat
        quant_C = self.quantize_tensor(C_q)
        
        # Comparer avec contraction réelle
        if isinstance(tensor_A, torch.Tensor):
            C_real = torch.einsum('ijk,jkl->il', tensor_A.float(), tensor_B.float())
        else:
            C_real = np.einsum('ijk,jkl->il', tensor_A.astype(np.float32), 
                              tensor_B.astype(np.float32))
        
        error = np.abs(quant_C['dequantized'] - (C_real if not isinstance(C_real, torch.Tensor) 
                                                 else C_real.numpy()))
        
        return {
            'quantized_result': quant_C,
            'real_result': C_real,
            'error': error,
            'mean_error': np.mean(error),
            'max_error': np.max(error)
        }

# Exemple
uniform_quant = UniformTensorQuantization(num_bits=8, symmetric=True)

# Test sur contraction simple
A = np.random.rand(10, 20, 15).astype(np.float32)
B = np.random.rand(20, 15, 25).astype(np.float32)

contraction_result = uniform_quant.simulate_quantized_contraction(A, B)

print("\n" + "="*70)
print("Quantification Uniforme")
print("="*70)
print(f"Erreur moyenne: {contraction_result['mean_error']:.2e}")
print(f"Erreur max: {contraction_result['max_error']:.2e}")
print(f"Erreur relative: {contraction_result['mean_error'] / np.mean(np.abs(contraction_result['real_result'])):.2%}")
```

---

## Quantification Non-Uniforme

### Quantification Adaptative par Tenseur

```python
class NonUniformTensorQuantization:
    """
    Quantification non-uniforme adaptative
    """
    
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
    
    def per_tensor_quantization(self, mps_tensors):
        """
        Quantification avec scale différent par tenseur
        
        Avantage: Adapte à distribution de chaque tenseur
        """
        quantized_tensors = []
        scales = []
        
        for i, tensor in enumerate(mps_tensors):
            # Calculer scale optimal pour ce tenseur
            tensor_abs_max = np.max(np.abs(tensor))
            scale = tensor_abs_max / (2 ** (self.num_bits - 1) - 1)
            
            # Quantifier
            quantized = np.round(tensor / scale)
            quantized = np.clip(quantized, -(2**(self.num_bits-1)-1), 2**(self.num_bits-1)-1)
            quantized = quantized.astype(np.int8)
            
            quantized_tensors.append(quantized)
            scales.append(scale)
        
        return {
            'quantized_tensors': quantized_tensors,
            'scales': scales
        }
    
    def per_channel_quantization(self, tensor, channel_dim=0):
        """
        Quantification par canal/channel
        
        Pour tenseur: quantifier chaque "tranche" indépendamment
        """
        scales = []
        quantized_slices = []
        
        # Parcourir dimension de canal
        for i in range(tensor.shape[channel_dim]):
            if channel_dim == 0:
                slice_tensor = tensor[i, ...]
            elif channel_dim == 1:
                slice_tensor = tensor[:, i, ...]
            else:
                # Généralisation
                indices = [slice(None)] * len(tensor.shape)
                indices[channel_dim] = i
                slice_tensor = tensor[tuple(indices)]
            
            # Quantifier cette tranche
            scale = np.max(np.abs(slice_tensor)) / (2 ** (self.num_bits - 1) - 1)
            quantized = np.round(slice_tensor / scale)
            quantized = np.clip(quantized, -(2**(self.num_bits-1)-1), 2**(self.num_bits-1)-1)
            
            scales.append(scale)
            quantized_slices.append(quantized.astype(np.int8))
        
        # Reconstruire tenseur
        quantized_tensor = np.stack(quantized_slices, axis=channel_dim)
        
        return {
            'quantized_tensor': quantized_tensor,
            'scales': np.array(scales)
        }
    
    def adaptive_bond_dimension_quantization(self, mps_tensors, bond_dims):
        """
        Quantification adaptative selon bond dimensions
        
        Stratégie: Plus de précision pour grandes bond dimensions (sensibles)
        """
        quantized_tensors = []
        bit_widths = []
        
        for i, (tensor, bond_dim) in enumerate(zip(mps_tensors, bond_dims)):
            # Plus de bits pour grandes bond dimensions
            if bond_dim > 64:
                bits = 16
            elif bond_dim > 32:
                bits = 12
            else:
                bits = 8
            
            # Quantifier avec nombre de bits adaptatif
            quantizer = UniformTensorQuantization(num_bits=bits)
            quant_result = quantizer.quantize_tensor(tensor)
            
            quantized_tensors.append(quant_result['quantized'])
            bit_widths.append(bits)
        
        return {
            'quantized_tensors': quantized_tensors,
            'bit_widths': bit_widths,
            'avg_bits': np.mean(bit_widths)
        }

non_uniform = NonUniformTensorQuantization(num_bits=8)

# Exemple per-tensor
mps_tensors = [np.random.rand(10, 20).astype(np.float32) for _ in range(5)]
per_tensor_result = non_uniform.per_tensor_quantization(mps_tensors)

print("\n" + "="*70)
print("Quantification Non-Uniforme")
print("="*70)
print(f"Scales par tenseur: {[f'{s:.4f}' for s in per_tensor_result['scales']]}")
```

---

## Quantification Hardware-Aware

### Adaptation aux Contraintes Hardware

```python
class HardwareAwareTensorQuantization:
    """
    Quantification adaptée aux contraintes hardware spécifiques
    """
    
    def __init__(self, hardware_type='fpga', target_precision='int8'):
        """
        Args:
            hardware_type: 'fpga', 'gpu', 'asic'
            target_precision: 'int4', 'int8', 'int16'
        """
        self.hardware_type = hardware_type
        self.target_precision = target_precision
        
        # Contraintes hardware
        self.hardware_constraints = {
            'fpga': {
                'supported_precisions': [4, 8, 16, 32],
                'dsp_efficiency': {4: 0.5, 8: 1.0, 16: 0.8, 32: 0.5},  # Relatif à int8
                'memory_efficiency': {4: 2.0, 8: 1.0, 16: 0.5, 32: 0.25}  # Relatif à int8
            },
            'gpu': {
                'supported_precisions': [8, 16, 32],
                'tensor_core_support': [8, 16],  # fp16/int8
                'memory_efficiency': {8: 1.0, 16: 0.5, 32: 0.25}
            }
        }
    
    def mixed_precision_quantization(self, mps_tensors, sensitivity_analysis):
        """
        Quantification mixed-precision selon sensibilité
        
        Args:
            sensitivity_analysis: Erreur introduite par quantification de chaque tenseur
        """
        quantized_tensors = []
        precisions = []
        
        for i, (tensor, sensitivity) in enumerate(zip(mps_tensors, sensitivity_analysis)):
            # Plus de précision pour tenseurs sensibles
            if sensitivity > 0.1:  # Seuil
                precision = 16
            elif sensitivity > 0.05:
                precision = 12
            else:
                precision = 8
            
            # Quantifier avec précision choisie
            quantizer = UniformTensorQuantization(num_bits=precision)
            quant_result = quantizer.quantize_tensor(tensor)
            
            quantized_tensors.append(quant_result['quantized'])
            precisions.append(precision)
        
        # Estimer ressources hardware
        resources = self.estimate_hardware_resources(quantized_tensors, precisions)
        
        return {
            'quantized_tensors': quantized_tensors,
            'precisions': precisions,
            'hardware_resources': resources
        }
    
    def estimate_hardware_resources(self, quantized_tensors, precisions):
        """
        Estime ressources hardware nécessaires
        """
        if self.hardware_type == 'fpga':
            # Estimer LUT, DSP, BRAM
            total_memory = 0
            dsp_efficiency = 1.0
            
            for tensor, precision in zip(quantized_tensors, precisions):
                elements = np.prod(tensor.shape)
                bytes_per_element = precision // 8
                total_memory += elements * bytes_per_element
                
                # DSP efficiency
                if precision in self.hardware_constraints['fpga']['dsp_efficiency']:
                    dsp_eff = self.hardware_constraints['fpga']['dsp_efficiency'][precision]
                    dsp_efficiency = min(dsp_efficiency, dsp_eff)
            
            return {
                'memory_mb': total_memory / (1024 ** 2),
                'dsp_efficiency': dsp_efficiency,
                'memory_efficiency': 1.0 / np.mean([self.hardware_constraints['fpga']['memory_efficiency'].get(p, 1.0) 
                                                   for p in precisions])
            }
        else:
            return {'memory_mb': 0}
    
    def hardware_optimized_quantization(self, mps_tensors, constraints):
        """
        Quantification optimisée pour contraintes hardware spécifiques
        
        Args:
            constraints: {'max_memory_mb': 10, 'max_latency_ns': 100000}
        """
        # Trouver précision qui respecte contraintes
        best_precision = 8
        best_result = None
        
        for precision in [4, 8, 16]:
            quantizer = UniformTensorQuantization(num_bits=precision)
            quantized = quantizer.quantize_mps_tensors(mps_tensors)
            
            # Estimer ressources
            resources = self.estimate_hardware_resources(
                quantized['quantized_tensors'], 
                [precision] * len(mps_tensors)
            )
            
            # Vérifier contraintes
            if resources['memory_mb'] <= constraints.get('max_memory_mb', float('inf')):
                # Évaluer qualité (erreur)
                total_error = self.evaluate_quantization_error(mps_tensors, 
                                                               quantized['quantized_tensors'],
                                                               quantized['scales'])
                
                if best_result is None or total_error < best_result['error']:
                    best_precision = precision
                    best_result = {
                        'quantized': quantized,
                        'precision': precision,
                        'resources': resources,
                        'error': total_error
                    }
        
        return best_result
    
    def evaluate_quantization_error(self, original_tensors, quantized_tensors, scales):
        """Évalue erreur de quantification"""
        total_error = 0
        for orig, quant, scale in zip(original_tensors, quantized_tensors, scales):
            dequant = quant.astype(np.float32) * scale
            error = np.mean(np.abs(orig - dequant))
            total_error += error
        return total_error / len(original_tensors)

# Exemple hardware-aware
hw_quant = HardwareAwareTensorQuantization(hardware_type='fpga', target_precision='int8')

mps_tensors = [np.random.rand(10, 20).astype(np.float32) for _ in range(5)]
sensitivity = [0.02, 0.08, 0.15, 0.05, 0.03]  # Sensibilité de chaque tenseur

mixed_prec_result = hw_quant.mixed_precision_quantization(mps_tensors, sensitivity)

print("\n" + "="*70)
print("Quantification Hardware-Aware")
print("="*70)
print(f"Précisions: {mixed_prec_result['precisions']}")
print(f"Mémoire estimée: {mixed_prec_result['hardware_resources']['memory_mb']:.2f} MB")
```

---

## Gestion de l'Accumulation d'Erreurs

### Techniques de Correction

```python
class ErrorAccumulationManagement:
    """
    Gestion de l'accumulation d'erreurs dans contractions quantifiées
    """
    
    def re_quantization_after_contractions(self, n_contractions, quantization_bits):
        """
        Re-quantifier après chaque N contractions
        
        Réduit accumulation en "resetant" erreurs périodiquement
        """
        return {
            'strategy': 'Re-quantification périodique',
            'frequency': f'Chaque {n_contractions} contractions',
            'benefit': 'Limite croissance erreurs',
            'cost': 'Opérations de quantification supplémentaires'
        }
    
    def error_correction_terms(self):
        """
        Ajouter termes de correction pour compenser erreurs systématiques
        """
        return {
            'strategy': 'Correction d\'erreur explicite',
            'method': 'Estimer erreur moyenne et soustraire',
            'benefit': 'Réduit biais systématique',
            'complexity': 'Augmente complexité computationnelle'
        }
    
    def adaptive_bond_dimension_reduction(self, mps_tensors, threshold=1e-6):
        """
        Réduire bond dimensions adaptativement pour limiter propagation erreurs
        
        Truncate petites valeurs singulières qui sont affectées par erreurs
        """
        truncated_tensors = []
        
        for tensor in mps_tensors:
            # SVD pour trouver valeurs importantes
            if len(tensor.shape) == 2:
                U, s, Vt = np.linalg.svd(tensor, full_matrices=False)
                
                # Truncate valeurs singulières petites (probablement bruit)
                mask = s > threshold
                s_trunc = s[mask]
                U_trunc = U[:, mask]
                Vt_trunc = Vt[mask, :]
                
                tensor_trunc = U_trunc @ np.diag(s_trunc) @ Vt_trunc
                truncated_tensors.append(tensor_trunc)
            else:
                truncated_tensors.append(tensor)
        
        return truncated_tensors

error_mgmt = ErrorAccumulationManagement()

print("\n" + "="*70)
print("Gestion Accumulation d'Erreurs")
print("="*70)
re_quant = error_mgmt.re_quantization_after_contractions(n_contractions=10, quantization_bits=8)
print(f"Stratégie: {re_quant['strategy']}")
print(f"Bénéfice: {re_quant['benefit']}")
```

---

## Quantification pour FPGA Spécifique

### Optimisation FPGA

```python
class FPGAOptimizedQuantization:
    """
    Optimisations spécifiques pour FPGA
    """
    
    def power_of_two_quantization(self, tensor):
        """
        Quantification avec scale puissance de 2
        
        Avantage FPGA: Multiplication devient shift (plus rapide)
        """
        # Trouver scale puissance de 2 proche
        max_val = np.max(np.abs(tensor))
        log2_scale = np.ceil(np.log2(max_val / (2**7 - 1)))  # Pour int8
        scale = 2 ** log2_scale
        
        # Quantifier
        quantized = np.round(tensor / scale)
        quantized = np.clip(quantized, -127, 127).astype(np.int8)
        
        return {
            'quantized': quantized,
            'scale': scale,
            'scale_is_power_of_2': True,
            'dequantization_method': 'bit_shift'  # Plus rapide sur FPGA
        }
    
    def fpga_resource_optimization(self, quantized_tensors, precisions):
        """
        Optimise utilisation ressources FPGA
        
        Stratégies:
        - Packing de données pour utiliser BRAM efficacement
        - Choisir précisions qui minimisent DSP usage
        """
        # Estimer ressources
        total_lut = 0
        total_dsp = 0
        total_bram = 0
        
        for tensor, precision in zip(quantized_tensors, precisions):
            elements = np.prod(tensor.shape)
            
            # BRAM: pack données pour utiliser largeur efficacement
            # Exemple: 2 int8 dans 1 int16 = utilisation BRAM × 2
            if precision == 8:
                packed_elements = elements // 2
                bram_usage = np.ceil(packed_elements / (36 * 1024 / 2))  # 2 bytes par mot
            else:
                bram_usage = np.ceil(elements * (precision // 8) / (36 * 1024))
            
            total_bram += bram_usage
            
            # DSP: dépend de précision et opérations
            if precision <= 8:
                dsp_per_op = 1
            elif precision <= 16:
                dsp_per_op = 2
            else:
                dsp_per_op = 4
            
            # Estimation simplifiée
            n_ops = elements  # Approximation
            total_dsp += n_ops * dsp_per_op
        
        return {
            'lut_estimate': total_lut,
            'dsp_estimate': total_dsp,
            'bram_estimate': total_bram,
            'optimization_tips': [
                'Utiliser powers of 2 pour scales',
                'Packing données dans BRAM',
                'Mixed precision selon sensibilité'
            ]
        }

fpga_quant = FPGAOptimizedQuantization()

# Exemple
tensor = np.random.rand(100, 200).astype(np.float32)
power2_result = fpga_quant.power_of_two_quantization(tensor)

print("\n" + "="*70)
print("Quantification Optimisée FPGA")
print("="*70)
print(f"Scale (puissance de 2): {power2_result['scale']}")
print(f"Déquantification: {power2_result['dequantization_method']} (très rapide)")
```

---

## Exercices

### Exercice 17.4.1
Implémentez une quantification uniforme pour un MPS et analysez l'accumulation d'erreurs sur 20 contractions.

### Exercice 17.4.2
Développez une stratégie de mixed-precision qui assigne plus de bits aux tenseurs avec grandes bond dimensions.

### Exercice 17.4.3
Comparez quantification standard vs puissance de 2 sur FPGA en termes de ressources et performance.

### Exercice 17.4.4
Implémentez une méthode de re-quantification périodique et mesurez son impact sur l'accumulation d'erreurs.

---

## Points Clés à Retenir

> 📌 **L'accumulation d'erreurs est critique dans réseaux de tenseurs avec nombreuses contractions**

> 📌 **Quantification uniforme est simple mais peut être sous-optimale**

> 📌 **Quantification non-uniforme (per-tensor, per-channel) peut améliorer qualité**

> 📌 **Quantification hardware-aware adapte précision aux contraintes hardware**

> 📌 **Re-quantification périodique limite accumulation d'erreurs**

> 📌 **Pour FPGA, powers-of-2 scales permettent optimisations (shifts au lieu de multiplications)**

---

*Section précédente : [17.3 Mapping sur Architectures Parallèles](./17_03_Mapping.md)*

