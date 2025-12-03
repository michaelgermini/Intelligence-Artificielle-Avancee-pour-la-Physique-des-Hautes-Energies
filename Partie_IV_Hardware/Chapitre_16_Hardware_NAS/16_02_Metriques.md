# 16.2 Métriques Hardware

---

## Introduction

Les **métriques hardware** sont essentielles pour le Hardware-Aware NAS. Elles permettent d'évaluer comment une architecture de réseau se comporte sur le hardware cible, en termes de **latence**, **consommation énergétique**, **utilisation des ressources**, et **coût de déploiement**.

Cette section présente les principales métriques hardware utilisées dans le NAS, leurs méthodes d'estimation, et leur intégration dans le processus de recherche d'architectures.

---

## Métriques Principales

### Vue d'Ensemble

```python
class HardwareMetrics:
    """
    Collection de métriques hardware pour évaluation NAS
    """
    
    def __init__(self):
        self.metrics = {
            'latency': {
                'description': 'Temps d\'exécution (inférence)',
                'unit': 'ns (nanosecondes) ou μs',
                'importance': 'Critique pour applications temps-réel',
                'constraints': 'Trigger HEP: < 100 ns'
            },
            'throughput': {
                'description': 'Nombre d\'inférences par seconde',
                'unit': 'inferences/sec',
                'importance': 'Important pour traitement batch',
                'constraints': 'Dépend de latence et parallélisme'
            },
            'energy': {
                'description': 'Consommation énergétique par inférence',
                'unit': 'pJ (picojoules) ou nJ',
                'importance': 'Critique pour edge devices, FPGA',
                'constraints': 'Budget énergétique limité'
            },
            'power': {
                'description': 'Puissance moyenne',
                'unit': 'W (watts)',
                'importance': 'Contraintes thermiques',
                'constraints': 'Dissipation thermique limitée'
            },
            'area': {
                'description': 'Surface occupée sur le chip',
                'unit': 'LUT, DSP, BRAM (FPGA) ou mm² (ASIC)',
                'importance': 'Contraintes physiques',
                'constraints': 'Ressources FPGA limitées'
            },
            'memory': {
                'description': 'Mémoire nécessaire (weights + activations)',
                'unit': 'MB ou KB',
                'importance': 'Contraintes mémoire',
                'constraints': 'Mémoire on-chip limitée'
            }
        }
    
    def display_metrics(self):
        """Affiche toutes les métriques"""
        print("\n" + "="*70)
        print("Hardware Metrics Overview")
        print("="*70)
        
        for metric, info in self.metrics.items():
            print(f"\n{metric.upper()}:")
            print(f"  Description: {info['description']}")
            print(f"  Unit: {info['unit']}")
            print(f"  Importance: {info['importance']}")
            print(f"  Constraints: {info['constraints']}")

metrics = HardwareMetrics()
metrics.display_metrics()
```

---

## Estimation de Latence

### Modèles Analytiques

```python
import torch
import torch.nn as nn
import numpy as np

class LatencyEstimator:
    """
    Estimateur de latence pour différents types de hardware
    """
    
    def __init__(self, hardware_type='fpga', clock_freq_mhz=200):
        """
        Args:
            hardware_type: 'fpga', 'gpu', 'cpu', 'asic'
            clock_freq_mhz: Fréquence d'horloge en MHz
        """
        self.hardware_type = hardware_type
        self.clock_period_ns = 1000.0 / clock_freq_mhz
        
        # Modèles de latence par type d'opération (cycles d'horloge)
        self.latency_models = {
            'fpga': {
                'linear': self._fpga_linear_latency,
                'conv2d': self._fpga_conv2d_latency,
                'bn': self._fpga_batch_norm_latency,
                'relu': self._fpga_activation_latency,
                'pool': self._fpga_pooling_latency
            },
            'gpu': {
                'linear': self._gpu_linear_latency,
                'conv2d': self._gpu_conv2d_latency,
                'bn': self._gpu_batch_norm_latency,
                'relu': self._gpu_activation_latency,
                'pool': self._gpu_pooling_latency
            }
        }
    
    def estimate_model_latency(self, model, input_shape=(1, 3, 224, 224)):
        """
        Estime la latence totale d'un modèle
        
        Args:
            model: Modèle PyTorch
            input_shape: Shape de l'input (batch, channels, height, width)
        
        Returns:
            Latence en nanosecondes
        """
        total_cycles = 0
        current_shape = input_shape
        
        model.eval()
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    cycles = self._estimate_layer_latency(module, current_shape, 'linear')
                    total_cycles += cycles
                    # Mise à jour shape (approximation)
                    current_shape = (current_shape[0], module.out_features)
                
                elif isinstance(module, nn.Conv2d):
                    cycles = self._estimate_layer_latency(module, current_shape, 'conv2d')
                    total_cycles += cycles
                    # Mise à jour shape
                    h_out = (current_shape[2] + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1) - 1) // module.stride[0] + 1
                    w_out = (current_shape[3] + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1) - 1) // module.stride[1] + 1
                    current_shape = (current_shape[0], module.out_channels, h_out, w_out)
                
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    cycles = self._estimate_layer_latency(module, current_shape, 'bn')
                    total_cycles += cycles
                
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                    cycles = self._estimate_layer_latency(module, current_shape, 'relu')
                    total_cycles += cycles
                
                elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                    cycles = self._estimate_layer_latency(module, current_shape, 'pool')
                    total_cycles += cycles
                    # Mise à jour shape
                    h_out = (current_shape[2] - module.kernel_size) // module.stride + 1
                    w_out = (current_shape[3] - module.kernel_size) // module.stride + 1
                    current_shape = (current_shape[0], current_shape[1], h_out, w_out)
        
        latency_ns = total_cycles * self.clock_period_ns
        return latency_ns
    
    def _estimate_layer_latency(self, module, input_shape, layer_type):
        """Estime la latence d'une couche"""
        if self.hardware_type in self.latency_models:
            estimator = self.latency_models[self.hardware_type][layer_type]
            return estimator(module, input_shape)
        else:
            return self._default_latency(module, input_shape, layer_type)
    
    def _fpga_linear_latency(self, module, input_shape):
        """
        Latence FPGA pour couche linéaire
        
        Approximation: dépend du nombre d'opérations et du parallélisme
        """
        # Nombre d'opérations MAC (Multiply-Accumulate)
        n_mac_ops = module.in_features * module.out_features
        
        # Parallélisme: nombre de MACs en parallèle (typiquement limité par ressources)
        # Hypothèse: parallélisme de 64 (peut être configuré)
        parallelism = 64
        n_cycles = np.ceil(n_mac_ops / parallelism)
        
        return int(n_cycles)
    
    def _fpga_conv2d_latency(self, module, input_shape):
        """
        Latence FPGA pour couche convolutive
        
        Plus complexe: dépend de la taille des kernels et du parallélisme
        """
        batch, in_channels, h_in, w_in = input_shape
        out_channels = module.out_channels
        
        # Opérations par output pixel
        kernel_ops = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
        
        # Nombre de pixels de sortie
        h_out = (h_in + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1) - 1) // module.stride[0] + 1
        w_out = (w_in + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1) - 1) // module.stride[1] + 1
        
        n_output_pixels = h_out * w_out * out_channels
        total_mac_ops = n_output_pixels * kernel_ops
        
        # Parallélisme: dépend de la configuration FPGA
        parallelism = 32  # MACs en parallèle
        n_cycles = np.ceil(total_mac_ops / parallelism)
        
        return int(n_cycles)
    
    def _fpga_batch_norm_latency(self, module, input_shape):
        """Latence pour BatchNorm (opérations simples)"""
        # BatchNorm: quelques cycles par élément (normalisation + scale + shift)
        n_elements = np.prod(input_shape)
        cycles_per_element = 3  # approximation
        return int(n_elements * cycles_per_element / 128)  # parallélisme
    
    def _fpga_activation_latency(self, module, input_shape):
        """Latence pour activation (très rapide)"""
        n_elements = np.prod(input_shape)
        # Activations: très peu de cycles, souvent pipelined
        return int(n_elements / 256)  # haut parallélisme
    
    def _fpga_pooling_latency(self, module, input_shape):
        """Latence pour pooling"""
        n_elements = np.prod(input_shape)
        return int(n_elements / 128)
    
    # GPU estimators (simplifiés)
    def _gpu_linear_latency(self, module, input_shape):
        """Latence GPU pour couche linéaire (beaucoup plus rapide)"""
        n_mac_ops = module.in_features * module.out_features
        # GPU: très haut parallélisme, latence principale = memory access
        return int(np.sqrt(n_mac_ops) * 0.1)  # approximation
    
    def _gpu_conv2d_latency(self, module, input_shape):
        """Latence GPU pour conv (optimisé)"""
        return int(self._fpga_conv2d_latency(module, input_shape) * 0.01)  # beaucoup plus rapide
    
    def _gpu_batch_norm_latency(self, module, input_shape):
        return int(self._fpga_batch_norm_latency(module, input_shape) * 0.1)
    
    def _gpu_activation_latency(self, module, input_shape):
        return 1  # très rapide sur GPU
    
    def _gpu_pooling_latency(self, module, input_shape):
        return int(self._fpga_pooling_latency(module, input_shape) * 0.1)
    
    def _default_latency(self, module, input_shape, layer_type):
        """Latence par défaut (approximation)"""
        return 1000  # placeholder

# Exemple d'utilisation
latency_est_fpga = LatencyEstimator('fpga', clock_freq_mhz=200)
latency_est_gpu = LatencyEstimator('gpu', clock_freq_mhz=1500)

# Test avec un modèle simple
test_model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 112 * 112, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

latency_fpga = latency_est_fpga.estimate_model_latency(test_model, (1, 3, 224, 224))
latency_gpu = latency_est_gpu.estimate_model_latency(test_model, (1, 3, 224, 224))

print(f"\nLatence estimée:")
print(f"  FPGA: {latency_fpga:.2f} ns ({latency_fpga/1000:.2f} μs)")
print(f"  GPU:  {latency_gpu:.2f} ns ({latency_gpu/1000:.2f} μs)")
print(f"  Ratio: {latency_fpga/latency_gpu:.1f}x plus rapide sur GPU")
```

### Modèles ML pour Prédiction de Latence

```python
class MLLatencyPredictor:
    """
    Prédicteur de latence basé sur Machine Learning
    
    Entraîné sur des mesures réelles pour améliorer la précision
    """
    
    def __init__(self):
        """
        Initialise le prédicteur ML (exemple simplifié)
        """
        # En pratique, entraîner un modèle sur des données réelles
        self.feature_extractor = self._create_feature_extractor()
        self.predictor = self._create_predictor_model()
    
    def _create_feature_extractor(self):
        """
        Extrait des caractéristiques d'une architecture
        """
        def extract_features(model, input_shape):
            """Extrait features pour prédiction"""
            features = {}
            
            # Caractéristiques globales
            total_params = sum(p.numel() for p in model.parameters())
            features['total_params'] = total_params
            features['num_layers'] = len(list(model.modules()))
            
            # Caractéristiques par type de couche
            n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
            n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            features['num_conv'] = n_conv
            features['num_linear'] = n_linear
            
            # Taille des poids
            total_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            features['model_memory_bytes'] = total_memory
            
            # Complexité computationnelle (FLOPs approximatifs)
            features['flops'] = self._estimate_flops(model, input_shape)
            
            return np.array([
                np.log10(features['total_params'] + 1),
                features['num_layers'],
                n_conv,
                n_linear,
                np.log10(features['model_memory_bytes'] + 1),
                np.log10(features['flops'] + 1)
            ])
        
        return extract_features
    
    def _estimate_flops(self, model, input_shape):
        """Estime le nombre de FLOPs"""
        flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                # Approximation: kernel_size^2 * in_channels * out_channels * output_size
                h_out = input_shape[2] // (module.stride[0] if isinstance(module.stride, tuple) else module.stride)
                w_out = input_shape[3] // (module.stride[1] if isinstance(module.stride, tuple) else module.stride)
                kernel_size = module.kernel_size[0] * module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size ** 2
                flops += kernel_size * module.in_channels * module.out_channels * h_out * w_out
        
        return flops
    
    def _create_predictor_model(self):
        """
        Crée un modèle ML simple pour prédire la latence
        
        En pratique, utiliser sklearn ou PyTorch pour un vrai modèle
        """
        # Approximation linéaire (en pratique, utiliser un modèle entraîné)
        # coefficients approximatifs basés sur corrélations typiques
        self.coefficients = np.array([0.5, 0.2, 0.3, 0.1, 0.15, 0.25])
        self.bias = 5.0
        
        def predict(features):
            """Prédit la latence depuis les features"""
            # Approximation: régression linéaire
            log_latency_ns = np.dot(features, self.coefficients) + self.bias
            latency_ns = 10 ** log_latency_ns
            return latency_ns
        
        return predict
    
    def predict_latency(self, model, input_shape):
        """
        Prédit la latence d'un modèle
        
        Args:
            model: Modèle PyTorch
            input_shape: Shape de l'input
        
        Returns:
            Latence prédite en ns
        """
        features = self.feature_extractor(model, input_shape)
        latency = self.predictor(features)
        return latency

# Test du prédicteur ML
ml_predictor = MLLatencyPredictor()
predicted_latency = ml_predictor.predict_latency(test_model, (1, 3, 224, 224))
print(f"\nLatence prédite (ML): {predicted_latency:.2f} ns")
```

---

## Estimation d'Énergie

### Modèles d'Énergie

```python
class EnergyEstimator:
    """
    Estimateur de consommation énergétique
    """
    
    def __init__(self, hardware_type='fpga', bitwidth=8):
        """
        Args:
            hardware_type: 'fpga', 'gpu', 'cpu', 'asic'
            bitwidth: Précision des opérations (8, 16, 32 bits)
        """
        self.hardware_type = hardware_type
        self.bitwidth = bitwidth
        
        # Énergie par opération (en picojoules)
        # Valeurs typiques pour FPGA à différentes précisions
        self.energy_per_op = {
            'fpga': {
                8: {'mult': 4.6, 'add': 0.9, 'memory_read': 1.0, 'memory_write': 2.0},
                16: {'mult': 18.0, 'add': 3.5, 'memory_read': 2.0, 'memory_write': 4.0},
                32: {'mult': 72.0, 'add': 14.0, 'memory_read': 4.0, 'memory_write': 8.0}
            },
            'gpu': {
                8: {'mult': 0.5, 'add': 0.2, 'memory_read': 10.0, 'memory_write': 20.0},
                16: {'mult': 1.0, 'add': 0.4, 'memory_read': 15.0, 'memory_write': 30.0},
                32: {'mult': 3.5, 'add': 1.2, 'memory_read': 20.0, 'memory_write': 40.0}
            }
        }
    
    def estimate_model_energy(self, model, input_shape=(1, 3, 224, 224)):
        """
        Estime l'énergie totale consommée pour une inférence
        
        Returns:
            Énergie en picojoules (pJ)
        """
        energy_ops = 0  # Énergie des opérations
        energy_memory = 0  # Énergie d'accès mémoire
        
        current_shape = input_shape
        
        # Parcours du modèle
        for module in model.modules():
            if isinstance(module, nn.Linear):
                e_ops, e_mem = self._estimate_linear_energy(module, current_shape)
                energy_ops += e_ops
                energy_memory += e_mem
                current_shape = (current_shape[0], module.out_features)
            
            elif isinstance(module, nn.Conv2d):
                e_ops, e_mem = self._estimate_conv2d_energy(module, current_shape)
                energy_ops += e_ops
                energy_memory += e_mem
                # Mise à jour shape
                h_out = (current_shape[2] + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1) - 1) // module.stride[0] + 1
                w_out = (current_shape[3] + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1) - 1) // module.stride[1] + 1
                current_shape = (current_shape[0], module.out_channels, h_out, w_out)
            
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                e_ops, e_mem = self._estimate_bn_energy(module, current_shape)
                energy_ops += e_ops
                energy_memory += e_mem
            
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                # Activations: peu d'énergie (comparaison simple)
                e_ops = np.prod(current_shape) * 0.1  # approximation
                energy_ops += e_ops
            
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                e_ops, e_mem = self._estimate_pool_energy(module, current_shape)
                energy_ops += e_ops
                energy_memory += e_mem
        
        total_energy = energy_ops + energy_memory
        return total_energy
    
    def _estimate_linear_energy(self, module, input_shape):
        """Estime l'énergie d'une couche linéaire"""
        energy_ops = self.energy_per_op[self.hardware_type][self.bitwidth]
        
        # Opérations MAC
        n_mults = module.in_features * module.out_features
        n_adds = module.out_features  # accumulations
        
        # Énergie opérations
        e_ops = n_mults * energy_ops['mult'] + n_adds * energy_ops['add']
        
        # Énergie mémoire: lecture poids + lecture inputs + écriture outputs
        n_weights = module.in_features * module.out_features
        n_inputs = np.prod(input_shape)
        n_outputs = module.out_features * input_shape[0]
        
        e_mem = (n_weights * energy_ops['memory_read'] + 
                 n_inputs * energy_ops['memory_read'] +
                 n_outputs * energy_ops['memory_write'])
        
        return e_ops, e_mem
    
    def _estimate_conv2d_energy(self, module, input_shape):
        """Estime l'énergie d'une couche convolutive"""
        energy_ops = self.energy_per_op[self.hardware_type][self.bitwidth]
        
        batch, in_channels, h_in, w_in = input_shape
        out_channels = module.out_channels
        
        # Opérations par output
        kernel_size = module.kernel_size[0] * module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size ** 2
        h_out = (h_in + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1) - 1) // module.stride[0] + 1
        w_out = (w_in + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1) - 1) // module.stride[1] + 1
        
        n_output_pixels = h_out * w_out * out_channels
        n_mults = n_output_pixels * kernel_size * in_channels
        n_adds = n_output_pixels
        
        e_ops = n_mults * energy_ops['mult'] + n_adds * energy_ops['add']
        
        # Énergie mémoire
        n_weights = kernel_size * in_channels * out_channels
        n_inputs = np.prod(input_shape)
        n_outputs = n_output_pixels * batch
        
        e_mem = (n_weights * energy_ops['memory_read'] +
                 n_inputs * energy_ops['memory_read'] +
                 n_outputs * energy_ops['memory_write'])
        
        return e_ops, e_mem
    
    def _estimate_bn_energy(self, module, input_shape):
        """Estime l'énergie pour BatchNorm"""
        energy_ops = self.energy_per_op[self.hardware_type][self.bitwidth]
        n_elements = np.prod(input_shape)
        
        # BatchNorm: normalisation, multiplication (scale), addition (shift)
        e_ops = n_elements * (2 * energy_ops['add'] + energy_ops['mult'])
        
        # Mémoire: lecture/write activations
        e_mem = n_elements * (energy_ops['memory_read'] + energy_ops['memory_write'])
        
        return e_ops, e_mem
    
    def _estimate_pool_energy(self, module, input_shape):
        """Estime l'énergie pour pooling"""
        energy_ops = self.energy_per_op[self.hardware_type][self.bitwidth]
        n_elements = np.prod(input_shape)
        
        # Pooling: comparaisons (pour max) ou additions (pour avg)
        e_ops = n_elements * energy_ops['add']
        e_mem = n_elements * (energy_ops['memory_read'] + energy_ops['memory_write'])
        
        return e_ops, e_mem

# Test de l'estimateur d'énergie
energy_est_fpga_8bit = EnergyEstimator('fpga', bitwidth=8)
energy_est_fpga_16bit = EnergyEstimator('fpga', bitwidth=16)

energy_8bit = energy_est_fpga_8bit.estimate_model_energy(test_model, (1, 3, 224, 224))
energy_16bit = energy_est_fpga_16bit.estimate_model_energy(test_model, (1, 3, 224, 224))

print(f"\nÉnergie par inférence:")
print(f"  FPGA 8-bit:  {energy_8bit/1e9:.2f} nJ ({energy_8bit/1e12:.2f} μJ)")
print(f"  FPGA 16-bit: {energy_16bit/1e9:.2f} nJ ({energy_16bit/1e12:.2f} μJ)")
print(f"  Ratio: {energy_16bit/energy_8bit:.2f}x plus d'énergie pour 16-bit")
```

---

## Estimation des Ressources (FPGA)

### Ressources FPGA

```python
class FPGAResourceEstimator:
    """
    Estimateur de ressources FPGA (LUT, DSP, BRAM)
    """
    
    def __init__(self):
        # Ressources nécessaires par opération (approximations)
        # Valeurs typiques pour différentes opérations
        
        # LUT (Look-Up Tables): pour logique combinatoire
        self.lut_per_op = {
            'linear_8bit': lambda in_feat, out_feat: in_feat * out_feat * 0.1,  # approximation
            'conv2d_8bit': lambda in_ch, out_ch, k: in_ch * out_ch * k * k * 0.05,
            'bn': lambda channels: channels * 2,
            'relu': lambda elements: elements * 0.01,
            'pool': lambda elements: elements * 0.5
        }
        
        # DSP (Digital Signal Processors): pour multiplications
        self.dsp_per_op = {
            'linear_8bit': lambda in_feat, out_feat: min(in_feat * out_feat, in_feat * out_feat / 8),  # peut être optimisé
            'conv2d_8bit': lambda in_ch, out_ch, k: min(in_ch * out_ch * k * k, (in_ch * out_ch * k * k) / 4),
            'bn': lambda channels: channels,
            'relu': lambda elements: 0,  # pas de DSP pour activations
            'pool': lambda elements: 0
        }
        
        # BRAM (Block RAM): pour stockage
        self.bram_per_op = {
            'weights_8bit': lambda num_weights: np.ceil(num_weights * 8 / (36 * 1024)),  # 36KB par BRAM
            'activations_8bit': lambda num_acts: np.ceil(num_acts * 8 / (36 * 1024))
        }
    
    def estimate_model_resources(self, model, input_shape=(1, 3, 224, 224)):
        """
        Estime les ressources FPGA nécessaires
        
        Returns:
            dict avec 'lut', 'dsp', 'bram'
        """
        total_lut = 0
        total_dsp = 0
        total_bram = 0
        
        current_shape = input_shape
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # LUT et DSP pour multiplications
                lut = self.lut_per_op['linear_8bit'](module.in_features, module.out_features)
                dsp = self.dsp_per_op['linear_8bit'](module.in_features, module.out_features)
                
                # BRAM pour poids et activations
                n_weights = module.in_features * module.out_features
                n_activations = np.prod(current_shape)
                
                bram_weights = self.bram_per_op['weights_8bit'](n_weights)
                bram_acts = self.bram_per_op['activations_8bit'](n_activations)
                
                total_lut += lut
                total_dsp += dsp
                total_bram += bram_weights + bram_acts
                
                current_shape = (current_shape[0], module.out_features)
            
            elif isinstance(module, nn.Conv2d):
                k_size = module.kernel_size[0] * module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size ** 2
                
                lut = self.lut_per_op['conv2d_8bit'](module.in_channels, module.out_channels, int(np.sqrt(k_size)))
                dsp = self.dsp_per_op['conv2d_8bit'](module.in_channels, module.out_channels, int(np.sqrt(k_size)))
                
                n_weights = k_size * module.in_channels * module.out_channels
                n_activations = np.prod(current_shape)
                
                bram_weights = self.bram_per_op['weights_8bit'](n_weights)
                bram_acts = self.bram_per_op['activations_8bit'](n_activations)
                
                total_lut += lut
                total_dsp += dsp
                total_bram += bram_weights + bram_acts
                
                # Mise à jour shape
                h_out = (current_shape[2] + 2*module.padding[0] - module.dilation[0]*(module.kernel_size[0]-1) - 1) // module.stride[0] + 1
                w_out = (current_shape[3] + 2*module.padding[1] - module.dilation[1]*(module.kernel_size[1]-1) - 1) // module.stride[1] + 1
                current_shape = (current_shape[0], module.out_channels, h_out, w_out)
            
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                channels = module.num_features
                lut += self.lut_per_op['bn'](channels)
                dsp += self.dsp_per_op['bn'](channels)
            
            elif isinstance(module, (nn.ReLU, nn.GELU)):
                n_elements = np.prod(current_shape)
                total_lut += self.lut_per_op['relu'](n_elements)
            
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
                n_elements = np.prod(current_shape)
                total_lut += self.lut_per_op['pool'](n_elements)
        
        return {
            'lut': int(total_lut),
            'dsp': int(total_dsp),
            'bram': int(total_bram)
        }
    
    def check_feasibility(self, model, fpga_constraints):
        """
        Vérifie si un modèle tient dans les contraintes FPGA
        
        Args:
            model: Modèle PyTorch
            fpga_constraints: dict avec 'max_lut', 'max_dsp', 'max_bram'
        
        Returns:
            (feasible: bool, resources: dict, utilization: dict)
        """
        resources = self.estimate_model_resources(model)
        
        utilization = {
            'lut': resources['lut'] / fpga_constraints['max_lut'],
            'dsp': resources['dsp'] / fpga_constraints['max_dsp'],
            'bram': resources['bram'] / fpga_constraints['max_bram']
        }
        
        feasible = (
            resources['lut'] <= fpga_constraints['max_lut'] and
            resources['dsp'] <= fpga_constraints['max_dsp'] and
            resources['bram'] <= fpga_constraints['max_bram']
        )
        
        return feasible, resources, utilization

# Test de l'estimateur de ressources
resource_est = FPGAResourceEstimator()
resources = resource_est.estimate_model_resources(test_model, (1, 3, 224, 224))

print(f"\nRessources FPGA estimées:")
print(f"  LUT:  {resources['lut']:,}")
print(f"  DSP:  {resources['dsp']:,}")
print(f"  BRAM: {resources['bram']:,}")

# Vérification de faisabilité (exemple: Xilinx Zynq UltraScale+)
fpga_constraints = {
    'max_lut': 548160,  # Zynq UltraScale+ XCZU9EG
    'max_dsp': 2520,
    'max_bram': 912
}

feasible, res, util = resource_est.check_feasibility(test_model, fpga_constraints)

print(f"\nFaisabilité sur FPGA:")
print(f"  Faisable: {feasible}")
print(f"  Utilisation LUT:  {util['lut']*100:.1f}%")
print(f"  Utilisation DSP:  {util['dsp']*100:.1f}%")
print(f"  Utilisation BRAM: {util['bram']*100:.1f}%")
```

---

## Intégration dans le NAS

### Classe Complète d'Évaluation Hardware

```python
class HardwareEvaluator:
    """
    Évaluateur complet de métriques hardware pour NAS
    """
    
    def __init__(self, hardware_type='fpga', bitwidth=8):
        self.latency_estimator = LatencyEstimator(hardware_type)
        self.energy_estimator = EnergyEstimator(hardware_type, bitwidth)
        
        if hardware_type == 'fpga':
            self.resource_estimator = FPGAResourceEstimator()
        else:
            self.resource_estimator = None
    
    def evaluate_architecture(self, model, input_shape=(1, 3, 224, 224), fpga_constraints=None):
        """
        Évalue complètement une architecture
        
        Returns:
            dict avec toutes les métriques hardware
        """
        metrics = {}
        
        # Latence
        metrics['latency_ns'] = self.latency_estimator.estimate_model_latency(model, input_shape)
        metrics['latency_us'] = metrics['latency_ns'] / 1000.0
        
        # Énergie
        metrics['energy_pj'] = self.energy_estimator.estimate_model_energy(model, input_shape)
        metrics['energy_nj'] = metrics['energy_pj'] / 1e3
        
        # Ressources (si FPGA)
        if self.resource_estimator:
            resources = self.resource_estimator.estimate_model_resources(model, input_shape)
            metrics['resources'] = resources
            
            # Vérification de faisabilité
            if fpga_constraints:
                feasible, res, util = self.resource_estimator.check_feasibility(model, fpga_constraints)
                metrics['feasible'] = feasible
                metrics['resource_utilization'] = util
        
        # Taille du modèle
        metrics['model_size_mb'] = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
        
        return metrics
    
    def compute_hardware_score(self, metrics, weights=None):
        """
        Calcule un score hardware combiné
        
        Args:
            metrics: dict de métriques
            weights: dict avec poids pour chaque métrique
        
        Returns:
            Score normalisé (plus bas = mieux)
        """
        if weights is None:
            weights = {
                'latency': 0.4,
                'energy': 0.3,
                'model_size': 0.2,
                'feasibility': 0.1
            }
        
        # Normalisation (exemples de valeurs de référence)
        latency_score = metrics['latency_us'] / 100.0  # référence: 100 μs
        energy_score = metrics['energy_nj'] / 1000.0  # référence: 1000 nJ
        size_score = metrics['model_size_mb'] / 10.0  # référence: 10 MB
        
        feasibility_score = 0.0
        if 'feasible' in metrics:
            feasibility_score = 0.0 if metrics['feasible'] else 1.0  # pénalité si infaisable
        
        score = (
            weights['latency'] * latency_score +
            weights['energy'] * energy_score +
            weights['model_size'] * size_score +
            weights['feasibility'] * feasibility_score
        )
        
        return score

# Exemple d'utilisation complète
hardware_eval = HardwareEvaluator('fpga', bitwidth=8)
metrics = hardware_eval.evaluate_architecture(test_model, (1, 3, 224, 224), fpga_constraints)

print("\n" + "="*70)
print("Évaluation Hardware Complète")
print("="*70)
print(f"\nLatence: {metrics['latency_us']:.2f} μs ({metrics['latency_ns']:.0f} ns)")
print(f"Énergie: {metrics['energy_nj']:.2f} nJ")
print(f"Taille modèle: {metrics['model_size_mb']:.2f} MB")
print(f"Paramètres: {metrics['num_parameters']:,}")

if 'resources' in metrics:
    print(f"\nRessources FPGA:")
    print(f"  LUT:  {metrics['resources']['lut']:,}")
    print(f"  DSP:  {metrics['resources']['dsp']:,}")
    print(f"  BRAM: {metrics['resources']['bram']:,}")

if 'feasible' in metrics:
    print(f"\nFaisable: {metrics['feasible']}")
    if 'resource_utilization' in metrics:
        util = metrics['resource_utilization']
        print(f"Utilisation: LUT {util['lut']*100:.1f}%, DSP {util['dsp']*100:.1f}%, BRAM {util['bram']*100:.1f}%")

score = hardware_eval.compute_hardware_score(metrics)
print(f"\nScore hardware combiné: {score:.4f} (plus bas = mieux)")
```

---

## Exercices

### Exercice 16.2.1
Implémentez un estimateur de latence pour un modèle quantifié (8-bit) en tenant compte du coût de déquantification.

### Exercice 16.2.2
Créez un prédicteur ML pour estimer la latence en entraînant un modèle sur des architectures variées avec leurs latences mesurées.

### Exercice 16.2.3
Comparez l'énergie consommée par un modèle en 8-bit vs 16-bit sur FPGA. Analysez le trade-off énergie/précision.

### Exercice 16.2.4
Implémentez une fonction qui vérifie si une architecture respecte des contraintes hardware strictes (latence < 100 ns, énergie < 100 nJ).

---

## Points Clés à Retenir

> 📌 **Les métriques hardware (latence, énergie, ressources) sont essentielles pour Hardware-Aware NAS**

> 📌 **L'estimation peut être analytique (modèles simples) ou basée sur ML (prédicteurs entraînés)**

> 📌 **La latence dépend fortement du parallélisme et de l'accès mémoire**

> 📌 **L'énergie varie significativement avec la précision (8-bit vs 16-bit vs 32-bit)**

> 📌 **Pour FPGA, les ressources (LUT, DSP, BRAM) sont souvent le facteur limitant**

> 📌 **L'intégration de toutes les métriques permet un score hardware combiné pour le NAS**

---

*Section précédente : [16.1 Principes du NAS](./16_01_Principes_NAS.md) | Section suivante : [16.3 Espaces de Recherche Contraints](./16_03_Espaces.md)*

