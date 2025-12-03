# 16.5 Co-design Modèle-Hardware

---

## Introduction

Le **co-design modèle-hardware** consiste à optimiser simultanément l'architecture du modèle de machine learning et la configuration du hardware cible. Au lieu d'optimiser le modèle puis de l'adapter au hardware, cette approche optimise les deux en parallèle pour obtenir de meilleures performances globales.

Cette section présente les principes du co-design, les méthodes pour optimiser simultanément modèle et hardware, et des applications pratiques pour FPGA et systèmes embarqués.

---

## Principes du Co-design

### Vue d'Ensemble

```python
class CoDesignPrinciples:
    """
    Principes fondamentaux du co-design modèle-hardware
    """
    
    def __init__(self):
        self.principles = {
            'joint_optimization': {
                'description': 'Optimisation simultanée modèle + hardware',
                'advantage': 'Meilleure solution globale que optimisation séparée',
                'challenge': 'Espace de recherche combiné très large',
                'example': 'Optimiser architecture CNN + parallélisme FPGA simultanément'
            },
            'hardware_aware_training': {
                'description': 'Entraînement avec contraintes hardware',
                'advantage': 'Modèle appris pour être efficace sur hardware cible',
                'challenge': 'Simulation hardware pendant entraînement',
                'example': 'Entraîner avec latence/énergie comme régularisation'
            },
            'adaptive_mapping': {
                'description': 'Mapping adaptatif du modèle sur hardware',
                'advantage': 'Utilisation optimale des ressources',
                'challenge': 'Trouver mapping optimal est complexe',
                'example': 'Répartir couches sur différents PEs (Processing Elements)'
            },
            'heterogeneous_computation': {
                'description': 'Utilisation de différents types de compute units',
                'advantage': 'Exploite avantages de chaque type',
                'challenge': 'Scheduling et synchronisation complexes',
                'example': 'Conv sur DSP, activation sur LUT, pooling sur BRAM'
            },
            'memory_hierarchy_optimization': {
                'description': 'Optimisation de la hiérarchie mémoire',
                'advantage': 'Réduit latence et énergie mémoire',
                'challenge': 'Trade-off complexe entre différentes mémoires',
                'example': 'Weights en BRAM, activations en cache, buffers optimisés'
            }
        }
    
    def display_principles(self):
        """Affiche les principes"""
        print("\n" + "="*70)
        print("Principes du Co-design Modèle-Hardware")
        print("="*70)
        
        for principle, info in self.principles.items():
            print(f"\n{principle.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Advantage: {info['advantage']}")
            print(f"  Challenge: {info['challenge']}")
            print(f"  Example: {info['example']}")

principles = CoDesignPrinciples()
principles.display_principles()
```

---

## Espace de Recherche Combiné

### Modèle + Hardware

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class CombinedSearchSpace:
    """
    Espace de recherche combiné: architecture modèle + configuration hardware
    """
    
    def __init__(self, model_search_space, hardware_config_space):
        """
        Args:
            model_search_space: Espace de recherche d'architectures
            hardware_config_space: Espace de recherche de configs hardware
        """
        self.model_space = model_search_space
        self.hardware_space = hardware_config_space
        
        # Espace combiné
        self.combined_space = {
            'model': self.model_space.base_space,
            'hardware': self.hardware_space.space
        }
    
    def sample_combined_config(self) -> Dict:
        """
        Génère une configuration combinée (modèle + hardware)
        """
        model_config = self.model_space.sample_random_config()
        hardware_config = self.hardware_space.sample_config()
        
        return {
            'model': model_config,
            'hardware': hardware_config
        }
    
    def create_model_from_combined_config(self, combined_config: Dict) -> nn.Module:
        """Crée modèle depuis config combinée"""
        return self.model_space.create_model_from_config(combined_config['model'])
    
    def evaluate_combined_config(self, combined_config: Dict, 
                                input_shape: Tuple = (1, 784)) -> Dict:
        """
        Évalue une configuration combinée
        
        Returns:
            Métriques combinées (performance, latence, énergie, etc.)
        """
        # Créer modèle
        model = self.create_model_from_combined_config(combined_config)
        
        # Simuler sur hardware configuré
        simulator = HardwareSimulator(combined_config['hardware'])
        metrics = simulator.simulate(model, input_shape)
        
        # Évaluer performance (proxy)
        performance = self._evaluate_performance_proxy(model, input_shape)
        
        return {
            'performance': performance,
            'latency_ns': metrics['latency_ns'],
            'energy_nj': metrics['energy_nj'],
            'hardware_utilization': metrics['utilization'],
            'combined_score': self._compute_combined_score(performance, metrics)
        }
    
    def _evaluate_performance_proxy(self, model: nn.Module, input_shape: Tuple) -> float:
        """Proxy de performance"""
        n_params = sum(p.numel() for p in model.parameters())
        return min(0.7 + (n_params / 1e6) * 0.2, 0.95)
    
    def _compute_combined_score(self, performance: float, metrics: Dict) -> float:
        """Score combiné"""
        latency_penalty = (metrics['latency_ns'] / 1e6) * 0.3
        energy_penalty = (metrics['energy_nj'] / 1e3) * 0.2
        return performance - latency_penalty - energy_penalty


class HardwareConfigSpace:
    """
    Espace de recherche de configurations hardware
    """
    
    def __init__(self):
        self.space = {
            'parallelism': {
                'pe_count': [4, 8, 16, 32, 64],  # Processing Elements
                'dataflow': ['systolic', 'output_stationary', 'weight_stationary'],
                'tile_sizes': [(8, 8), (16, 16), (32, 32)]
            },
            'memory': {
                'buffer_size': [1024, 2048, 4096, 8192],  # bytes
                'memory_banks': [1, 2, 4, 8],
                'data_reuse': [True, False]
            },
            'precision': {
                'weight_bits': [8, 16],
                'activation_bits': [8, 16],
                'accumulator_bits': [16, 32]
            },
            'frequency': {
                'clock_mhz': [100, 150, 200, 250, 300]
            }
        }
    
    def sample_config(self) -> Dict:
        """Génère une configuration hardware"""
        config = {}
        
        # Parallélisme
        config['pe_count'] = np.random.choice(self.space['parallelism']['pe_count'])
        config['dataflow'] = np.random.choice(self.space['parallelism']['dataflow'])
        config['tile_size'] = np.random.choice(self.space['parallelism']['tile_sizes'])
        
        # Mémoire
        config['buffer_size'] = np.random.choice(self.space['memory']['buffer_size'])
        config['memory_banks'] = np.random.choice(self.space['memory']['memory_banks'])
        config['data_reuse'] = np.random.choice(self.space['memory']['data_reuse'])
        
        # Précision
        config['weight_bits'] = np.random.choice(self.space['precision']['weight_bits'])
        config['activation_bits'] = np.random.choice(self.space['precision']['activation_bits'])
        config['accumulator_bits'] = np.random.choice(self.space['precision']['accumulator_bits'])
        
        # Fréquence
        config['clock_mhz'] = np.random.choice(self.space['frequency']['clock_mhz'])
        
        return config


class HardwareSimulator:
    """
    Simulateur hardware pour évaluer configurations
    """
    
    def __init__(self, hardware_config: Dict):
        self.config = hardware_config
        self.clock_period_ns = 1000.0 / hardware_config['clock_mhz']
    
    def simulate(self, model: nn.Module, input_shape: Tuple) -> Dict:
        """
        Simule l'exécution du modèle sur hardware configuré
        
        Returns:
            Métriques hardware
        """
        # Estimer latence basée sur parallélisme
        latency_ns = self._estimate_latency(model, input_shape)
        
        # Estimer énergie
        energy_nj = self._estimate_energy(model, input_shape)
        
        # Estimer utilisation ressources
        utilization = self._estimate_utilization(model, input_shape)
        
        return {
            'latency_ns': latency_ns,
            'latency_us': latency_ns / 1000.0,
            'energy_nj': energy_nj,
            'utilization': utilization
        }
    
    def _estimate_latency(self, model: nn.Module, input_shape: Tuple) -> float:
        """Estime latence avec parallélisme configuré"""
        total_cycles = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Opérations MAC
                n_mac = module.in_features * module.out_features
                
                # Cycles avec parallélisme
                cycles = np.ceil(n_mac / self.config['pe_count'])
                total_cycles += cycles
        
        latency_ns = total_cycles * self.clock_period_ns
        return latency_ns
    
    def _estimate_energy(self, model: nn.Module, input_shape: Tuple) -> float:
        """Estime énergie avec précision configurée"""
        # Énergie par opération dépend de précision
        energy_per_mult = {
            8: 4.6,   # pJ
            16: 18.0
        }
        
        energy_mult = energy_per_mult.get(self.config['weight_bits'], 4.6)
        
        total_energy_pj = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                n_mac = module.in_features * module.out_features
                total_energy_pj += n_mac * energy_mult
        
        return total_energy_pj / 1e3  # nJ
    
    def _estimate_utilization(self, model: nn.Module, input_shape: Tuple) -> Dict:
        """Estime utilisation des ressources"""
        # Simplifié: utilisation basée sur PE
        total_ops = sum(module.in_features * module.out_features 
                       for module in model.modules() 
                       if isinstance(module, nn.Linear))
        
        pe_utilization = min(1.0, total_ops / (self.config['pe_count'] * 1000))
        
        return {
            'pe_utilization': pe_utilization,
            'memory_utilization': 0.6,  # approximation
            'overall': pe_utilization * 0.7 + 0.6 * 0.3
        }
```

---

## Optimisation Jointe

### Algorithme de Co-optimisation

```python
class JointOptimizationNAS:
    """
    NAS avec optimisation jointe modèle-hardware
    """
    
    def __init__(self, combined_search_space, constraints: Dict):
        self.search_space = combined_search_space
        self.constraints = constraints
        
        self.evaluation_history = []
    
    def optimize(self, n_iterations: int = 200, 
                 input_shape: Tuple = (1, 784)) -> Dict:
        """
        Optimise simultanément modèle et hardware
        
        Stratégie: recherche alternée avec coordination
        """
        # Initialisation
        best_combined = None
        best_score = -float('inf')
        
        # Phase 1: Exploration large
        print("Phase 1: Exploration large...")
        for i in range(n_iterations // 2):
            combined_config = self.search_space.sample_combined_config()
            
            # Évaluer
            metrics = self.search_space.evaluate_combined_config(combined_config, input_shape)
            
            self.evaluation_history.append({
                'config': combined_config,
                'metrics': metrics
            })
            
            if metrics['combined_score'] > best_score:
                best_score = metrics['combined_score']
                best_combined = {
                    'config': combined_config,
                    'metrics': metrics
                }
            
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{n_iterations//2}: Best score = {best_score:.4f}")
        
        # Phase 2: Raffinement local
        print("\nPhase 2: Raffinement local...")
        current_config = best_combined['config']
        
        for i in range(n_iterations // 2):
            # Alterner entre optimisation modèle et hardware
            if i % 2 == 0:
                # Optimiser modèle autour de config hardware actuelle
                new_config = self._optimize_model_local(current_config)
            else:
                # Optimiser hardware autour de modèle actuel
                new_config = self._optimize_hardware_local(current_config)
            
            # Évaluer
            metrics = self.search_space.evaluate_combined_config(new_config, input_shape)
            
            if metrics['combined_score'] > best_score:
                best_score = metrics['combined_score']
                best_combined = {
                    'config': new_config,
                    'metrics': metrics
                }
                current_config = new_config
            
            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{n_iterations//2}: Best score = {best_score:.4f}")
        
        # Créer modèle final
        final_model = self.search_space.create_model_from_combined_config(best_combined['config'])
        
        return {
            'model': final_model,
            'model_config': best_combined['config']['model'],
            'hardware_config': best_combined['config']['hardware'],
            'metrics': best_combined['metrics'],
            'score': best_score
        }
    
    def _optimize_model_local(self, current_config: Dict) -> Dict:
        """Mutation locale du modèle"""
        new_config = current_config.copy()
        
        # Mutation du modèle
        model_config = new_config['model'].copy()
        
        if 'num_layers' in model_config:
            current_layers = model_config['num_layers']
            model_config['num_layers'] = np.random.choice([
                max(3, current_layers - 1),
                current_layers,
                min(8, current_layers + 1)
            ])
        
        new_config['model'] = model_config
        return new_config
    
    def _optimize_hardware_local(self, current_config: Dict) -> Dict:
        """Mutation locale du hardware"""
        new_config = current_config.copy()
        
        # Mutation du hardware
        hw_config = new_config['hardware'].copy()
        
        if 'pe_count' in hw_config:
            current_pe = hw_config['pe_count']
            options = [max(4, current_pe - 8), current_pe, min(64, current_pe + 8)]
            hw_config['pe_count'] = np.random.choice(options)
        
        if 'clock_mhz' in hw_config:
            current_freq = hw_config['clock_mhz']
            options = [max(100, current_freq - 50), current_freq, min(300, current_freq + 50)]
            hw_config['clock_mhz'] = np.random.choice(options)
        
        new_config['hardware'] = hw_config
        return new_config
```

---

## Hardware-Aware Training

### Entraînement avec Contraintes Hardware

```python
class HardwareAwareTraining:
    """
    Entraînement de modèle avec régularisation hardware
    """
    
    def __init__(self, model: nn.Module, hardware_simulator, 
                 hardware_weight: float = 0.1):
        """
        Args:
            model: Modèle à entraîner
            hardware_simulator: HardwareSimulator
            hardware_weight: Poids de la régularisation hardware
        """
        self.model = model
        self.hardware_sim = hardware_simulator
        self.hardware_weight = hardware_weight
    
    def train(self, train_loader, val_loader, epochs: int = 50,
              lr: float = 0.001, input_shape: Tuple = (1, 784)):
        """
        Entraîne avec régularisation hardware
        """
        import torch.optim as optim
        import torch.nn.functional as F
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward
                output = self.model(data)
                
                # Loss standard
                loss_standard = criterion(output, target)
                
                # Pénalité hardware (tous les N batches pour efficacité)
                if batch_idx % 10 == 0:
                    hardware_metrics = self.hardware_sim.simulate(self.model, input_shape)
                    hardware_penalty = self._compute_hardware_penalty(hardware_metrics)
                else:
                    hardware_penalty = 0
                
                # Loss combinée
                loss = loss_standard + self.hardware_weight * hardware_penalty
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            if (epoch + 1) % 10 == 0:
                acc = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {train_loss/len(train_loader):.4f}, "
                      f"Val Loss = {val_loss/len(val_loader):.4f}, "
                      f"Val Acc = {acc:.2f}%")
    
    def _compute_hardware_penalty(self, metrics: Dict) -> torch.Tensor:
        """
        Calcule pénalité hardware (plus bas = mieux)
        """
        latency_penalty = metrics['latency_us'] / 100.0  # normalisé
        energy_penalty = metrics['energy_nj'] / 1000.0  # normalisé
        
        return torch.tensor(latency_penalty + energy_penalty)
```

---

## Mapping Adaptatif sur FPGA

### Optimisation du Mapping

```python
class AdaptiveFPGAMapping:
    """
    Mapping adaptatif de modèle sur FPGA avec optimisation
    """
    
    def __init__(self, model: nn.Module, fpga_constraints: Dict):
        """
        Args:
            model: Modèle à mapper
            fpga_constraints: Contraintes FPGA (LUT, DSP, BRAM)
        """
        self.model = model
        self.fpga_constraints = fpga_constraints
        
        # Stratégies de mapping
        self.mapping_strategies = {
            'layer_wise': self._map_layer_wise,
            'tensor_slicing': self._map_tensor_slicing,
            'pipelined': self._map_pipelined
        }
    
    def find_optimal_mapping(self, input_shape: Tuple = (1, 784)) -> Dict:
        """
        Trouve le mapping optimal
        
        Returns:
            Configuration de mapping optimale
        """
        best_mapping = None
        best_latency = float('inf')
        
        # Tester différentes stratégies
        for strategy_name, strategy_fn in self.mapping_strategies.items():
            mapping = strategy_fn(input_shape)
            
            # Évaluer mapping
            latency = self._evaluate_mapping(mapping, input_shape)
            
            if latency < best_latency:
                best_latency = latency
                best_mapping = {
                    'strategy': strategy_name,
                    'mapping': mapping,
                    'latency_ns': latency
                }
        
        return best_mapping
    
    def _map_layer_wise(self, input_shape: Tuple) -> Dict:
        """
        Mapping couche par couche (séquentiel)
        """
        mapping = {
            'type': 'layer_wise',
            'layers': []
        }
        
        current_shape = input_shape
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_mapping = {
                    'name': name,
                    'type': 'linear',
                    'pe_count': min(64, module.out_features),  # PEs pour cette couche
                    'tile_size': (8, 8),
                    'input_shape': current_shape,
                    'output_shape': (current_shape[0], module.out_features)
                }
                mapping['layers'].append(layer_mapping)
                current_shape = layer_mapping['output_shape']
        
        return mapping
    
    def _map_tensor_slicing(self, input_shape: Tuple) -> Dict:
        """
        Mapping avec découpage de tenseurs (parallélisme spatial)
        """
        mapping = {
            'type': 'tensor_slicing',
            'slices': 4,  # Nombre de slices
            'layers': []
        }
        
        # Similar à layer_wise mais avec slicing
        current_shape = input_shape
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_mapping = {
                    'name': name,
                    'type': 'linear_sliced',
                    'slices': mapping['slices'],
                    'pe_per_slice': 16,
                    'input_shape': current_shape,
                    'output_shape': (current_shape[0], module.out_features)
                }
                mapping['layers'].append(layer_mapping)
                current_shape = layer_mapping['output_shape']
        
        return mapping
    
    def _map_pipelined(self, input_shape: Tuple) -> Dict:
        """
        Mapping pipeliné (overlapping computation)
        """
        mapping = {
            'type': 'pipelined',
            'pipeline_stages': 3,
            'layers': []
        }
        
        # Mapping avec pipeline stages
        current_shape = input_shape
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layer_mapping = {
                    'name': name,
                    'type': 'linear_pipelined',
                    'pipeline_stage': len(mapping['layers']) % mapping['pipeline_stages'],
                    'pe_count': 32,
                    'input_shape': current_shape,
                    'output_shape': (current_shape[0], module.out_features)
                }
                mapping['layers'].append(layer_mapping)
                current_shape = layer_mapping['output_shape']
        
        return mapping
    
    def _evaluate_mapping(self, mapping: Dict, input_shape: Tuple) -> float:
        """
        Évalue la latence d'un mapping
        
        Returns:
            Latence en nanosecondes
        """
        # Simplifié: estimation basée sur type de mapping
        total_latency = 0
        
        if mapping['type'] == 'layer_wise':
            # Latence séquentielle
            for layer in mapping['layers']:
                if layer['type'] == 'linear':
                    n_ops = layer['input_shape'][1] * layer['output_shape'][1]
                    cycles = np.ceil(n_ops / layer['pe_count'])
                    total_latency += cycles * 5  # 5 ns par cycle (200 MHz)
        
        elif mapping['type'] == 'tensor_slicing':
            # Latence avec parallélisme
            for layer in mapping['layers']:
                n_ops = layer['input_shape'][1] * layer['output_shape'][1]
                cycles = np.ceil(n_ops / (layer['pe_per_slice'] * layer['slices']))
                total_latency += cycles * 5
        
        elif mapping['type'] == 'pipelined':
            # Latence pipelinée (overlap)
            max_stage_latency = 0
            for layer in mapping['layers']:
                n_ops = layer['input_shape'][1] * layer['output_shape'][1]
                cycles = np.ceil(n_ops / layer['pe_count'])
                stage_latency = cycles * 5
                max_stage_latency = max(max_stage_latency, stage_latency)
            total_latency = max_stage_latency * mapping['pipeline_stages']
        
        return total_latency
```

---

## Applications Pratiques

### Cas d'Usage: Trigger HEP

```python
class HEPTriggerCoDesign:
    """
    Co-design pour système de trigger HEP
    """
    
    def __init__(self):
        self.requirements = {
            'max_latency_ns': 100000,  # 100 μs
            'max_energy_nj': 500,      # 500 nJ
            'target_accuracy': 0.95,    # 95%
            'fpga_family': 'Xilinx Zynq UltraScale+'
        }
    
    def design_trigger_system(self):
        """
        Conçoit un système de trigger optimisé
        """
        print("\n" + "="*70)
        print("Co-design Système de Trigger HEP")
        print("="*70)
        
        # 1. Définir espace de recherche
        print("\n1. Définition de l'espace de recherche...")
        model_space = ConstrainedSearchSpace(self.requirements, None)  # simplifié
        hw_space = HardwareConfigSpace()
        combined_space = CombinedSearchSpace(model_space, hw_space)
        
        # 2. Optimisation jointe
        print("\n2. Optimisation jointe modèle-hardware...")
        optimizer = JointOptimizationNAS(combined_space, self.requirements)
        result = optimizer.optimize(n_iterations=100)
        
        # 3. Résultats
        print("\n3. Résultats du co-design:")
        print(f"  Architecture modèle: {result['model_config']}")
        print(f"  Configuration hardware:")
        print(f"    - PE count: {result['hardware_config']['pe_count']}")
        print(f"    - Clock: {result['hardware_config']['clock_mhz']} MHz")
        print(f"    - Dataflow: {result['hardware_config']['dataflow']}")
        print(f"  Métriques:")
        print(f"    - Latence: {result['metrics']['latency_us']:.2f} μs")
        print(f"    - Énergie: {result['metrics']['energy_nj']:.2f} nJ")
        print(f"    - Performance: {result['metrics']['performance']:.4f}")
        print(f"    - Score combiné: {result['score']:.4f}")
        
        # 4. Vérification contraintes
        print("\n4. Vérification des contraintes:")
        constraints_met = (
            result['metrics']['latency_ns'] <= self.requirements['max_latency_ns'] and
            result['metrics']['energy_nj'] <= self.requirements['max_energy_nj']
        )
        print(f"  Contraintes respectées: {constraints_met}")
        
        return result
```

---

## Exercices

### Exercice 16.5.1
Implémentez un système de co-design pour un modèle de classification d'images avec contraintes FPGA. Comparez avec optimisation séparée modèle/hardware.

### Exercice 16.5.2
Créez un entraînement hardware-aware qui intègre la latence FPGA comme régularisation et comparez avec entraînement standard.

### Exercice 16.5.3
Développez un algorithme de mapping adaptatif qui trouve automatiquement la meilleure répartition des couches sur les ressources FPGA.

### Exercice 16.5.4
Analysez le trade-off entre différentes stratégies de mapping (layer-wise, tensor slicing, pipelined) pour un modèle donné.

---

## Points Clés à Retenir

> 📌 **Le co-design optimise simultanément modèle et hardware pour meilleure solution globale**

> 📌 **L'entraînement hardware-aware intègre contraintes hardware pendant l'apprentissage**

> 📌 **Le mapping adaptatif optimise l'utilisation des ressources hardware**

> 📌 **Les stratégies de mapping (layer-wise, slicing, pipelined) offrent différents trade-offs**

> 📌 **Le co-design est particulièrement important pour applications avec contraintes strictes (trigger HEP)**

> 📌 **L'espace de recherche combiné est très large mais peut être exploré efficacement avec bonnes stratégies**

---

*Section précédente : [16.4 Algorithmes de Recherche Efficaces](./16_04_Algorithmes.md)*

