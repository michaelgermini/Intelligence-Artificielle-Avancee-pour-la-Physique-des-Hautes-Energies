# 17.3 Mapping sur Architectures Parallèles

---

## Introduction

Le **mapping de réseaux de tenseurs sur architectures parallèles** consiste à répartir les calculs et données sur plusieurs unités de calcul (CPU cores, GPU streaming multiprocessors, FPGA Processing Elements) pour exploiter le parallélisme.

Cette section présente les stratégies de mapping pour différents types d'architectures parallèles, incluant le parallélisme de données, de modèles, et hybride.

---

## Types de Parallélisme

### Classification

```python
class ParallelismTypes:
    """
    Types de parallélisme pour réseaux de tenseurs
    """
    
    def __init__(self):
        self.parallelism_types = {
            'data_parallelism': {
                'description': 'Répartir données sur différentes unités',
                'example': 'Chaque GPU traite un batch différent',
                'best_for': 'Batch processing, données indépendantes',
                'communication': 'Faible (gradients seulement)'
            },
            'model_parallelism': {
                'description': 'Répartir modèle sur différentes unités',
                'example': 'Chaque GPU contient une partie du réseau',
                'best_for': 'Modèles trop grands pour un device',
                'communication': 'Élevée (activations entre unités)'
            },
            'tensor_parallelism': {
                'description': 'Paralléliser opérations tensorielles',
                'example': 'Découper contractions sur plusieurs unités',
                'best_for': 'Contractions grandes',
                'communication': 'Modérée (résultats partiels)'
            },
            'pipeline_parallelism': {
                'description': 'Pipeline de calculs séquentiels',
                'example': 'Chaque device traite une étape du pipeline',
                'best_for': 'Séquences de calculs',
                'communication': 'Modérée (activation forwarding)'
            }
        }
    
    def display_types(self):
        """Affiche les types de parallélisme"""
        print("\n" + "="*70)
        print("Types de Parallélisme")
        print("="*70)
        
        for ptype, info in self.parallelism_types.items():
            print(f"\n{ptype.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Example: {info['example']}")
            print(f"  Best for: {info['best_for']}")
            print(f"  Communication: {info['communication']}")

parallelism = ParallelismTypes()
parallelism.display_types()
```

---

## Mapping sur GPU Multi-Cards

### Data Parallelism

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class GPUMultiCardMapping:
    """
    Mapping sur GPU multi-cards
    """
    
    def __init__(self, n_gpus=4):
        """
        Args:
            n_gpus: Nombre de GPUs disponibles
        """
        self.n_gpus = n_gpus
        self.devices = [f'cuda:{i}' for i in range(n_gpus)]
    
    def data_parallel_tensor_network(self, model, batch_size=64):
        """
        Data parallelism: chaque GPU traite un sous-batch
        """
        batch_per_gpu = batch_size // self.n_gpus
        
        def forward_data_parallel(inputs):
            """Forward pass distribué"""
            outputs = []
            
            # Répartir batch sur GPUs
            for i, device in enumerate(self.devices):
                start_idx = i * batch_per_gpu
                end_idx = start_idx + batch_per_gpu
                batch_slice = inputs[start_idx:end_idx].to(device)
                
                # Forward sur ce GPU
                model_device = model.to(device)
                output_slice = model_device(batch_slice)
                outputs.append(output_slice.cpu())
            
            # Combiner résultats
            return torch.cat(outputs, dim=0)
        
        return forward_data_parallel
    
    def model_parallel_tensor_network(self, tensor_layers):
        """
        Model parallelism: répartir couches tensorielles sur GPUs
        
        Pour réseau de tenseurs: répartir tenseurs sur différents GPUs
        """
        layers_per_gpu = len(tensor_layers) // self.n_gpus
        
        gpu_layers = {}
        for i, device in enumerate(self.devices):
            start_idx = i * layers_per_gpu
            end_idx = start_idx + layers_per_gpu if i < self.n_gpus - 1 else len(tensor_layers)
            gpu_layers[device] = tensor_layers[start_idx:end_idx]
        
        def forward_model_parallel(inputs):
            """Forward avec modèle distribué"""
            current_input = inputs.to(self.devices[0])
            
            for device in self.devices:
                layers = gpu_layers[device]
                current_input = current_input.to(device)
                
                # Appliquer couches sur ce GPU
                for layer in layers:
                    current_input = layer(current_input)
                
                # Transférer vers prochain GPU si nécessaire
                if device != self.devices[-1]:
                    current_input = current_input.to(self.devices[self.devices.index(device) + 1])
            
            return current_input
        
        return forward_model_parallel
    
    def tensor_parallel_contraction(self, tensor_A, tensor_B, n_splits=4):
        """
        Paralléliser une contraction tensorielle sur plusieurs GPUs
        
        Stratégie: découper tenseurs et calculer par morceaux
        """
        # Pour A[i,j,k] * B[j,k,l] → C[i,l]
        # Découper sur dimension i
        
        if len(tensor_A.shape) == 3 and len(tensor_B.shape) == 3:
            i_dim, j_dim, k_dim = tensor_A.shape
            _, _, l_dim = tensor_B.shape
            
            split_size = i_dim // n_splits
            results = []
            
            for gpu_idx, device in enumerate(self.devices[:n_splits]):
                start_i = gpu_idx * split_size
                end_i = start_i + split_size if gpu_idx < n_splits - 1 else i_dim
                
                # Slice de A
                A_slice = tensor_A[start_i:end_i, :, :].to(device)
                B_device = tensor_B.to(device)
                
                # Contraction sur ce GPU
                C_slice = torch.einsum('ijk,jkl->il', A_slice, B_device)
                results.append(C_slice.cpu())
            
            # Combiner résultats
            C = torch.cat(results, dim=0)
            return C
        else:
            # Fallback
            return torch.einsum('ijk,jkl->il', tensor_A, tensor_B)

# Exemple
if torch.cuda.device_count() >= 2:
    gpu_mapper = GPUMultiCardMapping(n_gpus=min(2, torch.cuda.device_count()))
    print(f"\nMapping sur {gpu_mapper.n_gpus} GPUs")
else:
    print("\nGPUs multiples non disponibles")
```

---

## Mapping sur FPGA Multi-Chip

### Distribution Spatiale

```python
class FPGAMultiChipMapping:
    """
    Mapping sur plusieurs FPGAs
    """
    
    def __init__(self, n_fpgas=4, pe_per_fpga=64):
        """
        Args:
            n_fpgas: Nombre de FPGAs
            pe_per_fpga: PEs par FPGA
        """
        self.n_fpgas = n_fpgas
        self.pe_per_fpga = pe_per_fpga
        self.total_pe = n_fpgas * pe_per_fpga
    
    def spatial_mapping_mps(self, mps_tensors, bond_dimensions):
        """
        Mapping spatial d'un MPS sur plusieurs FPGAs
        
        Stratégie: Répartir tenseurs MPS sur FPGAs différents
        """
        n_tensors = len(mps_tensors)
        tensors_per_fpga = n_tensors // self.n_fpgas
        
        mapping = {}
        for fpga_id in range(self.n_fpgas):
            start_idx = fpga_id * tensors_per_fpga
            end_idx = start_idx + tensors_per_fpga if fpga_id < self.n_fpgas - 1 else n_tensors
            
            mapping[f'fpga_{fpga_id}'] = {
                'tensor_indices': list(range(start_idx, end_idx)),
                'pe_count': self.pe_per_fpga,
                'bond_dimensions': bond_dimensions[start_idx:end_idx+1]
            }
        
        return mapping
    
    def pipeline_mapping(self, contraction_sequence):
        """
        Mapping pipeline: chaque FPGA traite une étape
        
        Avantage: Throughput élevé avec plusieurs événements en pipeline
        """
        n_steps = len(contraction_sequence)
        steps_per_fpga = max(1, n_steps // self.n_fpgas)
        
        pipeline = {}
        for fpga_id in range(self.n_fpgas):
            start_step = fpga_id * steps_per_fpga
            end_step = start_step + steps_per_fpga if fpga_id < self.n_fpgas - 1 else n_steps
            
            pipeline[f'fpga_{fpga_id}'] = {
                'contraction_steps': contraction_sequence[start_step:end_step],
                'pipeline_stage': fpga_id,
                'pe_count': self.pe_per_fpga
            }
        
        # Estimer latence pipeline
        latency_per_stage = 100  # ns (exemple)
        pipeline_latency = latency_per_stage * len(pipeline)
        pipeline_throughput = 1.0 / (latency_per_stage * 1e-9)  # events/sec
        
        return {
            'pipeline': pipeline,
            'latency_ns': pipeline_latency,
            'throughput_events_per_sec': pipeline_throughput
        }
    
    def estimate_inter_fpga_communication(self, mapping):
        """
        Estime la communication nécessaire entre FPGAs
        """
        communication = {
            'data_transfer_size_bytes': 0,
            'bandwidth_required_gbps': 0,
            'latency_overhead_ns': 0
        }
        
        # Simplifié: estime selon mapping
        # En pratique, dépend de la structure du réseau
        
        return communication

fpga_mapper = FPGAMultiChipMapping(n_fpgas=4, pe_per_fpga=64)

# Exemple MPS
mps_tensors = list(range(10))  # 10 tenseurs
bond_dims = [10] * 11

spatial_map = fpga_mapper.spatial_mapping_mps(mps_tensors, bond_dims)

print("\n" + "="*70)
print("Mapping Spatial MPS sur FPGAs")
print("="*70)
for fpga, config in spatial_map.items():
    print(f"\n{fpga}:")
    print(f"  Tenseurs: {config['tensor_indices']}")
    print(f"  PEs: {config['pe_count']}")
```

---

## Mapping Hybride CPU-GPU

### Partitionnement Adaptatif

```python
class HybridCPUGPUMapping:
    """
    Mapping hybride sur CPU et GPU
    """
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
    
    def adaptive_mapping(self, contraction_sequence, tensor_sizes):
        """
        Mapping adaptatif: assigne à CPU ou GPU selon taille/complexité
        """
        mapping = []
        
        for i, (contraction, size_A, size_B) in enumerate(zip(contraction_sequence, 
                                                               tensor_sizes[:-1], 
                                                               tensor_sizes[1:])):
            # Heuristique: GPU pour grandes opérations, CPU pour petites
            total_size = size_A * size_B
            
            if self.has_gpu and total_size > 1e6:  # Seuil
                device = 'gpu'
            else:
                device = 'cpu'
            
            mapping.append({
                'contraction': contraction,
                'device': device,
                'size': total_size
            })
        
        return mapping
    
    def load_balancing_mapping(self, contractions, n_cpu_cores=8, n_gpus=1):
        """
        Mapping avec équilibrage de charge
        
        Répartit contractions pour minimiser temps total
        """
        # Estimer temps pour chaque contraction
        contraction_times = []
        for contraction, size_A, size_B in zip(contractions, 
                                               tensor_sizes[:-1], 
                                               tensor_sizes[1:]):
            # Estimation simplifiée
            ops = size_A * size_B
            time_cpu = ops / (n_cpu_cores * 1e9)  # Gops/sec par core
            time_gpu = ops / (n_gpus * 1e12) if self.has_gpu else float('inf')  # Tops/sec
            
            contraction_times.append({
                'contraction': contraction,
                'time_cpu': time_cpu,
                'time_gpu': time_gpu,
                'best_device': 'gpu' if time_gpu < time_cpu else 'cpu'
            })
        
        # Assigner avec équilibrage
        cpu_load = 0
        gpu_load = 0
        mapping = []
        
        for ct in sorted(contraction_times, key=lambda x: max(x['time_cpu'], x['time_gpu']), 
                        reverse=True):
            if ct['best_device'] == 'gpu' and gpu_load <= cpu_load:
                mapping.append({'contraction': ct['contraction'], 'device': 'gpu'})
                gpu_load += ct['time_gpu']
            else:
                mapping.append({'contraction': ct['contraction'], 'device': 'cpu'})
                cpu_load += ct['time_cpu']
        
        return {
            'mapping': mapping,
            'cpu_load': cpu_load,
            'gpu_load': gpu_load,
            'total_time': max(cpu_load, gpu_load)
        }

hybrid_mapper = HybridCPUGPUMapping()
```

---

## Communication et Synchronisation

### Gestion des Données Distribuées

```python
class CommunicationOptimization:
    """
    Optimisation de la communication dans mapping distribué
    """
    
    def reduce_communication_overhead(self, mapping):
        """
        Réduit overhead de communication
        
        Stratégies:
        - Fusionner communications
        - Pipeline communication/computation
        - Compression des données
        """
        strategies = {
            'communication_fusion': {
                'description': 'Fusionner plusieurs transfers en un',
                'benefit': 'Réduit latence et overhead'
            },
            'pipeline_comm_compute': {
                'description': 'Overlap communication et computation',
                'benefit': 'Masque latence communication'
            },
            'data_compression': {
                'description': 'Compresser données transférées',
                'benefit': 'Réduit bandwidth nécessaire'
            },
            'local_accumulation': {
                'description': 'Accumuler localement avant communication',
                'benefit': 'Réduit nombre de communications'
            }
        }
        
        return strategies
    
    def allreduce_optimization(self, n_devices):
        """
        Optimise AllReduce (somme sur tous devices)
        
        Important pour data parallelism et accumulation de gradients
        """
        # Ring AllReduce: O(n) communication au lieu de O(n²)
        ring_steps = n_devices - 1
        
        return {
            'algorithm': 'Ring AllReduce',
            'communication_steps': ring_steps,
            'bandwidth_efficient': True
        }

comm_opt = CommunicationOptimization()
strategies = comm_opt.reduce_communication_overhead({})
print("\nStratégies d'optimisation communication:")
for strategy, info in strategies.items():
    print(f"  {strategy}: {info['description']}")
```

---

## Cas d'Usage: Réseau MPS Distribué

### Exemple Complet

```python
class DistributedMPSExample:
    """
    Exemple complet de mapping distribué d'un MPS
    """
    
    def map_mps_to_hardware(self, n_tensors=20, bond_dim=32, 
                           n_fpgas=4, batch_size=100):
        """
        Mapping complet d'un MPS sur hardware distribué
        """
        # 1. Décomposer MPS
        mps_structure = {
            'n_tensors': n_tensors,
            'bond_dimension': bond_dim,
            'physical_dim': 2  # Exemple: spin-1/2
        }
        
        # 2. Ordonnancement optimal (séquentiel pour MPS)
        contraction_order = [(i, i+1) for i in range(n_tensors - 1)]
        
        # 3. Mapping spatial: répartir tenseurs sur FPGAs
        fpga_mapper = FPGAMultiChipMapping(n_fpgas=n_fpgas)
        spatial_mapping = fpga_mapper.spatial_mapping_mps(
            list(range(n_tensors)), [bond_dim] * (n_tensors + 1)
        )
        
        # 4. Estimer performance
        contractions_per_fpga = n_tensors // n_fpgas
        latency_per_contraction_ns = 100  # Exemple
        total_latency_ns = contractions_per_fpga * latency_per_contraction_ns
        
        # 5. Communication inter-FPGA
        comm_overhead_ns = (n_fpgas - 1) * 50  # Exemple
        
        total_latency = total_latency_ns + comm_overhead_ns
        
        return {
            'mps_structure': mps_structure,
            'contraction_order': contraction_order,
            'spatial_mapping': spatial_mapping,
            'estimated_latency_ns': total_latency,
            'estimated_throughput': batch_size / (total_latency * 1e-9)
        }

# Exemple
distributed_example = DistributedMPSExample()
result = distributed_example.map_mps_to_hardware()

print("\n" + "="*70)
print("Exemple: MPS Distribué")
print("="*70)
print(f"Latence estimée: {result['estimated_latency_ns']/1000:.2f} μs")
print(f"Throughput: {result['estimated_throughput']:.2e} events/sec")
```

---

## Exercices

### Exercice 17.3.1
Implémentez un mapping data-parallel pour un réseau de tenseurs sur 4 GPUs et mesurez le speedup.

### Exercice 17.3.2
Concevez un mapping pipeline d'un MPS sur 4 FPGAs et estimez le throughput avec events en pipeline.

### Exercice 17.3.3
Créez un système de mapping adaptatif qui assigne automatiquement contractions à CPU ou GPU selon leur taille.

### Exercice 17.3.4
Optimisez la communication dans un mapping distribué en utilisant Ring AllReduce et comparez avec communication naive.

---

## Points Clés à Retenir

> 📌 **Data parallelism est simple mais nécessite batch processing**

> 📌 **Model/tensor parallelism permet de traiter modèles plus grands mais avec plus de communication**

> 📌 **Pipeline parallelism améliore throughput pour séquences de calculs**

> 📌 **Mapping hybride CPU-GPU peut optimiser utilisation ressources hétérogènes**

> 📌 **Communication est souvent le bottleneck dans mapping distribué**

> 📌 **Optimisations (Ring AllReduce, fusion, pipelining) sont essentielles**

---

*Section précédente : [17.2 Ordonnancement Optimal](./17_02_Ordonnancement.md) | Section suivante : [17.4 Quantification Hardware-Aware](./17_04_Quantification.md)*

