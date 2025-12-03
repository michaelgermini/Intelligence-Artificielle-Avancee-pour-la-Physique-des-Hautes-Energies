# 17.1 Implémentation Efficace des Contractions Tensorielles

---

## Introduction

Les **contractions tensorielles** sont l'opération fondamentale des réseaux de tenseurs. Une implémentation efficace sur hardware nécessite une compréhension approfondie des patterns d'accès mémoire, de la vectorisation, et des optimisations spécifiques à chaque type de hardware (FPGA, GPU, CPU).

Cette section présente les techniques d'implémentation optimisées pour différentes architectures, incluant les optimisations de cache, la vectorisation SIMD, et les architectures spécialisées pour contractions.

---

## Fondements des Contractions Tensorielles

### Définition Mathématique

```python
import numpy as np
import torch

class TensorContraction:
    """
    Implémentation de contractions tensorielles
    """
    
    def __init__(self):
        self.definition = """
        Contraction tensorielle:
        
        Soit deux tenseurs A[i₁,...,iₘ] et B[j₁,...,jₙ] avec indices partagés.
        La contraction sur indices k₁,...,kₚ produit:
        
        C[free_indices] = Σ_{k₁,...,kₚ} A[...] * B[...]
        
        Complexité: O(∏ dims_free × ∏ dims_contracted)
        """
    
    def einsum_notation(self):
        """
        Notation Einstein pour contractions
        """
        examples = {
            'matrix_mult': {
                'notation': 'ij,jk->ik',
                'description': 'Multiplication matricielle standard',
                'example': 'A[i,j] * B[j,k] → C[i,k]'
            },
            'batched_matmul': {
                'notation': 'bij,bjk->bik',
                'description': 'Multiplication matricielle batchée',
                'example': 'A[b,i,j] * B[b,j,k] → C[b,i,k]'
            },
            'trace': {
                'notation': 'ii->',
                'description': 'Trace d\'une matrice',
                'example': 'A[i,i] → scalaire'
            },
            'outer_product': {
                'notation': 'i,j->ij',
                'description': 'Produit extérieur',
                'example': 'a[i] * b[j] → C[i,j]'
            },
            'tensor_contraction': {
                'notation': 'ijk,jkl->il',
                'description': 'Contraction sur indices j et k',
                'example': 'A[i,j,k] * B[j,k,l] → C[i,l]'
            }
        }
        return examples

# Exemple pratique
def simple_contraction():
    """Exemple simple de contraction"""
    # Tenseur A: shape (10, 20, 15)
    A = np.random.rand(10, 20, 15).astype(np.float32)
    
    # Tenseur B: shape (20, 15, 25)
    B = np.random.rand(20, 15, 25).astype(np.float32)
    
    # Contraction: A[i,j,k] * B[j,k,l] → C[i,l]
    # Contracter sur j et k
    C = np.einsum('ijk,jkl->il', A, B)
    
    print(f"Shape A: {A.shape}")
    print(f"Shape B: {B.shape}")
    print(f"Shape C (résultat): {C.shape}")
    
    # Vérification
    C_manual = np.zeros((10, 25), dtype=np.float32)
    for i in range(10):
        for l in range(25):
            for j in range(20):
                for k in range(15):
                    C_manual[i, l] += A[i, j, k] * B[j, k, l]
    
    print(f"\nErreur max: {np.max(np.abs(C - C_manual)):.2e}")
    
    # Complexité
    ops = 10 * 25 * 20 * 15
    print(f"Opérations: {ops:,}")

simple_contraction()
```

---

## Implémentations CPU Optimisées

### Optimisation avec Cache Blocking

```python
class OptimizedCPUTensorContraction:
    """
    Contractions optimisées pour CPU avec cache blocking
    """
    
    def __init__(self, block_size=32):
        """
        Args:
            block_size: Taille des blocs pour cache blocking (en éléments)
        """
        self.block_size = block_size
    
    def blocked_contraction(self, A, B, contract_dims_A, contract_dims_B):
        """
        Contraction avec cache blocking
        
        Stratégie: Découper en blocs qui tiennent dans le cache L2/L3
        """
        # Déterminer dimensions libres et contractées
        shape_A = A.shape
        shape_B = B.shape
        
        # Pour simplicité, supposons contraction standard
        # A[i,j,k] * B[j,k,l] → C[i,l]
        if len(shape_A) == 3 and len(shape_B) == 3:
            i_dim, j_dim, k_dim = shape_A
            _, _, l_dim = shape_B
            
            C = np.zeros((i_dim, l_dim), dtype=A.dtype)
            
            # Blocking sur dimensions i et l
            for i_block in range(0, i_dim, self.block_size):
                i_end = min(i_block + self.block_size, i_dim)
                
                for l_block in range(0, l_dim, self.block_size):
                    l_end = min(l_block + self.block_size, l_dim)
                    
                    # Blocking sur dimension j
                    for j_block in range(0, j_dim, self.block_size):
                        j_end = min(j_block + self.block_size, j_dim)
                        
                        # Calculer bloc de résultat
                        for i in range(i_block, i_end):
                            for l in range(l_block, l_end):
                                for j in range(j_block, j_end):
                                    for k in range(k_dim):
                                        C[i, l] += A[i, j, k] * B[j, k, l]
            
            return C
        else:
            # Fallback vers einsum
            return np.einsum('ijk,jkl->il', A, B)
    
    def vectorized_contraction(self, A, B):
        """
        Contraction vectorisée (utilise opérations vectorielles du CPU)
        """
        # Pour A[i,j,k] * B[j,k,l] → C[i,l]
        # Vectoriser sur dimension k (innermost)
        
        shape_A = A.shape
        shape_B = B.shape
        
        if len(shape_A) == 3 and len(shape_B) == 3:
            i_dim, j_dim, k_dim = shape_A
            _, _, l_dim = shape_B
            
            C = np.zeros((i_dim, l_dim), dtype=A.dtype)
            
            # Vectorisation: traiter plusieurs k en parallèle
            # NumPy utilise automatiquement SIMD si disponible
            for i in range(i_dim):
                for l in range(l_dim):
                    # Vectoriser sur j et k
                    # A[i, :, :] * B[:, :, l] puis sommer
                    temp = np.sum(A[i, :, :] * B[:, :, l], axis=(0, 1))
                    C[i, l] = temp
            
            return C
        else:
            return np.einsum('ijk,jkl->il', A, B)

# Test de performance
def benchmark_contractions():
    """Compare différentes implémentations"""
    import time
    
    # Créer tenseurs de test
    size = 64
    A = np.random.rand(size, size, size).astype(np.float32)
    B = np.random.rand(size, size, size).astype(np.float32)
    
    # 1. Einsum (optimisé NumPy)
    start = time.time()
    C1 = np.einsum('ijk,jkl->il', A, B)
    time_einsum = time.time() - start
    
    # 2. Blocked
    cpu_opt = OptimizedCPUTensorContraction(block_size=16)
    start = time.time()
    C2 = cpu_opt.blocked_contraction(A, B, [1, 2], [0, 1])
    time_blocked = time.time() - start
    
    # 3. Vectorized
    start = time.time()
    C3 = cpu_opt.vectorized_contraction(A, B)
    time_vectorized = time.time() - start
    
    print("\n" + "="*70)
    print("Benchmark Contractions CPU")
    print("="*70)
    print(f"Einsum (NumPy):     {time_einsum*1000:.2f} ms")
    print(f"Blocked:            {time_blocked*1000:.2f} ms")
    print(f"Vectorized:         {time_vectorized*1000:.2f} ms")
    
    # Vérifier exactitude
    print(f"\nErreur max (blocked):    {np.max(np.abs(C1 - C2)):.2e}")
    print(f"Erreur max (vectorized): {np.max(np.abs(C1 - C3)):.2e}")

# benchmark_contractions()
```

---

## Implémentations GPU Optimisées

### Utilisation de CuPy/PyTorch

```python
class OptimizedGPUTensorContraction:
    """
    Contractions optimisées pour GPU
    """
    
    def __init__(self, device='cuda'):
        """
        Args:
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.use_torch = torch.cuda.is_available() if device == 'cuda' else False
    
    def gpu_einsum(self, A, B, pattern='ijk,jkl->il'):
        """
        Contraction sur GPU avec einsum optimisé
        """
        if self.use_torch:
            A_t = torch.from_numpy(A).cuda() if isinstance(A, np.ndarray) else A
            B_t = torch.from_numpy(B).cuda() if isinstance(B, np.ndarray) else B
            
            # PyTorch einsum optimisé pour GPU
            C_t = torch.einsum(pattern, A_t, B_t)
            
            # Synchroniser pour timing précis
            torch.cuda.synchronize()
            
            return C_t.cpu().numpy() if isinstance(A, np.ndarray) else C_t
        else:
            # Fallback CPU
            return np.einsum(pattern, A, B)
    
    def batched_gpu_contraction(self, A_batch, B_batch):
        """
        Contraction batchée sur GPU (efficace pour traitement parallèle)
        
        A_batch: shape (batch, i, j, k)
        B_batch: shape (batch, j, k, l)
        Résultat: (batch, i, l)
        """
        if self.use_torch:
            pattern = 'bijkl,bjklm->bilm' if len(A_batch.shape) == 5 else 'bijk,bjkl->bil'
            
            A_t = torch.from_numpy(A_batch).cuda() if isinstance(A_batch, np.ndarray) else A_batch
            B_t = torch.from_numpy(B_batch).cuda() if isinstance(B_batch, np.ndarray) else B_batch
            
            C_t = torch.einsum(pattern, A_t, B_t)
            torch.cuda.synchronize()
            
            return C_t.cpu().numpy() if isinstance(A_batch, np.ndarray) else C_t
        else:
            # CPU fallback
            results = []
            for b in range(A_batch.shape[0]):
                C_b = np.einsum('ijk,jkl->il', A_batch[b], B_batch[b])
                results.append(C_b)
            return np.stack(results)
    
    def optimized_tensor_cores(self, A, B):
        """
        Utilise Tensor Cores (NVIDIA V100+) pour contractions
        
        Nécessite dimensions multiples de 8/16 pour fp16
        """
        if not self.use_torch:
            return self.gpu_einsum(A, B)
        
        A_t = torch.from_numpy(A).cuda().half() if isinstance(A, np.ndarray) else A.half()
        B_t = torch.from_numpy(B).cuda().half() if isinstance(B, np.ndarray) else B.half()
        
        # Tensor Cores activés automatiquement pour certaines opérations
        # MatMul avec dimensions multiples de 8/16
        # Convertir contraction en séquence de matmuls si possible
        
        # Exemple: A[i,j,k] * B[j,k,l] → C[i,l]
        # Reshape: A -> (i, j*k), B -> (j*k, l)
        # MatMul -> (i, l)
        shape_A = A_t.shape
        shape_B = B_t.shape
        
        if len(shape_A) == 3 and len(shape_B) == 3:
            i, j, k = shape_A
            _, _, l = shape_B
            
            A_reshaped = A_t.view(i, j * k)
            B_reshaped = B_t.view(j * k, l)
            
            # MatMul utilise Tensor Cores si disponible
            C_t = torch.matmul(A_reshaped, B_reshaped)
            torch.cuda.synchronize()
            
            return C_t.cpu().float().numpy() if isinstance(A, np.ndarray) else C_t.float()
        else:
            return self.gpu_einsum(A, B)

# Exemple GPU
def gpu_contraction_example():
    """Exemple d'utilisation GPU"""
    if torch.cuda.is_available():
        gpu_opt = OptimizedGPUTensorContraction(device='cuda')
        
        # Créer tenseurs
        size = 128
        A = np.random.rand(size, size, size).astype(np.float32)
        B = np.random.rand(size, size, size).astype(np.float32)
        
        # Contraction GPU
        import time
        start = time.time()
        C_gpu = gpu_opt.gpu_einsum(A, B)
        time_gpu = time.time() - start
        
        # Contraction CPU pour comparaison
        start = time.time()
        C_cpu = np.einsum('ijk,jkl->il', A, B)
        time_cpu = time.time() - start
        
        print(f"\nGPU contraction: {time_gpu*1000:.2f} ms")
        print(f"CPU contraction: {time_cpu*1000:.2f} ms")
        print(f"Speedup: {time_cpu/time_gpu:.2f}x")
        print(f"Erreur max: {np.max(np.abs(C_gpu - C_cpu)):.2e}")
    else:
        print("CUDA non disponible")
```

---

## Implémentations FPGA Optimisées

### Architecture Systolic pour Contractions

```python
class FPGATensorContraction:
    """
    Implémentation FPGA de contractions tensorielles
    """
    
    def __init__(self, pe_count=64, data_width=16):
        """
        Args:
            pe_count: Nombre de Processing Elements (PEs)
            data_width: Largeur de données en bits (8, 16, 32)
        """
        self.pe_count = pe_count
        self.data_width = data_width
        
        # Configuration hardware
        self.config = {
            'clock_freq_mhz': 200,
            'dsp_per_pe': 1,
            'lut_per_pe': 500,
            'bram_per_pe': 1
        }
    
    def systolic_array_contraction(self, A_shape, B_shape, contraction_pattern):
        """
        Conçoit une architecture systolic pour contraction
        
        Architecture Systolic:
        - Array de PEs organisés en grille
        - Données circulent en pipeline
        - Très efficace pour opérations régulières
        """
        architecture = {
            'type': 'systolic_array',
            'pe_array_size': (int(np.sqrt(self.pe_count)), int(np.sqrt(self.pe_count))),
            'dataflow': 'weight_stationary',  # ou 'output_stationary', 'input_stationary'
            'pipeline_stages': 10,
            'initiation_interval': 1  # Nouvelle contraction chaque cycle (idéal)
        }
        
        # Estimer latence
        i_dim, j_dim, k_dim = A_shape
        _, _, l_dim = B_shape
        
        # Nombre de cycles pour remplir le pipeline
        fill_cycles = architecture['pipeline_stages']
        
        # Cycles de calcul
        # Avec systolic array de taille N×N, traiter M éléments nécessite ~M/N cycles
        compute_cycles = max(i_dim * l_dim / self.pe_count, j_dim * k_dim / self.pe_count)
        
        total_cycles = fill_cycles + compute_cycles
        latency_ns = total_cycles * (1000.0 / self.config['clock_freq_mhz'])
        
        architecture['estimated_latency_ns'] = latency_ns
        architecture['estimated_throughput_ops_per_sec'] = (i_dim * l_dim * j_dim * k_dim) / (latency_ns * 1e-9)
        
        return architecture
    
    def estimate_resources(self, A_shape, B_shape):
        """
        Estime les ressources FPGA nécessaires
        """
        i_dim, j_dim, k_dim = A_shape
        _, _, l_dim = B_shape
        
        # Nombre de PEs
        n_pe = self.pe_count
        
        # Ressources par PE
        luts_per_pe = self.config['lut_per_pe']
        dsps_per_pe = self.config['dsp_per_pe']
        brams_per_pe = self.config['bram_per_pe']
        
        # Mémoire pour buffers
        # Buffer A: i × j × k éléments
        buffer_A_size = i_dim * j_dim * k_dim * (self.data_width // 8)
        bram_A = np.ceil(buffer_A_size / (36 * 1024))  # 36 KB par BRAM
        
        # Buffer B: j × k × l éléments
        buffer_B_size = j_dim * k_dim * l_dim * (self.data_width // 8)
        bram_B = np.ceil(buffer_B_size / (36 * 1024))
        
        # Buffer résultat: i × l éléments
        buffer_C_size = i_dim * l_dim * (self.data_width // 8)
        bram_C = np.ceil(buffer_C_size / (36 * 1024))
        
        resources = {
            'lut': n_pe * luts_per_pe + 10000,  # + overhead
            'dsp': n_pe * dsps_per_pe,
            'bram': bram_A + bram_B + bram_C + (n_pe * brams_per_pe),
            'buffer_memory_mb': (buffer_A_size + buffer_B_size + buffer_C_size) / (1024**2)
        }
        
        return resources
    
    def dataflow_optimization(self, contraction_pattern):
        """
        Optimise le dataflow pour réduire accès mémoire
        """
        dataflow_strategies = {
            'weight_stationary': {
                'description': 'Poids (B) restent dans PEs, A et résultats circulent',
                'pros': ['Réutilise B plusieurs fois', 'Réduit accès mémoire B'],
                'cons': ['Nécessite plus de BRAM pour B'],
                'best_for': 'B petit, A grand'
            },
            'output_stationary': {
                'description': 'Résultats accumulent dans PEs',
                'pros': ['Réduit écritures mémoire résultat'],
                'cons': ['Plus complexe à gérer'],
                'best_for': 'Résultat petit'
            },
            'input_stationary': {
                'description': 'Inputs (A) restent, B et résultats circulent',
                'pros': ['Réutilise A'],
                'cons': ['Nécessite plus de BRAM pour A'],
                'best_for': 'A petit, B grand'
            }
        }
        
        return dataflow_strategies

# Exemple FPGA
fpga_contract = FPGATensorContraction(pe_count=64, data_width=16)

A_shape = (100, 50, 30)
B_shape = (50, 30, 25)

architecture = fpga_contract.systolic_array_contraction(A_shape, B_shape, 'ijk,jkl->il')
resources = fpga_contract.estimate_resources(A_shape, B_shape)

print("\n" + "="*70)
print("Architecture FPGA pour Contraction")
print("="*70)
print(f"Type: {architecture['type']}")
print(f"PE Array: {architecture['pe_array_size']}")
print(f"Dataflow: {architecture['dataflow']}")
print(f"Latence estimée: {architecture['estimated_latency_ns']/1000:.2f} μs")
print(f"Throughput: {architecture['estimated_throughput_ops_per_sec']/1e9:.2f} GOps/sec")
print(f"\nRessources:")
print(f"  LUT:  {resources['lut']:,}")
print(f"  DSP:  {resources['dsp']:,}")
print(f"  BRAM: {resources['bram']:,}")
print(f"  Mémoire buffers: {resources['buffer_memory_mb']:.2f} MB")
```

---

## Optimisations Avancées

### Tiling et Fusion de Contractions

```python
class AdvancedTensorContractionOptimizations:
    """
    Optimisations avancées pour contractions
    """
    
    def contraction_fusion(self, contraction_sequence):
        """
        Fusionne plusieurs contractions consécutives
        
        Avantage: Réduit accès mémoire intermédiaire
        """
        # Exemple: (A * B) * C peut être fusionné en une opération
        # si structure permet
        
        fused = {
            'original_ops': len(contraction_sequence),
            'fused_ops': 1,  # Si fusion possible
            'memory_saved': 'Tenseurs intermédiaires évités'
        }
        
        return fused
    
    def tiled_contraction(self, A, B, tile_sizes):
        """
        Contraction avec tiling (découpage en tuiles)
        
        Avantage: Meilleure utilisation cache, parallélisation
        """
        # Implémentation simplifiée
        shape_A = A.shape
        shape_B = B.shape
        
        if len(shape_A) == 3 and len(shape_B) == 3:
            i_dim, j_dim, k_dim = shape_A
            _, _, l_dim = shape_B
            
            tile_i, tile_j, tile_k, tile_l = tile_sizes
            
            C = np.zeros((i_dim, l_dim), dtype=A.dtype)
            
            # Parcourir en tuiles
            for i_tile in range(0, i_dim, tile_i):
                i_end = min(i_tile + tile_i, i_dim)
                for l_tile in range(0, l_dim, tile_l):
                    l_end = min(l_tile + tile_l, l_dim)
                    for j_tile in range(0, j_dim, tile_j):
                        j_end = min(j_tile + tile_j, j_dim)
                        for k_tile in range(0, k_dim, tile_k):
                            k_end = min(k_tile + tile_k, k_dim)
                            
                            # Contraction sur tuile
                            A_tile = A[i_tile:i_end, j_tile:j_end, k_tile:k_end]
                            B_tile = B[j_tile:j_end, k_tile:k_end, l_tile:l_end]
                            
                            C_tile = np.einsum('ijk,jkl->il', A_tile, B_tile)
                            C[i_tile:i_end, l_tile:l_end] += C_tile
            
            return C
        else:
            return np.einsum('ijk,jkl->il', A, B)
    
    def sparse_contraction(self, A_sparse, B_sparse, sparsity_pattern):
        """
        Optimise contractions pour tenseurs creux (sparse)
        
        Skip les calculs sur zéros
        """
        # En pratique, utiliser structures de données sparse (CSR, COO)
        # et algorithmes adaptés
        
        return {
            'complexity_reduction': 'Proportionnel au sparsity',
            'implementation': 'Requiert structures sparse spécialisées'
        }

# Exemple tiling
def tiling_example():
    """Exemple de contraction avec tiling"""
    opt = AdvancedTensorContractionOptimizations()
    
    A = np.random.rand(64, 64, 64).astype(np.float32)
    B = np.random.rand(64, 64, 64).astype(np.float32)
    
    # Sans tiling
    C1 = np.einsum('ijk,jkl->il', A, B)
    
    # Avec tiling
    tile_sizes = (16, 16, 16, 16)
    C2 = opt.tiled_contraction(A, B, tile_sizes)
    
    print(f"Erreur max (tiling): {np.max(np.abs(C1 - C2)):.2e}")

# tiling_example()
```

---

## Exercices

### Exercice 17.1.1
Implémentez une contraction optimisée avec cache blocking pour CPU et comparez avec einsum standard.

### Exercice 17.1.2
Créez une architecture systolic array pour FPGA pour une contraction A[100,50,30] * B[50,30,25] et estimez les ressources.

### Exercice 17.1.3
Optimisez une séquence de contractions en utilisant la fusion pour réduire l'utilisation mémoire.

### Exercice 17.1.4
Comparez les performances GPU vs CPU pour des contractions batchées sur différents tailles de batch.

---

## Points Clés à Retenir

> 📌 **Les contractions CPU bénéficient de cache blocking et vectorisation SIMD**

> 📌 **Les GPU exploitent le parallélisme massif et les Tensor Cores pour contractions**

> 📌 **Les FPGA utilisent des architectures systolic pour efficacité énergétique**

> 📌 **Le choix du dataflow (weight/input/output stationary) affecte l'utilisation mémoire**

> 📌 **Le tiling et la fusion réduisent accès mémoire et améliorent cache locality**

> 📌 **Les optimisations doivent être adaptées au hardware cible et au pattern de contraction**

---

*Section précédente : [17.0 Introduction](./17_introduction.md) | Section suivante : [17.2 Ordonnancement Optimal des Contractions](./17_02_Ordonnancement.md)*

