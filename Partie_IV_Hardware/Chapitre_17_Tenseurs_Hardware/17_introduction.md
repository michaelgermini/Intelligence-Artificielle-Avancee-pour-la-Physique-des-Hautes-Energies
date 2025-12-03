# Chapitre 17 : Déploiement de Réseaux de Tenseurs sur Hardware

---

## Introduction

Le déploiement de **réseaux de tenseurs** (Tensor Networks) sur hardware présente des défis et opportunités uniques. Contrairement aux réseaux de neurones traditionnels, les réseaux de tenseurs utilisent des **contractions tensorielles** comme opérations fondamentales, ce qui nécessite des optimisations spécifiques pour les architectures FPGA, GPU et ASIC.

Ce chapitre couvre les techniques d'implémentation efficace des contractions tensorielles, l'ordonnancement optimal des opérations, le mapping sur architectures parallèles, et la quantification spécifique aux tenseurs.

---

## Plan du Chapitre

1. [Implémentation Efficace des Contractions Tensorielles](./17_01_Contractions.md)
2. [Ordonnancement Optimal des Contractions](./17_02_Ordonnancement.md)
3. [Mapping sur Architectures Parallèles](./17_03_Mapping.md)
4. [Quantification Hardware-Aware pour Tenseurs](./17_04_Quantification.md)

---

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│        Déploiement de Réseaux de Tenseurs sur Hardware         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Tensor     │───▶│  Contraction│───▶│  Hardware   │        │
│  │  Network    │    │  Optimizer  │    │  Mapping    │        │
│  └─────────────┘    └─────────────┘    └──────┬──────┘        │
│                                                 │                │
│                                                 ▼                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  Schedule   │◀───│  Resource   │◀───│  Parallel   │        │
│  │  Optimizer  │    │  Allocator  │    │  Execution  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Défis Spécifiques aux Réseaux de Tenseurs

### Comparaison avec Réseaux de Neurones

```python
import torch
import numpy as np

class TensorNetworkChallenges:
    """
    Défis spécifiques au déploiement de réseaux de tenseurs
    """
    
    def __init__(self):
        self.challenges = {
            'contraction_complexity': {
                'description': 'Complexité exponentielle des contractions',
                'example': 'Contraction de N tenseurs: O(2^N) ordonnancements possibles',
                'impact': 'Nécessite optimisation d\'ordonnancement'
            },
            'memory_explosion': {
                'description': 'Tenseurs intermédiaires peuvent être très grands',
                'example': 'Contraction A[i,j,k] * B[j,k,l] → C[i,l] nécessite stockage temporaire',
                'impact': 'Gestion mémoire critique sur FPGA (BRAM limité)'
            },
            'irregular_access': {
                'description': 'Patterns d\'accès mémoire irréguliers',
                'example': 'Contractions varient selon structure du réseau',
                'impact': 'Difficile à optimiser avec cache classique'
            },
            'precision_requirements': {
                'description': 'Sensibilité aux erreurs numériques',
                'example': 'Accumulation d\'erreurs dans contractions longues',
                'impact': 'Quantification plus délicate que réseaux classiques'
            }
        }
    
    def display_challenges(self):
        """Affiche les défis"""
        print("\n" + "="*70)
        print("Défis du Déploiement de Réseaux de Tenseurs")
        print("="*70)
        
        for challenge, info in self.challenges.items():
            print(f"\n{challenge.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Example: {info['example']}")
            print(f"  Impact: {info['impact']}")

challenges = TensorNetworkChallenges()
challenges.display_challenges()
```

---

## Opérations de Base : Contractions Tensorielles

### Exemple Simple

```python
import numpy as np

def tensor_contraction_example():
    """
    Exemple de contraction tensorielle
    """
    # Tenseur 3D: A[i, j, k]
    A = np.random.rand(10, 20, 15)
    
    # Tenseur 2D: B[j, k]
    B = np.random.rand(20, 15)
    
    # Contraction sur indices j et k
    # C[i] = sum_j sum_k A[i,j,k] * B[j,k]
    C = np.einsum('ijk,jk->i', A, B)
    
    print(f"Shape A: {A.shape}")
    print(f"Shape B: {B.shape}")
    print(f"Shape C (contraction): {C.shape}")
    
    # Complexité computationnelle
    complexity = A.shape[0] * A.shape[1] * A.shape[2] * B.shape[1]
    print(f"\nComplexité computationnelle: {complexity:,} opérations")
    
    # Complexité mémoire (tenseur intermédiaire)
    memory_temp = A.shape[0] * A.shape[1] * A.shape[2]  # Si on stocke A
    print(f"Mémoire temporaire nécessaire: ~{memory_temp * 4 / 1024:.2f} KB (float32)")

tensor_contraction_example()
```

---

## Structures de Réseaux de Tenseurs

### Types Principaux

```python
class TensorNetworkTypes:
    """
    Types principaux de réseaux de tenseurs pour déploiement hardware
    """
    
    def __init__(self):
        self.network_types = {
            'MPS': {
                'full_name': 'Matrix Product State',
                'structure': 'Chaîne linéaire de tenseurs',
                'deployment': 'Pipeline naturel, ordonnancement simple',
                'complexity': 'O(d^3) par contraction (d = bond dimension)'
            },
            'PEPS': {
                'full_name': 'Projected Entangled Pair State',
                'structure': 'Grille 2D de tenseurs',
                'deployment': 'Complexe, nécessite optimisations',
                'complexity': 'O(d^10) pour contraction exacte'
            },
            'TT': {
                'full_name': 'Tensor Train',
                'structure': 'Décomposition en chaîne',
                'deployment': 'Similaire à MPS, pipeline efficace',
                'complexity': 'O(d^3) par contraction'
            },
            'Tucker': {
                'full_name': 'Tucker Decomposition',
                'structure': 'Tenseur core + facteurs',
                'deployment': 'Contractions multiples, réutilisable',
                'complexity': 'O(d^N + d^2) (N = ordre)'
            }
        }
    
    def display_types(self):
        """Affiche les types"""
        print("\n" + "="*70)
        print("Types de Réseaux de Tenseurs")
        print("="*70)
        
        for net_type, info in self.network_types.items():
            print(f"\n{net_type} ({info['full_name']}):")
            print(f"  Structure: {info['structure']}")
            print(f"  Déploiement: {info['deployment']}")
            print(f"  Complexité: {info['complexity']}")

network_types = TensorNetworkTypes()
network_types.display_types()
```

---

## Métriques de Performance Hardware

### Latence, Throughput, Mémoire

```python
class HardwareMetricsTensor:
    """
    Métriques hardware spécifiques aux réseaux de tenseurs
    """
    
    def estimate_contraction_latency(self, shape_A, shape_B, contracted_dims, 
                                    hardware_type='fpga', parallelism=64):
        """
        Estime la latence d'une contraction
        
        Args:
            shape_A: Shape du tenseur A
            shape_B: Shape du tenseur B
            contracted_dims: Dimensions contractées (ex: [1, 2])
            hardware_type: 'fpga', 'gpu', 'cpu'
            parallelism: Nombre d'opérations en parallèle
        """
        # Calculer dimensions de sortie
        free_dims_A = [i for i in range(len(shape_A)) if i not in contracted_dims]
        free_dims_B = [i for i in range(len(shape_B)) if i not in contracted_dims]
        
        # Nombre d'opérations MAC
        n_free_A = np.prod([shape_A[i] for i in free_dims_A])
        n_free_B = np.prod([shape_B[i] for i in free_dims_B])
        n_contracted = np.prod([shape_A[i] for i in contracted_dims])
        
        total_ops = n_free_A * n_free_B * n_contracted
        
        # Latence selon hardware
        if hardware_type == 'fpga':
            cycles = np.ceil(total_ops / parallelism)
            clock_period_ns = 5  # 200 MHz
            latency_ns = cycles * clock_period_ns
        elif hardware_type == 'gpu':
            # GPU: beaucoup plus rapide
            latency_ns = total_ops / (parallelism * 1000)  # approximation
        else:  # CPU
            latency_ns = total_ops / (parallelism * 100)
        
        return {
            'total_ops': total_ops,
            'latency_ns': latency_ns,
            'throughput_ops_per_sec': total_ops / (latency_ns * 1e-9)
        }
    
    def estimate_memory_requirements(self, shapes, contraction_order):
        """
        Estime les besoins mémoire pour une séquence de contractions
        """
        memory_timeline = []
        current_tensors = list(shapes)
        
        for step, (i, j) in enumerate(contraction_order):
            # Calculer shape du résultat
            shape_result = self._compute_result_shape(current_tensors[i], 
                                                     current_tensors[j])
            
            # Mémoire nécessaire à ce step
            memory_step = {
                'step': step,
                'input_memory': (np.prod(current_tensors[i]) + 
                               np.prod(current_tensors[j])) * 4,  # float32
                'output_memory': np.prod(shape_result) * 4,
                'peak_memory': (np.prod(current_tensors[i]) + 
                              np.prod(current_tensors[j]) + 
                              np.prod(shape_result)) * 4
            }
            memory_timeline.append(memory_step)
            
            # Mettre à jour: remplacer i et j par résultat
            current_tensors = ([current_tensors[k] for k in range(len(current_tensors)) 
                              if k not in [i, j]] + [shape_result])
        
        return memory_timeline
    
    def _compute_result_shape(self, shape_A, shape_B):
        """Calcule la shape du résultat d'une contraction"""
        # Simplifié: suppose contraction sur dernières dims de A et premières de B
        if len(shape_A) == 0 or len(shape_B) == 0:
            return tuple()
        # Contraction simple: A[:-1] + B[1:]
        result = shape_A[:-1] + shape_B[1:]
        return result if result else (1,)

# Exemple
metrics = HardwareMetricsTensor()

# Exemple: contraction de deux tenseurs
shape_A = (100, 50, 30)
shape_B = (30, 20)
contracted = [2]  # Contracter dernière dim de A avec première de B

result = metrics.estimate_contraction_latency(shape_A, shape_B, contracted)
print(f"\nEstimation Contraction:")
print(f"  Opérations totales: {result['total_ops']:,}")
print(f"  Latence FPGA: {result['latency_ns']/1000:.2f} μs")
print(f"  Throughput: {result['throughput_ops_per_sec']/1e9:.2f} GOps/sec")
```

---

## Applications en Physique des Hautes Énergies

### Utilisation dans Triggers

```python
class HEPTensorNetworkApplications:
    """
    Applications des réseaux de tenseurs en HEP
    """
    
    def __init__(self):
        self.applications = {
            'jet_tagging': {
                'description': 'Classification de jets avec MPS/TT',
                'requirements': 'Latence < 100 ns, précision > 95%',
                'hardware': 'FPGA pour trigger L1'
            },
            'event_classification': {
                'description': 'Classification d\'événements avec PEPS',
                'requirements': 'Throughput 40 MHz, mémoire limitée',
                'hardware': 'FPGA avec optimisation mémoire'
            },
            'anomaly_detection': {
                'description': 'Détection d\'anomalies avec Tensor Train',
                'requirements': 'Latence faible, faible puissance',
                'hardware': 'FPGA edge device'
            }
        }
    
    def display_applications(self):
        """Affiche les applications"""
        print("\n" + "="*70)
        print("Applications Réseaux de Tenseurs en HEP")
        print("="*70)
        
        for app, info in self.applications.items():
            print(f"\n{app.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Requirements: {info['requirements']}")
            print(f"  Hardware: {info['hardware']}")

applications = HEPTensorNetworkApplications()
applications.display_applications()
```

---

## Exercices

### Exercice 17.0.1
Calculez la complexité computationnelle et mémoire d'une contraction entre un tenseur A[100, 50, 30] et B[30, 20] sur la dimension commune.

### Exercice 17.0.2
Comparez les besoins mémoire pour deux ordonnancements différents de contractions dans un réseau MPS à 10 tenseurs.

---

## Points Clés à Retenir

> 📌 **Les réseaux de tenseurs utilisent des contractions comme opérations fondamentales**

> 📌 **L'ordonnancement des contractions affecte drastiquement la complexité et la mémoire**

> 📌 **Le mapping sur hardware parallèle nécessite des optimisations spécifiques**

> 📌 **La quantification des tenseurs est plus délicate que pour les réseaux classiques**

> 📌 **Les applications HEP nécessitent latence ultra-faible et throughput élevé**

---

*Section suivante : [17.1 Implémentation Efficace des Contractions Tensorielles](./17_01_Contractions.md)*

