# 18.1 Architecture du Système de Trigger du LHC

---

## Introduction

L'architecture du système de trigger du LHC représente l'un des systèmes informatiques les plus complexes au monde. Elle doit traiter 40 millions d'interactions par seconde et réduire ce flux à environ 1-5 kHz d'événements stockés, tout en préservant les événements physiques intéressants.

Cette section détaille l'architecture globale, les composants principaux, et les flux de données depuis la détection jusqu'au stockage.

---

## Architecture Globale

### Vue d'Ensemble Multi-Niveaux

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TriggerLevel:
    """
    Représentation d'un niveau de trigger
    """
    name: str
    latency_us: float
    reduction_factor: float
    input_rate_hz: float
    output_rate_hz: float
    hardware: str
    processing_time_us: float
    
    def compute_output_rate(self):
        """Calcule le taux de sortie"""
        return self.input_rate_hz / self.reduction_factor

class LHCTriggerArchitecture:
    """
    Architecture complète du système de trigger du LHC
    """
    
    def __init__(self):
        """
        Initialise l'architecture avec les paramètres typiques
        """
        # Collision rate: 40 MHz (25 ns entre collisions)
        collision_rate = 40e6  # Hz
        
        # Level-1 Trigger
        self.l1_trigger = TriggerLevel(
            name="Level-1 Trigger",
            latency_us=4.0,
            reduction_factor=400,  # 40 MHz → 100 kHz
            input_rate_hz=collision_rate,
            output_rate_hz=100e3,
            hardware="FPGA/ASIC",
            processing_time_us=3.5
        )
        
        # High-Level Trigger (HLT)
        self.hlt_trigger = TriggerLevel(
            name="High-Level Trigger",
            latency_us=300e3,  # 300 ms
            reduction_factor=100,  # 100 kHz → 1 kHz
            input_rate_hz=100e3,
            output_rate_hz=1e3,
            hardware="CPU/GPU Farm",
            processing_time_us=250e3
        )
        
        # Storage
        self.storage_rate = 1e3  # 1 kHz
        
    def display_architecture(self):
        """Affiche l'architecture complète"""
        print("\n" + "="*70)
        print("Architecture du Système de Trigger du LHC")
        print("="*70)
        
        print(f"\n{'Niveau':<20} {'Latence':<12} {'Taux Input':<15} {'Taux Output':<15} {'Hardware'}")
        print("-" * 70)
        
        print(f"{'Collisions':<20} {'-':<12} {'-':<15} {self.l1_trigger.input_rate_hz/1e6:.1f} MHz{'':>7}")
        print(f"{self.l1_trigger.name:<20} {self.l1_trigger.latency_us:<11.1f} μs "
              f"{self.l1_trigger.input_rate_hz/1e3:>8.1f} kHz{self.l1_trigger.output_rate_hz/1e3:>10.1f} kHz  "
              f"{self.l1_trigger.hardware}")
        print(f"{self.hlt_trigger.name:<20} {self.hlt_trigger.latency_us/1e3:<11.1f} ms "
              f"{self.hlt_trigger.input_rate_hz/1e3:>8.1f} kHz{self.hlt_trigger.output_rate_hz/1e3:>10.1f} kHz  "
              f"{self.hlt_trigger.hardware}")
        print(f"{'Stockage':<20} {'-':<12} {self.storage_rate:<15.1f} Hz{'':>15}")
        
        print(f"\nRéduction totale: {self.l1_trigger.input_rate_hz / self.storage_rate:.0f}×")
        print(f"Latence totale: {self.l1_trigger.latency_us + self.hlt_trigger.latency_us:.1f} μs")

architecture = LHCTriggerArchitecture()
architecture.display_architecture()
```

---

## Composants du Level-1 Trigger

### Systèmes de Détection et Traitement

```python
class L1TriggerComponents:
    """
    Composants du Level-1 Trigger
    """
    
    def __init__(self):
        self.components = {
            'calorimeter_trigger': {
                'description': 'Traitement des données calorimétriques',
                'inputs': ['ECAL clusters', 'HCAL towers'],
                'outputs': ['Electron candidates', 'Photon candidates', 'Jet candidates', 'MET'],
                'latency_budget_us': 1.5,
                'hardware': 'FPGA (Virtex UltraScale+)',
                'processing': 'Clustering, thresholding, energy sums'
            },
            'muon_trigger': {
                'description': 'Détection et reconstruction de muons',
                'inputs': ['RPC hits', 'CSC hits', 'DT hits'],
                'outputs': ['Muon candidates (pᵀ, η, φ)'],
                'latency_budget_us': 1.0,
                'hardware': 'ASIC custom',
                'processing': 'Track finding, pT assignment'
            },
            'global_trigger': {
                'description': 'Décision globale L1',
                'inputs': ['Calorimeter objects', 'Muon objects', 'L1 conditions'],
                'outputs': ['L1 accept/reject', 'L1 bits'],
                'latency_budget_us': 0.5,
                'hardware': 'FPGA',
                'processing': 'Trigger menu logic, rate control'
            },
            'data_flow_control': {
                'description': 'Contrôle du flux de données',
                'inputs': ['Raw detector data'],
                'outputs': ['Buffered events for HLT'],
                'latency_budget_us': 1.0,
                'hardware': 'FPGA + Memory',
                'processing': 'Event buffering, pipeline management'
            }
        }
    
    def display_components(self):
        """Affiche les composants"""
        print("\n" + "="*70)
        print("Composants du Level-1 Trigger")
        print("="*70)
        
        for comp_name, comp_info in self.components.items():
            print(f"\n{comp_name.replace('_', ' ').title()}:")
            print(f"  Description: {comp_info['description']}")
            print(f"  Inputs: {', '.join(comp_info['inputs'])}")
            print(f"  Outputs: {', '.join(comp_info['outputs'])}")
            print(f"  Latence budget: {comp_info['latency_budget_us']} μs")
            print(f"  Hardware: {comp_info['hardware']}")
            print(f"  Processing: {comp_info['processing']}")

l1_components = L1TriggerComponents()
l1_components.display_components()
```

---

## Pipeline de Traitement L1

### Séquence Temporelle

```python
class L1Pipeline:
    """
    Pipeline de traitement Level-1
    """
    
    def __init__(self):
        self.pipeline_stages = [
            {
                'name': 'Detector Readout',
                'time_ns': 0,
                'duration_ns': 250,  # 10 cycles à 40 MHz
                'description': 'Lecture des détecteurs'
            },
            {
                'name': 'Calorimeter Processing',
                'time_ns': 250,
                'duration_ns': 1500,  # 60 cycles
                'description': 'Clustering, jet finding'
            },
            {
                'name': 'Muon Processing',
                'time_ns': 500,
                'duration_ns': 1000,  # 40 cycles
                'description': 'Muon track finding'
            },
            {
                'name': 'Global Decision',
                'time_ns': 1750,
                'duration_ns': 500,  # 20 cycles
                'description': 'Décision L1'
            },
            {
                'name': 'Data Buffer',
                'time_ns': 2250,
                'duration_ns': 1750,  # 70 cycles
                'description': 'Buffer pour HLT'
            }
        ]
        
        self.total_latency_ns = 4000  # 4 μs
    
    def visualize_pipeline(self):
        """Visualise le pipeline temporel"""
        print("\n" + "="*70)
        print("Pipeline Level-1 Trigger (Timeline)")
        print("="*70)
        
        # Timeline en nanosecondes
        timeline_scale = 50  # 50 ns par caractère
        
        for stage in self.pipeline_stages:
            start_char = int(stage['time_ns'] / timeline_scale)
            width_char = int(stage['duration_ns'] / timeline_scale)
            
            timeline = ' ' * start_char + '█' * width_char
            print(f"{stage['name']:<30} {timeline} ({stage['duration_ns']} ns)")
        
        print(f"\n{'Total Latency':<30} {'█' * int(self.total_latency_ns / timeline_scale)} ({self.total_latency_ns} ns = 4 μs)")
    
    def check_timing_constraints(self):
        """Vérifie les contraintes temporelles"""
        total_pipeline_time = sum(stage['duration_ns'] for stage in self.pipeline_stages)
        max_overlap = max(stage['time_ns'] + stage['duration_ns'] for stage in self.pipeline_stages)
        
        print(f"\nAnalyse des Contraintes Temporelles:")
        print(f"  Temps total pipeline: {total_pipeline_time / 1000:.2f} μs")
        print(f"  Latence totale (avec overlap): {max_overlap / 1000:.2f} μs")
        print(f"  Contrainte: < 4.0 μs")
        print(f"  Marge: {4000 - max_overlap:.0f} ns")
        
        return max_overlap <= 4000

l1_pipeline = L1Pipeline()
l1_pipeline.visualize_pipeline()
l1_pipeline.check_timing_constraints()
```

---

## Architecture HLT

### Farm de Processeurs

```python
class HLTArchitecture:
    """
    Architecture du High-Level Trigger
    """
    
    def __init__(self, n_nodes=1000, cores_per_node=32):
        """
        Args:
            n_nodes: Nombre de nœuds de calcul
            cores_per_node: Cores par nœud
        """
        self.n_nodes = n_nodes
        self.cores_per_node = cores_per_node
        self.total_cores = n_nodes * cores_per_node
        
        # Architecture
        self.architecture = {
            'input_rate_hz': 100e3,  # 100 kHz depuis L1
            'target_output_rate_hz': 1e3,  # 1 kHz
            'event_size_mb': 1.5,  # ~1.5 MB par événement
            'processing_time_ms': 250,  # 250 ms par événement en moyenne
            'throughput_gbps': 1000  # 1 TB/s
        }
    
    def compute_requirements(self):
        """Calcule les besoins en ressources"""
        # Événements par seconde
        events_per_sec = self.architecture['input_rate_hz']
        
        # Bandwidth nécessaire
        bandwidth_gbps = events_per_sec * self.architecture['event_size_mb'] * 8 / 1000
        
        # CPU cores nécessaires (si un core traite un événement à la fois)
        cores_needed = events_per_sec * (self.architecture['processing_time_ms'] / 1000)
        
        # Avec parallélisme
        cores_with_parallelism = cores_needed / 0.8  # 80% d'utilisation
        
        return {
            'bandwidth_gbps': bandwidth_gbps,
            'cores_needed': cores_needed,
            'cores_with_parallelism': cores_with_parallelism,
            'current_capacity': self.total_cores,
            'utilization': cores_with_parallelism / self.total_cores
        }
    
    def hlt_processing_stages(self):
        """Stages de traitement HLT"""
        stages = [
            {
                'name': 'Event Builder',
                'time_ms': 10,
                'description': 'Reconstruction événement depuis fragments'
            },
            {
                'name': 'Track Reconstruction',
                'time_ms': 80,
                'description': 'Reconstruction des traces dans tracker'
            },
            {
                'name': 'Calorimeter Reconstruction',
                'time_ms': 50,
                'description': 'Reconstruction clusters calorimétriques'
            },
            {
                'name': 'Muon Reconstruction',
                'time_ms': 30,
                'description': 'Reconstruction des muons'
            },
            {
                'name': 'Jet Reconstruction',
                'time_ms': 40,
                'description': 'Reconstruction des jets'
            },
            {
                'name': 'ML Inference',
                'time_ms': 30,
                'description': 'Inference des modèles ML (b-tagging, etc.)'
            },
            {
                'name': 'Trigger Decision',
                'time_ms': 10,
                'description': 'Décision finale HLT'
            }
        ]
        
        total_time = sum(stage['time_ms'] for stage in stages)
        
        return stages, total_time
    
    def display_architecture(self):
        """Affiche l'architecture HLT"""
        print("\n" + "="*70)
        print("Architecture High-Level Trigger")
        print("="*70)
        
        print(f"\nRessources:")
        print(f"  Nœuds: {self.n_nodes}")
        print(f"  Cores par nœud: {self.cores_per_node}")
        print(f"  Total cores: {self.total_cores:,}")
        
        requirements = self.compute_requirements()
        print(f"\nBesoins:")
        print(f"  Bandwidth: {requirements['bandwidth_gbps']:.1f} Gbps")
        print(f"  Cores nécessaires: {requirements['cores_with_parallelism']:.0f}")
        print(f"  Utilisation: {requirements['utilization']*100:.1f}%")
        
        stages, total_time = self.hlt_processing_stages()
        print(f"\nStages de Traitement (total: {total_time} ms):")
        for stage in stages:
            print(f"  {stage['name']:<30} {stage['time_ms']:>5} ms  {stage['description']}")

hlt_arch = HLTArchitecture()
hlt_arch.display_architecture()
```

---

## Flux de Données et Buffers

### Gestion des Buffers

```python
class DataFlowManagement:
    """
    Gestion du flux de données dans le système de trigger
    """
    
    def __init__(self):
        self.buffers = {
            'l1_input_buffer': {
                'size_events': 160,  # 40 MHz × 4 μs = 160 événements en pipeline
                'size_mb': 240,  # 1.5 MB × 160
                'location': 'On-detector electronics',
                'technology': 'FPGA memory'
            },
            'l1_output_buffer': {
                'size_events': 400,  # Buffer pour HLT
                'size_mb': 600,
                'location': 'Central trigger system',
                'technology': 'DDR4 RAM'
            },
            'hlt_input_buffer': {
                'size_events': 30000,  # 100 kHz × 300 ms
                'size_mb': 45000,
                'location': 'HLT farm',
                'technology': 'SSD + RAM'
            }
        }
    
    def compute_buffer_requirements(self, rate_hz, latency_us, event_size_mb):
        """
        Calcule les besoins en buffer
        
        Args:
            rate_hz: Taux d'événements par seconde
            latency_us: Latence en microsecondes
            event_size_mb: Taille d'un événement en MB
        """
        # Nombre d'événements en pipeline
        events_in_pipeline = rate_hz * (latency_us / 1e6)
        
        # Taille totale du buffer
        buffer_size_mb = events_in_pipeline * event_size_mb
        
        # Avec marge de sécurité (2×)
        buffer_size_with_margin = buffer_size_mb * 2
        
        return {
            'events_in_pipeline': events_in_pipeline,
            'buffer_size_mb': buffer_size_mb,
            'buffer_size_with_margin_mb': buffer_size_with_margin
        }
    
    def analyze_data_flow(self):
        """Analyse le flux de données complet"""
        print("\n" + "="*70)
        print("Analyse du Flux de Données")
        print("="*70)
        
        # Collisions → L1
        l1_buffer = self.compute_buffer_requirements(40e6, 4, 1.5)
        print(f"\nBuffer L1 Input:")
        print(f"  Événements en pipeline: {l1_buffer['events_in_pipeline']:.0f}")
        print(f"  Taille buffer: {l1_buffer['buffer_size_with_margin_mb']:.1f} MB")
        
        # L1 → HLT
        hlt_buffer = self.compute_buffer_requirements(100e3, 300e3, 1.5)
        print(f"\nBuffer HLT Input:")
        print(f"  Événements en pipeline: {hlt_buffer['events_in_pipeline']:.0f}")
        print(f"  Taille buffer: {hlt_buffer['buffer_size_with_margin_mb']:.1f} MB")
        
        print(f"\nTaille totale des buffers:")
        total_buffer_mb = l1_buffer['buffer_size_with_margin_mb'] + hlt_buffer['buffer_size_with_margin_mb']
        print(f"  Total: {total_buffer_mb:.1f} MB ({total_buffer_mb/1024:.2f} GB)")

data_flow = DataFlowManagement()
data_flow.analyze_data_flow()
```

---

## Intégration Détecteur-Trigger

### Interface Détecteur-Trigger

```python
class DetectorTriggerInterface:
    """
    Interface entre les détecteurs et le système de trigger
    """
    
    def __init__(self):
        self.detector_channels = {
            'ECAL': {
                'channels': 61200,  # CMS ECAL crystals
                'readout_rate_mhz': 40,
                'bits_per_sample': 12,
                'data_rate_gbps': 61200 * 40e6 * 12 / 8 / 1e9  # ~36.7 Gbps
            },
            'HCAL': {
                'channels': 7000,
                'readout_rate_mhz': 40,
                'bits_per_sample': 10,
                'data_rate_gbps': 7000 * 40e6 * 10 / 8 / 1e9  # ~3.5 Gbps
            },
            'Tracker': {
                'channels': 100000000,  # 100M pixels/strips
                'readout_rate_mhz': 40,
                'bits_per_sample': 8,  # Compressed
                'data_rate_gbps': 100000000 * 40e6 * 8 / 8 / 1e9  # ~32 TB/s (nécessite compression)
            },
            'Muon': {
                'channels': 250000,
                'readout_rate_mhz': 40,
                'bits_per_sample': 8,
                'data_rate_gbps': 250000 * 40e6 * 8 / 8 / 1e9  # ~0.08 Gbps
            }
        }
    
    def compute_total_bandwidth(self):
        """Calcule la bande passante totale"""
        total_gbps = sum(det['data_rate_gbps'] for det in self.detector_channels.values())
        return total_gbps
    
    def trigger_data_reduction(self):
        """Analyse la réduction de données par le trigger"""
        total_raw_gbps = self.compute_total_bandwidth()
        
        # Après L1: seulement événements acceptés (100 kHz)
        l1_reduction = 400  # 40 MHz → 100 kHz
        data_after_l1_gbps = total_raw_gbps / l1_reduction
        
        # Après HLT: événements stockés (1 kHz)
        hlt_reduction = 100  # 100 kHz → 1 kHz
        data_after_hlt_gbps = data_after_l1_gbps / hlt_reduction
        
        return {
            'raw_data_gbps': total_raw_gbps,
            'after_l1_gbps': data_after_l1_gbps,
            'after_hlt_gbps': data_after_hlt_gbps,
            'total_reduction': total_raw_gbps / data_after_hlt_gbps
        }
    
    def display_interface(self):
        """Affiche l'interface détecteur-trigger"""
        print("\n" + "="*70)
        print("Interface Détecteur-Trigger")
        print("="*70)
        
        print(f"\n{'Détecteur':<15} {'Canaux':<15} {'Data Rate (Gbps)':<20}")
        print("-" * 50)
        for det_name, det_info in self.detector_channels.items():
            print(f"{det_name:<15} {det_info['channels']:<15,} {det_info['data_rate_gbps']:<20.2f}")
        
        reduction = self.trigger_data_reduction()
        print(f"\nRéduction de Données:")
        print(f"  Raw data: {reduction['raw_data_gbps']:.1f} Gbps")
        print(f"  Après L1: {reduction['after_l1_gbps']:.1f} Gbps")
        print(f"  Après HLT: {reduction['after_hlt_gbps']:.1f} Gbps")
        print(f"  Réduction totale: {reduction['total_reduction']:.0f}×")

detector_interface = DetectorTriggerInterface()
detector_interface.display_interface()
```

---

## Exercices

### Exercice 18.1.1
Calculez la taille minimale des buffers nécessaires si le taux de collisions augmente à 60 MHz (Run 4 du LHC).

### Exercice 18.1.2
Dimensionnez une architecture HLT pour traiter 200 kHz d'événements L1 avec une latence maximale de 200 ms.

### Exercice 18.1.3
Analysez l'impact sur la bande passante si tous les canaux du tracker doivent être lus au lieu d'une lecture sélective.

### Exercice 18.1.4
Concevez un système de buffers avec redondance pour tolérer les pannes d'un nœud HLT.

---

## Points Clés à Retenir

> 📌 **L'architecture trigger est multi-niveaux: L1 (hardware) → HLT (software)**

> 📌 **Le L1 réduit 40 MHz → 100 kHz en 4 μs avec FPGA/ASIC**

> 📌 **Le HLT réduit 100 kHz → 1 kHz en ~300 ms avec farm CPU/GPU**

> 📌 **Les buffers sont critiques pour gérer les latences et pics de charge**

> 📌 **L'interface détecteur-trigger doit gérer des TB/s de données brutes**

> 📌 **L'architecture doit être scalable et tolérante aux pannes**

---

*Section précédente : [18.0 Introduction](./18_introduction.md) | Section suivante : [18.2 Level-1 Trigger](./18_02_L1_Trigger.md)*

