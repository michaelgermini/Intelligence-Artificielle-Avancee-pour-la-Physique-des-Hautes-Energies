# 18.2 Level-1 Trigger et Contraintes Temps Réel

---

## Introduction

Le **Level-1 (L1) Trigger** est le premier niveau de sélection des événements au LHC. Il fonctionne entièrement en hardware (FPGA, ASIC) et doit prendre une décision en **moins de 4 microsecondes** sur chaque croisement de faisceaux (40 MHz). Cette contrainte temporelle extrême impose des limitations strictes sur les algorithmes utilisables.

Cette section détaille les algorithmes L1, les contraintes hardware, et l'intégration de l'IA dans ce contexte ultra-contraint.

---

## Contraintes du L1 Trigger

### Spécifications Temporelles

```python
import numpy as np
from typing import Dict, List, Tuple

class L1Constraints:
    """
    Contraintes du Level-1 Trigger
    """
    
    def __init__(self):
        self.constraints = {
            'timing': {
                'total_latency_us': 4.0,
                'collision_rate_hz': 40e6,  # 40 MHz
                'time_between_collisions_ns': 25.0,  # 25 ns
                'pipeline_depth': 160,  # 40 MHz × 4 μs
                'initiation_interval': 1  # Nouvelle décision chaque cycle
            },
            'hardware': {
                'technology': 'FPGA (Xilinx UltraScale+) ou ASIC',
                'clock_frequency_mhz': 200,  # 200 MHz typique
                'clock_period_ns': 5.0,
                'cycles_available': 800,  # 4 μs / 5 ns
                'lut_budget': 500000,
                'dsp_budget': 2000,
                'bram_budget': 1000
            },
            'data': {
                'input_granularity': 'Calorimeter towers (0.1×0.1 en η×φ)',
                'muon_stations': 'RPC, CSC, DT hits',
                'output': 'L1 objects (ET, η, φ, qual)',
                'bandwidth_input_gbps': 100,
                'bandwidth_output_gbps': 10
            },
            'performance': {
                'target_efficiency_signal': 0.95,  # 95% efficacité signal
                'target_rate_output_khz': 100,  # 100 kHz
                'false_positive_rate': '< 0.1%'
            }
        }
    
    def display_constraints(self):
        """Affiche les contraintes"""
        print("\n" + "="*70)
        print("Contraintes Level-1 Trigger")
        print("="*70)
        
        for category, items in self.constraints.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for key, value in items.items():
                if isinstance(value, float):
                    if value >= 1e6:
                        print(f"  {key}: {value/1e6:.1f} MHz")
                    elif value >= 1e3:
                        print(f"  {key}: {value/1e3:.1f} kHz")
                    elif value >= 1:
                        print(f"  {key}: {value:.1f}")
                    else:
                        print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

l1_constraints = L1Constraints()
l1_constraints.display_constraints()
```

---

## Algorithmes L1 Classiques

### Calorimeter Trigger

```python
class L1CalorimeterAlgorithm:
    """
    Algorithmes de trigger pour le calorimètre
    """
    
    def __init__(self):
        self.tower_size = (0.1, 0.1)  # Δη, Δφ
        
    def clustering_2x2(self, calo_towers: np.ndarray, threshold_et: float = 20.0):
        """
        Clustering 2×2 dans le calorimètre
        
        Algorithme simple et rapide pour FPGA
        """
        n_eta, n_phi = calo_towers.shape
        clusters = []
        
        # Parcourir fenêtres 2×2
        for i in range(n_eta - 1):
            for j in range(n_phi - 1):
                # Somme ET dans fenêtre 2×2
                et_sum = (calo_towers[i, j] + calo_towers[i+1, j] +
                         calo_towers[i, j+1] + calo_towers[i+1, j+1])
                
                if et_sum > threshold_et:
                    # Position (barycentre approximatif)
                    eta_center = i + 0.5
                    phi_center = j + 0.5
                    
                    clusters.append({
                        'et': et_sum,
                        'eta_idx': eta_center,
                        'phi_idx': phi_center,
                        'type': 'cluster'
                    })
        
        return clusters
    
    def electron_identification(self, calo_towers: np.ndarray, 
                               ecal_towers: np.ndarray,
                               hcal_towers: np.ndarray,
                               ecal_threshold: float = 15.0,
                               isolation_threshold: float = 5.0):
        """
        Identification d'électrons
        
        Critères:
        - ET élevé dans ECAL
        - Isolation (pas d'énergie dans HCAL autour)
        """
        electrons = []
        n_eta, n_phi = calo_towers.shape
        
        for i in range(1, n_eta - 1):
            for j in range(1, n_phi - 1):
                # ET dans ECAL
                ecal_et = ecal_towers[i, j]
                
                if ecal_et > ecal_threshold:
                    # Isolation: vérifier HCAL autour
                    hcal_sum = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            hcal_sum += hcal_towers[i + di, j + dj]
                    
                    # Électron si isolé
                    if hcal_sum < isolation_threshold:
                        electrons.append({
                            'et': ecal_et,
                            'eta_idx': i,
                            'phi_idx': j,
                            'hcal_isolation': hcal_sum
                        })
        
        return electrons
    
    def jet_finding_simple(self, calo_towers: np.ndarray, 
                          cone_size: int = 4,
                          threshold_et: float = 30.0):
        """
        Trouve des jets avec algorithme simple (cone)
        
        Cones de taille fixe pour simplifier hardware
        """
        n_eta, n_phi = calo_towers.shape
        jets = []
        used_towers = np.zeros_like(calo_towers, dtype=bool)
        
        # Chercher maxima locaux
        for i in range(cone_size, n_eta - cone_size):
            for j in range(cone_size, n_phi - cone_size):
                if used_towers[i, j]:
                    continue
                
                # Somme ET dans cone
                et_sum = 0
                for di in range(-cone_size, cone_size + 1):
                    for dj in range(-cone_size, cone_size + 1):
                        r2 = di**2 + dj**2
                        if r2 <= cone_size**2:
                            et_sum += calo_towers[i + di, j + dj]
                
                if et_sum > threshold_et:
                    jets.append({
                        'et': et_sum,
                        'eta_idx': i,
                        'phi_idx': j,
                        'cone_size': cone_size
                    })
                    
                    # Marquer tours utilisées
                    for di in range(-cone_size, cone_size + 1):
                        for dj in range(-cone_size, cone_size + 1):
                            r2 = di**2 + dj**2
                            if r2 <= cone_size**2:
                                used_towers[i + di, j + dj] = True
        
        return jets
    
    def missing_et_calculation(self, calo_towers: np.ndarray):
        """
        Calcule l'énergie manquante transverse (MET)
        """
        n_eta, n_phi = calo_towers.shape
        
        # Somme vectorielle de l'ET
        met_x = 0
        met_y = 0
        
        for i in range(n_eta):
            for j in range(n_phi):
                et = calo_towers[i, j]
                # φ correspond à angle autour du faisceau
                phi = 2 * np.pi * j / n_phi
                met_x += et * np.cos(phi)
                met_y += et * np.sin(phi)
        
        met_magnitude = np.sqrt(met_x**2 + met_y**2)
        met_phi = np.arctan2(met_y, met_x)
        
        return {
            'met': met_magnitude,
            'met_phi': met_phi,
            'met_x': met_x,
            'met_y': met_y
        }

# Exemple d'utilisation
calo_algo = L1CalorimeterAlgorithm()

# Simuler des tours calorimétriques
calo_towers = np.random.rand(72, 36) * 10  # 72×36 tours
ecal_towers = calo_towers * 0.7
hcal_towers = calo_towers * 0.3

clusters = calo_algo.clustering_2x2(calo_towers, threshold_et=20.0)
electrons = calo_algo.electron_identification(calo_towers, ecal_towers, hcal_towers)
jets = calo_algo.jet_finding_simple(calo_towers, cone_size=4, threshold_et=30.0)
met = calo_algo.missing_et_calculation(calo_towers)

print(f"\nL1 Calorimeter Results:")
print(f"  Clusters trouvés: {len(clusters)}")
print(f"  Electrons: {len(electrons)}")
print(f"  Jets: {len(jets)}")
print(f"  MET: {met['met']:.1f} GeV")
```

---

## Muon Trigger L1

### Reconstruction de Muons

```python
class L1MuonAlgorithm:
    """
    Algorithmes de trigger pour muons
    """
    
    def __init__(self):
        self.muon_stations = ['RPC', 'CSC', 'DT']  # Résistive, Cathode Strip, Drift Tube
    
    def muon_segment_finding(self, rpc_hits: np.ndarray, 
                            csc_hits: np.ndarray,
                            dt_hits: np.ndarray):
        """
        Trouve des segments de muons dans les stations
        
        Simple pattern matching pour hardware
        """
        muon_segments = []
        
        # Recherche dans RPC (résistif - rapide)
        n_stations, n_strips = rpc_hits.shape
        
        for station in range(n_stations):
            for strip in range(1, n_strips - 1):
                # Pattern simple: 3 hits consécutifs
                if (rpc_hits[station, strip-1] and 
                    rpc_hits[station, strip] and 
                    rpc_hits[station, strip+1]):
                    
                    muon_segments.append({
                        'station': station,
                        'strip': strip,
                        'type': 'RPC',
                        'quality': 1
                    })
        
        # Recherche dans CSC
        n_csc_stations, n_wires = csc_hits.shape
        for station in range(n_csc_stations):
            for wire in range(n_wires):
                if csc_hits[station, wire]:
                    # Association avec segments RPC si possible
                    muon_segments.append({
                        'station': station,
                        'wire': wire,
                        'type': 'CSC',
                        'quality': 2
                    })
        
        return muon_segments
    
    def muon_track_finding(self, segments: List[Dict], 
                          min_stations: int = 3):
        """
        Trouve des traces de muons en associant segments
        
        Algorithme simplifié pour FPGA
        """
        muon_tracks = []
        
        # Grouper segments par position
        # Simplification: recherche dans fenêtres
        used_segments = set()
        
        for i, seg1 in enumerate(segments):
            if i in used_segments:
                continue
            
            track_segments = [seg1]
            
            # Chercher segments compatibles dans autres stations
            for j, seg2 in enumerate(segments):
                if j <= i or j in used_segments:
                    continue
                
                # Vérifier compatibilité (simplifié)
                if self._segments_compatible(seg1, seg2):
                    track_segments.append(seg2)
            
            # Muon si au moins min_stations
            if len(track_segments) >= min_stations:
                # Calculer pT approximatif (basé sur courbure)
                pt_estimate = self._estimate_pt(track_segments)
                
                muon_tracks.append({
                    'segments': track_segments,
                    'pt_estimate': pt_estimate,
                    'quality': len(track_segments)
                })
                
                # Marquer segments utilisés
                used_segments.update([segments.index(s) for s in track_segments])
        
        return muon_tracks
    
    def _segments_compatible(self, seg1: Dict, seg2: Dict) -> bool:
        """Vérifie si deux segments sont compatibles"""
        # Simplification: compatibles si stations différentes
        return seg1['station'] != seg2.get('station', -1)
    
    def _estimate_pt(self, segments: List[Dict]) -> float:
        """Estime pT du muon (simplifié)"""
        # Plus de segments = plus de pT (approximation)
        return 20.0 * len(segments)  # GeV

# Exemple
muon_algo = L1MuonAlgorithm()

# Simuler hits
rpc_hits = np.random.rand(4, 100) > 0.95  # 4 stations, sparse hits
csc_hits = np.random.rand(4, 100) > 0.97
dt_hits = np.random.rand(4, 100) > 0.96

segments = muon_algo.muon_segment_finding(rpc_hits, csc_hits, dt_hits)
tracks = muon_algo.muon_track_finding(segments, min_stations=3)

print(f"\nL1 Muon Results:")
print(f"  Segments trouvés: {len(segments)}")
print(f"  Traces de muons: {len(tracks)}")
if tracks:
    print(f"  pT moyen: {np.mean([t['pt_estimate'] for t in tracks]):.1f} GeV")
```

---

## Machine Learning dans le L1

### Réseaux de Neurones pour FPGA

```python
import torch
import torch.nn as nn

class L1MLTrigger(nn.Module):
    """
    Réseau de neurones optimisé pour L1 Trigger sur FPGA
    """
    
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=1, 
                 bitwidth=8):
        """
        Args:
            input_dim: Dimension des features input
            hidden_dim: Dimension couche cachée (limité par ressources)
            output_dim: Dimension sortie (1 pour classification binaire)
            bitwidth: Précision (8 bits typique pour L1)
        """
        super().__init__()
        
        # Architecture minimale pour respecter contraintes
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Activation compatible FPGA (ReLU)
        self.activation = nn.ReLU()
        
        self.bitwidth = bitwidth
    
    def forward(self, x):
        """Forward pass"""
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Probabilité
        return x
    
    def estimate_hardware_resources(self):
        """Estime les ressources FPGA nécessaires"""
        # Nombre de MACs
        macs_fc1 = self.fc1.in_features * self.fc1.out_features
        macs_fc2 = self.fc2.in_features * self.fc2.out_features
        total_macs = macs_fc1 + macs_fc2
        
        # DSP slices (1 par MAC pour 8-bit)
        dsp_slices = total_macs
        
        # LUTs (approximation)
        luts = total_macs * 50  # Overhead pour logique
        
        # BRAM (pour poids et activations)
        n_params = sum(p.numel() for p in self.parameters())
        bram_weights = np.ceil(n_params * (self.bitwidth / 8) / (36 * 1024))
        bram_activations = np.ceil(self.fc1.out_features * (self.bitwidth / 8) / (36 * 1024))
        total_bram = bram_weights + bram_activations
        
        return {
            'macs': total_macs,
            'dsp_slices': dsp_slices,
            'luts': luts,
            'bram': total_bram,
            'parameters': n_params
        }
    
    def estimate_latency_ns(self, clock_freq_mhz=200):
        """
        Estime la latence en nanosecondes
        
        Hypothèse: fully pipelined
        """
        clock_period_ns = 1000.0 / clock_freq_mhz
        
        # Profondeur du pipeline (cycles)
        # Couche 1: input buffer + MAC + activation
        cycles_layer1 = 10  # Approximation
        # Couche 2: MAC + sigmoid
        cycles_layer2 = 5
        
        total_cycles = cycles_layer1 + cycles_layer2
        latency_ns = total_cycles * clock_period_ns
        
        return latency_ns

class L1MLOptimizer:
    """
    Outils d'optimisation pour ML dans L1
    """
    
    @staticmethod
    def quantize_model_for_l1(model: nn.Module, bitwidth: int = 8):
        """
        Quantifie le modèle pour déploiement L1
        """
        # En pratique, utiliser torch.quantization
        # Ici: simulation simplifiée
        
        quantized_state = {}
        scales = {}
        
        for name, param in model.named_parameters():
            # Calculer scale
            param_max = torch.max(torch.abs(param))
            scale = param_max / (2 ** (bitwidth - 1) - 1)
            
            # Quantifier
            quantized = torch.round(param / scale)
            quantized = torch.clamp(quantized, 
                                   -(2 ** (bitwidth - 1) - 1),
                                   2 ** (bitwidth - 1) - 1)
            
            quantized_state[name] = quantized.int()
            scales[name] = scale
        
        return quantized_state, scales
    
    @staticmethod
    def prune_for_l1(model: nn.Module, sparsity: float = 0.5):
        """
        Prune le modèle pour réduire complexité
        """
        # Pruning magnitude-based simplifié
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Calculer seuil
                threshold = torch.quantile(torch.abs(param), sparsity)
                
                # Mask
                mask = torch.abs(param) > threshold
                param.data *= mask.float()
        
        return model

# Créer et analyser modèle L1
l1_ml_model = L1MLTrigger(input_dim=64, hidden_dim=32, output_dim=1, bitwidth=8)

resources = l1_ml_model.estimate_hardware_resources()
latency = l1_ml_model.estimate_latency_ns()

print("\n" + "="*70)
print("Modèle ML pour L1 Trigger")
print("="*70)
print(f"Paramètres: {resources['parameters']}")
print(f"MACs: {resources['macs']:,}")
print(f"Ressources FPGA estimées:")
print(f"  DSP slices: {resources['dsp_slices']:,}")
print(f"  LUTs: {resources['luts']:,}")
print(f"  BRAM: {resources['bram']:.0f}")
print(f"Latence estimée: {latency:.1f} ns")

# Vérifier contraintes
l1_constraints = L1Constraints()
print(f"\nVérification contraintes:")
print(f"  Latence < 4 μs: {latency < 4000} ({latency/1000:.2f} μs)")
print(f"  DSP < 2000: {resources['dsp_slices'] < 2000} ({resources['dsp_slices']})")
print(f"  LUT < 500k: {resources['luts'] < 500000} ({resources['luts']:,})")
```

---

## Global Trigger Logic

### Décision Finale L1

```python
class L1GlobalTrigger:
    """
    Logique du trigger global L1
    """
    
    def __init__(self):
        self.trigger_menu = {
            'single_electron': {
                'threshold_et': 25.0,  # GeV
                'prescale': 1
            },
            'single_muon': {
                'threshold_pt': 20.0,  # GeV
                'prescale': 1
            },
            'di_jet': {
                'threshold_et': 50.0,  # GeV par jet
                'prescale': 1
            },
            'met': {
                'threshold': 100.0,  # GeV
                'prescale': 1
            },
            'electron_muon': {
                'electron_et': 15.0,
                'muon_pt': 15.0,
                'prescale': 1
            }
        }
    
    def evaluate_trigger(self, l1_objects: Dict) -> Dict:
        """
        Évalue tous les chemins de trigger
        
        Returns:
            dict avec décisions par chemin
        """
        decisions = {}
        
        # Single electron
        electrons = l1_objects.get('electrons', [])
        single_electron_passed = any(e['et'] > self.trigger_menu['single_electron']['threshold_et'] 
                                    for e in electrons)
        decisions['single_electron'] = single_electron_passed
        
        # Single muon
        muons = l1_objects.get('muons', [])
        single_muon_passed = any(m['pt_estimate'] > self.trigger_menu['single_muon']['threshold_pt'] 
                                for m in muons)
        decisions['single_muon'] = single_muon_passed
        
        # Di-jet
        jets = l1_objects.get('jets', [])
        jets_above_threshold = [j for j in jets if j['et'] > self.trigger_menu['di_jet']['threshold_et']]
        di_jet_passed = len(jets_above_threshold) >= 2
        decisions['di_jet'] = di_jet_passed
        
        # MET
        met = l1_objects.get('met', {})
        met_passed = met.get('met', 0) > self.trigger_menu['met']['threshold']
        decisions['met'] = met_passed
        
        # Electron + muon
        electron_muon_passed = (single_electron_passed and single_muon_passed and
                               any(e['et'] > self.trigger_menu['electron_muon']['electron_et'] for e in electrons) and
                               any(m['pt_estimate'] > self.trigger_menu['electron_muon']['muon_pt'] for m in muons))
        decisions['electron_muon'] = electron_muon_passed
        
        # Décision finale L1
        l1_accept = any(decisions.values())
        
        return {
            'l1_accept': l1_accept,
            'decisions': decisions,
            'l1_bits': self._encode_l1_bits(decisions)
        }
    
    def _encode_l1_bits(self, decisions: Dict) -> int:
        """Encode les décisions en bits L1"""
        bits = 0
        bit_position = 0
        
        for path_name in sorted(decisions.keys()):
            if decisions[path_name]:
                bits |= (1 << bit_position)
            bit_position += 1
        
        return bits
    
    def apply_prescale(self, decisions: Dict, random_seed: int = None) -> Dict:
        """Applique les prescales"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        prescaled_decisions = decisions.copy()
        
        for path_name, decision in decisions.items():
            if decision and path_name in self.trigger_menu:
                prescale = self.trigger_menu[path_name].get('prescale', 1)
                if prescale > 1:
                    # Garde seulement 1/prescale événements
                    if np.random.random() > (1.0 / prescale):
                        prescaled_decisions[path_name] = False
        
        return prescaled_decisions

# Exemple
global_trigger = L1GlobalTrigger()

# Simuler objets L1
l1_objects = {
    'electrons': [{'et': 30.0, 'eta_idx': 10, 'phi_idx': 5}],
    'muons': [{'pt_estimate': 25.0, 'quality': 3}],
    'jets': [
        {'et': 60.0, 'eta_idx': 5, 'phi_idx': 10},
        {'et': 55.0, 'eta_idx': 7, 'phi_idx': 12}
    ],
    'met': {'met': 120.0, 'met_phi': 0.5}
}

result = global_trigger.evaluate_trigger(l1_objects)
print(f"\nL1 Global Trigger Decision:")
print(f"  L1 Accept: {result['l1_accept']}")
print(f"  Decisions par chemin:")
for path, decision in result['decisions'].items():
    print(f"    {path}: {decision}")
print(f"  L1 Bits: {result['l1_bits']:05b}")
```

---

## Exercices

### Exercice 18.2.1
Implémentez un algorithme de clustering adaptatif qui ajuste la taille de la fenêtre selon la densité d'énergie.

### Exercice 18.2.2
Concevez un réseau ML pour identification d'électrons avec moins de 500 paramètres et latence < 50 ns.

### Exercice 18.2.3
Optimisez un menu de trigger L1 pour maximiser l'efficacité signal tout en respectant un budget de 100 kHz.

### Exercice 18.2.4
Analysez l'impact de la quantification 4-bit vs 8-bit sur les performances et ressources FPGA.

---

## Points Clés à Retenir

> 📌 **Le L1 Trigger doit décider en < 4 μs avec hardware dédié (FPGA/ASIC)**

> 📌 **Les algorithmes classiques (clustering, pattern matching) sont rapides mais limités**

> 📌 **L'IA dans L1 nécessite quantification agressive (4-8 bits) et architectures minimales**

> 📌 **La pipeline architecture permet traitement continu à 40 MHz**

> 📌 **Le menu de trigger et prescales contrôlent le taux de sortie**

> 📌 **Les ressources FPGA (LUT, DSP, BRAM) limitent la complexité des modèles ML**

---

*Section précédente : [18.1 Architecture](./18_01_Architecture.md) | Section suivante : [18.3 High-Level Trigger](./18_03_HLT.md)*

