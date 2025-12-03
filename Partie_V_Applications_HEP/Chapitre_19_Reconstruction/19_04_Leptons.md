# 19.4 Identification de Leptons

---

## Introduction

L'**identification de leptons** (électrons et muons) est fondamentale en physique des hautes énergies. Les leptons sont des signatures propres de nombreux processus physiques (désintégration du W, Z, Higgs, etc.) car ils interagissent faiblement et laissent des signaux clairs dans les détecteurs.

Cette section présente les techniques d'identification d'électrons et de muons, incluant les méthodes classiques et les approches basées sur le machine learning.

---

## Types de Leptons

### Signatures Détecteur

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class LeptonSignatures:
    """
    Signatures des différents leptons dans le détecteur
    """
    
    def __init__(self):
        self.lepton_types = {
            'electron': {
                'detector_signals': [
                    'Track dans tracker',
                    'Cluster dans ECAL',
                    'Pas de signal HCAL (ou très faible)',
                    'Match track-ECAL'
                ],
                'key_features': [
                    'E/p ratio (énergie ECAL / momentum tracker)',
                    'Shower shape dans ECAL',
                    'Isolation (pas d'autres particules proches)',
                    'Track quality'
                ],
                'backgrounds': [
                    'Pions chargés (pion fake)',
                    'Photons conversions'
                ]
            },
            'muon': {
                'detector_signals': [
                    'Track dans tracker',
                    'Track dans détecteur muon',
                    'Pas de signal ECAL significatif',
                    'Pas de shower dans calorimètre'
                ],
                'key_features': [
                    'Chi2 de la trace',
                    'Nombre de stations muon',
                    'Isolation',
                    'pT du muon'
                ],
                'backgrounds': [
                    'Pions (qui passent à travers)',
                    'Autres hadrons'
                ]
            },
            'tau': {
                'detector_signals': [
                    'Track (1 ou 3)',
                    'Narrow jet',
                    'Signal dans calorimètres'
                ],
                'key_features': [
                    'Multiplicité de traces',
                    'Masse invariante',
                    'Isolation'
                ],
                'backgrounds': [
                    'Jets QCD',
                    'Jets hadroniques'
                ]
            }
        }
    
    def display_signatures(self):
        """Affiche les signatures"""
        print("\n" + "="*70)
        print("Signatures de Leptons dans le Détecteur")
        print("="*70)
        
        for lepton, info in self.lepton_types.items():
            print(f"\n{lepton.upper()}:")
            print(f"  Signaux détecteur:")
            for sig in info['detector_signals']:
                print(f"    • {sig}")
            print(f"  Features clés:")
            for feat in info['key_features']:
                print(f"    • {feat}")
            print(f"  Backgrounds principaux:")
            for bg in info['backgrounds']:
                print(f"    • {bg}")

signatures = LeptonSignatures()
signatures.display_signatures()
```

---

## Identification d'Électrons

### Features et Algorithmes

```python
class ElectronIdentification:
    """
    Identification d'électrons
    """
    
    def compute_ecal_features(self, ecal_cluster: np.ndarray,
                             track_momentum: float) -> Dict:
        """
        Features du cluster ECAL
        
        Args:
            ecal_cluster: énergies dans différentes couches/segments
            track_momentum: momentum de la trace associée
        """
        # Énergie totale du cluster
        cluster_energy = np.sum(ecal_cluster)
        
        # E/p ratio (devrait être ~1 pour électrons)
        e_over_p = cluster_energy / track_momentum if track_momentum > 0 else 0
        
        # Shower shape: largeur du shower
        if len(ecal_cluster.shape) == 2:  # Image 2D
            # Calculer barycentre
            eta_bins, phi_bins = ecal_cluster.shape
            eta_center = np.sum(np.arange(eta_bins)[:, None] * ecal_cluster) / cluster_energy
            phi_center = np.sum(np.arange(phi_bins)[None, :] * ecal_cluster) / cluster_energy
            
            # Largeur RMS
            eta_coords = np.arange(eta_bins)
            phi_coords = np.arange(phi_bins)
            eta_grid, phi_grid = np.meshgrid(eta_coords, phi_coords, indexing='ij')
            
            eta_rms = np.sqrt(np.sum((eta_grid - eta_center)**2 * ecal_cluster) / cluster_energy)
            phi_rms = np.sqrt(np.sum((phi_grid - phi_center)**2 * ecal_cluster) / cluster_energy)
            
            width = np.sqrt(eta_rms**2 + phi_rms**2)
        else:
            width = 0.1  # Placeholder
        
        # Fraction d'énergie dans couche centrale
        if len(ecal_cluster) > 1:
            central_fraction = ecal_cluster[len(ecal_cluster)//2] / cluster_energy
        else:
            central_fraction = 1.0
        
        return {
            'cluster_energy': cluster_energy,
            'e_over_p': e_over_p,
            'shower_width': width,
            'central_fraction': central_fraction
        }
    
    def compute_isolation_features(self, electron_eta: float, electron_phi: float,
                                  other_particles: np.ndarray,
                                  cone_size: float = 0.3) -> Dict:
        """
        Features d'isolation
        
        Électron isolé = pas d'autres particules proches
        """
        # Calculer distances aux autres particules
        iso_energy = 0
        iso_pt = 0
        n_tracks_nearby = 0
        
        for particle in other_particles:
            p_eta, p_phi = particle[1], particle[2]  # η, φ
            p_pt = particle[0]
            p_energy = particle[3] if len(particle) > 3 else p_pt
            
            # Distance ΔR
            deta = electron_eta - p_eta
            dphi = self._delta_phi(electron_phi, p_phi)
            dr = np.sqrt(deta**2 + dphi**2)
            
            if dr < cone_size:
                iso_energy += p_energy
                iso_pt += p_pt
                n_tracks_nearby += 1
        
        # Isolation relative
        electron_pt = other_particles[0][0] if len(other_particles) > 0 else 1.0
        rel_iso_pt = iso_pt / electron_pt if electron_pt > 0 else 0
        
        return {
            'isolation_energy': iso_energy,
            'isolation_pt': iso_pt,
            'relative_iso_pt': rel_iso_pt,
            'n_tracks_in_cone': n_tracks_nearby
        }
    
    def compute_track_features(self, track: np.ndarray) -> Dict:
        """
        Features de la trace associée
        """
        # Qualité de la trace (simplifié)
        track_chi2 = 2.0  # Placeholder
        n_hits = len(track) if hasattr(track, '__len__') else 10
        
        # Impact parameter (devrait être petit pour électrons primaires)
        impact_parameter = 0.01  # Placeholder
        
        return {
            'track_chi2': track_chi2,
            'n_hits': n_hits,
            'impact_parameter': impact_parameter
        }
    
    def compute_all_electron_features(self, ecal_cluster: np.ndarray,
                                     track: np.ndarray,
                                     track_momentum: float,
                                     electron_eta: float,
                                     electron_phi: float,
                                     other_particles: np.ndarray) -> np.ndarray:
        """
        Calcule toutes les features pour identification électron
        """
        features = []
        
        # Features ECAL
        ecal_feat = self.compute_ecal_features(ecal_cluster, track_momentum)
        features.extend([
            ecal_feat['cluster_energy'],
            ecal_feat['e_over_p'],
            ecal_feat['shower_width'],
            ecal_feat['central_fraction']
        ])
        
        # Features isolation
        iso_feat = self.compute_isolation_features(electron_eta, electron_phi, other_particles)
        features.extend([
            iso_feat['relative_iso_pt'],
            iso_feat['n_tracks_in_cone']
        ])
        
        # Features track
        track_feat = self.compute_track_features(track)
        features.extend([
            track_feat['track_chi2'],
            track_feat['n_hits'],
            track_feat['impact_parameter']
        ])
        
        # Cinématique
        features.extend([track_momentum, electron_eta])
        
        return np.array(features)
    
    def _delta_phi(self, phi1: float, phi2: float) -> float:
        """Calcule Δφ"""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return abs(dphi)

class ElectronIDModel(nn.Module):
    """
    Modèle ML pour identification d'électrons
    """
    
    def __init__(self, n_features=11):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probabilité électron
        )
    
    def forward(self, features):
        """Retourne probabilité que candidat soit un électron"""
        return self.classifier(features)

# Calculer features
electron_id = ElectronIdentification()

# Simuler données
ecal_cluster = np.random.rand(5, 5) * 10  # Cluster 5×5
track_momentum = 25.0  # GeV
track = np.random.rand(20, 6)  # 20 hits
electron_eta, electron_phi = 0.5, 0.3
other_particles = np.random.rand(10, 4)  # Autres particules

features = electron_id.compute_all_electron_features(
    ecal_cluster, track, track_momentum, electron_eta, electron_phi, other_particles
)

print(f"\nFeatures Identification Électron ({len(features)} features):")
print(f"  ECAL: energy, E/p, width, central_fraction")
print(f"  Isolation: rel_iso_pt, n_tracks_nearby")
print(f"  Track: chi2, n_hits, impact_parameter")
print(f"  Cinématique: pT, eta")

# Créer modèle
electron_model = ElectronIDModel(n_features=len(features))
```

---

## Identification de Muons

### Features et Modèles

```python
class MuonIdentification:
    """
    Identification de muons
    """
    
    def compute_muon_detector_features(self, muon_stations: np.ndarray) -> Dict:
        """
        Features du détecteur muon
        
        Args:
            muon_stations: (n_stations,) hits dans chaque station
        """
        # Nombre de stations avec hits
        n_stations_with_hits = np.sum(muon_stations > 0)
        
        # Qualité de la trace muon (chi2)
        # Approximation: meilleure qualité = plus de stations
        chi2 = 10.0 / (n_stations_with_hits + 1)
        
        # Fraction de stations hitées
        station_efficiency = n_stations_with_hits / len(muon_stations)
        
        return {
            'n_stations': n_stations_with_hits,
            'chi2': chi2,
            'station_efficiency': station_efficiency
        }
    
    def compute_calorimeter_deposit(self, ecal_energy: float,
                                   hcal_energy: float) -> Dict:
        """
        Énergie déposée dans calorimètres (devrait être faible pour muons)
        """
        total_calo_energy = ecal_energy + hcal_energy
        
        # Fraction ECAL/HCAL (muons déposent peu dans ECAL)
        ecal_fraction = ecal_energy / total_calo_energy if total_calo_energy > 0 else 0
        
        return {
            'total_calo_energy': total_calo_energy,
            'ecal_energy': ecal_energy,
            'hcal_energy': hcal_energy,
            'ecal_fraction': ecal_fraction
        }
    
    def compute_isolation_features(self, muon_eta: float, muon_phi: float,
                                  other_particles: np.ndarray,
                                  cone_size: float = 0.3) -> Dict:
        """
        Isolation du muon (similaire à électrons)
        """
        iso_pt = 0
        iso_energy = 0
        
        for particle in other_particles:
            p_eta, p_phi = particle[1], particle[2]
            p_pt = particle[0]
            p_energy = particle[3] if len(particle) > 3 else p_pt
            
            deta = muon_eta - p_eta
            dphi = self._delta_phi(muon_phi, p_phi)
            dr = np.sqrt(deta**2 + dphi**2)
            
            if dr < cone_size:
                iso_pt += p_pt
                iso_energy += p_energy
        
        muon_pt = 20.0  # Placeholder
        rel_iso_pt = iso_pt / muon_pt if muon_pt > 0 else 0
        
        return {
            'isolation_pt': iso_pt,
            'isolation_energy': iso_energy,
            'relative_iso_pt': rel_iso_pt
        }
    
    def compute_tracker_features(self, tracker_track: np.ndarray) -> Dict:
        """
        Features de la trace dans le tracker
        """
        n_hits = len(tracker_track) if hasattr(tracker_track, '__len__') else 0
        track_chi2 = 3.0  # Placeholder
        
        return {
            'n_tracker_hits': n_hits,
            'tracker_chi2': track_chi2
        }
    
    def compute_all_muon_features(self, muon_stations: np.ndarray,
                                 ecal_energy: float,
                                 hcal_energy: float,
                                 muon_eta: float,
                                 muon_phi: float,
                                 other_particles: np.ndarray,
                                 tracker_track: np.ndarray) -> np.ndarray:
        """
        Calcule toutes les features pour identification muon
        """
        features = []
        
        # Features détecteur muon
        muon_det_feat = self.compute_muon_detector_features(muon_stations)
        features.extend([
            muon_det_feat['n_stations'],
            muon_det_feat['chi2'],
            muon_det_feat['station_efficiency']
        ])
        
        # Features calorimètre
        calo_feat = self.compute_calorimeter_deposit(ecal_energy, hcal_energy)
        features.extend([
            calo_feat['total_calo_energy'],
            calo_feat['ecal_fraction']
        ])
        
        # Features isolation
        iso_feat = self.compute_isolation_features(muon_eta, muon_phi, other_particles)
        features.append(iso_feat['relative_iso_pt'])
        
        # Features tracker
        tracker_feat = self.compute_tracker_features(tracker_track)
        features.extend([
            tracker_feat['n_tracker_hits'],
            tracker_feat['tracker_chi2']
        ])
        
        # Cinématique
        features.extend([muon_eta])
        
        return np.array(features)
    
    def _delta_phi(self, phi1: float, phi2: float) -> float:
        """Calcule Δφ"""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return abs(dphi)

class MuonIDModel(nn.Module):
    """
    Modèle ML pour identification de muons
    """
    
    def __init__(self, n_features=9):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probabilité muon
        )
    
    def forward(self, features):
        return self.classifier(features)

# Calculer features
muon_id = MuonIdentification()

# Simuler données
muon_stations = np.array([1, 1, 1, 1, 0])  # 4 stations sur 5
ecal_energy = 0.5  # GeV (faible pour muon)
hcal_energy = 1.0  # GeV
muon_eta, muon_phi = 0.3, 0.2
other_particles = np.random.rand(8, 4)
tracker_track = np.random.rand(15, 6)

features = muon_id.compute_all_muon_features(
    muon_stations, ecal_energy, hcal_energy,
    muon_eta, muon_phi, other_particles, tracker_track
)

print(f"\nFeatures Identification Muon ({len(features)} features):")
print(f"  Détecteur muon: n_stations, chi2, efficiency")
print(f"  Calorimètre: total_energy, ecal_fraction")
print(f"  Isolation: rel_iso_pt")
print(f"  Tracker: n_hits, chi2")
print(f"  Cinématique: eta")

# Créer modèle
muon_model = MuonIDModel(n_features=len(features))
```

---

## Identification de Taus

### Reconstruction de Leptons Tau

```python
class TauIdentification:
    """
    Identification de leptons tau
    
    Taus désintègrent en 1 ou 3 traces + neutrinos
    """
    
    def compute_tau_features(self, tracks: np.ndarray,
                            calorimeter_cluster: np.ndarray) -> Dict:
        """
        Features pour identification tau
        
        Taus: narrow jet avec 1 ou 3 traces
        """
        n_tracks = len(tracks)
        
        # Masse invariante des traces
        if n_tracks > 0:
            momenta = tracks[:, 3:6] if tracks.shape[1] > 5 else np.random.randn(n_tracks, 3)
            total_momentum = np.sum(momenta, axis=0)
            p = np.linalg.norm(total_momentum)
            # Approximation masse
            mass = p * 0.1  # Placeholder
        else:
            mass = 0.0
        
        # Largeur du jet (tau = narrow)
        if len(calorimeter_cluster.shape) == 2:
            # Calculer largeur
            width = np.std(calorimeter_cluster)  # Simplifié
        else:
            width = 0.1
        
        # Ratio énergie trace / énergie totale
        track_energy = np.sum(tracks[:, 3]) if tracks.shape[1] > 3 else 0
        cluster_energy = np.sum(calorimeter_cluster) if hasattr(calorimeter_cluster, 'sum') else 1.0
        track_energy_fraction = track_energy / cluster_energy if cluster_energy > 0 else 0
        
        return {
            'n_tracks': n_tracks,
            'invariant_mass': mass,
            'jet_width': width,
            'track_energy_fraction': track_energy_fraction,
            'is_1prong': n_tracks == 1,  # 1 trace
            'is_3prong': n_tracks == 3   # 3 traces
        }

tau_id = TauIdentification()
```

---

## Modèles Unifiés

### Classification Multi-Lepton

```python
class UnifiedLeptonClassifier(nn.Module):
    """
    Classificateur unifié pour tous types de leptons
    """
    
    def __init__(self, n_features=20, n_classes=4):
        """
        Classes: [electron, muon, tau, fake]
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, n_classes)
        )
    
    def forward(self, features):
        """
        Returns logits pour [electron, muon, tau, fake]
        """
        return self.classifier(features)
    
    def predict_lepton_type(self, features):
        """Prédit le type de lepton"""
        with torch.no_grad():
            logits = self.forward(features)
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1)
            return predicted_class, probs

unified_model = UnifiedLeptonClassifier()
```

---

## Exercices

### Exercice 19.4.1
Implémentez un système complet d'identification d'électrons qui combine features ECAL, tracker, et isolation.

### Exercice 19.4.2
Entraînez un modèle pour distinguer muons vrais des pions qui traversent les détecteurs (pion fake).

### Exercice 19.4.3
Développez un classificateur qui identifie simultanément électrons, muons, et taus.

### Exercice 19.4.4
Analysez l'importance relative des différentes features (isolation, track quality, etc.) pour l'identification de leptons.

---

## Points Clés à Retenir

> 📌 **Les électrons sont identifiés par match track-ECAL et shower shape**

> 📌 **Les muons sont identifiés par traces dans détecteur muon et faible dépôt calorimétrique**

> 📌 **Les taus sont identifiés par narrow jets avec 1 ou 3 traces**

> 📌 **L'isolation est cruciale pour tous les leptons (réduit fakes)**

> 📌 **Le ML améliore significativement la discrimination contre les fakes**

> 📌 **Les features combinent informations tracker, calorimètre, et isolation**

---

*Section précédente : [19.3 Tagging de Saveurs](./19_03_Tagging.md) | Section suivante : [19.5 Reconstruction de l'Énergie Manquante](./19_05_MET.md)*

