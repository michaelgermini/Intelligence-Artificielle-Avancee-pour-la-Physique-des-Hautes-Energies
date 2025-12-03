# 19.3 Tagging de Saveurs

---

## Introduction

Le **tagging de saveurs** (flavor tagging) consiste à identifier la saveur d'un quark à l'origine d'un jet. Le **b-tagging** (identification de quarks b) est particulièrement crucial car les quarks b sont produits dans les désintégrations du top quark et du boson de Higgs, deux particules fondamentales étudiées au LHC.

Cette section présente les techniques de tagging, en particulier le b-tagging et c-tagging, et comment le machine learning améliore significativement ces performances.

---

## Signatures des Quarks Lourds

### Propriétés Distinctives

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class FlavorTaggingBasics:
    """
    Bases du tagging de saveurs
    """
    
    def __init__(self):
        self.flavor_signatures = {
            'b_quark': {
                'lifetime': '~1.5 ps (longue)',
                'key_signature': 'Vertex secondaire déplacé',
                'features': [
                    'Secondary vertex position',
                    'Track impact parameter',
                    'Charged track multiplicity',
                    'Jet shape'
                ],
                'importance': 'Critique pour Higgs, top'
            },
            'c_quark': {
                'lifetime': '~0.5 ps (moyenne)',
                'key_signature': 'Vertex secondaire plus proche',
                'features': [
                    'Secondary vertex (closer)',
                    'Track impact parameter',
                    'Charmed hadron decay products'
                ],
                'importance': 'Important pour certaines analyses'
            },
            'light_quark': {
                'lifetime': '~0.01 ps (courte)',
                'key_signature': 'Pas de vertex secondaire',
                'features': [
                    'No secondary vertex',
                    'Low impact parameters',
                    'Different jet shape'
                ],
                'importance': 'Background principal'
            }
        }
    
    def display_signatures(self):
        """Affiche les signatures"""
        print("\n" + "="*70)
        print("Signatures des Quarks Lourds")
        print("="*70)
        
        for flavor, info in self.flavor_signatures.items():
            print(f"\n{flavor.upper().replace('_', ' ')}:")
            print(f"  Lifetime: {info['lifetime']}")
            print(f"  Signature clé: {info['key_signature']}")
            print(f"  Features: {', '.join(info['features'])}")
            print(f"  Importance: {info['importance']}")

basics = FlavorTaggingBasics()
basics.display_signatures()
```

---

## Features pour B-Tagging

### Vertex Secondaire et Impact Parameter

```python
class BTaggingFeatures:
    """
    Calcul de features pour b-tagging
    """
    
    def compute_secondary_vertex_features(self, tracks: np.ndarray,
                                         primary_vertex: np.ndarray) -> Dict:
        """
        Calcule features liées au vertex secondaire
        
        Args:
            tracks: (n_tracks, 6) [x, y, z, px, py, pz]
            primary_vertex: (3,) position vertex primaire
        """
        # Trouver vertex secondaire (simplifié: intersection des traces)
        # En pratique: utiliser algorithmes de fitting
        
        # Approximation: barycentre des points d'approche le plus proche
        secondary_vertex = np.mean(tracks[:, :3], axis=0)
        
        # Distance 3D du vertex secondaire au primaire
        sv_distance = np.linalg.norm(secondary_vertex - primary_vertex)
        
        # Distance transverse (significance)
        sv_xy = secondary_vertex[:2] - primary_vertex[:2]
        sv_xy_distance = np.linalg.norm(sv_xy)
        
        # Erreur sur distance (approximation)
        sv_distance_err = 0.01 * sv_distance  # 1% erreur relative
        sv_significance = sv_xy_distance / sv_distance_err if sv_distance_err > 0 else 0
        
        # Nombre de traces au vertex
        n_tracks = len(tracks)
        
        # Masse invariante des traces (approximation de masse du B)
        total_momentum = np.sum(tracks[:, 3:6], axis=0)
        p = np.linalg.norm(total_momentum)
        # Approximation: E ≈ p pour hadrons lourds
        mass = p * 0.1  # Approximation simplifiée
        
        return {
            'sv_distance': sv_distance,
            'sv_xy_distance': sv_xy_distance,
            'sv_significance': sv_significance,
            'n_tracks_at_sv': n_tracks,
            'sv_mass': mass
        }
    
    def compute_impact_parameter_features(self, tracks: np.ndarray,
                                         primary_vertex: np.ndarray) -> Dict:
        """
        Calcule features liées à l'impact parameter
        
        Impact parameter = distance d'approche le plus proche d'une trace au vertex primaire
        """
        impact_parameters = []
        impact_parameters_2d = []
        
        for track in tracks:
            # Paramétrisation de la trace (droite)
            track_pos = track[:3]
            track_dir = track[3:6]
            track_dir = track_dir / np.linalg.norm(track_dir) if np.linalg.norm(track_dir) > 0 else track_dir
            
            # Distance d'approche au vertex primaire
            vec_to_vertex = primary_vertex - track_pos
            # Projection sur direction perpendiculaire
            perp_component = vec_to_vertex - np.dot(vec_to_vertex, track_dir) * track_dir
            ip_3d = np.linalg.norm(perp_component)
            
            # Impact parameter 2D (transverse seulement)
            track_dir_xy = track_dir[:2]
            track_dir_xy = track_dir_xy / np.linalg.norm(track_dir_xy) if np.linalg.norm(track_dir_xy) > 0 else track_dir_xy
            vec_to_vertex_xy = (primary_vertex - track_pos)[:2]
            perp_component_xy = vec_to_vertex_xy - np.dot(vec_to_vertex_xy, track_dir_xy) * track_dir_xy
            ip_2d = np.linalg.norm(perp_component_xy)
            
            impact_parameters.append(ip_3d)
            impact_parameters_2d.append(ip_2d)
        
        # Statistiques
        return {
            'max_ip_3d': np.max(impact_parameters) if impact_parameters else 0,
            'max_ip_2d': np.max(impact_parameters_2d) if impact_parameters_2d else 0,
            'mean_ip_2d': np.mean(impact_parameters_2d) if impact_parameters_2d else 0,
            'n_tracks_with_large_ip': np.sum(np.array(impact_parameters_2d) > 0.1)
        }
    
    def compute_jet_shape_features(self, jet_particles: np.ndarray,
                                  secondary_vertex: np.ndarray) -> Dict:
        """
        Features de forme du jet liées au b-tagging
        """
        # Fraction d'énergie près du vertex secondaire
        sv_pos = secondary_vertex
        
        energy_near_sv = 0
        total_energy = 0
        
        for p in jet_particles:
            pos = p[:3] if len(p) > 3 else np.array([0, 0, 0])  # Approximation
            dist_to_sv = np.linalg.norm(pos - sv_pos)
            
            if dist_to_sv < 0.1:  # Dans 0.1 mm
                energy_near_sv += p[3] if len(p) > 3 else p[0]  # E ou pT
            
            total_energy += p[3] if len(p) > 3 else p[0]
        
        energy_fraction_near_sv = energy_near_sv / total_energy if total_energy > 0 else 0
        
        return {
            'energy_fraction_near_sv': energy_fraction_near_sv
        }
    
    def compute_all_btag_features(self, tracks: np.ndarray,
                                 jet_particles: np.ndarray,
                                 primary_vertex: np.ndarray) -> np.ndarray:
        """
        Calcule toutes les features pour b-tagging
        """
        features = []
        
        # Features vertex secondaire
        sv_features = self.compute_secondary_vertex_features(tracks, primary_vertex)
        features.extend([
            sv_features['sv_distance'],
            sv_features['sv_xy_distance'],
            sv_features['sv_significance'],
            sv_features['n_tracks_at_sv'],
            sv_features['sv_mass']
        ])
        
        # Features impact parameter
        ip_features = self.compute_impact_parameter_features(tracks, primary_vertex)
        features.extend([
            ip_features['max_ip_3d'],
            ip_features['max_ip_2d'],
            ip_features['mean_ip_2d'],
            ip_features['n_tracks_with_large_ip']
        ])
        
        # Features jet shape
        secondary_vertex_pos = np.mean(tracks[:, :3], axis=0)  # Approximation
        shape_features = self.compute_jet_shape_features(jet_particles, secondary_vertex_pos)
        features.append(shape_features['energy_fraction_near_sv'])
        
        # Features jet de base
        if len(jet_particles) > 0:
            jet_pt = np.sum(jet_particles[:, 0]) if jet_particles.shape[1] > 0 else 0
            jet_eta = np.mean(jet_particles[:, 1]) if jet_particles.shape[1] > 1 else 0
            features.extend([jet_pt, jet_eta])
        
        return np.array(features)

# Calculer features
btag_features = BTaggingFeatures()

# Simuler données
primary_vertex = np.array([0.0, 0.0, 0.0])
tracks = np.random.rand(5, 6)  # 5 traces
tracks[:, :3] *= 0.01  # Positions proches
tracks[:, 3:6] = np.random.randn(5, 3)  # Directions
jet_particles = np.random.rand(10, 4)  # Particules du jet

features = btag_features.compute_all_btag_features(tracks, jet_particles, primary_vertex)

print(f"\nFeatures B-Tagging ({len(features)} features):")
print(f"  Vertex secondaire: distance, significance, mass, n_tracks")
print(f"  Impact parameter: max_3d, max_2d, mean_2d, n_large_ip")
print(f"  Jet shape: energy_fraction_near_sv")
print(f"  Jet: pt, eta")
```

---

## Modèles de B-Tagging

### Architectures ML

```python
class AdvancedBTagger(nn.Module):
    """
    Modèle avancé de b-tagging
    """
    
    def __init__(self, vertex_features_dim=50, jet_features_dim=16):
        super().__init__()
        
        # Encodeur de features de vertex
        self.vertex_encoder = nn.Sequential(
            nn.Linear(vertex_features_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32)
        )
        
        # Encodeur de features de jet
        self.jet_encoder = nn.Sequential(
            nn.Linear(jet_features_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32)
        )
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),  # vertex + jet = 64
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 3)  # [light, charm, bottom]
        )
    
    def forward(self, vertex_features, jet_features):
        """
        Args:
            vertex_features: (batch, vertex_features_dim)
            jet_features: (batch, jet_features_dim)
        """
        v_encoded = self.vertex_encoder(vertex_features)
        j_encoded = self.jet_encoder(jet_features)
        
        combined = torch.cat([v_encoded, j_encoded], dim=-1)
        return self.classifier(combined)
    
    def predict_flavor(self, vertex_features, jet_features):
        """Retourne probabilités par saveur"""
        with torch.no_grad():
            logits = self.forward(vertex_features, jet_features)
            probs = F.softmax(logits, dim=-1)
            return probs

class DeepSetBTagger(nn.Module):
    """
    B-tagger utilisant Deep Set pour traiter ensemble de traces
    
    Permet nombre variable de traces par jet
    """
    
    def __init__(self, track_features_dim=6, jet_features_dim=16, hidden_dim=64):
        super().__init__()
        
        # Encodeur par trace
        self.track_encoder = nn.Sequential(
            nn.Linear(track_features_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Agrégation (Deep Set)
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combiner avec features jet
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + jet_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # [light, charm, bottom]
        )
    
    def forward(self, track_features, jet_features):
        """
        Args:
            track_features: (batch, n_tracks, track_features_dim) nombre variable
            jet_features: (batch, jet_features_dim)
        """
        # Encoder chaque trace
        track_embeddings = self.track_encoder(track_features)
        
        # Agrégation: moyenne pondérée (ou max pooling)
        aggregated = torch.mean(track_embeddings, dim=1)  # (batch, hidden_dim)
        aggregated = self.aggregation(aggregated)
        
        # Combiner avec features jet
        combined = torch.cat([aggregated, jet_features], dim=-1)
        
        return self.classifier(combined)

# Créer modèles
advanced_btagger = AdvancedBTagger(vertex_features_dim=50, jet_features_dim=16)
deepset_btagger = DeepSetBTagger(track_features_dim=6, jet_features_dim=16)

print(f"\nModèles B-Tagging:")
print(f"  Advanced: {sum(p.numel() for p in advanced_btagger.parameters()):,} paramètres")
print(f"  DeepSet: {sum(p.numel() for p in deepset_btagger.parameters()):,} paramètres")
```

---

## C-Tagging

### Identification de Quarks Charm

```python
class CTagger(nn.Module):
    """
    C-tagger: identification de quarks charm
    
    Plus difficile que b-tagging car lifetime plus courte
    """
    
    def __init__(self):
        super().__init__()
        
        # Similar à b-tagger mais optimisé pour signatures de charme
        self.classifier = nn.Sequential(
            nn.Linear(50, 128),  # Features combinées
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 2)  # [light, charm]
        )
    
    def compute_ctag_features(self, tracks, jet_particles, primary_vertex):
        """
        Features spécifiques pour c-tagging
        
        C-quarks: vertex secondaire plus proche, hadrons charmés
        """
        features = []
        
        # Features similaires à b-tagging mais avec focus sur distance plus courte
        btag_feat = BTaggingFeatures()
        all_features = btag_feat.compute_all_btag_features(tracks, jet_particles, primary_vertex)
        
        # Ajouter features spécifiques charm
        # Masse invariante (charm hadrons ~1.9 GeV)
        if len(tracks) >= 2:
            momenta = tracks[:, 3:6]
            total_p = np.sum(momenta, axis=0)
            mass = np.linalg.norm(total_p) * 0.05  # Approximation
            features.append(mass)
        else:
            features.append(0.0)
        
        return np.concatenate([all_features, np.array(features)])

ctagger = CTagger()
```

---

## Métriques de Performance

### Évaluation du Tagging

```python
class TaggingPerformanceMetrics:
    """
    Métriques pour évaluer performance de tagging
    """
    
    def compute_efficiency_vs_rejection(self, true_labels: np.ndarray,
                                       scores: np.ndarray,
                                       signal_class: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule courbe efficacité vs rejet (ROC)
        
        Args:
            true_labels: 0=light, 1=charm, 2=bottom
            scores: Scores de classification
            signal_class: Classe considérée comme signal (2 pour bottom)
        """
        signal_mask = true_labels == signal_class
        background_mask = true_labels != signal_class
        
        signal_scores = scores[signal_mask]
        background_scores = scores[background_mask]
        
        # Varier seuil
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        
        efficiencies = []
        rejections = []
        
        for threshold in thresholds:
            signal_passed = (signal_scores > threshold).mean()
            background_passed = (background_scores > threshold).mean()
            
            efficiencies.append(signal_passed)
            rejections.append(1.0 / background_passed if background_passed > 0 else float('inf'))
        
        return np.array(efficiencies), np.array(rejections), thresholds
    
    def compute_working_points(self, true_labels: np.ndarray,
                              scores: np.ndarray,
                              target_efficiencies: List[float] = [0.60, 0.70, 0.80],
                              signal_class: int = 2) -> Dict:
        """
        Calcule working points (seuils) pour efficacités cibles
        
        Returns:
            Dict avec seuils et rejets correspondants
        """
        signal_mask = true_labels == signal_class
        background_mask = true_labels != signal_class
        
        signal_scores = scores[signal_mask]
        background_scores = scores[background_mask]
        
        working_points = {}
        
        for target_eff in target_efficiencies:
            # Trouver seuil pour cette efficacité
            threshold = np.percentile(signal_scores, (1 - target_eff) * 100)
            
            # Calculer rejet à ce seuil
            background_passed = (background_scores > threshold).mean()
            rejection = 1.0 / background_passed if background_passed > 0 else float('inf')
            
            working_points[f'WP_{int(target_eff*100)}'] = {
                'efficiency': target_eff,
                'threshold': threshold,
                'background_rejection': rejection
            }
        
        return working_points

metrics = TaggingPerformanceMetrics()

# Simuler résultats
n_samples = 10000
true_labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])  # Mostly light
scores = np.random.rand(n_samples)  # Placeholder

working_points = metrics.compute_working_points(true_labels, scores)

print(f"\nWorking Points B-Tagging:")
for wp_name, wp_info in working_points.items():
    print(f"  {wp_name}: Eff={wp_info['efficiency']:.0%}, "
          f"Rej={wp_info['background_rejection']:.1f}")
```

---

## Exercices

### Exercice 19.3.1
Implémentez un calculateur de features de vertex secondaire complet qui trouve réellement le vertex par fitting de traces.

### Exercice 19.3.2
Entraînez un modèle de b-tagging et comparez les performances avec un modèle classique basé sur règles.

### Exercice 19.3.3
Développez un système de tagging multi-classe qui distingue simultanément light, charm, et bottom jets.

### Exercice 19.3.4
Analysez l'impact de chaque type de feature (vertex, impact parameter, jet shape) sur les performances de b-tagging.

---

## Points Clés à Retenir

> 📌 **Le b-tagging est crucial pour physique du Higgs et du top**

> 📌 **Les vertex secondaires déplacés sont la signature clé des quarks b**

> 📌 **L'impact parameter distingue traces de quarks lourds vs légers**

> 📌 **Les modèles ML combinent multiples sources d'information (vertex + jet + traces)**

> 📌 **Les Deep Sets permettent de gérer nombre variable de traces**

> 📌 **Les working points permettent compromis efficacité/pureté selon analyse**

---

*Section précédente : [19.2 Identification de Jets](./19_02_Jets.md) | Section suivante : [19.4 Identification de Leptons](./19_04_Leptons.md)*

