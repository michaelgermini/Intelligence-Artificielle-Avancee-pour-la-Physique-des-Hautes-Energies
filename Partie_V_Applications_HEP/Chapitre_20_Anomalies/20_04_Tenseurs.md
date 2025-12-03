# 20.4 Réseaux de Tenseurs pour la Détection d'Anomalies

---

## Introduction

Les **réseaux de tenseurs** offrent une approche unique pour la détection d'anomalies en exploitant leur structure compacte et leur capacité à capturer des corrélations complexes entre variables. Leur efficacité computationnelle les rend particulièrement attractifs pour les applications temps réel comme les triggers.

Cette section présente l'utilisation des réseaux de tenseurs (MPS, Tensor Train) pour la détection d'anomalies, incluant les autoencoders basés sur tenseurs et les méthodes spécifiques aux structures tensorielles.

---

## Avantages des Réseaux de Tenseurs

### Pourquoi les Tenseurs pour Anomalies ?

```python
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class TensorNetworkAdvantages:
    """
    Avantages des réseaux de tenseurs pour détection d'anomalies
    """
    
    def __init__(self):
        self.advantages = {
            'compression': {
                'description': 'Représentation compacte',
                'benefit': 'Moins de paramètres = moins de risque overfitting',
                'impact': 'Meilleure généralisation'
            },
            'interpretability': {
                'description': 'Structure explicite',
                'benefit': 'Bond dimensions révèlent complexité nécessaire',
                'impact': 'Compréhension des corrélations importantes'
            },
            'efficiency': {
                'description': 'Efficacité computationnelle',
                'benefit': 'Contractions rapides, déployable sur FPGA',
                'impact': 'Utilisable dans triggers temps réel'
            },
            'correlations': {
                'description': 'Capte corrélations complexes',
                'benefit': 'Structure tensorielle encode dépendances multi-variables',
                'impact': 'Détecte patterns subtils d\'anomalies'
            }
        }
    
    def display_advantages(self):
        """Affiche les avantages"""
        print("\n" + "="*70)
        print("Avantages Réseaux de Tenseurs pour Anomalies")
        print("="*70)
        
        for advantage, info in self.advantages.items():
            print(f"\n{advantage.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Bénéfice: {info['benefit']}")
            print(f"  Impact: {info['impact']}")

advantages = TensorNetworkAdvantages()
advantages.display_advantages()
```

---

## Tensor Train Autoencoder

### Architecture TT pour Autoencoder

```python
class TensorTrainAutoencoder(nn.Module):
    """
    Autoencodeur basé sur Tensor Train (TT)
    
    Utilise décomposition TT pour encodeur et décodeur
    """
    
    def __init__(self, input_dims=[4, 4, 4, 4], bond_dims=[2, 3, 2], latent_dim=8):
        """
        Args:
            input_dims: Dimensions de chaque mode de l'input tensorisé
            bond_dims: Bond dimensions pour TT
            latent_dim: Dimension espace latent (flatten)
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.bond_dims = bond_dims
        self.latent_dim = latent_dim
        self.input_size = np.prod(input_dims)
        
        # Encodeur: TT decomposition
        # Input tensorisé → compression TT → latent
        self.tt_encoder = self._create_tt_layers(input_dims, bond_dims, latent_dim)
        
        # Décodeur: reconstruction depuis latent
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size)
        )
    
    def _create_tt_layers(self, input_dims, bond_dims, latent_dim):
        """
        Crée couches TT pour encodeur
        
        Simplifié: utilise couches linéaires qui simulent compression TT
        """
        # En pratique: utiliser vraie décomposition TT
        # Ici: approximation avec MLP
        input_size = np.prod(input_dims)
        
        layers = nn.Sequential(
            nn.Linear(input_size, bond_dims[0] * input_dims[0]),
            nn.ReLU(),
            nn.Linear(bond_dims[0] * input_dims[0], bond_dims[1] * input_dims[1]),
            nn.ReLU(),
            nn.Linear(bond_dims[1] * input_dims[1], latent_dim)
        )
        
        return layers
    
    def forward(self, x):
        """Forward pass"""
        # Encoder avec TT
        encoded = self.tt_encoder(x.view(x.size(0), -1))
        
        # Decoder
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), *self.input_dims)
        
        return decoded
    
    def compute_anomaly_score(self, x):
        """Score d'anomalie"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            
            # Erreur de reconstruction
            error = torch.mean((x - reconstructed)**2, dim=tuple(range(1, len(x.shape))))
            
            return error

# Créer TT Autoencoder
tt_autoencoder = TensorTrainAutoencoder(
    input_dims=[5, 5, 4],
    bond_dims=[3, 2],
    latent_dim=10
)

print(f"\nTensor Train Autoencoder:")
print(f"  Input size: {tt_autoencoder.input_size}")
print(f"  Latent dim: {tt_autoencoder.latent_dim}")
print(f"  Paramètres: {sum(p.numel() for p in tt_autoencoder.parameters()):,}")

# Comparer avec autoencodeur standard
standard_ae = nn.Sequential(
    nn.Linear(tt_autoencoder.input_size, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.ReLU(),
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, tt_autoencoder.input_size)
)

print(f"  Autoencodeur standard: {sum(p.numel() for p in standard_ae.parameters()):,} paramètres")
print(f"  Compression: {sum(p.numel() for p in standard_ae.parameters()) / sum(p.numel() for p in tt_autoencoder.parameters()):.2f}×")
```

---

## MPS pour Modélisation de Distributions

### Matrix Product State pour Densité

```python
class MPSDensityModel:
    """
    Modèle de densité avec MPS (Matrix Product State)
    
    Modélise distribution jointe des features comme MPS
    """
    
    def __init__(self, n_features=10, bond_dim=4, n_categories_per_feature=10):
        """
        Args:
            n_features: Nombre de features
            bond_dim: Bond dimension du MPS
            n_categories_per_feature: Nombre de valeurs possibles par feature
        """
        super().__init__()
        
        self.n_features = n_features
        self.bond_dim = bond_dim
        self.n_categories = n_categories_per_feature
        
        # Créer tenseurs MPS
        # En pratique: utiliser vraie structure MPS
        # Ici: approximation
        
        # Tensors pour chaque site
        self.mps_tensors = nn.ModuleList()
        
        for i in range(n_features):
            if i == 0:
                # Premier site: (bond_dim, n_categories)
                tensor = nn.Parameter(torch.randn(bond_dim, n_categories_per_feature))
            elif i == n_features - 1:
                # Dernier site: (n_categories, bond_dim)
                tensor = nn.Parameter(torch.randn(n_categories_per_feature, bond_dim))
            else:
                # Sites intermédiaires: (bond_dim, n_categories, bond_dim)
                tensor = nn.Parameter(torch.randn(bond_dim, n_categories_per_feature, bond_dim))
            
            self.mps_tensors.append(nn.Parameter(tensor))
    
    def compute_log_probability(self, x_discrete):
        """
        Calcule log-probabilité d'une configuration
        
        Args:
            x_discrete: (batch, n_features) indices discrets
        """
        # Contraction MPS pour calculer probabilité
        # Simplifié: approximation linéaire
        
        # En pratique: vraie contraction MPS
        # log p(x) = log(contraction des tenseurs selon indices x)
        
        # Approximation
        log_prob = torch.zeros(x_discrete.shape[0])
        
        for i in range(self.n_features):
            feature_values = x_discrete[:, i]
            # Prendre valeurs correspondantes dans tenseurs
            # (Simplifié)
            log_prob += torch.randn(x_discrete.shape[0]) * 0.1  # Placeholder
        
        return log_prob
    
    def compute_anomaly_score(self, x_discrete):
        """
        Score d'anomalie = -log probabilité
        
        Anomalies = faibles probabilités
        """
        log_prob = self.compute_log_probability(x_discrete)
        anomaly_score = -log_prob
        
        return anomaly_score

mps_density = MPSDensityModel(n_features=8, bond_dim=4, n_categories_per_feature=10)
```

---

## Tensor Train pour Détection d'Anomalies Multi-Variées

### Modélisation de Corrélations

```python
class TTTAnomalyDetector:
    """
    Détecteur d'anomalies basé sur Tensor Train
    
    Modélise distribution jointe des features
    """
    
    def __init__(self, feature_dims, bond_dims):
        """
        Args:
            feature_dims: Dimensions de chaque feature (après discrétisation)
            bond_dims: Bond dimensions pour TT
        """
        self.feature_dims = feature_dims
        self.bond_dims = bond_dims
        self.n_features = len(feature_dims)
        
        # Tensors TT (core tensors)
        # Structure: T[i] a shape (r[i-1], d[i], r[i])
        self.tt_cores = []
        
        # Initialiser cores
        for i in range(self.n_features):
            if i == 0:
                shape = (1, feature_dims[i], bond_dims[i])
            elif i == self.n_features - 1:
                shape = (bond_dims[i-1], feature_dims[i], 1)
            else:
                shape = (bond_dims[i-1], feature_dims[i], bond_dims[i])
            
            core = nn.Parameter(torch.randn(*shape))
            self.tt_cores.append(core)
        
        self.tt_cores = nn.ParameterList(self.tt_cores)
    
    def compute_probability(self, x_indices):
        """
        Calcule probabilité d'une configuration
        
        Args:
            x_indices: (batch, n_features) indices pour chaque feature
        """
        # Contraction TT
        # Pour chaque configuration, contracter cores selon indices
        
        batch_size = x_indices.shape[0]
        probs = torch.ones(batch_size)
        
        # Contraction simplifiée (approximation)
        for i in range(self.n_features):
            feature_idx = x_indices[:, i]
            # Extraire slices correspondantes des cores
            # (Simplifié ici)
            probs = probs * torch.randn(batch_size).abs() * 0.1  # Placeholder
        
        return probs
    
    def compute_anomaly_score(self, x_indices):
        """
        Score d'anomalie depuis probabilité TT
        """
        probs = self.compute_probability(x_indices)
        anomaly_score = -torch.log(probs + 1e-10)  # -log prob
        
        return anomaly_score

tt_detector = TTTAnomalyDetector(
    feature_dims=[5, 5, 5, 5],
    bond_dims=[3, 3, 3]
)

print(f"\nTensor Train Anomaly Detector:")
print(f"  Features: {tt_detector.n_features}")
print(f"  Bond dims: {tt_detector.bond_dims}")
```

---

## Autoencoder Tensor Train pour Événements HEP

### Application Spécifique

```python
class HEPTensorAutoencoder(nn.Module):
    """
    Autoencodeur Tensor Train pour événements HEP
    
    Tensorise features selon structure physique
    """
    
    def __init__(self, n_jets=4, n_leptons=2, jet_features=8, lepton_features=4):
        """
        Tensorise selon: [jets, leptons, MET]
        
        Structure: Tensor avec modes = [jet_1, jet_2, ..., lepton_1, lepton_2, MET]
        """
        super().__init__()
        
        self.n_jets = n_jets
        self.n_leptons = n_leptons
        self.jet_features = jet_features
        self.lepton_features = lepton_features
        
        # Input tensorisé
        input_dims = [jet_features] * n_jets + [lepton_features] * n_leptons + [4]  # MET
        self.input_dims = input_dims
        self.input_size = np.prod(input_dims)
        
        # TT encoder avec bond dimensions adaptatives
        bond_dims = [min(8, dim) for dim in input_dims[:-1]]
        latent_dim = 16
        
        # Encodeur (simplifié: MLP qui simule TT compression)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_size)
        )
    
    def forward(self, event_dict):
        """
        Forward avec structure d'événement
        
        Args:
            event_dict: {
                'jets': (batch, n_jets, jet_features),
                'leptons': (batch, n_leptons, lepton_features),
                'met': (batch, 4)
            }
        """
        # Flatten selon structure tensorielle
        jets_flat = event_dict['jets'].view(event_dict['jets'].shape[0], -1)
        leptons_flat = event_dict['leptons'].view(event_dict['leptons'].shape[0], -1)
        met_flat = event_dict['met']
        
        x = torch.cat([jets_flat, leptons_flat, met_flat], dim=1)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        decoded = self.decoder(latent)
        
        # Reshape
        decoded_dict = {
            'jets': decoded[:, :self.n_jets * self.jet_features].view(
                -1, self.n_jets, self.jet_features
            ),
            'leptons': decoded[:, 
                self.n_jets * self.jet_features:
                self.n_jets * self.jet_features + self.n_leptons * self.lepton_features
            ].view(-1, self.n_leptons, self.lepton_features),
            'met': decoded[:, -4:]
        }
        
        return decoded_dict
    
    def compute_anomaly_score(self, event_dict):
        """Score d'anomalie par composante"""
        with torch.no_grad():
            decoded = self.forward(event_dict)
            
            scores = {}
            
            # Erreur par composante
            scores['jets'] = torch.mean((event_dict['jets'] - decoded['jets'])**2, dim=(1, 2))
            scores['leptons'] = torch.mean((event_dict['leptons'] - decoded['leptons'])**2, dim=(1, 2))
            scores['met'] = torch.mean((event_dict['met'] - decoded['met'])**2, dim=1)
            
            # Score total
            scores['total'] = scores['jets'] + scores['leptons'] + scores['met']
            
            return scores

hep_tt_ae = HEPTensorAutoencoder(n_jets=4, n_leptons=2)

print(f"\nHEP Tensor Autoencoder:")
print(f"  Input size: {hep_tt_ae.input_size}")
print(f"  Paramètres: {sum(p.numel() for p in hep_tt_ae.parameters()):,}")
```

---

## Exercices

### Exercice 20.4.1
Implémentez un autoencodeur Tensor Train complet avec vraie contraction TT pour encoder/décoder.

### Exercice 20.4.2
Comparez performance d'un autoencodeur TT vs autoencodeur standard sur données HEP simulées.

### Exercice 20.4.3
Analysez l'impact des bond dimensions sur capacité de détection d'anomalies.

### Exercice 20.4.4
Développez un modèle MPS qui modélise distribution jointe de features discrétisées et utilise pour détection d'anomalies.

---

## Points Clés à Retenir

> 📌 **Les réseaux de tenseurs offrent compression et efficacité computationnelle**

> 📌 **Les autoencoders TT peuvent encoder/décoder avec moins de paramètres**

> 📌 **Les MPS peuvent modéliser distributions jointes de features discrétisées**

> 📌 **La structure tensorielle capture corrélations multi-variables importantes**

> 📌 **L'efficacité permet déploiement sur FPGA pour triggers**

> 📌 **L'interprétabilité via bond dimensions révèle complexité nécessaire**

---

*Section précédente : [20.3 Méthodes Non Supervisées](./20_03_Non_Supervise.md) | Section suivante : [20.5 Quantification de l'Incertitude](./20_05_Incertitude.md)*

