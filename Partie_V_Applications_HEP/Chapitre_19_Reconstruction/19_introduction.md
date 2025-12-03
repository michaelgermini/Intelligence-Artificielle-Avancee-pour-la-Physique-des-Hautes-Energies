# Chapitre 19 : Reconstruction et Identification de Particules

---

## Introduction

La **reconstruction** consiste à reconstruire les propriétés des particules à partir des signaux bruts du détecteur. L'IA révolutionne cette étape en améliorant la précision et la vitesse d'identification.

---

## Plan du Chapitre

1. [Reconstruction de Traces](./19_01_Traces.md)
2. [Identification de Jets](./19_02_Jets.md)
3. [Tagging de Saveurs](./19_03_Tagging.md)
4. [Identification de Leptons](./19_04_Leptons.md)
5. [Reconstruction de l'Énergie Manquante](./19_05_MET.md)

---

## Vue d'Ensemble du Processus

```
┌─────────────────────────────────────────────────────────────────┐
│              Chaîne de Reconstruction                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Signaux Bruts                                                  │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────┐                                          │
│  │  Calibration     │  Correction des réponses                │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │  Reconstruction  │  Traces, clusters, jets                │
│  │  de Base         │                                         │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │  Identification  │  Type de particule (ML)                │
│  │  ML              │                                         │
│  └────────┬─────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌──────────────────┐                                          │
│  │  Reconstruction  │  Propriétés cinématiques                │
│  │  Finale          │                                         │
│  └──────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reconstruction de Traces avec GNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrackReconstructionGNN(nn.Module):
    """
    Reconstruction de traces avec Graph Neural Network
    
    Les hits forment les nœuds, les arêtes potentielles connectent
    les hits compatibles
    """
    
    def __init__(self, hit_features=3, hidden_dim=64):
        super().__init__()
        
        # Encodeur de hits
        self.hit_encoder = nn.Linear(hit_features, hidden_dim)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(3)
        ])
        
        # Classificateur d'arêtes (cette arête est vraie ?)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hits, edge_index):
        """
        Args:
            hits: (n_hits, hit_features) coordonnées des hits
            edge_index: (2, n_edges) indices des paires de hits
        """
        # Encode les hits
        x = self.hit_encoder(hits)
        
        # Message passing
        for mp in self.mp_layers:
            x = mp(x, edge_index)
        
        # Prédiction des arêtes
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_scores = self.edge_classifier(edge_features).squeeze()
        
        return edge_scores

class MessagePassingLayer(nn.Module):
    """Couche de message passing simple"""
    
    def __init__(self, dim):
        super().__init__()
        self.message_net = nn.Linear(2 * dim, dim)
        self.update_net = nn.Linear(2 * dim, dim)
        
    def forward(self, x, edge_index):
        src, dst = edge_index
        
        # Messages
        messages = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.message_net(messages)
        
        # Agrégation (simplifiée: somme)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        
        # Mise à jour
        updated = torch.cat([x, aggregated], dim=-1)
        updated = self.update_net(updated)
        
        return F.relu(updated)
```

---

## Jet Tagging

```python
class JetTagger(nn.Module):
    """
    Classification de jets: quark vs gluon, b-tagging, etc.
    """
    
    def __init__(self, n_features=16, n_classes=5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, n_classes)
        )
        
    def forward(self, jet_features):
        """
        Args:
            jet_features: (batch, n_features)
                features = [pt, eta, phi, mass, multiplicity, width, ...]
        """
        return self.classifier(jet_features)

# Features typiques pour jet tagging
jet_feature_names = [
    'pt', 'eta', 'phi', 'mass',           # Cinématique
    'multiplicity',                        # Nombre de constituants
    'width', 'pt_D',                       # Largeur
    'LHA', 'thrust',                       # Forme
    'C1', 'C2',                           # N-subjettiness
    'charged_multiplicity',                # Multiplicité chargée
    'neutral_fraction',                    # Fraction neutre
    'b_score', 'c_score'                   # Scores de tagging existants
]

print(f"Features de jet: {len(jet_feature_names)}")
for i, name in enumerate(jet_feature_names[:5]):
    print(f"  {i+1}. {name}")
```

---

## B-Tagging

```python
class BTagger(nn.Module):
    """
    B-tagger: identifie les jets provenant de quarks b
    
    Critique pour la physique du top et du Higgs
    """
    
    def __init__(self, input_dim=50):
        super().__init__()
        
        # Utilise les features de vertex secondaire
        self.vertex_features = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Utilise les features du jet
        self.jet_features = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # [light, charm, bottom]
        )
        
    def forward(self, vertex_features, jet_features):
        v = self.vertex_features(vertex_features)
        j = self.jet_features(jet_features)
        
        combined = torch.cat([v, j], dim=-1)
        return self.classifier(combined)

# Utilisation
b_tagger = BTagger()
vertex_feat = torch.randn(32, 50)  # Features de vertex
jet_feat = torch.randn(32, 16)     # Features de jet

output = b_tagger(vertex_feat, jet_feat)
probs = F.softmax(output, dim=-1)
print(f"Probabilités: light={probs[:, 0].mean():.2%}, "
      f"charm={probs[:, 1].mean():.2%}, "
      f"bottom={probs[:, 2].mean():.2%}")
```

---

## Points Clés à Retenir

> 📌 **Les GNN sont particulièrement adaptés à la reconstruction de traces**

> 📌 **Le jet tagging bénéficie grandement des techniques de deep learning**

> 📌 **Le b-tagging est crucial pour la physique du boson de Higgs**

> 📌 **L'IA améliore significativement la précision de reconstruction**

---

*Section suivante : [19.1 Reconstruction de Traces](./19_01_Traces.md)*

