# Chapitre 20 : Détection d'Anomalies et Nouvelle Physique

---

## Introduction

La **détection d'anomalies** est devenue un paradigme central dans la recherche de nouvelle physique au LHC. Au lieu de chercher des signaux spécifiques prédits par des modèles théoriques, cette approche cherche à identifier des événements "anormaux" qui pourraient révéler des processus inconnus ou des particules non découvertes.

Le machine learning, en particulier les méthodes non supervisées comme les autoencoders et les réseaux de tenseurs, joue un rôle crucial dans cette quête.

---

## Plan du Chapitre

1. [Recherche de Nouvelle Physique au LHC](./20_01_Nouvelle_Physique.md)
2. [Autoencoders pour la Détection d'Anomalies](./20_02_Autoencoders.md)
3. [Méthodes Non Supervisées](./20_03_Non_Supervise.md)
4. [Réseaux de Tenseurs pour la Détection d'Anomalies](./20_04_Tenseurs.md)
5. [Quantification de l'Incertitude](./20_05_Incertitude.md)

---

## Paradigme de Détection d'Anomalies

### Approche Classique vs Anomaly Detection

```python
import numpy as np
import torch
import torch.nn as nn

class AnomalyDetectionParadigm:
    """
    Comparaison des approches de recherche
    """
    
    def __init__(self):
        self.approaches = {
            'supervised_search': {
                'description': 'Recherche guidée par modèle théorique',
                'process': '1. Modèle prédit signal → 2. Chercher signal spécifique → 3. Test statistique',
                'advantages': ['Bien défini', 'Précis', 'Interprétable'],
                'disadvantages': ['Biaisé vers modèles connus', 'Manque signaux inattendus'],
                'example': 'Recherche du Higgs (prédit par SM)'
            },
            'anomaly_detection': {
                'description': 'Recherche d\'événements anormaux',
                'process': '1. Apprendre distribution background → 2. Identifier outliers → 3. Analyser anomalies',
                'advantages': ['Sans biais théorique', 'Découvre inattendu', 'Exploratoire'],
                'disadvantages': ['Difficile à interpréter', 'Nombreux faux positifs', 'Validation complexe'],
                'example': 'Variational Autoencoder (VAE) pour anomalies'
            }
        }
    
    def display_comparison(self):
        """Affiche la comparaison"""
        print("\n" + "="*70)
        print("Paradigmes de Recherche")
        print("="*70)
        
        for approach, info in self.approaches.items():
            print(f"\n{approach.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Process: {info['process']}")
            print(f"  Avantages:")
            for adv in info['advantages']:
                print(f"    + {adv}")
            print(f"  Inconvénients:")
            for disadv in info['disadvantages']:
                print(f"    - {disadv}")
            print(f"  Exemple: {info['example']}")

paradigm = AnomalyDetectionParadigm()
paradigm.display_comparison()
```

---

## Workflow de Détection d'Anomalies

### Pipeline Complet

```
┌─────────────────────────────────────────────────────────────────┐
│         Pipeline de Détection d'Anomalies                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Données Background (SM)                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────┐                                  │
│  │  Apprentissage           │                                  │
│  │  (Autoencoder, etc.)     │                                  │
│  └──────────┬───────────────┘                                  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────┐                                  │
│  │  Score d'Anomalie        │                                  │
│  │  (Reconstruction error)   │                                  │
│  └──────────┬───────────────┘                                  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────┐                                  │
│  │  Sélection Anomalies     │                                  │
│  │  (Outliers)               │                                  │
│  └──────────┬───────────────┘                                  │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────────┐                                  │
│  │  Analyse Physique        │                                  │
│  │  (Interprétation)         │                                  │
│  └──────────────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Types d'Anomalies

### Classification

```python
class AnomalyTypes:
    """
    Types d'anomalies en physique des particules
    """
    
    def __init__(self):
        self.types = {
            'local_anomalies': {
                'description': 'Événements individuels anormaux',
                'example': 'Événement avec distribution énergétique inhabituelle',
                'detection': 'Score d\'anomalie élevé pour événement spécifique'
            },
            'collective_anomalies': {
                'description': 'Patterns dans ensemble d\'événements',
                'example': 'Excès dans région spécifique de l\'espace des phases',
                'detection': 'Densité anormale dans certaines régions'
            },
            'temporal_anomalies': {
                'description': 'Évolutions temporelles anormales',
                'example': 'Changement dans distribution de données au fil du temps',
                'detection': 'Dérive dans distribution'
            },
            'distributional_anomalies': {
                'description': 'Changements dans distribution globale',
                'example': 'Distribution différente de celle attendue du SM',
                'detection': 'Divergence entre distributions observée/attendue'
            }
        }
    
    def display_types(self):
        """Affiche les types"""
        print("\n" + "="*70)
        print("Types d'Anomalies")
        print("="*70)
        
        for anom_type, info in self.types.items():
            print(f"\n{anom_type.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Exemple: {info['example']}")
            print(f"  Détection: {info['detection']}")

anomaly_types = AnomalyTypes()
anomaly_types.display_types()
```

---

## Défis de la Détection d'Anomalies

### Problèmes Spécifiques HEP

```python
class AnomalyDetectionChallenges:
    """
    Défis spécifiques à la détection d'anomalies en HEP
    """
    
    def __init__(self):
        self.challenges = {
            'high_dimensionality': {
                'description': 'Espaces de features de très haute dimension',
                'impact': 'Curse of dimensionality, difficulté d\'apprentissage',
                'solution': 'Dimensionality reduction, autoencoders'
            },
            'imbalanced_data': {
                'description': 'Background énorme vs signal potentiel rare',
                'impact': 'Difficile d\'apprendre caractéristiques signal',
                'solution': 'Méthodes non supervisées, pas besoin de labels signal'
            },
            'systematic_uncertainties': {
                'description': 'Incertitudes systématiques importantes',
                'impact': 'Anomalies peuvent être artefacts expérimentaux',
                'solution': 'Modélisation incertitudes, validation rigoureuse'
            },
            'interpretability': {
                'description': 'Interprétation physique des anomalies',
                'impact': 'Difficile de comprendre pourquoi événement est anormal',
                'solution': 'Visualisation, features importantes, analyse physique'
            },
            'validation': {
                'description': 'Validation sans connaissance du vrai signal',
                'impact': 'Comment savoir si anomalies sont réelles ?',
                'solution': 'Tests sur données connues, études de robustesse'
            }
        }
    
    def display_challenges(self):
        """Affiche les défis"""
        print("\n" + "="*70)
        print("Défis de la Détection d'Anomalies en HEP")
        print("="*70)
        
        for challenge, info in self.challenges.items():
            print(f"\n{challenge.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Impact: {info['impact']}")
            print(f"  Solution: {info['solution']}")

challenges = AnomalyDetectionChallenges()
challenges.display_challenges()
```

---

## Exemple Simple d'Autoencoder

### Démonstration Basique

```python
class SimpleAutoencoder(nn.Module):
    """
    Autoencodeur simple pour démonstration
    """
    
    def __init__(self, input_dim=100, latent_dim=20):
        super().__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Si données normalisées [0,1]
        )
    
    def forward(self, x):
        """Encode puis décode"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def compute_anomaly_score(self, x):
        """
        Score d'anomalie = erreur de reconstruction
        
        Plus l'erreur est grande, plus l'événement est anormal
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed)**2, dim=1)
            return error

# Exemple simple
autoencoder = SimpleAutoencoder(input_dim=50, latent_dim=10)

# Simuler données
background_events = torch.randn(1000, 50)  # 1000 événements background
anomaly_events = torch.randn(10, 50) * 2 + 5  # 10 événements anormaux (distribution différente)

# Calculer scores
bg_scores = autoencoder.compute_anomaly_score(background_events)
anom_scores = autoencoder.compute_anomaly_score(anomaly_events)

print(f"\nScores d'Anomalie (exemple):")
print(f"  Background: mean={bg_scores.mean():.4f}, std={bg_scores.std():.4f}")
print(f"  Anomalies: mean={anom_scores.mean():.4f}, std={anom_scores.std():.4f}")
```

---

## Exercices

### Exercice 20.0.1
Créez un autoencodeur simple et testez-le sur des données avec distribution normale et des anomalies simulées.

### Exercice 20.0.2
Analysez l'impact de la dimension de l'espace latent sur la capacité de détection d'anomalies.

---

## Points Clés à Retenir

> 📌 **La détection d'anomalies permet recherche sans biais théorique**

> 📌 **Les autoencoders apprennent distribution background et identifient outliers**

> 📌 **Les méthodes non supervisées sont cruciales (pas de labels signal)**

> 📌 **La validation est complexe car vraie nature des anomalies inconnue**

> 📌 **L'interprétabilité est importante pour comprendre anomalies trouvées**

---

*Section suivante : [20.1 Recherche de Nouvelle Physique](./20_01_Nouvelle_Physique.md)*

