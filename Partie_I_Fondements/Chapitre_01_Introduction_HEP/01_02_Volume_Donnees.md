# 1.2 Volume et Vélocité des Données en Physique des Particules

---

## Introduction

La physique des hautes énergies est entrée dans l'ère du **Big Data** bien avant que ce terme ne devienne populaire. Les expériences du LHC génèrent des volumes de données qui défient l'imagination et nécessitent des infrastructures de calcul distribuées à l'échelle mondiale.

---

## Les Quatre V du Big Data en HEP

### Volume

```
┌────────────────────────────────────────────────────────────────┐
│                    Hiérarchie des Données LHC                  │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Données brutes (avant trigger)     │  ~60 PB/s               │
│           ↓ Trigger L1              │                         │
│  Après L1 (~100 kHz)                │  ~150 GB/s              │
│           ↓ HLT                     │                         │
│  Données enregistrées               │  ~1-2 GB/s              │
│           ↓ Reconstruction          │                         │
│  Données analysables                │  ~100 PB/an             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

#### Statistiques de Stockage (2023)

| Expérience | Données brutes/an | Données reconstruites/an |
|------------|-------------------|-------------------------|
| ATLAS | ~50 PB | ~100 PB |
| CMS | ~50 PB | ~100 PB |
| ALICE | ~30 PB | ~50 PB |
| LHCb | ~20 PB | ~40 PB |

### Vélocité

La vélocité des données est dictée par la physique du LHC :

```python
# Calcul du débit de données
class LHCDataRate:
    BUNCH_CROSSING_FREQ = 40e6  # 40 MHz
    AVG_PILEUP = 50  # collisions par croisement
    RAW_EVENT_SIZE = 1.5e6  # bytes
    
    @classmethod
    def raw_data_rate(cls):
        """Débit brut théorique en bytes/s"""
        return cls.BUNCH_CROSSING_FREQ * cls.RAW_EVENT_SIZE
    
    @classmethod
    def collisions_per_second(cls):
        """Nombre de collisions par seconde"""
        return cls.BUNCH_CROSSING_FREQ * cls.AVG_PILEUP

# Résultats
print(f"Débit brut: {LHCDataRate.raw_data_rate() / 1e15:.1f} PB/s")
print(f"Collisions/s: {LHCDataRate.collisions_per_second():.2e}")
```

Output:
```
Débit brut: 60.0 PB/s
Collisions/s: 2.00e+09
```

### Variété

Les données HEP présentent une grande diversité :

1. **Données de détecteur** : Signaux électroniques bruts
2. **Données reconstruites** : Traces, clusters, jets
3. **Données simulées** : Monte Carlo
4. **Métadonnées** : Conditions de prise de données
5. **Données dérivées** : Formats d'analyse (AOD, NANO)

### Véracité

La qualité des données est critique :

- **Calibration** : Correction des réponses des détecteurs
- **Alignement** : Positionnement précis des sous-détecteurs
- **Data Quality** : Validation de chaque run de données

---

## Le Worldwide LHC Computing Grid (WLCG)

### Architecture Hiérarchique

Le WLCG est organisé en niveaux (Tiers) :

```
                        ┌─────────────┐
                        │   Tier-0    │
                        │    CERN     │
                        │  ~200 PB    │
                        └──────┬──────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
     ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
     │  Tier-1   │       │  Tier-1   │       │  Tier-1   │
     │   (13)    │       │   (13)    │       │   (13)    │
     │  ~50 PB   │       │  ~50 PB   │       │  ~50 PB   │
     └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
           │                   │                   │
     ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
     │  Tier-2   │       │  Tier-2   │       │  Tier-2   │
     │  (~160)   │       │  (~160)   │       │  (~160)   │
     └───────────┘       └───────────┘       └───────────┘
```

### Rôles des Différents Tiers

| Tier | Localisation | Rôle Principal | Capacité Typique |
|------|--------------|----------------|------------------|
| 0 | CERN | Reconstruction primaire, archivage | ~200 PB stockage |
| 1 | 13 centres nationaux | Reprocessing, stockage permanent | ~50 PB chacun |
| 2 | ~160 centres | Simulation, analyse utilisateur | ~5-20 PB chacun |
| 3 | Universités | Analyse locale | Variable |

### Capacité Totale du WLCG

```
┌─────────────────────────────────────────────────────────────┐
│              Ressources WLCG (2023)                         │
├─────────────────────────────────────────────────────────────┤
│  • Stockage total : > 1 Exabyte                            │
│  • Puissance de calcul : > 1 million de cœurs CPU          │
│  • Sites : > 170 dans 42 pays                              │
│  • Transfert de données : ~50 GB/s en continu              │
│  • Jobs par jour : > 2 millions                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Formats de Données

### Hiérarchie des Formats

```
RAW (Données brutes du détecteur)
    │
    ▼ Reconstruction
ESD/AOD (Event Summary Data / Analysis Object Data)
    │
    ▼ Dérivation
DAOD (Derived AOD - formats spécialisés)
    │
    ▼ Réduction finale
NANO/MINI (Formats compacts pour analyse)
```

### Exemple de Structure de Données

```python
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Particle:
    """Représentation d'une particule reconstruite"""
    pt: float      # Impulsion transverse (GeV)
    eta: float     # Pseudo-rapidité
    phi: float     # Angle azimuthal
    mass: float    # Masse (GeV)
    pdg_id: int    # Identifiant de particule
    charge: int    # Charge électrique

@dataclass
class Event:
    """Structure d'un événement de collision"""
    run_number: int
    event_number: int
    lumi_block: int
    
    # Collections de particules
    electrons: List[Particle]
    muons: List[Particle]
    photons: List[Particle]
    jets: List[Particle]
    
    # Variables globales
    met: float           # Énergie transverse manquante
    met_phi: float       # Direction du MET
    n_vertices: int      # Nombre de vertex primaires
    
    def total_particles(self) -> int:
        return (len(self.electrons) + len(self.muons) + 
                len(self.photons) + len(self.jets))

# Taille typique d'un événement en format NANO
# ~1-5 KB par événement (vs ~1.5 MB en RAW)
```

### Compression et Stockage

```python
# Comparaison des tailles de fichiers
data_formats = {
    'RAW': {'size_per_event_kb': 1500, 'compression': 1.0},
    'ESD': {'size_per_event_kb': 500, 'compression': 3.0},
    'AOD': {'size_per_event_kb': 100, 'compression': 15.0},
    'DAOD': {'size_per_event_kb': 20, 'compression': 75.0},
    'NANO': {'size_per_event_kb': 2, 'compression': 750.0},
}

print("Format | Taille/evt | Facteur de compression")
print("-" * 50)
for fmt, info in data_formats.items():
    print(f"{fmt:6} | {info['size_per_event_kb']:6} KB | {info['compression']:6.0f}x")
```

---

## Défis de Gestion des Données

### 1. Stockage à Long Terme

Les données du LHC doivent être préservées pour des décennies :

- **Durée de vie** : Les données du Run 1 (2010-2012) sont toujours analysées
- **Formats évolutifs** : Migration vers de nouveaux formats
- **Accessibilité** : Accès rapide pour la réanalyse

### 2. Transfert de Données

```python
# Calcul du temps de transfert
def transfer_time(data_size_pb, bandwidth_gbps):
    """
    Calcule le temps de transfert en heures
    
    Args:
        data_size_pb: Taille en pétaoctets
        bandwidth_gbps: Bande passante en Gb/s
    """
    data_bits = data_size_pb * 1e15 * 8  # Conversion en bits
    bandwidth_bps = bandwidth_gbps * 1e9
    time_seconds = data_bits / bandwidth_bps
    return time_seconds / 3600  # Conversion en heures

# Exemple : Transférer 1 PB à 100 Gb/s
time_hours = transfer_time(1, 100)
print(f"Temps pour transférer 1 PB à 100 Gb/s: {time_hours:.1f} heures")
# Output: Temps pour transférer 1 PB à 100 Gb/s: 22.2 heures
```

### 3. Traitement Distribué

Les analyses nécessitent un traitement massivement parallèle :

```python
# Exemple simplifié de job distribué
class DistributedAnalysis:
    def __init__(self, n_events, events_per_job=10000):
        self.n_events = n_events
        self.events_per_job = events_per_job
        self.n_jobs = n_events // events_per_job
    
    def estimate_walltime(self, time_per_event_ms, n_cores):
        """Estime le temps total d'analyse"""
        total_time_ms = self.n_events * time_per_event_ms
        walltime_hours = total_time_ms / (n_cores * 1000 * 3600)
        return walltime_hours

# Analyse de 1 milliard d'événements
analysis = DistributedAnalysis(n_events=1e9)
print(f"Nombre de jobs: {analysis.n_jobs:,}")

# Avec 10000 cœurs et 10ms par événement
walltime = analysis.estimate_walltime(time_per_event_ms=10, n_cores=10000)
print(f"Temps estimé: {walltime:.1f} heures")
```

---

## Évolution Future : HL-LHC

Le **High-Luminosity LHC** (prévu pour 2029) multipliera les défis :

```
┌─────────────────────────────────────────────────────────────────┐
│           Comparaison LHC vs HL-LHC                            │
├─────────────────────────────────────────────────────────────────┤
│                        │    LHC (Run 3)    │     HL-LHC        │
│  ──────────────────────┼───────────────────┼───────────────────│
│  Luminosité inst.      │    2×10³⁴         │    5-7.5×10³⁴     │
│  Pile-up moyen         │    ~50            │    ~140-200       │
│  Données/an            │    ~100 PB        │    ~500 PB        │
│  Stockage total prévu  │    ~1 EB          │    ~5 EB          │
└─────────────────────────────────────────────────────────────────┘
```

### Implications pour l'IA

L'augmentation du pile-up nécessite des algorithmes plus sophistiqués :

1. **Reconstruction plus complexe** : Plus de particules à démêler
2. **Meilleure discrimination** : Séparer signal du bruit de fond
3. **Efficacité accrue** : Traiter plus de données avec les mêmes ressources
4. **Compression agressive** : Réduire les besoins de stockage

---

## Exercices

### Exercice 1.2.1
Le WLCG transfère en moyenne 50 GB/s de données. Combien de temps faudrait-il pour transférer l'ensemble des données du Run 2 (~300 PB) à cette vitesse ?

### Exercice 1.2.2
Si le HL-LHC génère 5 fois plus de données que le LHC actuel, mais que le budget de stockage n'augmente que de 50%, quel facteur de compression supplémentaire faut-il atteindre ?

### Exercice 1.2.3
Un format NANO contient en moyenne 2 KB par événement. Combien d'événements peut-on stocker sur un disque de 10 TB ?

---

## Points Clés à Retenir

> 📌 **Le LHC génère ~100 PB de données par an, stockées sur le WLCG**

> 📌 **Le WLCG comprend plus de 170 sites dans 42 pays**

> 📌 **Les données passent par plusieurs formats, de RAW (~1.5 MB) à NANO (~2 KB)**

> 📌 **Le HL-LHC multipliera les défis par 5 à partir de 2029**

---

## Références

1. Bird, I. et al. "Update of the Computing Models of the WLCG and the LHC Experiments." CERN-LHCC-2014-014
2. ATLAS Collaboration. "ATLAS Computing and Data Handling." ATL-SOFT-PUB-2022-001
3. Albrecht, J. et al. "A Roadmap for HEP Software and Computing R&D for the 2020s." Comput Softw Big Sci 3, 7 (2019)

---

*Section suivante : [1.3 Défis du Traitement en Temps Réel](./01_03_Temps_Reel.md)*

