# 1.3 Défis du Traitement en Temps Réel

---

## Introduction

Le traitement en temps réel est le défi central de l'acquisition de données au LHC. Avec 40 millions de croisements de faisceaux par seconde, il est physiquement impossible de stocker toutes les données. Un système de **trigger** (déclenchement) intelligent doit décider en quelques microsecondes quels événements méritent d'être conservés.

---

## Le Problème Fondamental

### Contrainte de Bande Passante

```
┌─────────────────────────────────────────────────────────────────┐
│                  Le Goulot d'Étranglement                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENTRÉE                           SORTIE                        │
│  ══════                           ══════                        │
│  40 MHz × 1.5 MB = 60 PB/s   →    ~1 GB/s (stockage)           │
│                                                                 │
│  Facteur de réduction nécessaire : ~60 000 000                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Physique Rare vs Bruit de Fond

La plupart des collisions produisent des processus "ordinaires" :

```python
# Sections efficaces typiques au LHC (13 TeV)
cross_sections = {
    'Total inélastique': 80e-3,      # 80 mb (millibarns)
    'Production de jets': 1e-3,       # 1 mb
    'Production W': 200e-9,           # 200 nb (nanobarns)
    'Production Z': 60e-9,            # 60 nb
    'Production top': 800e-12,        # 800 pb (picobarns)
    'Production Higgs': 50e-12,       # 50 pb
    'Nouvelle physique (SUSY)': 1e-15 # ~1 fb (femtobarn) ou moins
}

# Taux de production à L = 2×10³⁴ cm⁻² s⁻¹
luminosity = 2e34  # cm⁻² s⁻¹

print("Processus            | Section eff. | Taux (Hz)")
print("-" * 55)
for process, sigma in cross_sections.items():
    # Conversion : 1 barn = 10⁻²⁴ cm²
    sigma_cm2 = sigma * 1e-24
    rate = luminosity * sigma_cm2
    print(f"{process:20} | {sigma:10.2e} b | {rate:10.2e}")
```

Output:
```
Processus            | Section eff. | Taux (Hz)
-------------------------------------------------------
Total inélastique    |   8.00e-02 b |   1.60e+09
Production de jets   |   1.00e-03 b |   2.00e+07
Production W         |   2.00e-07 b |   4.00e+03
Production Z         |   6.00e-08 b |   1.20e+03
Production top       |   8.00e-10 b |   1.60e+01
Production Higgs     |   5.00e-11 b |   1.00e+00
Nouvelle physique    |   1.00e-15 b |   2.00e-05
```

**Conclusion** : Un boson de Higgs est produit environ une fois par seconde, noyé dans 1.6 milliard de collisions inélastiques !

---

## Architecture du Système de Trigger

### Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    Système de Trigger ATLAS                     │
└─────────────────────────────────────────────────────────────────┘

    Collisions
        │
        │ 40 MHz (25 ns entre croisements)
        ▼
┌───────────────────┐
│    Level-1 (L1)   │ ◄── Hardware (FPGA, ASIC)
│   Latence: 2.5 μs │     Granularité réduite
│   Décision: 100ns │     Calorimètres + Muons
└─────────┬─────────┘
          │
          │ ~100 kHz (réduction ×400)
          ▼
┌───────────────────┐
│  High-Level (HLT) │ ◄── Software (CPU/GPU farm)
│  Latence: ~200 ms │     Reconstruction complète
│                   │     Algorithmes complexes
└─────────┬─────────┘
          │
          │ ~1-3 kHz (réduction ×100)
          ▼
    ┌───────────┐
    │ Stockage  │
    │ ~1 GB/s   │
    └───────────┘
```

### Level-1 Trigger : Contraintes Extrêmes

Le L1 doit prendre des décisions en **2.5 microsecondes** :

```python
class Level1Constraints:
    """Contraintes du trigger Level-1"""
    
    # Timing
    BUNCH_CROSSING_PERIOD = 25e-9      # 25 ns
    TOTAL_LATENCY = 2.5e-6             # 2.5 μs
    PIPELINE_DEPTH = 100               # 100 bunch crossings
    
    # Bande passante
    INPUT_RATE = 40e6                  # 40 MHz
    OUTPUT_RATE = 100e3                # 100 kHz max
    REDUCTION_FACTOR = INPUT_RATE / OUTPUT_RATE  # 400
    
    # Hardware
    TECHNOLOGY = "FPGA + ASIC"
    CLOCK_FREQ = 40e6                  # Synchrone avec LHC
    
    @classmethod
    def available_clock_cycles(cls):
        """Nombre de cycles disponibles pour la décision"""
        return int(cls.TOTAL_LATENCY * cls.CLOCK_FREQ)

print(f"Cycles disponibles: {Level1Constraints.available_clock_cycles()}")
# Output: Cycles disponibles: 100
```

### Algorithmes L1 Typiques

```python
# Pseudo-code d'un algorithme L1 simplifié
def l1_electron_trigger(calo_towers, threshold_et=20):
    """
    Trigger L1 pour électrons/photons
    
    Args:
        calo_towers: Grille de tours calorimétriques (η × φ)
        threshold_et: Seuil en énergie transverse (GeV)
    
    Returns:
        Liste des candidats électron/photon
    """
    candidates = []
    
    # Fenêtre glissante 2×2 tours
    for i in range(len(calo_towers) - 1):
        for j in range(len(calo_towers[0]) - 1):
            # Somme 2×2
            et_sum = (calo_towers[i][j] + calo_towers[i+1][j] +
                     calo_towers[i][j+1] + calo_towers[i+1][j+1])
            
            # Maximum local ?
            if et_sum > threshold_et and is_local_maximum(i, j, calo_towers):
                candidates.append({
                    'eta': get_eta(i),
                    'phi': get_phi(j),
                    'et': et_sum
                })
    
    return candidates

def l1_decision(electrons, muons, jets, met):
    """
    Décision finale L1 basée sur un menu de triggers
    """
    # Menu simplifié
    triggers = {
        'single_electron_25': len([e for e in electrons if e['et'] > 25]) >= 1,
        'single_muon_20': len([m for m in muons if m['pt'] > 20]) >= 1,
        'dijet_100': len([j for j in jets if j['et'] > 100]) >= 2,
        'met_50': met > 50,
    }
    
    # L'événement passe si au moins un trigger est satisfait
    return any(triggers.values())
```

---

## High-Level Trigger (HLT)

### Architecture Software

Le HLT dispose de plus de temps et de ressources :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ferme de Calcul HLT                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
│  │ Node 1  │  │ Node 2  │  │ Node 3  │  ...  │ Node N  │       │
│  │ 64 CPU  │  │ 64 CPU  │  │ 64 CPU  │       │ 64 CPU  │       │
│  │ 8 GPU   │  │ 8 GPU   │  │ 8 GPU   │       │ 8 GPU   │       │
│  └─────────┘  └─────────┘  └─────────┘       └─────────┘       │
│                                                                 │
│  Total ATLAS HLT: ~80,000 CPU cores, ~500 GPUs                 │
│  Latence moyenne: ~200 ms                                       │
│  Débit: ~100 kHz input → ~1-3 kHz output                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Reconstruction au HLT

```python
class HLTReconstruction:
    """Pipeline de reconstruction HLT simplifié"""
    
    def __init__(self, event_data):
        self.raw_data = event_data
        self.tracks = []
        self.vertices = []
        self.electrons = []
        self.muons = []
        self.jets = []
        
    def run_tracking(self):
        """Reconstruction des traces (étape la plus coûteuse)"""
        # Algorithme de tracking rapide
        # Utilise les seeds du L1 pour guider la reconstruction
        self.tracks = fast_track_reconstruction(
            self.raw_data['inner_detector'],
            regions_of_interest=self.raw_data['l1_rois']
        )
        
    def run_vertexing(self):
        """Reconstruction des vertex primaires"""
        self.vertices = vertex_finding(self.tracks)
        
    def run_electron_id(self):
        """Identification des électrons"""
        for track in self.tracks:
            if matches_calorimeter_cluster(track):
                if passes_electron_id(track):
                    self.electrons.append(build_electron(track))
                    
    def run_jet_finding(self):
        """Reconstruction des jets avec anti-kt"""
        self.jets = anti_kt_algorithm(
            self.raw_data['calorimeter'],
            R=0.4
        )
        
    def evaluate_triggers(self):
        """Évaluation des triggers HLT"""
        results = {}
        
        # Exemple de triggers HLT
        results['HLT_e26_tight'] = any(
            e.pt > 26 and e.passes_tight_id 
            for e in self.electrons
        )
        
        results['HLT_mu24_iloose'] = any(
            m.pt > 24 and m.is_isolated 
            for m in self.muons
        )
        
        return results
```

---

## Contraintes de Latence

### Budget Temporel

```
┌─────────────────────────────────────────────────────────────────┐
│              Budget de Latence Level-1 (ATLAS)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Composante                          │  Latence                 │
│  ─────────────────────────────────────────────────────────────  │
│  Temps de vol (détecteur → électr.)  │  ~100 ns                │
│  Câbles et fibres optiques           │  ~500 ns                │
│  Traitement calorimètre              │  ~400 ns                │
│  Traitement muons                    │  ~400 ns                │
│  Processeur central de trigger       │  ~200 ns                │
│  Retour de décision                  │  ~400 ns                │
│  ─────────────────────────────────────────────────────────────  │
│  TOTAL                               │  ~2.5 μs               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline et Buffers

Pour gérer la latence, les données sont stockées dans des **pipelines** :

```python
class TriggerPipeline:
    """Simulation d'un pipeline de trigger"""
    
    def __init__(self, depth=100):
        self.depth = depth  # Profondeur en bunch crossings
        self.buffer = [None] * depth
        self.write_ptr = 0
        self.decisions = {}
        
    def push_event(self, event_id, data):
        """Ajoute un événement au pipeline"""
        self.buffer[self.write_ptr] = {
            'id': event_id,
            'data': data,
            'timestamp': time.time()
        }
        self.write_ptr = (self.write_ptr + 1) % self.depth
        
    def get_event_for_decision(self, latency_bc=100):
        """Récupère l'événement prêt pour décision"""
        read_ptr = (self.write_ptr - latency_bc) % self.depth
        return self.buffer[read_ptr]
        
    def apply_decision(self, event_id, accept):
        """Applique la décision de trigger"""
        self.decisions[event_id] = accept
        if accept:
            # Transfère vers le HLT
            self.send_to_hlt(event_id)
```

---

## Rôle de l'IA dans le Trigger

### Pourquoi l'IA ?

Les algorithmes traditionnels atteignent leurs limites :

1. **Complexité croissante** : Plus de pile-up → plus de bruit
2. **Sélectivité** : Besoin de mieux discriminer signal/bruit
3. **Nouveaux signaux** : Recherche de physique non anticipée
4. **Efficacité** : Maximiser l'acceptance pour la physique rare

### Défis de l'IA Temps Réel

```python
# Contraintes pour un modèle ML au Level-1
class L1MLConstraints:
    MAX_LATENCY_NS = 100          # Contribution max à la latence
    MAX_PARAMETERS = 10000        # Limité par les ressources FPGA
    PRECISION = 'int8'            # Quantification nécessaire
    MAX_OPERATIONS = 100000       # FLOPs par inférence
    
    @staticmethod
    def is_model_compatible(model):
        """Vérifie si un modèle respecte les contraintes L1"""
        n_params = count_parameters(model)
        latency = estimate_fpga_latency(model)
        
        return (n_params <= L1MLConstraints.MAX_PARAMETERS and
                latency <= L1MLConstraints.MAX_LATENCY_NS)
```

### Techniques de Déploiement

| Technique | Avantage | Inconvénient |
|-----------|----------|--------------|
| Quantification | Réduit la latence | Perte de précision |
| Pruning | Moins de calculs | Complexité d'entraînement |
| Knowledge Distillation | Modèles compacts | Nécessite un teacher |
| Réseaux de Tenseurs | Compression théorique | Implémentation complexe |

---

## Études de Cas

### 1. Jet Tagging au L1

```python
# Exemple simplifié de b-tagging au L1
import numpy as np

class L1JetTagger:
    """Tagger de jets simplifié pour FPGA"""
    
    def __init__(self):
        # Réseau très petit pour FPGA
        self.weights_1 = np.random.randn(16, 8).astype(np.int8)
        self.weights_2 = np.random.randn(8, 2).astype(np.int8)
        
    def forward(self, jet_features):
        """
        Forward pass quantifié
        
        Args:
            jet_features: [n_constituents, n_tracks, ...]
        """
        # Couche 1 avec activation ReLU
        x = np.maximum(0, jet_features @ self.weights_1)
        
        # Couche 2 (sortie)
        logits = x @ self.weights_2
        
        # Softmax simplifié
        return logits.argmax()
        
    def count_operations(self):
        """Compte les opérations (MACs)"""
        ops_1 = 16 * 8  # Première couche
        ops_2 = 8 * 2   # Deuxième couche
        return ops_1 + ops_2
```

### 2. Anomaly Detection au HLT

```python
class HLTAnomalyDetector:
    """Détecteur d'anomalies pour le HLT"""
    
    def __init__(self, autoencoder_model):
        self.model = autoencoder_model
        self.threshold = None
        
    def set_threshold(self, calibration_data, percentile=99):
        """Calibre le seuil sur des données de référence"""
        reconstruction_errors = []
        for event in calibration_data:
            recon = self.model(event)
            error = np.mean((event - recon)**2)
            reconstruction_errors.append(error)
        
        self.threshold = np.percentile(reconstruction_errors, percentile)
        
    def is_anomalous(self, event):
        """Détecte si un événement est anormal"""
        recon = self.model(event)
        error = np.mean((event - recon)**2)
        return error > self.threshold
```

---

## Exercices

### Exercice 1.3.1
Si le L1 a une latence de 2.5 μs et que les croisements de faisceaux ont lieu toutes les 25 ns, combien d'événements sont "en vol" simultanément dans le système de trigger ?

### Exercice 1.3.2
Un modèle de classification a 10,000 paramètres en float32. Quelle réduction de mémoire obtient-on en le quantifiant en int8 ?

### Exercice 1.3.3
Le HLT traite 100 kHz d'événements avec une ferme de 80,000 cœurs. Quel est le temps CPU moyen disponible par événement ?

---

## Points Clés à Retenir

> 📌 **Le système de trigger doit réduire le débit de 40 MHz à ~1 kHz**

> 📌 **Le Level-1 dispose de seulement 2.5 μs pour décider**

> 📌 **L'IA permet d'améliorer la sélectivité mais doit respecter des contraintes strictes**

> 📌 **La quantification et la compression sont essentielles pour le déploiement temps réel**

---

## Références

1. ATLAS Collaboration. "The ATLAS Trigger System." JINST 15 (2020) P10004
2. CMS Collaboration. "The CMS Trigger System." JINST 12 (2017) P01020
3. Duarte, J. et al. "Fast inference of deep neural networks in FPGAs for particle physics." JINST 13 (2018) P07027

---

*Section suivante : [1.4 Rôle de l'IA dans la Recherche Fondamentale](./01_04_Role_IA.md)*

