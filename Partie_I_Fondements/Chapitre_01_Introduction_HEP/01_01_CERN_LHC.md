# 1.1 Le CERN et le Large Hadron Collider (LHC)

---

## Le CERN : Laboratoire Européen pour la Physique des Particules

### Présentation Générale

Le **CERN** (Organisation Européenne pour la Recherche Nucléaire) est le plus grand laboratoire de physique des particules au monde. Situé à la frontière franco-suisse près de Genève, il rassemble plus de 17 000 scientifiques de plus de 110 nationalités.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CERN en Chiffres                        │
├─────────────────────────────────────────────────────────────────┤
│  • Fondation : 1954                                             │
│  • États membres : 23                                           │
│  • Personnel : ~2 500 employés                                  │
│  • Utilisateurs : ~17 000 scientifiques                         │
│  • Budget annuel : ~1.2 milliard CHF                           │
│  • Circonférence du LHC : 27 km                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mission et Objectifs

Le CERN poursuit plusieurs objectifs fondamentaux :

1. **Recherche Fondamentale** : Comprendre les lois fondamentales de l'univers
2. **Développement Technologique** : Repousser les limites de la technologie
3. **Formation** : Former la prochaine génération de scientifiques
4. **Collaboration Internationale** : Unir les nations autour de la science

---

## Le Large Hadron Collider (LHC)

### Architecture du Collisionneur

Le **LHC** est l'accélérateur de particules le plus puissant jamais construit. Il accélère des protons (ou des ions lourds) à des vitesses proches de celle de la lumière, puis les fait entrer en collision.

```
                    ┌──────────────────────────────────────┐
                    │          Vue Schématique du LHC      │
                    └──────────────────────────────────────┘
                    
                              ATLAS    
                                ●
                               /│\
                              / │ \
                             /  │  \
                    ALICE ●─────┼─────● CMS
                             \  │  /
                              \ │ /
                               \│/
                                ●
                              LHCb
                              
                    ───────── 27 km de circonférence ─────────
```

### Caractéristiques Techniques

| Paramètre | Valeur |
|-----------|--------|
| Circonférence | 26.7 km |
| Énergie par faisceau | 6.5 TeV (Run 2) → 6.8 TeV (Run 3) |
| Énergie de collision | 13-13.6 TeV |
| Nombre de bunches | ~2 800 par faisceau |
| Protons par bunch | ~10¹¹ |
| Fréquence de croisement | 40 MHz |
| Luminosité instantanée | ~2 × 10³⁴ cm⁻² s⁻¹ |

### Le Concept de Luminosité

La **luminosité** est une mesure cruciale qui quantifie le taux de collisions :

$$\mathcal{L} = \frac{N_1 N_2 f n_b}{4\pi \sigma_x \sigma_y}$$

Où :
- $N_1, N_2$ : nombre de particules par bunch
- $f$ : fréquence de révolution
- $n_b$ : nombre de bunches
- $\sigma_x, \sigma_y$ : tailles transverses du faisceau

La **luminosité intégrée** (en fb⁻¹) représente le "nombre total de collisions" accumulées :

$$L_{int} = \int \mathcal{L}(t) \, dt$$

---

## Les Quatre Grandes Expériences

### ATLAS (A Toroidal LHC ApparatuS)

ATLAS est le plus grand détecteur de particules jamais construit :

- **Dimensions** : 46 m de long, 25 m de diamètre
- **Poids** : 7 000 tonnes
- **Canaux de lecture** : ~100 millions
- **Débit de données** : ~1 Po/s (avant filtrage)

```python
# Structure simplifiée du détecteur ATLAS
class ATLASDetector:
    def __init__(self):
        self.inner_detector = {
            'pixel': {'layers': 4, 'channels': 92_000_000},
            'SCT': {'layers': 4, 'channels': 6_300_000},
            'TRT': {'straws': 350_000}
        }
        self.calorimeters = {
            'electromagnetic': {'cells': 190_000},
            'hadronic': {'cells': 10_000}
        }
        self.muon_spectrometer = {
            'chambers': 1_200,
            'channels': 1_100_000
        }
```

### CMS (Compact Muon Solenoid)

CMS est caractérisé par son puissant solénoïde :

- **Champ magnétique** : 3.8 Tesla
- **Dimensions** : 21 m de long, 15 m de diamètre
- **Poids** : 14 000 tonnes
- **Canaux de lecture** : ~75 millions

### ALICE (A Large Ion Collider Experiment)

ALICE est spécialisé dans l'étude du plasma quark-gluon :

- **Focus** : Collisions d'ions lourds (Pb-Pb)
- **Objectif** : Recréer les conditions de l'univers primordial
- **Spécificité** : Identification de particules à bas momentum

### LHCb (LHC beauty)

LHCb étudie l'asymétrie matière-antimatière :

- **Focus** : Physique des quarks b et c
- **Géométrie** : Spectromètre vers l'avant
- **Objectif** : Comprendre la violation de CP

---

## Chaîne d'Accélération

Les protons passent par plusieurs étapes avant d'atteindre le LHC :

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   LINAC 4   │───▶│   PSB       │───▶│    PS      │
│   160 MeV   │    │   2 GeV     │    │   26 GeV   │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌─────────────────────────────────┐
│    LHC      │◀───│           SPS                   │
│   6.8 TeV   │    │          450 GeV                │
└─────────────┘    └─────────────────────────────────┘
```

### Détails de l'Accélération

1. **Source d'hydrogène** : Production de protons par ionisation
2. **LINAC 4** : Accélération linéaire jusqu'à 160 MeV
3. **PSB** (Proton Synchrotron Booster) : 2 GeV
4. **PS** (Proton Synchrotron) : 26 GeV
5. **SPS** (Super Proton Synchrotron) : 450 GeV
6. **LHC** : Accélération finale jusqu'à 6.8 TeV

---

## Implications pour le Calcul

### Volume de Données Brutes

À chaque croisement de faisceaux (40 millions de fois par seconde) :

```
Fréquence de croisement : 40 MHz
Taille moyenne d'un événement : ~1.5 MB
Débit brut théorique : 40 × 10⁶ × 1.5 × 10⁶ = 60 PB/s
```

Ce débit est **physiquement impossible** à stocker. D'où la nécessité d'un système de **trigger** intelligent.

### Nécessité de l'Intelligence Artificielle

Face à ces volumes, l'IA devient indispensable pour :

1. **Filtrage en temps réel** : Réduire 60 PB/s à ~1 GB/s
2. **Reconstruction rapide** : Identifier les particules en microsecondes
3. **Analyse offline** : Explorer des pétaoctets de données stockées
4. **Simulation** : Accélérer les simulations Monte Carlo

---

## Exercices

### Exercice 1.1.1
Calculez le nombre total de collisions proton-proton par seconde au LHC, sachant que :
- Fréquence de croisement : 40 MHz
- Nombre moyen de collisions par croisement (pile-up) : ~50

### Exercice 1.1.2
Si chaque événement après le trigger Level-1 fait 1.5 MB, et que le taux de sortie du L1 est de 100 kHz, quel est le débit de données à traiter par le High-Level Trigger ?

### Exercice 1.1.3
La luminosité intégrée du Run 2 du LHC était d'environ 140 fb⁻¹. Si la section efficace de production du boson de Higgs est d'environ 50 pb, combien de bosons de Higgs ont été produits ?

---

## Points Clés à Retenir

> 📌 **Le LHC produit des collisions à 40 MHz, générant un flux de données brutes de ~60 PB/s**

> 📌 **Seule une fraction infime (~1/40 000) des événements peut être stockée**

> 📌 **L'IA est essentielle pour le filtrage et l'analyse en temps réel**

> 📌 **Les quatre grandes expériences ont des objectifs complémentaires**

---

## Références

1. CERN. "About CERN." https://home.cern/about
2. ATLAS Collaboration. "The ATLAS Experiment at the CERN Large Hadron Collider." JINST 3 (2008) S08003
3. CMS Collaboration. "The CMS Experiment at the CERN LHC." JINST 3 (2008) S08004
4. Evans, L., Bryant, P. "LHC Machine." JINST 3 (2008) S08001

---

*Section suivante : [1.2 Volume et Vélocité des Données](./01_02_Volume_Donnees.md)*

