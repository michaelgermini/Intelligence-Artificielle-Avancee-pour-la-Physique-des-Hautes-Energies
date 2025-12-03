# 21.1 Simulation Monte Carlo en Physique des Particules

---

## Introduction

La **simulation Monte Carlo** est la méthode standard pour générer des événements simulés en physique des hautes énergies. Elle reproduit fidèlement les processus physiques depuis la collision initiale jusqu'à la réponse du détecteur. Cette section présente les principes de la simulation Monte Carlo, les outils utilisés, et les limitations qui motivent l'utilisation de l'IA.

---

## Principes de la Simulation Monte Carlo

### Génération Stochastique

```python
import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy import stats

class MonteCarloSimulation:
    """
    Principes de base de la simulation Monte Carlo
    """
    
    def __init__(self):
        self.stages = {
            'hard_scattering': {
                'description': 'Collision initiale (partons)',
                'tools': ['Pythia', 'Sherpa', 'MadGraph'],
                'output': 'Partons initiaux'
            },
            'parton_shower': {
                'description': 'Émission de gluons et quarks',
                'tools': ['Pythia shower', 'Herwig'],
                'output': 'Jets de partons'
            },
            'hadronisation': {
                'description': 'Formation de hadrons',
                'tools': ['Pythia', 'Herwig', 'Cluster model'],
                'output': 'Hadrons stables'
            },
            'detector': {
                'description': 'Interactions avec détecteur',
                'tools': ['GEANT4', 'GFlash'],
                'output': 'Signaux détecteur'
            },
            'digitization': {
                'description': 'Conversion en données brutes',
                'tools': ['Detector-specific'],
                'output': 'Données simulées'
            }
        }
    
    def display_stages(self):
        """Affiche les étapes"""
        print("\n" + "="*70)
        print("Étapes de Simulation Monte Carlo")
        print("="*70)
        
        for stage, info in self.stages.items():
            print(f"\n{stage.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Outils: {', '.join(info['tools'])}")
            print(f"  Sortie: {info['output']}")

mc_sim = MonteCarloSimulation()
mc_sim.display_stages()
```

---

## Génération d'Événements Hard Scattering

### Méthodes et Outils

```python
class HardScattering:
    """
    Simulation du processus hard scattering
    """
    
    def __init__(self):
        self.processes = {
            'pp_collision': {
                'description': 'Collision proton-proton',
                'center_of_mass_energy': '13.6 TeV (LHC Run 3)',
                'processes': ['Drell-Yan', 'QCD', 'Top', 'Higgs', 'BSM']
            },
            'matrix_elements': {
                'description': 'Calcul éléments de matrice',
                'method': 'Perturbation theory (LO, NLO, NNLO)',
                'tools': ['MadGraph', 'aMC@NLO', 'Powheg']
            }
        }
    
    def generate_event(self, process='drell_yan', n_events=1000):
        """
        Simule génération d'événement hard scattering
        
        (Simplifié: en pratique utilise Pythia/MadGraph)
        """
        if process == 'drell_yan':
            # Drell-Yan: pp → Z/γ* → ℓ+ℓ-
            # Simuler masse invariante du Z
            m_z = 91.2  # GeV
            width_z = 2.5  # GeV
            
            # Distribution Breit-Wigner pour masse Z
            masses = np.random.normal(m_z, width_z, n_events)
            
            # Angles de diffusion
            cos_theta = np.random.uniform(-1, 1, n_events)
            phi = np.random.uniform(0, 2*np.pi, n_events)
            
            return {
                'm_invariant': masses,
                'cos_theta': cos_theta,
                'phi': phi,
                'process': 'Drell-Yan'
            }
        
        elif process == 'qcd':
            # QCD dijets
            # Distribution en pT
            pT_min = 20  # GeV
            pT_max = 2000  # GeV
            
            # Distribution de pT (approximation)
            log_pT = np.random.uniform(np.log(pT_min), np.log(pT_max), n_events)
            pT = np.exp(log_pT)
            
            return {
                'pT': pT,
                'process': 'QCD dijets'
            }
        
        return None

hard_scatter = HardScattering()

print(f"\nHard Scattering Simulation:")
print(f"  Processus: pp collision @ 13.6 TeV")

# Générer événements Drell-Yan
dy_events = hard_scatter.generate_event('drell_yan', n_events=1000)
print(f"\nÉvénements Drell-Yan générés:")
print(f"  Nombre: {len(dy_events['m_invariant'])}")
print(f"  Masse invariante moyenne: {dy_events['m_invariant'].mean():.2f} GeV")
print(f"  Largeur: {dy_events['m_invariant'].std():.2f} GeV")
```

---

## Parton Shower et Hadronisation

### Évolution des Partons

```python
class PartonShower:
    """
    Simulation du parton shower et hadronisation
    """
    
    def __init__(self):
        self.shower_models = {
            'pythia': {
                'type': 'Ordered angular-ordered',
                'algorithm': 'DGLAP evolution',
                'parameters': ['α_s', 'cutoff scale']
            },
            'herwig': {
                'type': 'Coherent branching',
                'algorithm': 'Angular ordering',
                'parameters': ['Shower scale', 'matching']
            }
        }
        
        self.hadronization_models = {
            'lund_string': {
                'description': 'Modèle de corde de Lund (Pythia)',
                'principle': 'Formation de cordes entre quarks'
            },
            'cluster': {
                'description': 'Modèle de clusters (Herwig)',
                'principle': 'Formation puis décroissance de clusters'
            }
        }
    
    def simulate_shower(self, initial_parton, n_steps=10):
        """
        Simule parton shower
        
        Args:
            initial_parton: {pT, eta, phi, flavor}
        """
        partons = [initial_parton]
        current_parton = initial_parton
        
        for step in range(n_steps):
            # Probabilité d'émission gluon
            p_emission = 0.3  # Probabilité par étape
            
            if np.random.random() < p_emission:
                # Émettre gluon
                new_parton = {
                    'pT': current_parton['pT'] * 0.8,  # Perte d'énergie
                    'eta': current_parton['eta'] + np.random.normal(0, 0.1),
                    'phi': current_parton['phi'] + np.random.normal(0, 0.1),
                    'flavor': 'g'  # gluon
                }
                partons.append(new_parton)
                current_parton = new_parton
        
        return partons
    
    def hadronize(self, partons):
        """
        Simule hadronisation
        
        Transforme partons en hadrons
        """
        hadrons = []
        
        for parton in partons:
            # Simplifié: créer quelques hadrons
            if parton['flavor'] in ['u', 'd']:
                # Créer pion
                hadron = {
                    'type': 'π',
                    'pT': parton['pT'] * np.random.uniform(0.3, 0.7),
                    'eta': parton['eta'],
                    'phi': parton['phi']
                }
                hadrons.append(hadron)
        
        return hadrons

shower = PartonShower()

# Simuler shower
initial_quark = {'pT': 100, 'eta': 1.5, 'phi': 0.5, 'flavor': 'u'}
evolved_partons = shower.simulate_shower(initial_quark, n_steps=5)
hadrons = shower.hadronize(evolved_partons)

print(f"\nParton Shower et Hadronisation:")
print(f"  Partons initiaux: 1")
print(f"  Partons après shower: {len(evolved_partons)}")
print(f"  Hadrons produits: {len(hadrons)}")
```

---

## Simulation du Détecteur avec GEANT4

### Propagation dans le Détecteur

```python
class DetectorSimulation:
    """
    Simulation des interactions avec le détecteur (GEANT4)
    """
    
    def __init__(self):
        self.detector_components = {
            'tracker': {
                'type': 'Tracker (pixels + strips)',
                'material': 'Silicon',
                'purpose': 'Mesure trajectoires'
            },
            'calorimeter_em': {
                'type': 'Calorimètre électromagnétique',
                'material': 'Liquid Argon / Scintillator',
                'purpose': 'Mesure énergie électrons/photons'
            },
            'calorimeter_had': {
                'type': 'Calorimètre hadronique',
                'material': 'Iron + Scintillator',
                'purpose': 'Mesure énergie hadrons'
            },
            'muon_system': {
                'type': 'Détecteurs muons',
                'material': 'Drift tubes / RPC',
                'purpose': 'Identification muons'
            }
        }
    
    def simulate_particle_detector(self, particle, detector_component='tracker'):
        """
        Simule interaction particule avec détecteur
        
        (Simplifié: en pratique utilise GEANT4)
        """
        results = {
            'hits': [],
            'energy_deposit': 0,
            'detected': False
        }
        
        if detector_component == 'tracker':
            # Tracker: hits de position
            if abs(particle['eta']) < 2.5:  # Acceptation tracker
                results['detected'] = True
                # Créer hits le long de la trajectoire
                n_hits = np.random.poisson(10)  # Nombre moyen de hits
                for i in range(n_hits):
                    hit = {
                        'r': np.random.uniform(30, 1200),  # mm
                        'phi': particle['phi'],
                        'z': np.random.uniform(-3000, 3000),
                        'layer': i % 4
                    }
                    results['hits'].append(hit)
        
        elif detector_component == 'calorimeter_em':
            # Calorimètre EM: dépôt d'énergie
            if abs(particle['eta']) < 3.0:
                results['detected'] = True
                # Dépôt d'énergie (simplifié)
                if particle['type'] in ['e', 'γ']:
                    results['energy_deposit'] = particle['energy'] * 0.95  # Efficacité
                else:
                    results['energy_deposit'] = 0  # Pas d'interaction EM
        
        elif detector_component == 'calorimeter_had':
            # Calorimètre hadronique
            if abs(particle['eta']) < 3.0:
                results['detected'] = True
                if particle['type'] in ['π', 'p', 'n']:
                    results['energy_deposit'] = particle['energy'] * 0.6  # Efficacité plus faible
        
        elif detector_component == 'muon_system':
            # Système muon
            if abs(particle['eta']) < 2.4:
                if particle['type'] == 'μ':
                    results['detected'] = True
                    results['hits'] = [{'eta': particle['eta'], 'phi': particle['phi']}]
        
        return results
    
    def simulate_full_detector(self, particles):
        """
        Simule passage dans tous les composants
        """
        full_response = {
            'tracker_hits': [],
            'em_energy': [],
            'had_energy': [],
            'muon_hits': []
        }
        
        for particle in particles:
            # Tracker
            tracker_result = self.simulate_particle_detector(particle, 'tracker')
            if tracker_result['detected']:
                full_response['tracker_hits'].extend(tracker_result['hits'])
            
            # Calorimètre EM
            em_result = self.simulate_particle_detector(particle, 'calorimeter_em')
            if em_result['detected'] and em_result['energy_deposit'] > 0:
                full_response['em_energy'].append(em_result['energy_deposit'])
            
            # Calorimètre hadronique
            had_result = self.simulate_particle_detector(particle, 'calorimeter_had')
            if had_result['detected'] and had_result['energy_deposit'] > 0:
                full_response['had_energy'].append(had_result['energy_deposit'])
            
            # Muons
            muon_result = self.simulate_particle_detector(particle, 'muon_system')
            if muon_result['detected']:
                full_response['muon_hits'].extend(muon_result['hits'])
        
        return full_response

detector = DetectorSimulation()

# Simuler particule dans détecteur
electron = {'type': 'e', 'energy': 50, 'eta': 1.0, 'phi': 0.5, 'pT': 45}
response = detector.simulate_full_detector([electron])

print(f"\nSimulation Détecteur:")
print(f"  Hits tracker: {len(response['tracker_hits'])}")
print(f"  Énergie EM: {sum(response['em_energy']):.2f} GeV")
print(f"  Énergie had: {sum(response['had_energy']):.2f} GeV")
print(f"  Hits muons: {len(response['muon_hits'])}")
```

---

## Coût Computationnel

### Temps et Ressources

```python
class ComputationalCost:
    """
    Analyse du coût computationnel de la simulation MC
    """
    
    def __init__(self):
        self.cost_breakdown = {
            'hard_scattering': {
                'time_per_event': 0.01,  # secondes
                'percentage': 0.1
            },
            'parton_shower': {
                'time_per_event': 0.05,
                'percentage': 0.5
            },
            'hadronisation': {
                'time_per_event': 0.02,
                'percentage': 0.2
            },
            'detector_simulation': {
                'time_per_event': 0.8,  # GEANT4 est très coûteux
                'percentage': 80.0
            },
            'digitization': {
                'time_per_event': 0.1,
                'percentage': 10.0
            }
        }
    
    def compute_total_time(self, n_events):
        """Calcule temps total"""
        total_time_per_event = sum(
            stage['time_per_event'] for stage in self.cost_breakdown.values()
        )
        
        total_time = total_time_per_event * n_events
        
        return {
            'time_per_event': total_time_per_event,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'total_time_days': total_time / (3600 * 24)
        }
    
    def estimate_resources(self, n_events, n_cores=1000):
        """
        Estime ressources nécessaires
        
        Args:
            n_events: Nombre d'événements à simuler
            n_cores: Nombre de cores disponibles
        """
        total_time = self.compute_total_time(n_events)
        
        # Temps avec parallélisation
        parallel_time = total_time['total_time_hours'] / n_cores
        
        # Coût (approximatif: $0.01/heure/core)
        cost = total_time['total_time_hours'] * n_cores * 0.01
        
        return {
            'sequential_time_hours': total_time['total_time_hours'],
            'parallel_time_hours': parallel_time,
            'estimated_cost_usd': cost,
            'n_cores_needed': n_cores
        }

cost_analyzer = ComputationalCost()

# Analyser coût pour différentes tailles
scenarios = {
    'small': 1000000,  # 1M événements
    'medium': 100000000,  # 100M événements
    'large': 10000000000  # 10B événements
}

print(f"\nAnalyse Coût Computationnel:")
print(f"  Temps par événement: {sum(s['time_per_event'] for s in cost_analyzer.cost_breakdown.values()):.2f} s")
print(f"\n{'Scénario':<10} {'Événements':<15} {'Temps (h)':<15} {'Coût ($)':<15}")
print("-" * 55)

for scenario, n_events in scenarios.items():
    resources = cost_analyzer.estimate_resources(n_events, n_cores=1000)
    events_str = f"{n_events/1e6:.1f}M" if n_events < 1e9 else f"{n_events/1e9:.1f}B"
    print(f"{scenario:<10} {events_str:<15} {resources['parallel_time_hours']:<15.1f} {resources['estimated_cost_usd']:<15.0f}")
```

---

## Limitations et Motivation pour l'IA

### Défis de la Simulation MC

```python
class MCLimitations:
    """
    Limitations qui motivent utilisation de l'IA
    """
    
    def __init__(self):
        self.limitations = {
            'speed': {
                'problem': 'Très lent: minutes par événement',
                'impact': 'Limite nombre d\'événements simulés',
                'ia_solution': 'Génération 100-1000× plus rapide'
            },
            'computational_cost': {
                'problem': 'Très coûteux en ressources',
                'impact': 'Budget computationnel limité',
                'ia_solution': 'Coût réduit après entraînement'
            },
            'scalability': {
                'problem': 'Difficile de générer milliards d\'événements',
                'impact': 'Statistiques limitées pour processus rares',
                'ia_solution': 'Génération massive facile'
            },
            'flexibility': {
                'problem': 'Changements de détecteur nécessitent reconfiguration',
                'impact': 'Temps de développement long',
                'ia_solution': 'Adaptation rapide avec retraînement'
            },
            'preprocessing': {
                'problem': 'Chaque étape dépend de la précédente',
                'impact': 'Difficile d\'optimiser pipeline',
                'ia_solution': 'Modèles peuvent remplacer étapes individuelles'
            }
        }
    
    def display_limitations(self):
        """Affiche les limitations"""
        print("\n" + "="*70)
        print("Limitations Simulation MC et Solutions IA")
        print("="*70)
        
        for limitation, info in self.limitations.items():
            print(f"\n{limitation.replace('_', ' ').title()}:")
            print(f"  Problème: {info['problem']}")
            print(f"  Impact: {info['impact']}")
            print(f"  Solution IA: {info['ia_solution']}")

limitations = MCLimitations()
limitations.display_limitations()
```

---

## Exercices

### Exercice 21.1.1
Simulez un processus hard scattering simple (ex: Drell-Yan) et analysez la distribution de masse invariante.

### Exercice 21.1.2
Estimez le temps et coût nécessaires pour simuler 1 milliard d'événements avec différentes configurations de parallélisation.

### Exercice 21.1.3
Analysez la répartition du temps de calcul entre les différentes étapes de simulation MC.

### Exercice 21.1.4
Comparez les caractéristiques de simulation MC vs génération IA en termes de précision, vitesse, et flexibilité.

---

## Points Clés à Retenir

> 📌 **La simulation MC reproduit fidèlement processus physiques depuis collision jusqu'au détecteur**

> 📌 **GEANT4 est l'outil standard pour simulation détecteur mais très coûteux**

> 📌 **Le coût computationnel limite nombre d'événements simulés**

> 📌 **L'IA peut accélérer simulation 100-1000× tout en préservant propriétés essentielles**

> 📌 **Le compromis précision/vitesse doit être évalué soigneusement**

> 📌 **La simulation MC reste nécessaire pour validation et entraînement modèles IA**

---

*Section précédente : [21.0 Introduction](./21_introduction.md) | Section suivante : [21.2 GANs](./21_02_GANs.md)*

