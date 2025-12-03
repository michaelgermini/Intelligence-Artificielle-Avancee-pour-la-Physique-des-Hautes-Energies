# 20.1 Recherche de Nouvelle Physique au LHC

---

## Introduction

La **recherche de nouvelle physique** est l'un des objectifs principaux du LHC. Alors que le Modèle Standard décrit admirablement bien les interactions connues, il laisse de nombreuses questions ouvertes (matière noire, hiérarchie des masses, etc.). La détection d'anomalies offre une approche complémentaire aux recherches dirigées par des modèles théoriques spécifiques.

Cette section présente le contexte de la recherche de nouvelle physique, les défis associés, et comment l'anomaly detection s'intègre dans cette quête.

---

## Modèle Standard et Ses Limitations

### Vue d'Ensemble

```python
import numpy as np
from typing import Dict, List

class StandardModelLimitations:
    """
    Limitations du Modèle Standard
    """
    
    def __init__(self):
        self.limitations = {
            'dark_matter': {
                'description': 'Matière noire non expliquée',
                'evidence': 'Rotation galaxies, CMB, etc.',
                'search_strategies': [
                    'Recherche directe (WIMPs)',
                    'Production au LHC',
                    'Anomaly detection (signatures invisibles)'
                ]
            },
            'hierarchy_problem': {
                'description': 'Pourquoi masse Higgs si petite vs Planck ?',
                'evidence': 'Fine-tuning nécessaire',
                'search_strategies': [
                    'Supersymétrie',
                    'Dimensions supplémentaires',
                    'Anomalies dans production Higgs'
                ]
            },
            'neutrino_masses': {
                'description': 'Masses des neutrinos non dans SM minimal',
                'evidence': 'Oscillations neutrinos',
                'search_strategies': [
                    'See-saw mechanism',
                    'Anomalies dans désintégrations'
                ]
            },
            'cp_violation': {
                'description': 'CP violation insuffisante pour baryogenèse',
                'evidence': 'Asymétrie matière-antimatière',
                'search_strategies': [
                    'CP violation dans secteur Higgs',
                    'Anomalies dans distributions angulaires'
                ]
            }
        }
    
    def display_limitations(self):
        """Affiche les limitations"""
        print("\n" + "="*70)
        print("Limitations du Modèle Standard")
        print("="*70)
        
        for limitation, info in self.limitations.items():
            print(f"\n{limitation.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Évidence: {info['evidence']}")
            print(f"  Stratégies de recherche:")
            for strategy in info['search_strategies']:
                print(f"    • {strategy}")

sm_limits = StandardModelLimitations()
sm_limits.display_limitations()
```

---

## Approches de Recherche

### Recherche Guidée vs Exploratoire

```python
class SearchStrategies:
    """
    Stratégies de recherche de nouvelle physique
    """
    
    def __init__(self):
        self.strategies = {
            'model_directed': {
                'name': 'Recherche Guidée par Modèle',
                'description': 'Chercher signaux prédits par théories spécifiques',
                'examples': [
                    'Recherche SUSY (supersymétrie)',
                    'Recherche dimensions supplémentaires',
                    'Recherche WIMPs'
                ],
                'advantages': [
                    'Test précis de théories',
                    'Optimisation possible',
                    'Interprétation claire'
                ],
                'disadvantages': [
                    'Biaisé vers modèles testés',
                    'Manque signaux inattendus',
                    'Dépendant de prédictions théoriques'
                ]
            },
            'anomaly_driven': {
                'name': 'Recherche par Détection d\'Anomalies',
                'description': 'Identifier événements anormaux sans modèle spécifique',
                'examples': [
                    'Autoencoders sur données',
                    'Outlier detection',
                    'Covariate shift detection'
                ],
                'advantages': [
                    'Sans biais théorique',
                    'Découvre inattendu',
                    'Approche exploratoire'
                ],
                'disadvantages': [
                    'Interprétation difficile',
                    'Beaucoup de faux positifs',
                    'Validation complexe'
                ]
            },
            'hybrid': {
                'name': 'Approche Hybride',
                'description': 'Combiner recherche guidée et anomaly detection',
                'examples': [
                    'Anomaly detection dans régions spécifiques',
                    'Validation de modèles avec anomalies',
                    'Découverte guidée par domaines'
                ],
                'advantages': [
                    'Meilleur des deux mondes',
                    'Validation croisée'
                ],
                'disadvantages': [
                    'Plus complexe',
                    'Nécessite coordination'
                ]
            }
        }
    
    def display_strategies(self):
        """Affiche les stratégies"""
        print("\n" + "="*70)
        print("Stratégies de Recherche de Nouvelle Physique")
        print("="*70)
        
        for strategy, info in self.strategies.items():
            print(f"\n{info['name']}:")
            print(f"  Description: {info['description']}")
            print(f"  Exemples:")
            for ex in info['examples']:
                print(f"    • {ex}")
            print(f"  Avantages:")
            for adv in info['advantages']:
                print(f"    + {adv}")
            print(f"  Inconvénients:")
            for disadv in info['disadvantages']:
                print(f"    - {disadv}")

strategies = SearchStrategies()
strategies.display_strategies()
```

---

## Signaux de Nouvelle Physique Potentiels

### Signatures Génériques

```python
class NewPhysicsSignatures:
    """
    Signatures potentielles de nouvelle physique
    """
    
    def __init__(self):
        self.signatures = {
            'high_mass_resonances': {
                'description': 'Résonances à haute masse',
                'example': 'Z\' boson, gravitons Kaluza-Klein',
                'signature': 'Pic dans distribution de masse invariante',
                'detection': 'Anomalie dans distribution m(ℓℓ) ou m(jj)'
            },
            'missing_energy_patterns': {
                'description': 'Patterns spécifiques d\'énergie manquante',
                'example': 'Matière noire, neutrinos stériles',
                'signature': 'MET avec distributions caractéristiques',
                'detection': 'Anomalie dans distribution MET vs autres variables'
            },
            'unusual_jets': {
                'description': 'Jets avec propriétés inhabituelles',
                'example': 'Jets de particules exotiques',
                'signature': 'Shape, multiplicité, ou composition anormale',
                'detection': 'Jets avec features hors distribution background'
            },
            'rare_topologies': {
                'description': 'Topologies d\'événements rares',
                'example': 'Événements multi-leptoniques inhabituels',
                'signature': 'Combinaisons de leptons/jets rares',
                'detection': 'Événements dans régions peu peuplées'
            },
            'asymmetries': {
                'description': 'Asymétries inattendues',
                'example': 'CP violation dans secteur Higgs',
                'signature': 'Asymétries dans distributions angulaires',
                'detection': 'Déviation de symétrie attendue'
            }
        }
    
    def display_signatures(self):
        """Affiche les signatures"""
        print("\n" + "="*70)
        print("Signatures Potentielles de Nouvelle Physique")
        print("="*70)
        
        for sig, info in self.signatures.items():
            print(f"\n{sig.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Exemple: {info['example']}")
            print(f"  Signature: {info['signature']}")
            print(f"  Détection: {info['detection']}")

signatures = NewPhysicsSignatures()
signatures.display_signatures()
```

---

## Défis Statistiques

### Tests Statistiques et Significativité

```python
class StatisticalChallenges:
    """
    Défis statistiques dans recherche nouvelle physique
    """
    
    def __init__(self):
        self.challenges = {
            'look_elsewhere_effect': {
                'description': 'Effect de regarder ailleurs (multiple testing)',
                'problem': 'Beaucoup de tests → faux positifs',
                'solution': 'Correction multiple testing (Bonferroni, etc.)',
                'impact': 'Significativité réelle plus faible que apparente'
            },
            'trial_factor': {
                'description': 'Nombre élevé d\'analyses différentes',
                'problem': 'Chaque analyse = test indépendant',
                'solution': 'Pre-registration, calcul trial factor global',
                'impact': '5σ devient 3σ avec 1000 tests'
            },
            'systematic_uncertainties': {
                'description': 'Incertitudes systématiques dominantes',
                'problem': 'Difficile à quantifier, peut masquer signaux',
                'solution': 'Études systématiques, nuisance parameters',
                'impact': 'Limite sensibilité, complique interprétation'
            },
            'validation': {
                'description': 'Validation sans connaissance signal réel',
                'problem': 'Comment valider détection sans vrai signal ?',
                'solution': 'Tests sur données connues, closure tests',
                'impact': 'Confiance limitée dans résultats'
            }
        }
    
    def compute_trial_factor_penalty(self, n_tests: int, 
                                     nominal_significance: float = 5.0) -> Dict:
        """
        Calcule impact du trial factor sur significativité
        
        Args:
            n_tests: Nombre de tests effectués
            nominal_significance: Significativité nominale (en σ)
        """
        # P-value correspondante
        from scipy import stats
        p_value = stats.norm.sf(nominal_significance)
        
        # Correction Bonferroni
        p_corrected = min(1.0, p_value * n_tests)
        significance_corrected = stats.norm.isf(p_corrected)
        
        return {
            'nominal_significance_sigma': nominal_significance,
            'nominal_pvalue': p_value,
            'n_tests': n_tests,
            'corrected_pvalue': p_corrected,
            'corrected_significance_sigma': significance_corrected,
            'penalty': nominal_significance - significance_corrected
        }
    
    def display_challenges(self):
        """Affiche les défis"""
        print("\n" + "="*70)
        print("Défis Statistiques")
        print("="*70)
        
        for challenge, info in self.challenges.items():
            print(f"\n{challenge.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Problème: {info['problem']}")
            print(f"  Solution: {info['solution']}")
            print(f"  Impact: {info['impact']}")

stats_challenges = StatisticalChallenges()
stats_challenges.display_challenges()

# Exemple trial factor
trial_result = stats_challenges.compute_trial_factor_penalty(n_tests=100, nominal_significance=5.0)
print(f"\nExemple Trial Factor:")
print(f"  Significativité nominale: {trial_result['nominal_significance_sigma']:.1f}σ")
print(f"  Avec 100 tests: {trial_result['corrected_significance_sigma']:.2f}σ")
print(f"  Pénalité: {trial_result['penalty']:.2f}σ")
```

---

## Cas d'Usage: Recherche de Matière Noire

### Exemple Concret

```python
class DarkMatterSearch:
    """
    Recherche de matière noire avec anomaly detection
    """
    
    def __init__(self):
        self.search_strategy = {
            'signature': 'MET + jets (monojet)',
            'background': 'QCD + W/Z+jets',
            'challenge': 'Background énorme, signal faible',
            'approach': 'Anomaly detection sur distribution MET vs autres variables'
        }
    
    def monojet_signature(self):
        """
        Signature monojet pour matière noire
        
        WIMP + WIMP → jet + MET (invisible)
        """
        signature_features = {
            'met': 'Élevé (> 100 GeV)',
            'n_jets': '1 jet principal',
            'jet_pt': 'Élevé (> 100 GeV)',
            'dphi_jet_met': 'Grand (jet et MET opposés)',
            'no_leptons': 'Pas de leptons (pour réduire W+jets)'
        }
        
        return signature_features
    
    def background_characteristics(self):
        """
        Caractéristiques du background
        """
        backgrounds = {
            'qcd': {
                'met_source': 'Résolution calorimètre',
                'distribution': 'MET bas, gaussien',
                'separation': 'Relativement facile'
            },
            'w_jets': {
                'met_source': 'Neutrino du W',
                'distribution': 'MET modéré, correlation avec lepton',
                'separation': 'Difficile (peut ressembler signal)'
            },
            'z_jets': {
                'met_source': 'Résolution (Z → invisible rare)',
                'distribution': 'MET bas',
                'separation': 'Relativement facile'
            }
        }
        
        return backgrounds
    
    def anomaly_detection_approach(self):
        """
        Comment utiliser anomaly detection
        """
        approach = {
            'training': 'Entraîner sur données background (SM)',
            'features': ['MET', 'jet_pt', 'jet_eta', 'dphi_jet_met', 'n_jets'],
            'method': 'Autoencoder ou isolation forest',
            'selection': 'Événements avec score anomalie élevé',
            'analysis': 'Analyser propriétés des anomalies trouvées'
        }
        
        return approach

dm_search = DarkMatterSearch()

print("\n" + "="*70)
print("Recherche de Matière Noire")
print("="*70)

signature = dm_search.monojet_signature()
print(f"\nSignature Monojet:")
for feat, value in signature.items():
    print(f"  {feat}: {value}")

approach = dm_search.anomaly_detection_approach()
print(f"\nApproche Anomaly Detection:")
for step, desc in approach.items():
    print(f"  {step}: {desc}")
```

---

## Validation et Interprétation

### Méthodes de Validation

```python
class ValidationMethods:
    """
    Méthodes de validation pour anomaly detection
    """
    
    def closure_test(self, model, test_data):
        """
        Closure test: vérifier que modèle fonctionne sur données connues
        
        Teste sur signal injecté connu
        """
        # Injecter signal connu dans données
        # Vérifier que modèle le détecte
        
        return {
            'test_name': 'Closure test',
            'procedure': 'Inject signal connu, verify detection',
            'success_criteria': 'Signal detected avec bonne efficacité'
        }
    
    def sideband_validation(self, model, sideband_data):
        """
        Validation sur sideband (région de validation)
        
        Utilise région proche mais distincte de région signal
        """
        return {
            'test_name': 'Sideband validation',
            'procedure': 'Test on control region',
            'success_criteria': 'Background well modeled'
        }
    
    def robustness_checks(self):
        """
        Vérifications de robustesse
        """
        checks = [
            'Stabilité sous variations systématiques',
            'Indépendance de choix hyperparamètres',
            'Performance sur différents datasets',
            'Cohérence avec analyses classiques'
        ]
        
        return checks

validation = ValidationMethods()
```

---

## Exercices

### Exercice 20.1.1
Analysez l'impact du trial factor sur une recherche avec 1000 canaux différents testés simultanément.

### Exercice 20.1.2
Concevez une stratégie de recherche de matière noire combinant recherche guidée (monojet) et anomaly detection.

### Exercice 20.1.3
Développez un système de validation pour une méthode d'anomaly detection qui utilise des closure tests.

### Exercice 20.1.4
Comparez les avantages et inconvénients de recherche guidée vs anomaly detection pour différents types de nouvelle physique.

---

## Points Clés à Retenir

> 📌 **Le Modèle Standard a des limitations qui motivent recherche de nouvelle physique**

> 📌 **La recherche guidée teste modèles spécifiques, anomaly detection est exploratoire**

> 📌 **Le trial factor réduit significativité apparente avec nombreux tests**

> 📌 **Les incertitudes systématiques sont souvent limitantes**

> 📌 **La validation est complexe sans connaissance du vrai signal**

> 📌 **L'approche hybride combine avantages des deux méthodes**

---

*Section précédente : [20.0 Introduction](./20_introduction.md) | Section suivante : [20.2 Autoencoders](./20_02_Autoencoders.md)*

