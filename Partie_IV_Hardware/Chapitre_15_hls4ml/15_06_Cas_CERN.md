# 15.6 Études de Cas au CERN

---

## Introduction

Cette section présente des **études de cas réelles** d'utilisation de hls4ml au CERN, notamment dans les expériences CMS et ATLAS, démontrant l'efficacité du framework en conditions réelles.

---

## Cas d'Usage 1: CMS Level-1 Trigger - Jet Tagger

### Contexte

```python
class CMSJetTagger:
    """
    Étude de cas: CMS L1 Trigger - Classification de jets
    """
    
    def __init__(self):
        self.context = {
            'experiment': 'CMS',
            'application': 'Level-1 Trigger',
            'task': 'Classification de jets (b-jet, c-jet, light-jet)',
            'constraints': {
                'latency': '< 1 μs',
                'throughput': '40 MHz',
                'fpga': 'Xilinx Ultrascale+'
            },
            'model': {
                'type': 'MLP',
                'architecture': '3 dense layers: 16→32→16→3',
                'input_features': 16,
                'output_classes': 3
            }
        }
    
    def display_context(self):
        """Affiche le contexte"""
        print("\n" + "="*60)
        print("CMS L1 Trigger - Jet Tagger Case Study")
        print("="*60)
        
        print(f"\nExperiment: {self.context['experiment']}")
        print(f"Application: {self.context['application']}")
        print(f"Task: {self.context['task']}")
        
        print("\nConstraints:")
        for constraint, value in self.context['constraints'].items():
            print(f"  {constraint}: {value}")
        
        print("\nModel:")
        for key, value in self.context['model'].items():
            print(f"  {key}: {value}")

cms_case = CMSJetTagger()
cms_case.display_context()
```

---

### Implémentation

```python
class CMSJetTaggerImplementation:
    """
    Détails d'implémentation du CMS Jet Tagger
    """
    
    def __init__(self):
        self.implementation = {
            'model_config': {
                'precision': 'ap_fixed<12,4>',
                'reuse_factor': 1,  # Fully parallel
                'strategy': 'Latency'
            },
            'results': {
                'latency_ns': 450,  # < 1 μs requirement
                'throughput_mhz': 40,  # Meets requirement
                'accuracy': 0.96,  # 96% accuracy
                'resources': {
                    'lut': 8500,
                    'dsp': 128,
                    'bram': 12
                }
            },
            'performance': {
                'b_tagging_efficiency': 0.85,
                'background_rejection': 0.92,
                'meets_requirements': True
            }
        }
    
    def display_implementation(self):
        """Affiche l'implémentation"""
        print("\n" + "="*60)
        print("Implementation Details")
        print("="*60)
        
        print("\nModel Configuration:")
        for key, value in self.implementation['model_config'].items():
            print(f"  {key}: {value}")
        
        print("\nResults:")
        print(f"  Latency: {self.implementation['results']['latency_ns']} ns")
        print(f"  Throughput: {self.implementation['results']['throughput_mhz']} MHz")
        print(f"  Accuracy: {self.implementation['results']['accuracy']:.2%}")
        
        print("\nResources:")
        for resource, value in self.implementation['results']['resources'].items():
            print(f"  {resource.upper()}: {value:,}")
        
        print("\nPerformance:")
        print(f"  b-tagging efficiency: {self.implementation['performance']['b_tagging_efficiency']:.2%}")
        print(f"  Background rejection: {self.implementation['performance']['background_rejection']:.2%}")
        print(f"  Meets requirements: {self.implementation['performance']['meets_requirements']}")

implementation = CMSJetTaggerImplementation()
implementation.display_implementation()
```

---

## Cas d'Usage 2: ATLAS Trigger - Muon Identification

### Contexte

```python
class ATLASMuonID:
    """
    Étude de cas: ATLAS Trigger - Identification de muons
    """
    
    def __init__(self):
        self.context = {
            'experiment': 'ATLAS',
            'application': 'Level-1 Trigger',
            'task': 'Identification de muons',
            'constraints': {
                'latency': '< 2 μs',
                'throughput': '40 MHz',
                'fpga': 'Xilinx Kintex UltraScale'
            },
            'model': {
                'type': 'MLP',
                'architecture': '4 dense layers: 20→64→32→16→2',
                'input_features': 20,
                'output': 'Binary classification (muon / non-muon)'
            },
            'features': [
                'Track parameters',
                'Calorimeter deposits',
                'Muon spectrometer hits',
                'Transverse momentum'
            ]
        }
    
    def display_context(self):
        """Affiche le contexte"""
        print("\n" + "="*60)
        print("ATLAS L1 Trigger - Muon ID Case Study")
        print("="*60)
        
        for key, value in self.context.items():
            if isinstance(value, list):
                print(f"\n{key.replace('_', ' ').title()}:")
                for item in value:
                    print(f"  • {item}")
            elif isinstance(value, dict):
                print(f"\n{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

atlas_case = ATLASMuonID()
atlas_case.display_context()
```

---

### Résultats

```python
class ATLASMuonIDResults:
    """
    Résultats de l'implémentation ATLAS
    """
    
    def __init__(self):
        self.results = {
            'implementation': {
                'precision': 'ap_fixed<10,4>',
                'reuse_factor': 2,  # Resource sharing
                'optimization': 'Mixed precision per layer'
            },
            'performance': {
                'latency_ns': 1200,  # < 2 μs
                'throughput_mhz': 40,
                'muon_efficiency': 0.98,
                'fake_rate': 0.02,
                'meets_requirements': True
            },
            'resources': {
                'lut': 12000,
                'dsp': 200,
                'bram': 18,
                'utilization': {
                    'lut': 0.23,  # 23%
                    'dsp': 0.91,  # 91%
                    'bram': 0.03  # 3%
                }
            },
            'lessons_learned': [
                'Mixed precision crucial pour tenir dans ressources',
                'Resource sharing nécessaire pour grande architecture',
                'Validation sur données réelles essentielle'
            ]
        }
    
    def display_results(self):
        """Affiche les résultats"""
        print("\n" + "="*60)
        print("ATLAS Muon ID - Results")
        print("="*60)
        
        print("\nImplementation:")
        for key, value in self.results['implementation'].items():
            print(f"  {key}: {value}")
        
        print("\nPerformance:")
        for key, value in self.results['performance'].items():
            print(f"  {key}: {value}")
        
        print("\nResources:")
        print("  Usage:")
        for resource, value in self.results['resources'].items():
            if resource != 'utilization':
                print(f"    {resource.upper()}: {value:,}")
        print("  Utilization:")
        for resource, util in self.results['resources']['utilization'].items():
            print(f"    {resource.upper()}: {util:.1%}")
        
        print("\nLessons Learned:")
        for lesson in self.results['lessons_learned']:
            print(f"  • {lesson}")

atlas_results = ATLASMuonIDResults()
atlas_results.display_results()
```

---

## Cas d'Usage 3: CMS - Electron/Photon Classification

### Contexte et Résultats

```python
class CMSElectronPhoton:
    """
    Étude de cas: CMS - Classification électron/photon
    """
    
    def __init__(self):
        self.case_study = {
            'context': {
                'experiment': 'CMS',
                'application': 'Trigger optimization',
                'task': 'Distinguer électrons de photons',
                'importance': 'Critique pour analyses précises'
            },
            'model': {
                'type': 'CNN + MLP',
                'input': 'Calorimeter images (3×3 or 5×5)',
                'architecture': 'Conv2D blocks + Dense layers'
            },
            'implementation': {
                'precision': 'ap_fixed<8,2>',  # Très agressif
                'reuse_factor': 4,
                'optimization': 'Aggressive quantization + pruning'
            },
            'results': {
                'latency_ns': 800,
                'accuracy': 0.94,
                'electron_efficiency': 0.95,
                'photon_efficiency': 0.93,
                'resources': {
                    'lut': 15000,
                    'dsp': 350,
                    'bram': 25
                }
            },
            'key_insights': [
                'CNN sur FPGA réalisable avec optimisations agressives',
                'Quantification 8-bit acceptable pour cette tâche',
                'Feature engineering important pour réduire complexité'
            ]
        }
    
    def display_case_study(self):
        """Affiche l'étude de cas"""
        print("\n" + "="*60)
        print("CMS Electron/Photon Classification Case Study")
        print("="*60)
        
        for section, content in self.case_study.items():
            print(f"\n{section.replace('_', ' ').title()}:")
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (list, dict)):
                        print(f"  {key}:")
                        if isinstance(value, list):
                            for item in value:
                                print(f"    • {item}")
                        else:
                            for k, v in value.items():
                                print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {content}")

cms_eg_case = CMSElectronPhoton()
cms_eg_case.display_case_study()
```

---

## Analyse Comparative des Cas d'Usage

```python
class CaseStudyComparison:
    """
    Comparaison des études de cas
    """
    
    @staticmethod
    def compare_cases():
        """Compare les différents cas d'usage"""
        cases = {
            'CMS Jet Tagger': {
                'latency_ns': 450,
                'model_complexity': 'Simple (3 layers)',
                'precision': '12-bit',
                'reuse_factor': 1,
                'resources_dsp': 128,
                'key_factor': 'Fully parallel possible'
            },
            'ATLAS Muon ID': {
                'latency_ns': 1200,
                'model_complexity': 'Medium (4 layers)',
                'precision': '10-bit',
                'reuse_factor': 2,
                'resources_dsp': 200,
                'key_factor': 'Resource sharing needed'
            },
            'CMS e/γ': {
                'latency_ns': 800,
                'model_complexity': 'Complex (CNN+MLP)',
                'precision': '8-bit',
                'reuse_factor': 4,
                'resources_dsp': 350,
                'key_factor': 'Aggressive quantization'
            }
        }
        
        print("\n" + "="*60)
        print("Case Studies Comparison")
        print("="*60)
        
        print(f"\n{'Case':<20} | {'Latency':<10} | {'Complexity':<12} | {'Precision':<10} | {'RF':<4} | {'DSP':<6}")
        print("-" * 80)
        
        for case_name, metrics in cases.items():
            print(f"{case_name:<20} | {metrics['latency_ns']:<10} | "
                  f"{metrics['model_complexity']:<12} | {metrics['precision']:<10} | "
                  f"{metrics['reuse_factor']:<4} | {metrics['resources_dsp']:<6}")
        
        print("\nKey Factors:")
        for case_name, metrics in cases.items():
            print(f"  {case_name}: {metrics['key_factor']}")

CaseStudyComparison.compare_cases()
```

---

## Leçons Apprises

### Best Practices Identifiées

```python
class LessonsLearned:
    """
    Leçons apprises des études de cas
    """
    
    def __init__(self):
        self.lessons = {
            'model_design': [
                'Simplifier architecture si possible',
                'Éviter opérations complexes non supportées',
                'Tester compatibilité tôt'
            ],
            'optimization': [
                'Quantification souvent nécessaire',
                'Mixed precision par couche efficace',
                'Pruning peut aider pour grands modèles',
                'ReuseFactor adapté aux contraintes'
            ],
            'validation': [
                'Valider sur données réelles HEP',
                'Vérifier métriques physiques, pas seulement ML accuracy',
                'Mesurer latence sur hardware réel',
                'Tests end-to-end essentiels'
            ],
            'integration': [
                'Interface avec trigger système critique',
                'Preprocessing peut être goulot d\'étranglement',
                'Pipeline nécessaire pour throughput',
                'Monitoring en production important'
            ],
            'deployment': [
                'Itération nécessaire entre modélisation et FPGA',
                'Collaboration ML experts + FPGA experts',
                'Documentation des configurations importantes',
                'Version control pour modèles et bitstreams'
            ]
        }
    
    def display_lessons(self):
        """Affiche les leçons apprises"""
        print("\n" + "="*60)
        print("Lessons Learned from CERN Case Studies")
        print("="*60)
        
        for category, lessons in self.lessons.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for lesson in lessons:
                print(f"  • {lesson}")

lessons = LessonsLearned()
lessons.display_lessons()
```

---

## Défis Rencontrés et Solutions

```python
class ChallengesAndSolutions:
    """
    Défis rencontrés et solutions
    """
    
    def __init__(self):
        self.challenges = {
            'timing_closure': {
                'challenge': 'Respecter contraintes de timing strictes',
                'solutions': [
                    'Pipeline agressif',
                    'Réduire précision si acceptable',
                    'Optimiser chemins critiques',
                    'Partitionnement mémoire'
                ]
            },
            'resource_constraints': {
                'challenge': 'Modèles trop grands pour FPGA',
                'solutions': [
                    'Quantification agressive (8-bit, 4-bit)',
                    'Compression (pruning)',
                    'Architecture plus simple',
                    'Resource sharing (ReuseFactor > 1)'
                ]
            },
            'accuracy_preservation': {
                'challenge': 'Maintenir accuracy après optimisation',
                'solutions': [
                    'Quantization-aware training',
                    'Mixed precision par couche',
                    'Fine-tuning après quantification',
                    'Validation rigoureuse'
                ]
            },
            'integration_complexity': {
                'challenge': 'Intégration avec systèmes existants',
                'solutions': [
                    'Interfaces standardisées (AXI)',
                    'Documentation complète',
                    'Tests progressifs',
                    'Support de la communauté'
                ]
            }
        }
    
    def display_challenges(self):
        """Affiche les défis et solutions"""
        print("\n" + "="*60)
        print("Challenges and Solutions")
        print("="*60)
        
        for challenge, info in self.challenges.items():
            print(f"\n{challenge.replace('_', ' ').title()}:")
            print(f"  Challenge: {info['challenge']}")
            print(f"  Solutions:")
            for solution in info['solutions']:
                print(f"    • {solution}")

challenges = ChallengesAndSolutions()
challenges.display_challenges()
```

---

## Exercices

### Exercice 15.6.1
Analysez un des cas d'usage et proposez des optimisations supplémentaires possibles.

### Exercice 15.6.2
Concevez un modèle pour une nouvelle application HEP en vous inspirant des leçons apprises.

---

## Points Clés à Retenir

> 📌 **hls4ml validé en production dans CMS et ATLAS**

> 📌 **Modèles simples (MLP) fonctionnent bien, CNN possible avec optimisations**

> 📌 **Quantification souvent nécessaire pour tenir dans ressources**

> 📌 **Validation sur données réelles HEP essentielle**

> 📌 **Collaboration ML + FPGA experts cruciale**

> 📌 **Itération nécessaire pour trouver compromis optimal**

---

*Chapitre suivant : [Chapitre 16 - Hardware-Aware Neural Architecture Search](../Chapitre_16_Hardware_NAS/16_introduction.md)*

