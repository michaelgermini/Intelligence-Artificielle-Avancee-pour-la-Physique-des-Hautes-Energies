# 27.5 Licences et Propriété Intellectuelle

---

## Introduction

Comprendre les **licences open source** est crucial pour contribuer légalement et protéger votre travail. Cette section présente les types de licences, leurs implications, et comment choisir une licence appropriée pour votre projet.

---

## Types de Licences

### Classification des Licences

```python
"""
Catégories principales:

1. Licences permissives
   - MIT, Apache 2.0, BSD
   - Permet usage commercial
   - Modification libre
   - Peu de restrictions

2. Licences copyleft
   - GPL, AGPL
   - Modifications doivent être open source
   - "Share alike" principle

3. Licences faiblement copyleft
   - LGPL, MPL
   - Copyleft pour modifications du code lui-même
   - Permet linking avec code propriétaire
"""

class OpenSourceLicenses:
    """
    Vue d'ensemble licences open source
    """
    
    def __init__(self):
        self.licenses = {
            'MIT': {
                'type': 'Permissive',
                'description': 'Très permissive, très populaire',
                'requirements': [
                    'Inclure copyright et licence',
                    'Pas de garantie'
                ],
                'allows': [
                    'Usage commercial',
                    'Modification',
                    'Distribution',
                    'Sublicensing',
                    'Patent use'
                ],
                'prohibits': [],
                'best_for': 'Projets souhaitant adoption maximale'
            },
            'Apache_2.0': {
                'type': 'Permissive',
                'description': 'Similaire MIT avec clause patent',
                'requirements': [
                    'Inclure copyright et licence',
                    'Indiquer modifications',
                    'License file dans distribution'
                ],
                'allows': [
                    'Usage commercial',
                    'Modification',
                    'Distribution',
                    'Sublicensing',
                    'Patent grant explicite'
                ],
                'prohibits': [
                    'Usage nom pour endorsement'
                ],
                'best_for': 'Projets avec préoccupations patents'
            },
            'GPL_v3': {
                'type': 'Copyleft',
                'description': 'Copyleft fort, modifications doivent être GPL',
                'requirements': [
                    'Distribuer source code',
                    'Maintenir licence GPL',
                    'Modifications sous GPL'
                ],
                'allows': [
                    'Usage commercial',
                    'Modification',
                    'Distribution (avec source)'
                ],
                'prohibits': [
                    'Linking avec code propriétaire',
                    'Changer licence',
                    'Sublicensing différent'
                ],
                'best_for': 'Projets souhaitant garantir open source'
            },
            'BSD_3_Clause': {
                'type': 'Permissive',
                'description': 'Très permissive, clause non-endorsement',
                'requirements': [
                    'Inclure copyright',
                    'Disclaimer'
                ],
                'allows': [
                    'Usage commercial',
                    'Modification',
                    'Distribution'
                ],
                'prohibits': [
                    'Usage nom auteurs pour endorsement'
                ],
                'best_for': 'Similaire MIT avec clause non-endorsement'
            },
            'LGPL': {
                'type': 'Weak Copyleft',
                'description': 'Copyleft pour library, permet linking propriétaire',
                'requirements': [
                    'Modifications LGPL doivent rester LGPL',
                    'Source code pour modifications LGPL'
                ],
                'allows': [
                    'Linking avec code propriétaire',
                    'Usage commercial'
                ],
                'prohibits': [
                    'Changer licence modifications LGPL'
                ],
                'best_for': 'Bibliothèques utilisées par logiciels propriétaires'
            }
        }
    
    def display_licenses(self):
        """Affiche comparaison licences"""
        print("\n" + "="*70)
        print("Comparaison Licences Open Source")
        print("="*70)
        
        for license_name, info in self.licenses.items():
            print(f"\n{license_name}:")
            print(f"  Type: {info['type']}")
            print(f"  Description: {info['description']}")
            print(f"  Permet:")
            for allow in info['allows'][:3]:
                print(f"    + {allow}")
            if info['prohibits']:
                print(f"  Interdit:")
                for prohibit in info['prohibits']:
                    print(f"    - {prohibit}")
```

---

## Choisir une Licence

### Guide de Sélection

```python
class LicenseSelection:
    """
    Guide pour choisir licence
    """
    
    def __init__(self):
        self.selection_guide = {
            'maximize_adoption': {
                'recommended': ['MIT', 'Apache 2.0', 'BSD'],
                'reason': 'Licences permissives facilitent adoption'
            },
            'protect_open_source': {
                'recommended': ['GPL v3', 'AGPL'],
                'reason': 'Copyleft garantit modifications restent open'
            },
            'library_for_commercial': {
                'recommended': ['MIT', 'Apache 2.0', 'LGPL'],
                'reason': 'Permet usage dans logiciels commerciaux'
            },
            'patent_concerns': {
                'recommended': ['Apache 2.0'],
                'reason': 'Clause patent explicite'
            },
            'cloud_deployment': {
                'recommended': ['AGPL'],
                'reason': 'AGPL couvre services network'
            }
        }
    
    def choose_license(self, requirements: Dict) -> List[str]:
        """Suggère licences selon requirements"""
        candidates = []
        
        if requirements.get('commercial_use', False):
            candidates.extend(['MIT', 'Apache 2.0', 'BSD', 'LGPL'])
        
        if requirements.get('modifications_required_open', False):
            candidates.extend(['GPL v3', 'AGPL'])
        
        if requirements.get('library', False):
            candidates.append('LGPL')
        
        if requirements.get('patent_protection', False):
            candidates.append('Apache 2.0')
        
        # Retirer doublons
        return list(set(candidates))

license_selector = LicenseSelection()
```

---

## Compatibilité des Licences

### Compatibilité entre Licences

```python
class LicenseCompatibility:
    """
    Compatibilité entre licences
    """
    
    def __init__(self):
        self.compatibility_matrix = {
            'MIT': {
                'compatible_with': ['MIT', 'Apache 2.0', 'BSD', 'GPL'],
                'can_combine_with': 'Most licenses'
            },
            'Apache_2.0': {
                'compatible_with': ['MIT', 'Apache 2.0', 'GPL v3'],
                'can_combine_with': 'Most except GPL v2'
            },
            'GPL_v3': {
                'compatible_with': ['MIT', 'Apache 2.0', 'GPL v3'],
                'can_combine_with': 'GPL-compatible only'
            },
            'GPL_v2': {
                'compatible_with': ['GPL v2'],
                'can_combine_with': 'GPL v2 only'
            }
        }
    
    def check_compatibility(self, license1: str, license2: str) -> bool:
        """Vérifie compatibilité deux licences"""
        compat1 = self.compatibility_matrix.get(license1, {}).get('compatible_with', [])
        return license2 in compat1

compatibility_checker = LicenseCompatibility()
```

---

## Contribuer à Projets Existants

### Licences des Contributions

```python
class ContributingLicenses:
    """
    Licences lors de contributions
    """
    
    def __init__(self):
        self.guidelines = {
            'contributing_license': {
                'description': 'Contributions sous licence du projet',
                'requirement': 'Vérifier licence projet avant contribution',
                'implication': 'Votre contribution sera sous licence projet'
            },
            'cla': {
                'description': 'Contributor License Agreement',
                'purpose': 'Clarifier droits intellectuels',
                'types': [
                    'CLA individuelle',
                    'Corporate CLA',
                    'DCO (Developer Certificate of Origin)'
                ]
            },
            'copyright': {
                'description': 'Droits d\'auteur sur contributions',
                'practice': 'Garde copyright, donne licence au projet',
                'alternative': 'Copyright peut être assigné au projet'
            }
        }
    
    def understand_contribution_terms(self):
        """Comprendre termes contribution"""
        terms = {
            'what_you_keep': [
                'Droits d\'auteur sur votre code',
                'Droit d\'utiliser votre code ailleurs'
            ],
            'what_you_grant': [
                'Licence au projet d\'utiliser votre code',
                'Droit de distribuer sous licence projet',
                'Droit de modifier et distribuer modifications'
            ],
            'typical_clauses': [
                'Vous avez droit de contribuer',
                'Code est votre propre travail',
                'Vous accordez licence nécessaire'
            ]
        }
        return terms
```

---

## Propriété Intellectuelle

### Droits et Responsabilités

```python
class IntellectualProperty:
    """
    Propriété intellectuelle dans open source
    """
    
    def __init__(self):
        self.ip_aspects = {
            'copyright': {
                'description': 'Droit d\'auteur sur code',
                'protection': 'Automatique dès création',
                'duration': 'Vie auteur + 70 ans (varie par pays)',
                'rights': [
                    'Reproduction',
                    'Distribution',
                    'Modification',
                    'Public performance'
                ]
            },
            'patents': {
                'description': 'Protection inventions',
                'relevance': 'Certaines licences incluent clauses patents',
                'apache_2.0': 'Grant explicite droits patents',
                'gpl_v3': 'Clause défensive contre brevets'
            },
            'trademarks': {
                'description': 'Noms et logos',
                'protection': 'Séparé de licence code',
                'practice': 'Licences généralement ne couvrent pas trademarks'
            }
        }
    
    def understand_ip_implications(self):
        """Implications propriété intellectuelle"""
        implications = {
            'contributing': [
                'Vous gardez copyright',
                'Vous accordez licence au projet',
                'Ne pas contribuer code sans droit'
            ],
            'using': [
                'Respecter termes licence',
                'Inclure notices copyright',
                'Respecter conditions redistribution'
            ],
            'modifying': [
                'Modifications sous licence originale',
                'Indiquer modifications si requis',
                'Respecter compatibilité licence'
            ]
        }
        return implications
```

---

## Licences pour Votre Projet

### Créer et Ajouter Licence

```python
class AddingLicense:
    """
    Ajouter licence à votre projet
    """
    
    def create_license_file(self, license_type: str = 'MIT'):
        """Crée fichier LICENSE"""
        licenses = {
            'MIT': """MIT License

Copyright (c) [YEAR] [YOUR NAME]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
            'Apache_2.0': """
# Apache License 2.0 - voir apache.org/licenses/LICENSE-2.0
"""
        }
        
        return licenses.get(license_type, licenses['MIT'])
    
    def add_license_to_project(self):
        """Guide ajout licence"""
        steps = [
            'Créer fichier LICENSE dans racine projet',
            'Copier texte licence appropriée',
            'Remplacer [YEAR] et [YOUR NAME]',
            'Ajouter badge licence dans README',
            'Spécifier licence dans setup.py/pyproject.toml'
        ]
        return steps

license_adder = AddingLicense()
```

---

## Exercices

### Exercice 27.5.1
Comparez licences MIT, Apache 2.0, et GPL v3 et identifiez différences clés.

### Exercice 27.5.2
Choisissez licence appropriée pour projet hypothétique selon requirements.

### Exercice 27.5.3
Vérifiez compatibilité de licence pour combiner code de différents projets.

### Exercice 27.5.4
Créez fichier LICENSE pour un projet et ajoutez badges dans README.

---

## Points Clés à Retenir

> 📌 **Licences permissives (MIT, Apache) maximisent adoption**

> 📌 **Licences copyleft (GPL) garantissent open source**

> 📌 **Compatibilité des licences importante lors combinaison code**

> 📌 **Contributions sont généralement sous licence du projet**

> 📌 **Comprendre copyright et patents est important**

> 📌 **CLA/DCO clarifient droits intellectuels**

---

*Section précédente : [27.4 Code Review](./27_04_Code_Review.md)*

