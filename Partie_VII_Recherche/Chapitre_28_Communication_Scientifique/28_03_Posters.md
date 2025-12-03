# 28.3 Posters Scientifiques

---

## Introduction

Les **posters scientifiques** sont un format de communication qui permet interaction détaillée avec audience lors de sessions poster. Un bon poster présente recherche de manière visuelle et accessible, facilitant discussions approfondies. Cette section présente les principes de design et création de posters efficaces.

---

## Principes de Design

### Structure et Organisation

```python
class PosterDesign:
    """
    Design de posters scientifiques
    """
    
    def __init__(self):
        self.poster_structure = {
            'title_section': {
                'location': 'Top center',
                'content': [
                    'Titre (grand, visible de loin)',
                    'Auteurs et affiliations',
                    'Institution logos',
                    'Contact information'
                ],
                'height': '10-15%'
            },
            'introduction': {
                'location': 'Top left',
                'content': [
                    'Contexte',
                    'Problème',
                    'Objectifs'
                ],
                'width': '45-50%'
            },
            'methods': {
                'location': 'Left middle',
                'content': [
                    'Approche',
                    'Méthodologie',
                    'Setup expérimental'
                ],
                'width': '45-50%'
            },
            'results': {
                'location': 'Center/Right',
                'content': [
                    'Résultats principaux',
                    'Figures clés',
                    'Comparaisons'
                ],
                'width': '50-60%'
            },
            'conclusion': {
                'location': 'Bottom',
                'content': [
                    'Contributions',
                    'Impact',
                    'Future work'
                ],
                'width': 'Full width'
            },
            'acknowledgments': {
                'location': 'Bottom right',
                'content': [
                    'Funding',
                    'Collaborations',
                    'Thanks'
                ],
                'width': '30-40%'
            }
        }
    
    def design_principles(self):
        """Principes design poster"""
        principles = {
            'visual_hierarchy': [
                'Titre visible de 3-5 mètres',
                'Sections clairement délimitées',
                'Flux de lecture naturel (gauche-droite, haut-bas)',
                'Important éléments mis en évidence'
            ],
            'readability': [
                'Police minimale 24pt (32-44pt pour titres)',
                'Contraste élevé',
                'Éviter textes longs (paragraphes courts)',
                'Beaucoup d\'espace blanc'
            ],
            'visuals': [
                'Figures grandes et claires',
                'Graphiques simples',
                'Utiliser couleurs efficacement',
                'Minimiser texte, maximiser visuels'
            ],
            'balance': [
                'Distribution équilibrée contenu',
                'Pas de sections trop denses',
                'Cohérence visuelle',
                'Marges appropriées'
            ]
        }
        return principles

poster_designer = PosterDesign()
```

---

## Dimensions et Formats

### Standards de Taille

```python
class PosterSpecifications:
    """
    Spécifications techniques posters
    """
    
    def __init__(self):
        self.sizes = {
            'common': {
                'A0': {
                    'dimensions': '841mm × 1189mm (33.1" × 46.8")',
                    'usage': 'Très commun',
                    'portrait': True,
                    'landscape': True
                },
                'A1': {
                    'dimensions': '594mm × 841mm (23.4" × 33.1")',
                    'usage': 'Commun',
                    'portrait': True,
                    'landscape': True
                },
                'custom_large': {
                    'dimensions': '36" × 48" ou 90cm × 120cm',
                    'usage': 'Certaines conférences',
                    'check': 'Vérifier requirements conférence'
                }
            }
        }
        
        self.resolution = {
            'printing': '300 DPI minimum',
            'screens': '72-150 DPI suffisant',
            'recommended': '300 DPI pour flexibilité'
        }
    
    def check_conference_requirements(self):
        """Vérifier requirements conférence"""
        checklist = [
            'Dimensions exactes requises',
            'Orientation (portrait/landscape)',
            'Marges minimales',
            'Format fichier (PDF recommandé)',
            'Résolution minimale',
            'Deadline soumission',
            'Méthode affichage (pins, velcro, etc.)'
        ]
        return checklist

specs = PosterSpecifications()
```

---

## Contenu du Poster

### Éléments Clés

```python
class PosterContent:
    """
    Contenu efficace pour poster
    """
    
    def create_content_outline(self):
        """Outline contenu poster"""
        outline = {
            'title': {
                'length': '1 ligne si possible',
                'style': 'Descriptif mais concis',
                'font_size': '72-96pt'
            },
            'sections': {
                'introduction': {
                    'length': '150-200 mots',
                    'key_points': [
                        'Contexte (2-3 phrases)',
                        'Problème (1-2 phrases)',
                        'Objectifs (1 phrase)'
                    ]
                },
                'methods': {
                    'length': '200-250 mots',
                    'key_points': [
                        'Approche générale',
                        'Innovations clés',
                        'Setup expérimental',
                        'Détails techniques importants'
                    ],
                    'visuals': 'Diagrammes architecture, algorithmes'
                },
                'results': {
                    'length': '250-300 mots',
                    'key_points': [
                        'Résultats principaux',
                        'Comparaisons',
                        'Analyses',
                        'Takeaways'
                    ],
                    'visuals': 'Graphiques, tableaux, figures principales'
                },
                'conclusion': {
                    'length': '100-150 mots',
                    'key_points': [
                        'Contributions principales',
                        'Impact',
                        'Directions futures'
                    ]
                }
            },
            'figures': {
                'number': '3-5 figures principales',
                'size': 'Grandes (20-30% poster)',
                'captions': 'Brèves mais informatives',
                'location': 'Centre ou droite (high visibility)'
            }
        }
        return outline
    
    def text_guidelines(self):
        """Guidelines texte"""
        guidelines = {
            'length': [
                'Beaucoup moins texte que article',
                'Bullet points plutôt que paragraphes',
                'Maximum 800-1000 mots total',
                'Focus sur visuels'
            ],
            'style': [
                'Phrases courtes',
                'Langage accessible',
                'Éviter jargon excessif',
                'Actif plutôt que passif'
            ],
            'hierarchy': [
                'Titres sections: 44-60pt',
                'Sous-titres: 32-36pt',
                'Corps texte: 24-28pt',
                'Captions: 18-20pt'
            ]
        }
        return guidelines

content_guide = PosterContent()
```

---

## Création Technique

### Outils et Workflow

```python
class PosterCreationTools:
    """
    Outils création posters
    """
    
    def __init__(self):
        self.tools = {
            'adobe_illustrator': {
                'pros': ['Professionnel', 'Flexibilité design', 'Vector graphics'],
                'cons': ['Coûteux', 'Courbe apprentissage'],
                'best_for': 'Design professionnel complet'
            },
            'inkscape': {
                'pros': ['Gratuit', 'Vector graphics', 'Open source'],
                'cons': ['Interface moins polie'],
                'best_for': 'Alternative gratuite à Illustrator'
            },
            'powerpoint_keynote': {
                'pros': ['Facile', 'Familiar', 'Templates disponibles'],
                'cons': ['Moins professionnel', 'Limitations design'],
                'best_for': 'Posters simples ou débutants'
            },
            'latex': {
                'pros': ['Qualité typographie', 'Formules math', 'Versioning'],
                'cons': ['Courbe apprentissage', 'Moins flexible'],
                'packages': ['beamerposter', 'tikzposter'],
                'best_for': 'Posters avec beaucoup de mathématiques'
            },
            'canva': {
                'pros': ['Templates', 'Facile', 'Online'],
                'cons': ['Moins flexible', 'Quality variable'],
                'best_for': 'Posters simples, templates pré-faits'
            }
        }
    
    def creation_workflow(self):
        """Workflow création poster"""
        workflow = [
            '1. Créer outline et structure',
            '2. Préparer figures (haute résolution)',
            '3. Écrire texte (réduire depuis article)',
            '4. Créer layout dans outil choisi',
            '5. Ajouter contenu section par section',
            '6. Ajuster taille polices et spacing',
            '7. Review et révisions',
            '8. Export PDF haute résolution',
            '9. Pre-print review (impression test)'
        ]
        return workflow
    
    def quality_checklist(self):
        """Checklist qualité poster"""
        checklist = {
            'content': [
                'Toutes sections présentes',
                'Texte concis et clair',
                'Figures claires et grandes',
                'Pas d\'erreurs typographiques'
            ],
            'design': [
                'Lisibilité de distance (3-5m)',
                'Hiérarchie visuelle claire',
                'Couleurs cohérentes',
                'Espaces blancs appropriés'
            ],
            'technical': [
                'Dimensions correctes',
                'Résolution suffisante (300 DPI)',
                'Marges respectées',
                'Format PDF correct'
            ]
        }
        return checklist

creation_tools = PosterCreationTools()
```

---

## Présentation du Poster

### Session Poster

```python
class PosterPresentation:
    """
    Présentation lors session poster
    """
    
    def __init__(self):
        self.presentation_guide = {
            'preparation': [
                'Avoir version courte (2-3 min)',
                'Avoir version longue (5-10 min)',
                'Préparer selon niveau intérêt visiteur',
                'Anticiper questions communes'
            ],
            'during_presentation': {
                'opening': [
                    'Salutation amicale',
                    'Demander intérêt visiteur',
                    'Adapter niveau explication'
                ],
                'delivery': [
                    'Pointer vers sections poster',
                    'Parler clairement mais pas trop fort',
                    'Engager avec questions',
                    'Rester près du poster'
                ],
                'engagement': [
                    'Poser questions à audience',
                    'Encourager questions',
                    'Discuter applications',
                    'Échanger contacts si intéressant'
                ]
            },
            'handling_questions': [
                'Écouter complètement question',
                'Répondre directement',
                'Utiliser poster comme support visuel',
                'Si pas sûr, proposer discussion après',
                'Reconnaître limitations honnêtement'
            ]
        }
    
    def elevator_pitch(self):
        """Version courte présentation"""
        template = """
Elevator Pitch (2 minutes):

"Hi, I'm [Name] from [Institution]. 

Our research addresses [problem] in [domain].

Current methods have limitations: [brief limitation].

We propose [approach] that [key innovation].

Our results show [main result] which enables [impact].

Would you like to know more about [specific aspect]?"
"""
        return template
    
    def networking_tips(self):
        """Conseils networking"""
        tips = [
            'Avoir cartes visite ou QR code',
            'Échanger contacts avec intéressants',
            'Poser questions sur leur recherche',
            'Discuter collaborations potentielles',
            'Prendre notes après sessions',
            'Follow-up après conférence'
        ]
        return tips

poster_presenter = PosterPresentation()
```

---

## Exercices

### Exercice 28.3.1
Créez outline complet de poster avec sections et emplacement figures.

### Exercice 28.3.2
Concevez layout de poster avec structure visuelle claire.

### Exercice 28.3.3
Préparez version courte (2 min) et longue (5 min) de présentation poster.

### Exercice 28.3.4
Créez poster complet sur votre recherche et pratiquez présentation.

---

## Points Clés à Retenir

> 📌 **Posters doivent être lisibles de distance (3-5 mètres)**

> 📌 **Beaucoup moins texte que article - focus sur visuels**

> 📌 **Structure claire guide lecture naturelle**

> 📌 **Figures grandes et claires sont essentielles**

> 📌 **Présentation interactive permet discussions approfondies**

> 📌 **Adaptation au niveau visiteur améliore communication**

---

*Section précédente : [28.2 Présentations](./28_02_Presentations.md) | Section suivante : [28.4 Collaboration](./28_04_Collaboration.md)*

