# 28.2 Présentations en Conférences

---

## Introduction

Les **présentations en conférences** sont un moyen essentiel de communiquer la recherche oralement. Une présentation efficace engage l'audience, communique clairement les idées, et permet recevoir feedback immédiat. Cette section présente les techniques pour créer et donner des présentations scientifiques efficaces.

---

## Structure d'une Présentation

### Organisation Standard

```python
class PresentationStructure:
    """
    Structure présentation scientifique
    """
    
    def __init__(self, duration_minutes: int = 15):
        self.duration = duration_minutes
        self.structure = {
            'title_slide': {
                'duration': 0.5,
                'content': [
                    'Titre article',
                    'Auteurs et affiliations',
                    'Logo institution',
                    'Date et venue'
                ]
            },
            'outline': {
                'duration': 0.5,
                'content': [
                    'Motivation',
                    'Approche',
                    'Résultats',
                    'Conclusion'
                ]
            },
            'motivation': {
                'duration': 2,
                'content': [
                    'Problème adressé',
                    'Pourquoi important',
                    'Limitations approches existantes'
                ]
            },
            'related_work': {
                'duration': 1,
                'content': [
                    'Travaux existants pertinents',
                    'Positionnement',
                    'Contribution'
                ]
            },
            'methodology': {
                'duration': 4,
                'content': [
                    'Approche proposée',
                    'Innovations clés',
                    'Détails techniques (si temps)'
                ]
            },
            'results': {
                'duration': 5,
                'content': [
                    'Résultats principaux',
                    'Comparaisons',
                    'Visualisations'
                ]
            },
            'conclusion': {
                'duration': 2,
                'content': [
                    'Contributions',
                    'Impact',
                    'Directions futures'
                ]
            },
            'questions': {
                'duration': 'Remaining',
                'content': [
                    'Merci',
                    'Questions?',
                    'Contact info'
                ]
            }
        }
    
    def calculate_slide_allocation(self):
        """Calcule nombre slides par section"""
        # Règle générale: 1-2 minutes par slide
        total_slides = self.duration // 2
        
        allocation = {
            'title': 1,
            'outline': 1,
            'motivation': 2,
            'related_work': 1,
            'methodology': 4,
            'results': 5,
            'conclusion': 1,
            'questions': 1
        }
        
        return allocation

presentation_structure = PresentationStructure(15)
```

---

## Design des Slides

### Principes Visuels

```python
class SlideDesign:
    """
    Design efficace de slides
    """
    
    def __init__(self):
        self.design_principles = {
            'simplicity': {
                'description': 'Simplicité visuelle',
                'guidelines': [
                    'Maximum 1 idée principale par slide',
                    'Utiliser liste à puces (4-6 items max)',
                    'Éviter texte dense',
                    'Beaucoup d\'espace blanc'
                ]
            },
            'readability': {
                'description': 'Lisibilité',
                'guidelines': [
                    'Police lisible (24pt minimum pour texte)',
                    'Contraste élevé (noir sur blanc)',
                    'Titres plus grands (36-44pt)',
                    'Éviter polices décoratives'
                ]
            },
            'visuals': {
                'description': 'Utilisation visuels',
                'guidelines': [
                    'Préférer figures à texte',
                    'Diagrammes clairs',
                    'Graphiques simples',
                    'Photographies si approprié'
                ]
            },
            'consistency': {
                'description': 'Cohérence',
                'guidelines': [
                    'Template uniforme',
                    'Couleurs cohérentes',
                    'Style figures similaire',
                    'Formatting uniforme'
                ]
            }
        }
    
    def slide_templates(self):
        """Templates pour différents types slides"""
        templates = {
            'title': """
Title Slide:
- Large title (centered)
- Authors (smaller)
- Affiliations
- Conference name/date
""",
            'content': """
Content Slide:
- Title (top)
- 1 main point or figure
- Supporting points (if needed)
- Clean, uncluttered
""",
            'comparison': """
Comparison Slide:
- Two columns or split screen
- Clear labels
- Key differences highlighted
- Side-by-side comparison
""",
            'results': """
Results Slide:
- Clear figure/chart
- Descriptive title
- Key takeaways (1-2 bullets)
- Units and labels clear
"""
        }
        return templates

slide_designer = SlideDesign()
```

---

## Techniques de Présentation

### Livraison Efficace

```python
class PresentationDelivery:
    """
    Techniques pour présentation orale
    """
    
    def __init__(self):
        self.delivery_techniques = {
            'voice': {
                'volume': 'Loud enough for all to hear',
                'pace': 'Slower than conversation (pause often)',
                'variation': 'Vary tone to emphasize points',
                'clarity': 'Clear articulation, avoid filler words'
            },
            'body_language': {
                'eye_contact': 'Make eye contact with audience',
                'posture': 'Stand straight, open posture',
                'movement': 'Move naturally, don\'t hide behind podium',
                'gestures': 'Use gestures to emphasize points'
            },
            'nervousness': {
                'preparation': 'Practice multiple times',
                'breathing': 'Deep breaths before starting',
                'focus': 'Focus on message, not on self',
                'recovery': 'If mistake, acknowledge and continue'
            },
            'engagement': {
                'questions': 'Ask rhetorical questions',
                'stories': 'Use examples or anecdotes',
                'interaction': 'Engage with audience when possible',
                'enthusiasm': 'Show passion for your work'
            }
        }
    
    def preparation_checklist(self):
        """Checklist préparation"""
        checklist = [
            'Slides finalisés et testés',
            'Présentation pratiquée (multiple fois)',
            'Timing vérifié',
            'Backup plan (PDF, USB)',
            'Technologie testée (projecteur, clicker)',
            'Questions anticipées préparées',
            'Confortable avec contenu',
            'Sleep bien avant'
        ]
        return checklist
    
    def handle_questions(self):
        """Guide gestion questions"""
        strategies = {
            'listening': 'Écouter question complète avant répondre',
            'clarification': 'Si question pas claire, demander clarification',
            'direct': 'Répondre directement si possible',
            'honest': 'Si pas sûr, être honnête, offrir follow-up',
            'time': 'Garder réponses concises (1-2 minutes)',
            'difficult': 'Pour questions difficiles, reconnaître validité, proposer discussion après'
        }
        return strategies

delivery_guide = PresentationDelivery()
```

---

## Pratique et Répétition

### Préparation

```python
class PresentationPractice:
    """
    Pratique pour présentation
    """
    
    def practice_routine(self):
        """Routine de pratique"""
        routine = {
            'first_practice': {
                'when': '1-2 semaines avant',
                'focus': 'Organisation et flow',
                'duration': 'Full presentation'
            },
            'multiple_practices': {
                'when': 'Daily leading up to presentation',
                'focus': 'Smooth delivery, timing',
                'duration': 'Full presentation'
            },
            'recorded_practice': {
                'when': 'Few days before',
                'focus': 'Watch yourself, identify issues',
                'benefits': [
                    'Identify filler words',
                    'Check timing',
                    'See body language',
                    'Hear voice quality'
                ]
            },
            'practice_with_audience': {
                'when': 'Before presentation',
                'audience': 'Colleagues, lab members',
                'benefits': [
                    'Receive feedback',
                    'Practice Q&A',
                    'Get comfortable',
                    'Identify confusing parts'
                ]
            }
        }
        return routine
    
    def timing_practice(self):
        """Pratique timing"""
        tips = [
            'Pratiquer avec timer',
            'Identifier sections à accélérer/ralentir',
            'Préparer "skip" slides si court de temps',
            'Préparer "deep dive" slides si surplus temps',
            'Pause pour questions si temps permet'
        ]
        return tips
```

---

## Outils et Logiciels

### Options de Présentation

```python
class PresentationTools:
    """
    Outils pour créations présentations
    """
    
    def __init__(self):
        self.tools = {
            'latex_beamer': {
                'pros': [
                    'Qualité typographique excellente',
                    'Formules mathématiques parfaites',
                    'Versioning facile',
                    'Cohérence automatique'
                ],
                'cons': [
                    'Courbe apprentissage',
                    'Moins flexible design',
                    'Plus lent pour modifications'
                ],
                'best_for': 'Présentations avec beaucoup de maths'
            },
            'powerpoint_keynote': {
                'pros': [
                    'Facile à utiliser',
                    'Design flexible',
                    'Animations',
                    'Widely compatible'
                ],
                'cons': [
                    'Qualité typographie variable',
                    'Formules mathématiques moins bonnes',
                    'Versioning moins facile'
                ],
                'best_for': 'Présentations générales'
            },
            'revealjs': {
                'pros': [
                    'Web-based',
                    'Interactive',
                    'Versioning avec Git',
                    'Markdown support'
                ],
                'cons': [
                    'Nécessite serveur/web',
                    'Moins d\'outils design'
                ],
                'best_for': 'Présentations web/interactives'
            }
        }
    
    def recommendations(self):
        """Recommandations"""
        return {
            'scientific_presentations': 'LaTeX Beamer pour qualité',
            'quick_presentations': 'PowerPoint/Keynote pour rapidité',
            'collaboration': 'Google Slides pour collaboration temps réel',
            'interactive': 'Reveal.js pour présentation web'
        }

tools = PresentationTools()
```

---

## Exercices

### Exercice 28.2.1
Créez présentation de 15 minutes sur votre recherche avec slides structurés.

### Exercice 28.2.2
Pratiquez présentation devant caméra et analysez votre performance.

### Exercice 28.2.3
Créez version "backup" de présentation avec slides supplémentaires pour questions.

### Exercice 28.2.4
Préparez réponses à 5 questions potentielles sur votre recherche.

---

## Points Clés à Retenir

> 📌 **Structure claire (motivation → approach → results → conclusion)**

> 📌 **Slides simples avec 1 idée principale par slide**

> 📌 **Pratique multiple fois avant présentation réelle**

> 📌 **Timing critique - respecter limite temps**

> 📌 **Engagement avec audience améliore communication**

> 📌 **Préparation Q&A permet répondre confortablement**

---

*Section précédente : [28.1 Articles](./28_01_Articles.md) | Section suivante : [28.3 Posters](./28_03_Posters.md)*

