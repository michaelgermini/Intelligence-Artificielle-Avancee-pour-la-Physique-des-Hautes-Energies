# 27.4 Code Review et Collaboration

---

## Introduction

Le **code review** est une pratique essentielle dans le développement open source qui améliore la qualité du code, facilite l'apprentissage, et maintient la cohérence du projet. Cette section présente comment effectuer et recevoir des code reviews constructifs.

---

## Principes du Code Review

### Objectifs

```python
"""
Objectifs du code review:

1. Qualité du code
   - Détecter bugs
   - Améliorer design
   - Assurer standards

2. Partage de connaissances
   - Apprentissage mutuel
   - Transfer d'expertise
   - Documentation implicite

3. Cohérence projet
   - Style uniforme
   - Patterns consistants
   - Architecture alignée

4. Sécurité
   - Détecter vulnérabilités
   - Vérifier pratiques sûres
"""
```

---

## Recevoir Code Review

### Comment Répondre aux Feedbacks

```python
class ReceivingCodeReview:
    """
    Bonnes pratiques pour recevoir reviews
    """
    
    def __init__(self):
        self.best_practices = {
            'before_submission': [
                'Auto-review votre code d\'abord',
                'Vérifier tests passent',
                'Vérifier linting',
                'Documentation à jour',
                'Commits propres'
            ],
            'during_review': [
                'Rester ouvert aux feedbacks',
                'Ne pas prendre critiques personnellement',
                'Poser questions si commentaires pas clairs',
                'Mercier reviewers pour leur temps',
                'Répondre à tous commentaires'
            ],
            'addressing_feedback': [
                'Implémenter changements suggérés',
                'Ou expliquer pourquoi pas possible',
                'Faire changements en commits séparés',
                'Référencer commentaires dans commits',
                'Demander re-review après changements'
            ]
        }
    
    def respond_to_comments(self):
        """Guide pour répondre commentaires"""
        strategies = {
            'agree_and_implement': """
            "Good catch! Fixed in latest commit."
            + Implémenter changement
            """,
            'disagree_with_reason': """
            "I considered that approach, but chose X because Y.
            What do you think?"
            + Expliquer raisonnement
            """,
            'need_clarification': """
            "Could you clarify what you mean by X?
            I want to make sure I understand correctly."
            + Poser questions spécifiques
            """,
            'partial_agreement': """
            "I agree with part A, but for part B I think we should
            consider alternative Z. What's your take?"
            + Proposer alternative
            """
        }
        return strategies

review_receiver = ReceivingCodeReview()
```

---

## Effectuer Code Review

### Checklist de Review

```python
class CodeReviewChecklist:
    """
    Checklist pour effectuer code review
    """
    
    def __init__(self):
        self.checklist = {
            'functionality': [
                'Code fait ce qu\'il prétend faire?',
                'Edge cases gérés?',
                'Error handling approprié?',
                'Performance acceptable?'
            ],
            'code_quality': [
                'Code lisible et maintenable?',
                'Suis standards du projet?',
                'Pas de duplication?',
                'Nommage clair?',
                'Commentaires nécessaires présents?'
            ],
            'testing': [
                'Tests présents pour nouveau code?',
                'Tests couvrent cas principaux?',
                'Tests passent?',
                'Couverture suffisante?'
            ],
            'documentation': [
                'Docstrings présents?',
                'Documentation à jour?',
                'Exemples clairs?',
                'Changelog mis à jour?'
            ],
            'security': [
                'Input validation?',
                'Pas de secrets hardcodés?',
                'Gestion mémoire correcte?',
                'Pas de vulnérabilités connues?'
            ]
        }
    
    def review_pull_request(self, pr_url: str):
        """Processus review PR"""
        steps = [
            "Lire description PR et comprendre objectif",
            "Tester localement si possible",
            "Vérifier chaque point de checklist",
            "Donner feedback constructif et spécifique",
            "Approuver si tout bon, ou demander changements"
        ]
        return steps

reviewer = CodeReviewChecklist()
```

---

## Donner Feedback Constructif

### Techniques de Communication

```python
class ConstructiveFeedback:
    """
    Donner feedback constructif
    """
    
    def __init__(self):
        self.feedback_principles = {
            'be_specific': {
                'bad': 'This code is confusing',
                'good': 'This function is 50 lines long and does 3 things. Consider splitting into helper functions.'
            },
            'be_helpful': {
                'bad': 'This is wrong',
                'good': 'This approach might have issues with edge case X. Consider using Y instead, which handles it better.'
            },
            'be_respectful': {
                'bad': 'This is a terrible design',
                'good': 'I see the intent, but wonder if we could improve this by...'
            },
            'suggest_alternatives': {
                'bad': "Don't do this",
                'good': 'Have you considered using [alternative]? It might be more efficient/maintainable because...'
            },
            'ask_questions': {
                'bad': 'This is unclear',
                'good': 'Could you help me understand the reasoning behind this approach?'
            }
        }
    
    def format_review_comment(self, suggestion_type: str, 
                            location: str, 
                            issue: str,
                            suggestion: str = None):
        """Formate commentaire review"""
        comment = f"**{suggestion_type}** ({location}):\n"
        comment += f"{issue}\n"
        
        if suggestion:
            comment += f"\n**Suggestion:**\n```python\n{suggestion}\n```"
        
        return comment

feedback_giver = ConstructiveFeedback()
```

---

## Types de Reviews

### Approvals et Requests for Changes

```python
class ReviewTypes:
    """
    Types de reviews GitHub/GitLab
    """
    
    def __init__(self):
        self.review_types = {
            'approve': {
                'when': 'Code est bon et prêt',
                'message': 'Looks good! Ready to merge.',
                'action': 'Maintainer peut merger'
            },
            'request_changes': {
                'when': 'Changements nécessaires avant merge',
                'message': 'Please address these comments before merging.',
                'action': 'Auteur doit faire changements'
            },
            'comment': {
                'when': 'Questions ou suggestions non-blocking',
                'message': 'Good work! A few suggestions for consideration.',
                'action': 'Discussion continue'
            }
        }
    
    def when_to_approve(self):
        """Critères pour approval"""
        criteria = [
            'Code fonctionne correctement',
            'Tests passent',
            'Documentation à jour',
            'Suis standards projet',
            'Pas de bugs évidents',
            'Performance acceptable',
            'Sécurité vérifiée'
        ]
        return criteria
```

---

## Collaboration Efficace

### Communication dans Reviews

```python
class CollaborationInReviews:
    """
    Collaboration efficace dans reviews
    """
    
    def __init__(self):
        self.collaboration_tips = {
            'timeliness': {
                'description': 'Répondre rapidement aux PRs',
                'guideline': 'Répondre dans 1-2 jours si possible',
                'impact': 'Maintenir momentum projet'
            },
            'conversation': {
                'description': 'Reviews sont conversations',
                'guideline': 'Poser questions, discuter alternatives',
                'impact': 'Apprentissage mutuel'
            },
            'acknowledgment': {
                'description': 'Reconnaître bon travail',
                'guideline': 'Approuver et féliciter quand code bon',
                'impact': 'Motivation contributeurs'
            },
            'consensus': {
                'description': 'Trouver consensus',
                'guideline': 'Si désaccord, discuter ouvertement',
                'impact': 'Décisions collectives meilleures'
            }
        }
    
    def handle_disagreements(self):
        """Gérer désaccords dans reviews"""
        strategies = [
            "Clarifier objectifs communs",
            "Discuter trade-offs objectivement",
            "Proposer compromis",
            "Implémenter et comparer si possible",
            "Demander opinion tierce si nécessaire",
            "Respecter décision maintainer final"
        ]
        return strategies
```

---

## Outils de Review

### GitHub et GitLab Features

```python
class ReviewTools:
    """
    Outils pour code review
    """
    
    def __init__(self):
        self.tools = {
            'github_features': {
                'suggestions': 'Suggérer changements inline directement',
                'reviews': 'Reviews avec approve/request changes/comment',
                'conversation': 'Threads de discussion',
                'assignees': 'Assigner reviewers',
                'labels': 'Organiser PRs avec labels',
                'templates': 'Templates pour PRs et issues',
                'checks': 'CI/CD checks avant merge'
            },
            'gitlab_features': {
                'merge_requests': 'Équivalent PRs GitHub',
                'draft_requests': 'MRs en draft pour feedback précoce',
                'approval_rules': 'Règles approbation configurables',
                'inline_comments': 'Commentaires inline',
                'review_apps': 'Déployer MR pour test'
            },
            'tools': {
                'reviewboard': 'Outils review dédiés',
                'phabricator': 'Platform complète',
                'gerrit': 'Review workflow spécifique'
            }
        }
    
    def use_github_suggestions(self):
        """Comment utiliser suggestions GitHub"""
        example = """
# Dans review comment:
```
Here's a suggestion:

```suggestion:path/to/file.py
// Suggested code here
```

This would improve X because Y.
```

# Auteur peut accepter suggestion avec un clic
"""
        return example
```

---

## Exercices

### Exercice 27.4.1
Reviewez un Pull Request open (sur projet open source) et donnez feedback constructif.

### Exercice 27.4.2
Recevez review sur votre PR et pratiquez répondre aux commentaires.

### Exercice 27.4.3
Créez checklist personnelle pour vos propres code reviews.

### Exercice 27.4.4
Participez à discussion dans PR et négociez changements proposés.

---

## Points Clés à Retenir

> 📌 **Code review améliore qualité et facilite apprentissage**

> 📌 **Feedback constructif et spécifique est plus utile**

> 📌 **Rester ouvert et respectueux dans reviews**

> 📌 **Répondre à tous commentaires montre professionnalisme**

> 📌 **Reviews sont conversations, pas jugements**

> 📌 **Outils (suggestions, inline comments) facilitent reviews**

---

*Section précédente : [27.3 Documentation et Tests](./27_03_Documentation_Tests.md) | Section suivante : [27.5 Licences](./27_05_Licences.md)*

