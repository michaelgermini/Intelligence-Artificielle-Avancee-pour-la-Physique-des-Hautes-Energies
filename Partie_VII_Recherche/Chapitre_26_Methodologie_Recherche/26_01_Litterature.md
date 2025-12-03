# 26.1 Revue de Littérature Systématique

---

## Introduction

La **revue de littérature systématique** est une étape fondamentale de tout projet de recherche. Elle permet de comprendre l'état de l'art, d'identifier les gaps de connaissance, et d'éviter la duplication de travaux existants. Cette section présente les méthodes pour mener une revue de littérature efficace et systématique.

---

## Objectifs d'une Revue de Littérature

### Pourquoi Faire une Revue ?

```python
"""
Objectifs d'une revue de littérature:

1. Comprendre état de l'art
   - Méthodes existantes
   - Résultats obtenus
   - Limitations actuelles

2. Identifier gaps
   - Questions non résolues
   - Méthodes non explorées
   - Applications manquantes

3. Éviter duplication
   - Connaître travaux existants
   - Identifier nouveauté contribution

4. Justifier recherche
   - Montrer besoin de nouvelle méthode
   - Positionner contribution
"""
```

---

## Processus Systématique

### Étapes d'une Revue

```python
class LiteratureReview:
    """
    Processus de revue de littérature systématique
    """
    
    def __init__(self, topic: str):
        self.topic = topic
        self.papers = []
        self.summary = {}
    
    def conduct_review(self):
        """
        Mène revue systématique
        """
        # 1. Définir scope
        scope = self.define_scope()
        
        # 2. Sources de recherche
        sources = self.identify_sources()
        
        # 3. Mots-clés et stratégie recherche
        keywords = self.create_keywords()
        search_strategy = self.create_search_strategy(keywords)
        
        # 4. Recherche et collecte
        papers = self.search_papers(search_strategy, sources)
        
        # 5. Sélection (inclusion/exclusion)
        selected_papers = self.filter_papers(papers)
        
        # 6. Extraction données
        extracted_data = self.extract_data(selected_papers)
        
        # 7. Synthèse
        synthesis = self.synthesize_findings(extracted_data)
        
        return synthesis
    
    def define_scope(self):
        """Définit scope de la revue"""
        return {
            'research_questions': [
                "Quelles méthodes de compression existent?",
                "Quels résultats ont été obtenus?",
                "Quelles sont les limitations?"
            ],
            'inclusion_criteria': [
                "Papiers sur compression modèles DL",
                "Applications en HEP",
                "Publiés après 2020"
            ],
            'exclusion_criteria': [
                "Travaux non publiés",
                "Applications non-HEP"
            ]
        }
    
    def identify_sources(self):
        """Identifie sources de recherche"""
        return {
            'databases': [
                'arXiv',
                'Google Scholar',
                'IEEE Xplore',
                'ACM Digital Library',
                'Semantic Scholar'
            ],
            'venues': [
                'NeurIPS',
                'ICML',
                'ICLR',
                'HEP conferences'
            ],
            'journals': [
                'JMLR',
                'Machine Learning',
                'Physical Review'
            ]
        }
    
    def create_keywords(self):
        """Crée mots-clés de recherche"""
        return {
            'main_terms': ['model compression', 'neural network compression'],
            'related_terms': ['pruning', 'quantization', 'distillation'],
            'domain_terms': ['high energy physics', 'particle physics', 'LHC'],
            'technique_terms': ['tensor networks', 'low-rank approximation']
        }
    
    def create_search_strategy(self, keywords):
        """Crée stratégie de recherche"""
        # Exemples de requêtes
        queries = [
            f"{keywords['main_terms'][0]} AND {keywords['domain_terms'][0]}",
            f"{keywords['related_terms'][0]} AND tensor networks",
            # ... autres combinaisons
        ]
        return queries
    
    def filter_papers(self, papers):
        """Filtre papiers selon critères"""
        selected = []
        
        for paper in papers:
            # Vérifier critères inclusion
            if self.meets_criteria(paper):
                selected.append(paper)
        
        return selected
    
    def extract_data(self, papers):
        """Extrait données clés de chaque papier"""
        extracted = []
        
        for paper in papers:
            data = {
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'method': self.extract_method(paper),
                'results': self.extract_results(paper),
                'limitations': self.extract_limitations(paper),
                'dataset': self.extract_dataset(paper)
            }
            extracted.append(data)
        
        return extracted
    
    def synthesize_findings(self, extracted_data):
        """Synthétise résultats"""
        return {
            'timeline': self.create_timeline(extracted_data),
            'method_comparison': self.compare_methods(extracted_data),
            'gap_analysis': self.identify_gaps(extracted_data),
            'trends': self.identify_trends(extracted_data)
        }
```

---

## Organisation et Documentation

### Structure de Documentation

```python
class LiteratureReviewDocument:
    """
    Structure pour documenter revue de littérature
    """
    
    def create_review_document(self):
        """Crée document structuré"""
        document = {
            'introduction': {
                'context': 'Contexte et motivation',
                'objectives': 'Objectifs de la revue',
                'scope': 'Portée et limitations'
            },
            'methodology': {
                'search_strategy': 'Stratégie de recherche',
                'selection_criteria': 'Critères inclusion/exclusion',
                'data_extraction': 'Méthode extraction données'
            },
            'results': {
                'overview': 'Vue d\'ensemble papiers trouvés',
                'categorization': 'Catégorisation des méthodes',
                'comparison': 'Comparaison des approches'
            },
            'analysis': {
                'strengths': 'Forces des méthodes existantes',
                'weaknesses': 'Faiblesses et limitations',
                'gaps': 'Gaps identifiés'
            },
            'conclusion': {
                'summary': 'Résumé des findings',
                'implications': 'Implications pour recherche',
                'future_directions': 'Directions futures'
            }
        }
        
        return document
```

---

## Outils et Ressources

### Bibliothèques et Outils

```python
"""
Outils pour revue de littérature:

1. Gestion bibliographique
   - Zotero: Gestion références
   - Mendeley: Organisation et annotation
   - BibTeX: Format standard

2. Recherche
   - arXiv: Préprints
   - Google Scholar: Recherche large
   - Semantic Scholar: AI-powered search
   - Connected Papers: Visualisation connexions

3. Organisation
   - Notion: Documentation structurée
   - Obsidian: Knowledge graph
   - Paperpile: Gestion collaborative

4. Analyse
   - VOSviewer: Visualisation réseaux
   - CiteSpace: Analyse citations
"""

class LiteratureTools:
    """
    Outils recommandés
    """
    
    def __init__(self):
        self.tools = {
            'reference_management': {
                'zotero': 'Gestion références, intégration navigateur',
                'mendeley': 'Organisation, annotation PDF',
                'bibtex': 'Format standard LaTeX'
            },
            'search_engines': {
                'arxiv': 'Préprints scientifiques',
                'google_scholar': 'Recherche académique large',
                'semantic_scholar': 'Recherche avec IA',
                'connected_papers': 'Visualisation connexions'
            },
            'organization': {
                'notion': 'Documentation structurée',
                'obsidian': 'Knowledge graph, liens',
                'paperpile': 'Collaboration équipe'
            }
        }
    
    def display_tools(self):
        """Affiche outils"""
        print("\n" + "="*70)
        print("Outils pour Revue de Littérature")
        print("="*70)
        
        for category, tools in self.tools.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for tool, desc in tools.items():
                print(f"  • {tool}: {desc}")
```

---

## Analyse Comparative

### Comparaison de Méthodes

```python
class MethodComparison:
    """
    Comparaison systématique de méthodes
    """
    
    def create_comparison_table(self, papers):
        """Crée tableau comparatif"""
        comparison = {
            'methods': [],
            'metrics': ['accuracy', 'compression_ratio', 'speedup', 'complexity']
        }
        
        for paper in papers:
            method_data = {
                'name': paper['method'],
                'metrics': {
                    'accuracy': paper['results'].get('accuracy'),
                    'compression': paper['results'].get('compression_ratio'),
                    'speedup': paper['results'].get('speedup'),
                    'complexity': self.estimate_complexity(paper)
                },
                'pros': paper.get('strengths', []),
                'cons': paper.get('limitations', [])
            }
            comparison['methods'].append(method_data)
        
        return comparison
    
    def identify_trends(self, papers):
        """Identifie tendances temporelles"""
        by_year = {}
        
        for paper in papers:
            year = paper['year']
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(paper)
        
        trends = {
            'method_popularity': self.analyze_method_popularity(by_year),
            'performance_improvements': self.analyze_performance_trends(by_year),
            'emerging_techniques': self.identify_emerging(by_year)
        }
        
        return trends
```

---

## Exercices

### Exercice 26.1.1
Menez une revue de littérature systématique sur un sujet spécifique (ex: pruning methods).

### Exercice 26.1.2
Créez une base de données structurée de papiers avec extraction de métadonnées.

### Exercice 26.1.3
Développez un tableau comparatif de méthodes existantes avec leurs forces/faiblesses.

### Exercice 26.1.4
Identifiez les gaps dans la littérature pour votre domaine de recherche.

---

## Points Clés à Retenir

> 📌 **Revue systématique évite duplication et identifie gaps**

> 📌 **Stratégie de recherche claire optimise résultats**

> 📌 **Critères inclusion/exclusion garantissent pertinence**

> 📌 **Documentation structurée facilite synthèse**

> 📌 **Outils bibliographiques (Zotero, Mendeley) simplifient gestion**

> 📌 **Analyse comparative révèle forces/faiblesses méthodes**

---

*Section précédente : [26.0 Introduction](./26_introduction.md) | Section suivante : [26.2 Design d'Expériences](./26_02_Design_Experiences.md)*

