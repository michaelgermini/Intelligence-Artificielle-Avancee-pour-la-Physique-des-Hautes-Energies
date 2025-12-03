# 26.5 Gestion de Projets de Recherche

---

## Introduction

La **gestion efficace de projets de recherche** est essentielle pour mener des projets à bien, respecter les délais, et collaborer efficacement. Cette section présente les outils et méthodes pour gérer des projets de recherche, incluant planification, suivi, collaboration, et gestion des ressources.

---

## Planification de Projet

### Structure et Timeline

```python
from datetime import datetime, timedelta
from typing import List, Dict

class ResearchProjectPlan:
    """
    Planification de projet de recherche
    """
    
    def __init__(self, project_name: str, duration_months: int = 12):
        self.project_name = project_name
        self.duration_months = duration_months
        self.phases = []
        self.milestones = []
    
    def define_phases(self):
        """Définit phases du projet"""
        self.phases = [
            {
                'name': 'Literature Review',
                'duration_weeks': 4,
                'deliverables': ['Review document', 'Gap analysis'],
                'dependencies': []
            },
            {
                'name': 'Methodology Design',
                'duration_weeks': 3,
                'deliverables': ['Experimental design', 'Implementation plan'],
                'dependencies': ['Literature Review']
            },
            {
                'name': 'Implementation',
                'duration_weeks': 8,
                'deliverables': ['Code', 'Baseline results'],
                'dependencies': ['Methodology Design']
            },
            {
                'name': 'Experimentation',
                'duration_weeks': 6,
                'deliverables': ['Experimental results', 'Analysis'],
                'dependencies': ['Implementation']
            },
            {
                'name': 'Writing',
                'duration_weeks': 4,
                'deliverables': ['Paper draft', 'Figures'],
                'dependencies': ['Experimentation']
            }
        ]
        
        return self.phases
    
    def create_timeline(self, start_date: datetime):
        """Crée timeline avec dates"""
        timeline = []
        current_date = start_date
        
        for phase in self.phases:
            end_date = current_date + timedelta(weeks=phase['duration_weeks'])
            timeline.append({
                'phase': phase['name'],
                'start': current_date,
                'end': end_date,
                'duration': phase['duration_weeks'],
                'deliverables': phase['deliverables']
            })
            current_date = end_date
        
        return timeline
    
    def identify_milestones(self):
        """Identifie milestones clés"""
        self.milestones = [
            {
                'name': 'Literature Review Complete',
                'date': None,  # Calculé depuis timeline
                'criteria': 'Review document approved'
            },
            {
                'name': 'Baseline Results',
                'date': None,
                'criteria': 'Baseline model trained and evaluated'
            },
            {
                'name': 'Experiments Complete',
                'date': None,
                'criteria': 'All experiments run, results analyzed'
            },
            {
                'name': 'Paper Submission',
                'date': None,
                'criteria': 'Paper submitted to venue'
            }
        ]
        
        return self.milestones
```

---

## Gestion des Tâches

### Organisation et Priorisation

```python
class TaskManagement:
    """
    Gestion des tâches de recherche
    """
    
    def __init__(self):
        self.tasks = []
        self.current_tasks = []
        self.completed_tasks = []
    
    def create_task(self, title: str, description: str, 
                   priority: str = 'medium', 
                   estimated_hours: float = 0,
                   dependencies: List[str] = None):
        """Crée nouvelle tâche"""
        task = {
            'id': len(self.tasks) + 1,
            'title': title,
            'description': description,
            'priority': priority,  # 'high', 'medium', 'low'
            'status': 'todo',
            'estimated_hours': estimated_hours,
            'actual_hours': 0,
            'dependencies': dependencies or [],
            'assigned_to': None,
            'due_date': None,
            'created_date': datetime.now()
        }
        
        self.tasks.append(task)
        self.current_tasks.append(task)
        
        return task
    
    def prioritize_tasks(self):
        """Priorise tâches"""
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        
        self.current_tasks.sort(
            key=lambda t: (
                -priority_order.get(t['priority'], 0),
                len(t['dependencies'])
            )
        )
        
        return self.current_tasks
    
    def get_next_tasks(self, n: int = 5):
        """Récupère prochaines tâches à faire"""
        prioritized = self.prioritize_tasks()
        
        # Filtrer tâches sans dépendances non complétées
        ready_tasks = []
        for task in prioritized:
            if task['status'] != 'todo':
                continue
            
            # Vérifier dépendances
            deps_complete = all(
                any(t['id'] == dep and t['status'] == 'completed' 
                    for t in self.tasks)
                for dep in task.get('dependencies', [])
            )
            
            if deps_complete or not task.get('dependencies'):
                ready_tasks.append(task)
        
        return ready_tasks[:n]

# Exemple
task_mgr = TaskManagement()

task_mgr.create_task(
    "Literature review on tensor compression",
    "Review papers on tensor network compression",
    priority='high',
    estimated_hours=20
)

task_mgr.create_task(
    "Implement baseline model",
    "Train and evaluate baseline ResNet",
    priority='high',
    estimated_hours=8,
    dependencies=[]
)

next_tasks = task_mgr.get_next_tasks(3)
print(f"Next tasks: {[t['title'] for t in next_tasks]}")
```

---

## Collaboration

### Gestion d'Équipe

```python
class CollaborationManagement:
    """
    Gestion de collaboration recherche
    """
    
    def __init__(self):
        self.team_members = []
        self.roles = {}
        self.communication_channels = {}
    
    def setup_team(self, members: List[Dict]):
        """Configure équipe"""
        self.team_members = members
        
        # Assigner rôles
        roles = ['lead', 'researcher', 'developer', 'reviewer']
        for i, member in enumerate(members):
            self.roles[member['name']] = roles[i % len(roles)]
        
        return self.team_members
    
    def assign_responsibilities(self):
        """Assigne responsabilités"""
        responsibilities = {
            'lead': [
                'Project planning',
                'Coordination',
                'Paper writing (main)'
            ],
            'researcher': [
                'Literature review',
                'Experiments',
                'Analysis'
            ],
            'developer': [
                'Code implementation',
                'Code review',
                'Documentation'
            ],
            'reviewer': [
                'Code review',
                'Paper review',
                'Quality assurance'
            ]
        }
        
        return responsibilities
    
    def setup_communication(self):
        """Configure canaux communication"""
        self.communication_channels = {
            'github': {
                'purpose': 'Code collaboration, issues, PRs',
                'best_practices': [
                    'Use issues for tasks',
                    'PR reviews required',
                    'Clear commit messages'
                ]
            },
            'slack_teams': {
                'purpose': 'Daily communication',
                'best_practices': [
                    'Daily standups',
                    'Channel organization',
                    'Document decisions'
                ]
            },
            'shared_docs': {
                'purpose': 'Documentation collaborative',
                'tools': ['Google Docs', 'Notion', 'Overleaf'],
                'best_practices': [
                    'Version control',
                    'Comments for feedback',
                    'Clear ownership'
                ]
            },
            'meetings': {
                'frequency': 'Weekly',
                'agenda': [
                    'Progress updates',
                    'Blockers',
                    'Next steps'
                ]
            }
        }
        
        return self.communication_channels
```

---

## Suivi de Progression

### Métriques et Reporting

```python
class ProgressTracking:
    """
    Suivi de progression projet
    """
    
    def __init__(self, project_plan: ResearchProjectPlan):
        self.project_plan = project_plan
        self.progress_log = []
    
    def track_progress(self, phase: str, completion_pct: float, 
                     notes: str = ""):
        """Track progression phase"""
        entry = {
            'date': datetime.now(),
            'phase': phase,
            'completion': completion_pct,
            'notes': notes
        }
        self.progress_log.append(entry)
        
        return entry
    
    def calculate_overall_progress(self) -> float:
        """Calcule progression globale"""
        if not self.progress_log:
            return 0.0
        
        # Poids par phase (basé sur durée)
        phase_weights = {
            phase['name']: phase['duration_weeks']
            for phase in self.project_plan.phases
        }
        total_weight = sum(phase_weights.values())
        
        # Progression pondérée
        weighted_progress = 0.0
        for phase_name, weight in phase_weights.items():
            phase_entries = [e for e in self.progress_log 
                           if e['phase'] == phase_name]
            if phase_entries:
                latest_completion = max(phase_entries, 
                                      key=lambda x: x['date'])['completion']
                weighted_progress += latest_completion * (weight / total_weight)
        
        return weighted_progress
    
    def generate_progress_report(self) -> str:
        """Génère rapport progression"""
        overall = self.calculate_overall_progress()
        
        report = f"\n{'='*70}\n"
        report += f"Progress Report: {self.project_plan.project_name}\n"
        report += f"{'='*70}\n"
        report += f"\nOverall Progress: {overall:.1f}%\n\n"
        
        report += "Phase Progress:\n"
        for phase in self.project_plan.phases:
            phase_entries = [e for e in self.progress_log 
                           if e['phase'] == phase['name']]
            if phase_entries:
                latest = max(phase_entries, key=lambda x: x['date'])
                report += f"  {phase['name']}: {latest['completion']:.1f}%\n"
        
        return report

# Exemple
project = ResearchProjectPlan("Tensor Compression Research", duration_months=6)
project.define_phases()

tracker = ProgressTracking(project)
tracker.track_progress("Literature Review", 75.0, "Most papers reviewed")
tracker.track_progress("Implementation", 50.0, "Baseline implemented")

print(tracker.generate_progress_report())
```

---

## Gestion des Ressources

### Budget et Infrastructure

```python
class ResourceManagement:
    """
    Gestion ressources projet recherche
    """
    
    def __init__(self):
        self.resources = {
            'compute': {
                'gpus': [],
                'clusters': [],
                'cloud_credits': 0
            },
            'storage': {
                'local': 0,
                'cloud': 0,
                'backup': 0
            },
            'software': {
                'licenses': [],
                'subscriptions': []
            },
            'budget': {
                'allocated': 0,
                'spent': 0,
                'remaining': 0
            }
        }
    
    def estimate_compute_needs(self, experiments: List[Dict]) -> Dict:
        """Estime besoins computationnels"""
        total_gpu_hours = 0
        
        for exp in experiments:
            # Estimation basée sur durée entraînement
            gpu_hours_per_exp = (
                exp.get('training_hours', 10) * 
                exp.get('n_replications', 5) *
                exp.get('n_configurations', 3)
            )
            total_gpu_hours += gpu_hours_per_exp
        
        return {
            'total_gpu_hours': total_gpu_hours,
            'estimated_cost': total_gpu_hours * 1.0,  # $/GPU-hour
            'timeline': 'Based on experiment schedule'
        }
    
    def track_resource_usage(self):
        """Track utilisation ressources"""
        usage = {
            'compute': {
                'gpu_hours_used': 0,
                'gpu_hours_remaining': 0
            },
            'storage': {
                'data_size_gb': 0,
                'model_checkpoints_gb': 0,
                'logs_gb': 0
            },
            'budget': {
                'compute_cost': 0,
                'storage_cost': 0,
                'software_cost': 0
            }
        }
        
        return usage
```

---

## Exercices

### Exercice 26.5.1
Créez un plan de projet de recherche avec phases, milestones, et timeline.

### Exercice 26.5.2
Organisez vos tâches de recherche avec système de priorisation.

### Exercice 26.5.3
Configurez collaboration pour équipe de recherche (GitHub, communication).

### Exercice 26.5.4
Estimez besoins computationnels pour série d'expériences planifiées.

---

## Points Clés à Retenir

> 📌 **Planification claire avec phases et milestones guide projet**

> 📌 **Gestion de tâches avec priorités optimise productivité**

> 📌 **Communication efficace est cruciale pour collaboration**

> 📌 **Suivi de progression permet ajustements en temps réel**

> 📌 **Gestion ressources évite dépassements budgétaires**

> 📌 **Outils appropriés (GitHub, project management) facilitent organisation**

---

*Section précédente : [26.4 Reproductibilité](./26_04_Reproductibilite.md)*

