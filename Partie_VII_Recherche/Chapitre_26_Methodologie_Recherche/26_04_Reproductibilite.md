# 26.4 Reproductibilité et Bonnes Pratiques

---

## Introduction

La **reproductibilité** est un pilier de la recherche scientifique. Elle permet à d'autres chercheurs de vérifier et de construire sur vos résultats. Cette section présente les pratiques pour garantir la reproductibilité, incluant versioning, documentation, et partage de code et données.

---

## Principes de Reproductibilité

### Définition et Importance

```python
"""
Types de reproductibilité:

1. Reproductibilité computationnelle
   - Même code + même données = mêmes résultats
   - Nécessite: versioning code, données, environnement

2. Reproductibilité méthodologique
   - Méthodes clairement documentées
   - Autres peuvent suivre procédure

3. Reproductibilité empirique
   - Expériences peuvent être répétées
   - Résultats similaires obtenus
"""

class ReproducibilityFramework:
    """
    Framework pour garantir reproductibilité
    """
    
    def __init__(self):
        self.components = {
            'code_versioning': {
                'tool': 'Git',
                'requirements': ['requirements.txt', 'environment.yml'],
                'best_practices': ['Commit fréquent', 'Tags pour versions']
            },
            'data_management': {
                'tool': 'DVC, Git LFS',
                'requirements': ['Data versioning', 'Metadata'],
                'best_practices': ['Hash checksums', 'Provenance tracking']
            },
            'environment': {
                'tool': 'Docker, Conda',
                'requirements': ['Containerization', 'Environment files'],
                'best_practices': ['Lock dependencies', 'Document versions']
            },
            'documentation': {
                'tool': 'README, Jupyter notebooks',
                'requirements': ['Setup instructions', 'Usage examples'],
                'best_practices': ['Clear instructions', 'Reproducible notebooks']
            }
        }
```

---

## Version Control et Code

### Gestion de Versions

```python
"""
Stratégies pour reproductibilité code:

1. Versioning strict
   - Tags pour versions publiées
   - Branches pour expériences
   - Commit messages descriptifs

2. Dependencies fixes
   - requirements.txt avec versions exactes
   - environment.yml pour conda
   - Pipfile.lock pour pipenv

3. Configuration externalisée
   - Config files (YAML, JSON)
   - Pas de hardcoding
   - Paramètres versionnés
"""

class CodeReproducibility:
    """
    Garantir reproductibilité code
    """
    
    def create_reproducible_setup(self):
        """Crée setup reproductible"""
        setup = {
            'requirements_txt': self.generate_requirements(),
            'environment_yml': self.generate_conda_env(),
            'dockerfile': self.generate_dockerfile(),
            'config_template': self.create_config_template()
        }
        return setup
    
    def generate_requirements(self):
        """Génère requirements.txt avec versions"""
        requirements = """# Core dependencies
torch==2.0.1
numpy==1.24.3
scipy==1.10.1
pandas==2.0.3

# Deep learning
tensorflow==2.13.0
keras==2.13.1

# Scientific computing
scikit-learn==1.3.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Development
jupyter==1.0.0
pytest==7.4.0
"""
        return requirements
    
    def generate_conda_env(self):
        """Génère environment.yml"""
        env_yml = """name: research_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - numpy=1.24.3
  - pip
  - pip:
    - tensorflow==2.13.0
    - jupyter==1.0.0
"""
        return env_yml
    
    def create_config_template(self):
        """Crée template de configuration"""
        config_template = {
            'experiment_name': 'experiment_001',
            'random_seed': 42,
            'model': {
                'type': 'resnet18',
                'pretrained': True
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            },
            'compression': {
                'method': 'pruning',
                'sparsity': 0.5
            },
            'data': {
                'dataset': 'CIFAR-10',
                'data_path': './data'
            }
        }
        return config_template
```

---

## Gestion des Données

### Versioning et Partage

```python
class DataReproducibility:
    """
    Gestion données pour reproductibilité
    """
    
    def __init__(self):
        self.data_protocol = {
            'storage': {
                'raw_data': 'Immutable, versioned',
                'processed_data': 'Hash checksums',
                'splits': 'Fixed random seeds'
            },
            'documentation': {
                'metadata': 'Data provenance',
                'preprocessing': 'Steps documented',
                'licenses': 'Usage rights'
            },
            'sharing': {
                'formats': 'Standard formats (HDF5, parquet)',
                'access': 'DOI, persistent URLs',
                'licenses': 'Clear usage rights'
            }
        }
    
    def create_data_manifest(self, data_path: str):
        """Crée manifeste de données"""
        import hashlib
        from pathlib import Path
        
        manifest = {
            'data_path': data_path,
            'files': [],
            'checksums': {},
            'metadata': {}
        }
        
        data_dir = Path(data_path)
        for file_path in data_dir.rglob('*'):
            if file_path.is_file():
                # Calculer hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                manifest['files'].append(str(file_path.relative_to(data_dir)))
                manifest['checksums'][str(file_path.relative_to(data_dir))] = file_hash
        
        return manifest
    
    def verify_data_integrity(self, data_path: str, manifest: Dict):
        """Vérifie intégrité données"""
        from pathlib import Path
        import hashlib
        
        data_dir = Path(data_path)
        verified = True
        issues = []
        
        for file_rel_path, expected_hash in manifest['checksums'].items():
            file_path = data_dir / file_rel_path
            
            if not file_path.exists():
                verified = False
                issues.append(f"Missing: {file_rel_path}")
                continue
            
            with open(file_path, 'rb') as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            
            if actual_hash != expected_hash:
                verified = False
                issues.append(f"Hash mismatch: {file_rel_path}")
        
        return {
            'verified': verified,
            'issues': issues
        }
```

---

## Documentation et Notebooks

### Documentation Reproductible

```python
class DocumentationReproducibility:
    """
    Documentation pour reproductibilité
    """
    
    def create_readme_template(self):
        """Crée template README"""
        readme = """# Project Title

## Description
Brief description of the project and its objectives.

## Installation

### Requirements
- Python 3.10+
- See requirements.txt for dependencies

### Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate research_env

# Or use pip
pip install -r requirements.txt
```

## Usage

### Running Experiments
```bash
python train.py --config configs/experiment_001.yaml
```

### Reproducing Results
1. Set random seed: `--seed 42`
2. Use exact versions from requirements.txt
3. Follow steps in EXPERIMENTS.md

## Data
- Data location: `./data`
- Data manifest: `data_manifest.json`
- Download script: `scripts/download_data.sh`

## Results
- Model checkpoints: `./checkpoints`
- Experiment logs: `./logs`
- Figures: `./figures`

## Citation
If you use this code, please cite:
```
@article{...}
```

## License
MIT License
"""
        return readme
    
    def create_experiment_log(self, experiment_name: str):
        """Crée log d'expérience"""
        log = {
            'experiment_name': experiment_name,
            'date': '2024-01-01',
            'researcher': 'Your Name',
            'objective': 'Description of experiment objective',
            'configuration': 'configs/experiment_001.yaml',
            'git_commit': 'abc123def456',
            'environment': {
                'python_version': '3.10.0',
                'torch_version': '2.0.1',
                'cuda_version': '11.8'
            },
            'results': {
                'metrics': {},
                'artifacts': []
            },
            'notes': 'Any additional notes'
        }
        return log

# Exemple notebook reproductible
class ReproducibleNotebook:
    """
    Bonnes pratiques notebooks Jupyter
    """
    
    def notebook_best_practices(self):
        """Présente bonnes pratiques"""
        practices = [
            "Fixer random seeds en début notebook",
            "Définir toutes variables avant exécution",
            "Éviter variables globales persistantes",
            "Documenter chaque cellule",
            "Utiliser markdown pour explications",
            "Sauvegarder sorties intermédiaires",
            "Versionner notebooks (Git)",
            "Nettoyer output avant commit",
            "Utiliser environment.yml pour dépendances",
            "Documenter ordre d'exécution"
        ]
        return practices
    
    def example_reproducible_notebook(self):
        """Exemple structure notebook"""
        notebook_structure = """
# Cellule 1: Imports et setup
import torch
import numpy as np
import random

# Fixer seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Cellule 2: Configuration
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# Cellule 3: Chargement données
# (avec documentation)

# Cellule 4: Modèle
# (avec documentation)

# Cellule 5: Entraînement
# (avec documentation)

# Cellule 6: Résultats
# (avec visualisations)
"""
        return notebook_structure
```

---

## Checklist de Reproductibilité

### Vérifications

```python
class ReproducibilityChecklist:
    """
    Checklist pour vérifier reproductibilité
    """
    
    def __init__(self):
        self.checklist = {
            'code': [
                'Code versionné dans Git',
                'README avec instructions claires',
                'Requirements avec versions exactes',
                'Configuration externalisée',
                'Random seeds fixés',
                'Pas de hardcoding'
            ],
            'data': [
                'Données versionnées ou accessibles',
                'Checksums pour vérification intégrité',
                'Manifeste de données',
                'Documentation preprocessing',
                'Licences documentées'
            ],
            'environment': [
                'Environment file (conda/pip)',
                'Docker container (optionnel)',
                'Versions dépendances spécifiées',
                'Instructions installation claires'
            ],
            'documentation': [
                'README complet',
                'Documentation fonctions principales',
                'Exemples d\'utilisation',
                'Log d\'expériences',
                'Notes sur choix de design'
            ],
            'results': [
                'Résultats versionnés',
                'Figures avec code génération',
                'Métriques documentées',
                'Modèles sauvegardés'
            ]
        }
    
    def verify_project(self, project_path: str) -> Dict:
        """Vérifie projet pour reproductibilité"""
        results = {}
        
        for category, items in self.checklist.items():
            category_results = []
            for item in items:
                # Vérifier item (simplifié)
                verified = self.check_item(project_path, category, item)
                category_results.append({
                    'item': item,
                    'verified': verified
                })
            results[category] = category_results
        
        return results
    
    def generate_report(self, verification_results: Dict) -> str:
        """Génère rapport de vérification"""
        report = "\n" + "="*70 + "\n"
        report += "Reproducibility Checklist Report\n"
        report += "="*70 + "\n\n"
        
        for category, items in verification_results.items():
            report += f"{category.upper()}:\n"
            for item_result in items:
                status = "✓" if item_result['verified'] else "✗"
                report += f"  {status} {item_result['item']}\n"
            report += "\n"
        
        return report
```

---

## Exercices

### Exercice 26.4.1
Créez un projet reproductible avec requirements.txt, README, et configuration.

### Exercice 26.4.2
Générez un manifeste de données avec checksums pour vos données.

### Exercice 26.4.3
Créez un notebook Jupyter reproductible suivant bonnes pratiques.

### Exercice 26.4.4
Vérifiez un projet existant avec checklist de reproductibilité.

---

## Points Clés à Retenir

> 📌 **Versioning strict (Git) est essentiel pour reproductibilité**

> 📌 **Dépendances fixées garantissent environnement identique**

> 📌 **Documentation claire permet reproduction par autres**

> 📌 **Checksums permettent vérifier intégrité données**

> 📌 **Random seeds fixes garantissent reproductibilité résultats**

> 📌 **Configuration externalisée facilite reproduction**

---

*Section précédente : [26.3 Analyse Statistique](./26_03_Analyse_Statistique.md) | Section suivante : [26.5 Gestion Projets](./26_05_Gestion_Projets.md)*

