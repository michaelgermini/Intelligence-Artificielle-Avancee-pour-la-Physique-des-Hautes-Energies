# 22.1 Environnement de Développement

---

## Introduction

Un environnement de développement bien configuré est essentiel pour la productivité et la reproductibilité en deep learning. Cette section présente les outils et pratiques pour configurer un environnement Python professionnel, incluant gestion de dépendances, environnements virtuels, et outils de développement.

---

## Gestionnaires d'Environnement

### Conda vs venv

```python
"""
Gestionnaires d'environnement Python

1. conda (Anaconda/Miniconda)
   - Gère Python + packages + dépendances système
   - Excellent pour packages scientifiques (NumPy, SciPy)
   - Support CUDA intégré
   
2. venv (built-in Python)
   - Simple et léger
   - Uniquement gestion packages Python
   - Nécessite installation séparée des dépendances système
   
3. pip + virtualenv
   - Standard Python
   - Flexible mais plus manuel
"""

# Exemple création environnement conda
"""
conda create -n hep_dl python=3.10
conda activate hep_dl
conda install numpy pytorch tensorflow -c conda-forge
"""

# Exemple création environnement venv
"""
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
pip install numpy torch tensorflow
"""

class EnvironmentManager:
    """
    Gestionnaire d'environnement
    """
    
    def __init__(self):
        self.recommended_packages = {
            'core': ['numpy', 'scipy', 'pandas', 'matplotlib'],
            'deep_learning': ['torch', 'tensorflow', 'keras'],
            'development': ['jupyter', 'ipython', 'pytest', 'black', 'flake8'],
            'scientific': ['scikit-learn', 'scikit-image', 'h5py'],
            'visualization': ['seaborn', 'plotly', 'bokeh']
        }
    
    def display_packages(self):
        """Affiche packages recommandés"""
        print("\n" + "="*70)
        print("Packages Recommandés par Catégorie")
        print("="*70)
        
        for category, packages in self.recommended_packages.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for pkg in packages:
                print(f"  • {pkg}")

env_manager = EnvironmentManager()
env_manager.display_packages()
```

---

## Installation PyTorch

### Avec CUDA Support

```python
"""
Installation PyTorch avec support GPU

1. Vérifier version CUDA disponible
   nvidia-smi  # Affiche version CUDA installée

2. Installer PyTorch avec CUDA
   # Pour CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # Pour CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CPU seulement
   pip install torch torchvision torchaudio
"""

import torch

def check_pytorch_installation():
    """Vérifie installation PyTorch"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available - using CPU")

check_pytorch_installation()
```

---

## Installation TensorFlow

### Configuration GPU

```python
"""
Installation TensorFlow

1. Installer TensorFlow
   pip install tensorflow  # CPU
   pip install tensorflow-gpu  # GPU (ancienne méthode)
   pip install tensorflow[and-cuda]  # Nouvelle méthode avec CUDA

2. Vérifier installation
   python -c "import tensorflow as tf; print(tf.__version__)"
"""

try:
    import tensorflow as tf
    
    def check_tensorflow_installation():
        """Vérifie installation TensorFlow"""
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        if len(tf.config.list_physical_devices('GPU')) > 0:
            print("GPU devices:")
            for gpu in tf.config.list_physical_devices('GPU'):
                print(f"  • {gpu}")
    
    check_tensorflow_installation()
except ImportError:
    print("TensorFlow not installed")
```

---

## Configuration Jupyter

### Notebooks et Extensions

```python
"""
Configuration Jupyter Notebook

1. Installation
   pip install jupyter jupyterlab
   
2. Extensions utiles
   pip install jupyter_contrib_nbextensions
   jupyter contrib nbextension install --user
   
3. Kernel pour environnement virtuel
   python -m ipykernel install --user --name=hep_dl --display-name="HEP DL"
"""

class JupyterSetup:
    """
    Configuration Jupyter
    """
    
    def __init__(self):
        self.useful_extensions = {
            'code_folding': 'Plier/déplier sections de code',
            'variable_inspector': 'Inspecter variables globales',
            'toc2': 'Table des matières automatique',
            'execute_time': 'Afficher temps d\'exécution',
            'widgetsnbextension': 'Support widgets interactifs'
        }
        
        self.best_practices = [
            'Utiliser cellules markdown pour documentation',
            'Éviter variables globales persistantes',
            'Restart kernel régulièrement',
            'Utiliser .ipynb_checkpoints/ pour sauvegardes',
            'Convertir en .py pour code production'
        ]
    
    def display_setup(self):
        """Affiche configuration"""
        print("\n" + "="*70)
        print("Extensions Jupyter Utiles")
        print("="*70)
        
        for ext, desc in self.useful_extensions.items():
            print(f"  • {ext}: {desc}")
        
        print("\n" + "="*70)
        print("Bonnes Pratiques")
        print("="*70)
        
        for i, practice in enumerate(self.best_practices, 1):
            print(f"{i}. {practice}")

jupyter_setup = JupyterSetup()
jupyter_setup.display_setup()
```

---

## Requirements.txt et Environment Files

### Gestion des Dépendances

```python
"""
Fichier requirements.txt

Exemple pour projet deep learning HEP:

torch>=2.0.0
tensorflow>=2.13.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
tensorly>=0.8.0
"""

class DependencyManagement:
    """
    Gestion des dépendances
    """
    
    def generate_requirements(self, environment_name='hep_dl'):
        """Génère requirements.txt depuis environnement"""
        requirements = f"""# Requirements for {environment_name}
# Core scientific computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Deep Learning
torch>=2.0.0
tensorflow>=2.13.0
keras>=2.13.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
jupyter>=1.0.0
ipython>=8.12.0
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Scientific libraries
scikit-learn>=1.3.0
scikit-image>=0.21.0
h5py>=3.9.0

# Tensor networks
tensorly>=0.8.0
"""
        return requirements
    
    def generate_conda_env(self, environment_name='hep_dl'):
        """Génère fichier environment.yml pour conda"""
        conda_env = f"""name: {environment_name}
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - scipy
  - pandas
  - matplotlib
  - jupyter
  - pip
  - pip:
    - torch>=2.0.0
    - tensorflow>=2.13.0
    - keras>=2.13.0
    - tensorly>=0.8.0
"""
        return conda_env

dep_manager = DependencyManagement()

print("\n" + "="*70)
print("Exemple requirements.txt")
print("="*70)
print(dep_manager.generate_requirements())

print("\n" + "="*70)
print("Exemple environment.yml (conda)")
print("="*70)
print(dep_manager.generate_conda_env())
```

---

## Outils de Développement

### Linters et Formatters

```python
"""
Outils de développement

1. black: Formateur de code automatique
   pip install black
   black .

2. flake8: Linter pour style et erreurs
   pip install flake8
   flake8 .

3. mypy: Type checking
   pip install mypy
   mypy .

4. pytest: Tests unitaires
   pip install pytest
   pytest tests/
"""

class DevTools:
    """
    Outils de développement
    """
    
    def __init__(self):
        self.tools = {
            'black': {
                'purpose': 'Formatage automatique de code',
                'config': '# pyproject.toml\n[tool.black]\nline-length = 88'
            },
            'flake8': {
                'purpose': 'Détection erreurs et style',
                'config': '# .flake8\n[flake8]\nmax-line-length = 88\nexclude = .git,__pycache__,venv'
            },
            'mypy': {
                'purpose': 'Type checking statique',
                'config': '# mypy.ini\n[mypy]\nignore_missing_imports = True'
            },
            'pytest': {
                'purpose': 'Tests unitaires',
                'config': '# pytest.ini\n[pytest]\ntestpaths = tests'
            }
        }
    
    def display_tools(self):
        """Affiche outils"""
        print("\n" + "="*70)
        print("Outils de Développement")
        print("="*70)
        
        for tool, info in self.tools.items():
            print(f"\n{tool}:")
            print(f"  Purpose: {info['purpose']}")
            print(f"  Config: {info['config']}")

dev_tools = DevTools()
dev_tools.display_tools()
```

---

## Configuration IDE

### VS Code et PyCharm

```python
"""
Configuration IDE recommandée

VS Code:
- Extension Python
- Extension Jupyter
- Extension Pylance (type checking)
- Extension Black Formatter

PyCharm:
- Interpréteur Python configuré
- Support Jupyter intégré
- Débogueur intégré
- Git intégré
"""

class IDESetup:
    """
    Configuration IDE
    """
    
    def __init__(self):
        self.vscode_extensions = [
            'ms-python.python',
            'ms-python.vscode-pylance',
            'ms-toolsai.jupyter',
            'ms-python.black-formatter',
            'ms-python.flake8',
            'ms-python.mypy-type-checker'
        ]
        
        self.recommended_settings = {
            'python.formatting.provider': 'black',
            'python.linting.enabled': True,
            'python.linting.flake8Enabled': True,
            'editor.formatOnSave': True
        }
    
    def display_vscode_setup(self):
        """Affiche configuration VS Code"""
        print("\n" + "="*70)
        print("Extensions VS Code Recommandées")
        print("="*70)
        
        for ext in self.vscode_extensions:
            print(f"  • {ext}")
        
        print("\n" + "="*70)
        print("Settings Recommandés (settings.json)")
        print("="*70)
        
        import json
        print(json.dumps(self.recommended_settings, indent=2))

ide_setup = IDESetup()
ide_setup.display_vscode_setup()
```

---

## Version Control (Git)

### Workflow Git

```python
"""
Workflow Git recommandé

1. Initialiser repository
   git init
   git add .
   git commit -m "Initial commit"

2. Créer .gitignore
   # Python
   __pycache__/
   *.py[cod]
   *.so
   .Python
   venv/
   env/
   
   # Jupyter
   .ipynb_checkpoints/
   
   # Models
   *.pth
   *.h5
   models/
   checkpoints/
   
   # Data
   data/
   datasets/
   *.h5
   *.hdf5
"""

class GitSetup:
    """
    Configuration Git
    """
    
    def generate_gitignore(self):
        """Génère .gitignore pour projet DL"""
        gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/
.conda/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Models and checkpoints
*.pth
*.pt
*.h5
*.hdf5
models/
checkpoints/
weights/

# Data
data/
datasets/
*.csv
*.parquet
*.h5
*.hdf5

# Logs
logs/
*.log
tensorboard/

# OS
.DS_Store
Thumbs.db
"""
        return gitignore

git_setup = GitSetup()
print("\n" + "="*70)
print(".gitignore pour Projet Deep Learning")
print("="*70)
print(git_setup.generate_gitignore())
```

---

## Exercices

### Exercice 22.1.1
Créez un environnement conda avec Python 3.10, installez PyTorch avec support CUDA, et vérifiez que le GPU est accessible.

### Exercice 22.1.2
Configurez un environnement virtuel avec venv, créez un fichier requirements.txt avec toutes les dépendances nécessaires, et installez-les.

### Exercice 22.1.3
Configurez Jupyter Notebook avec un kernel pour votre environnement, et créez un notebook de test simple.

### Exercice 22.1.4
Configurez VS Code ou PyCharm avec les extensions recommandées et testez le formatage automatique avec black.

---

## Points Clés à Retenir

> 📌 **Conda est excellent pour packages scientifiques et dépendances système**

> 📌 **venv est plus léger mais nécessite gestion manuelle dépendances**

> 📌 **Le support GPU nécessite installation spécifique CUDA + frameworks**

> 📌 **Jupyter est essentiel pour développement interactif**

> 📌 **Les outils de développement (black, flake8, pytest) améliorent qualité code**

> 📌 **Git et .gitignore sont essentiels pour version control**

---

*Section précédente : [22.0 Introduction](./22_introduction.md) | Section suivante : [22.2 NumPy](./22_02_NumPy.md)*

