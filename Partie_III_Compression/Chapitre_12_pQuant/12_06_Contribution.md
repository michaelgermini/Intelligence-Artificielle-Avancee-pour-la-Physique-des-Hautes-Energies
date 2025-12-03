# 12.6 Contribution Open-Source et Bonnes Pratiques

---

## Introduction

pQuant est un projet **open-source** développé au CERN. Cette section présente comment contribuer au projet, les bonnes pratiques de développement, et les standards de code attendus.

---

## Structure du Projet

### Organisation du Repository

```
pquant/
├── README.md              # Documentation principale
├── CONTRIBUTING.md        # Guide de contribution
├── LICENSE                # Licence (Apache 2.0 ou similaire)
├── setup.py              # Installation
├── requirements.txt      # Dépendances
├── .github/
│   └── workflows/        # CI/CD
├── tests/                # Tests unitaires
├── examples/             # Exemples d'utilisation
├── docs/                 # Documentation Sphinx
└── pquant/               # Code source
```

---

## Workflow de Contribution

### 1. Fork et Clone

```bash
# Fork le repository sur GitHub
# Clone votre fork
git clone https://github.com/votre-username/pquant.git
cd pquant

# Ajoute le repository original comme upstream
git remote add upstream https://github.com/cern/pquant.git
```

### 2. Créer une Branche

```bash
# Crée une branche pour votre feature
git checkout -b feature/ma-nouvelle-feature

# Ou pour un bugfix
git checkout -b fix/correction-bug
```

### 3. Développement

```python
# Respecter les standards de code
# - PEP 8 pour Python
# - Docstrings pour toutes les fonctions
# - Tests unitaires pour nouvelles fonctionnalités

def nouvelle_methode_compression(config):
    """
    Nouvelle méthode de compression
    
    Args:
        config: Configuration dict
    
    Returns:
        Modèle compressé
    
    Raises:
        ValueError: Si config invalide
    """
    # Implémentation
    pass
```

### 4. Tests

```python
# tests/test_nouvelle_methode.py

import unittest
import torch
from pquant.methods.new_method import NewCompressionMethod

class TestNewCompression(unittest.TestCase):
    """Tests pour nouvelle méthode"""
    
    def setUp(self):
        """Setup avant chaque test"""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        )
    
    def test_compression_basic(self):
        """Test basique de compression"""
        config = {'param': 64}
        method = NewCompressionMethod(config)
        
        compressed = method.compress(self.model)
        
        # Vérifications
        self.assertIsNotNone(compressed)
        # ...
    
    def test_compression_ratio(self):
        """Test du ratio de compression"""
        # ...
```

### 5. Commit et Push

```bash
# Commit avec message clair
git commit -m "feat: Add new compression method X"

# Push vers votre fork
git push origin feature/ma-nouvelle-feature
```

### 6. Pull Request

- Créez une PR sur GitHub
- Description claire de la modification
- Référence les issues liées
- Attendez la revue de code

---

## Standards de Code

### Style de Code

```python
# PEP 8 compliance
# Utiliser black pour formatting automatique
# black pquant/

# Type hints recommandés
from typing import Dict, Any, Optional, List

def compress_model(
    model: torch.nn.Module,
    config: Dict[str, Any],
    train_loader: Optional[DataLoader] = None
) -> torch.nn.Module:
    """
    Compresse un modèle
    
    Args:
        model: Modèle PyTorch à compresser
        config: Configuration de compression
        train_loader: DataLoader pour calibration (optionnel)
    
    Returns:
        Modèle compressé
    
    Raises:
        ValueError: Si la configuration est invalide
        RuntimeError: Si la compression échoue
    """
    pass
```

### Docstrings

```python
def complex_function(param1, param2, param3=None):
    """
    Description courte et claire
    
    Description détaillée si nécessaire.
    Peut s'étendre sur plusieurs lignes.
    
    Args:
        param1 (type): Description
        param2 (type): Description
        param3 (Optional[type]): Description optionnelle
    
    Returns:
        type: Description de la valeur retournée
    
    Raises:
        ValueError: Quand se produit l'erreur
    
    Example:
        >>> result = complex_function(1, 2, param3=3)
        >>> print(result)
        6
    """
    pass
```

---

## Tests

### Structure des Tests

```python
# tests/
#   ├── test_core/
#   │   ├── test_compression_method.py
#   │   └── test_layer_adapter.py
#   ├── test_methods/
#   │   ├── test_low_rank.py
#   │   ├── test_quantization.py
#   │   └── test_tensor_networks.py
#   └── test_pipelines/
#       └── test_compression_pipeline.py
```

### Exemple de Test Complet

```python
import unittest
import torch
from pquant.methods.low_rank import LowRankCompression

class TestLowRankCompression(unittest.TestCase):
    """Tests complets pour LowRankCompression"""
    
    def setUp(self):
        """Setup pour chaque test"""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )
        self.config = {'rank': 32}
    
    def test_compression_ratio(self):
        """Vérifie le ratio de compression"""
        method = LowRankCompression(self.config)
        compressed = method.compress(self.model)
        
        info = method.get_compression_info(self.model, compressed)
        
        self.assertGreater(info['compression_ratio'], 1.0)
        self.assertLessEqual(info['compressed_params'], 
                           info['original_params'])
    
    def test_forward_consistency(self):
        """Vérifie que le forward fonctionne"""
        method = LowRankCompression(self.config)
        compressed = method.compress(self.model)
        
        x = torch.randn(10, 100)
        
        with torch.no_grad():
            y_original = self.model(x)
            y_compressed = compressed(x)
        
        # Les outputs doivent avoir la même shape
        self.assertEqual(y_original.shape, y_compressed.shape)
    
    def test_error_threshold(self):
        """Test avec seuil d'erreur"""
        config = {'rank': 32, 'error_threshold': 0.01}
        method = LowRankCompression(config)
        compressed = method.compress(self.model)
        
        # Vérifie que l'erreur est sous le seuil
        # ...

# Exécution des tests
if __name__ == '__main__':
    unittest.main()
```

---

## Documentation

### Documentation du Code

```python
class CompressionMethod:
    """
    Classe de base pour toutes les méthodes de compression
    
    Cette classe définit l'interface standard que toutes les méthodes
    de compression doivent implémenter.
    
    Attributes:
        config (Dict[str, Any]): Configuration de la méthode
        name (str): Nom de la méthode
    
    Example:
        >>> config = {'rank': 64}
        >>> method = LowRankCompression(config)
        >>> compressed = method.compress(model)
    """
    
    def compress(self, model):
        """
        Compresse un modèle
        
        Cette méthode doit être implémentée par toutes les sous-classes.
        """
        raise NotImplementedError
```

### Documentation Sphinx

```rst
.. _low-rank-compression:

Low-Rank Compression
====================

La compression par rang faible utilise la décomposition SVD...

Usage
-----

.. code-block:: python

    from pquant import LowRankCompression
    
    compressor = LowRankCompression({'rank': 64})
    compressed = compressor.compress(model)

API Reference
-------------

.. autoclass:: pquant.methods.low_rank.LowRankCompression
   :members:
   :undoc-members:
```

---

## CI/CD

### GitHub Actions

```yaml
# .github/workflows/tests.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=pquant --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Code Review Guidelines

### Checklist pour PRs

- [ ] Code respecte PEP 8
- [ ] Docstrings pour toutes les fonctions publiques
- [ ] Tests unitaires ajoutés/modifiés
- [ ] Tous les tests passent
- [ ] Documentation mise à jour
- [ ] Pas de breaking changes (ou documentés)
- [ ] Exemples d'utilisation si nouvelle fonctionnalité

---

## Exercices

### Exercice 12.6.1
Créez un test complet pour une nouvelle méthode de compression.

### Exercice 12.6.2
Rédigez la documentation pour une fonctionnalité existante.

---

## Points Clés à Retenir

> 📌 **Respecter les standards de code (PEP 8, type hints, docstrings)**

> 📌 **Toujours ajouter des tests pour nouvelles fonctionnalités**

> 📌 **Documentation claire et complète**

> 📌 **Suivre le workflow Git standard (fork, branch, PR)**

> 📌 **Code review est important pour maintenir la qualité**

---

*Chapitre suivant : [Chapitre 13 - Introduction aux FPGAs](../Partie_IV_Hardware/Chapitre_13_FPGA_Introduction/13_introduction.md)*

