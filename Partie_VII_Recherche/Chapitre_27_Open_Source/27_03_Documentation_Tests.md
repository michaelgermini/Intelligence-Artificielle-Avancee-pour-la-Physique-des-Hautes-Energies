# 27.3 Documentation et Tests

---

## Introduction

La **documentation** et les **tests** sont essentiels pour la qualité et la maintenabilité des projets open source. Une bonne documentation facilite l'adoption, tandis que des tests complets garantissent la stabilité. Cette section présente les pratiques pour écrire documentation et tests de qualité.

---

## Types de Documentation

### Documentation Complète

```python
"""
Types de documentation:

1. README
   - Vue d'ensemble projet
   - Installation rapide
   - Exemple basique

2. API Documentation
   - Documentation fonctions/classes
   - Paramètres et retours
   - Exemples d'utilisation

3. Tutorials
   - Guides pas à pas
   - Cas d'usage communs
   - Workflows complets

4. Contributing Guide
   - Comment contribuer
   - Standards du projet
   - Process de review

5. Changelog
   - Historique des changements
   - Versions et features
"""
```

---

## README de Qualité

### Structure Recommandée

```python
class READMEStructure:
    """
    Structure recommandée pour README
    """
    
    def create_readme_template(self):
        """Crée template README"""
        readme = """# Project Name

Brief one-line description of the project.

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- See requirements.txt for full list

### Install from PyPI
```bash
pip install project-name
```

### Install from source
```bash
git clone https://github.com/user/project.git
cd project
pip install -e .
```

## Quick Start

```python
import project

# Simple example
result = project.example_function(input_data)
print(result)
```

## Documentation

Full documentation available at: https://project.readthedocs.io

## Examples

See [examples/](examples/) directory for more examples.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

If you use this project, please cite:
```
@article{...}
```

## Acknowledgments

Thanks to all contributors!
"""
        return readme
```

---

## Documentation API

### Docstrings Standards

```python
def well_documented_function(param1: int, param2: str, 
                            optional_param: float = 1.0) -> Dict[str, float]:
    """
    Brief description of what the function does.
    
    More detailed description if needed. Can span multiple lines.
    Explain any important details, algorithms used, etc.
    
    Args:
        param1 (int): Description of param1. Can include examples.
            Example: `param1 = 42`
        param2 (str): Description of param2. Can mention constraints.
            Example: `param2 = "example"`
        optional_param (float, optional): Description of optional parameter.
            Defaults to 1.0.
    
    Returns:
        Dict[str, float]: Description of return value.
            Keys:
                - 'key1': Description of key1
                - 'key2': Description of key2
    
    Raises:
        ValueError: When param1 is negative.
        TypeError: When param2 is not a string.
    
    Example:
        >>> result = well_documented_function(10, "test", optional_param=2.0)
        >>> print(result['key1'])
        42.0
    
    Note:
        Additional notes about the function if needed.
    
    See Also:
        related_function: Description of related function
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    if not isinstance(param2, str):
        raise TypeError("param2 must be a string")
    
    return {'key1': param1 * optional_param, 'key2': len(param2)}
```

---

## Tests Unitaires

### Structure de Tests

```python
import unittest
import pytest
import torch
import torch.nn as nn

class CompressionTests(unittest.TestCase):
    """
    Tests unitaires pour module de compression
    """
    
    def setUp(self):
        """Setup pour chaque test"""
        self.model = nn.Linear(10, 5)
        self.test_input = torch.randn(32, 10)
    
    def test_pruning_preserves_shape(self):
        """Test que pruning préserve shape output"""
        from torch.nn.utils import prune
        
        # Appliquer pruning
        prune.l1_unstructured(self.model, name='weight', amount=0.5)
        
        # Vérifier shape préservé
        output = self.model(self.test_input)
        self.assertEqual(output.shape, (32, 5))
    
    def test_quantization_accuracy(self):
        """Test que quantification maintient accuracy acceptable"""
        # Entraîner modèle
        # ...
        
        # Quantifier
        quantized = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Comparer outputs
        original_output = self.model(self.test_input)
        quantized_output = quantized(self.test_input)
        
        # Vérifier similarité
        mse = torch.mean((original_output - quantized_output)**2)
        self.assertLess(mse.item(), 0.01)
    
    def tearDown(self):
        """Cleanup après chaque test"""
        pass

# Tests avec pytest (plus moderne)
class TestCompressionPytest:
    """Tests avec pytest"""
    
    def test_pruning_sparsity(self):
        """Test sparsité après pruning"""
        from torch.nn.utils import prune
        
        model = nn.Linear(10, 5)
        prune.l1_unstructured(model, name='weight', amount=0.5)
        
        sparsity = (model.weight == 0).float().mean()
        assert abs(sparsity.item() - 0.5) < 0.1  # Tolérance
    
    @pytest.mark.parametrize("sparsity", [0.1, 0.3, 0.5, 0.7])
    def test_pruning_different_sparsities(self, sparsity):
        """Test pruning avec différents taux"""
        from torch.nn.utils import prune
        
        model = nn.Linear(10, 5)
        prune.l1_unstructured(model, name='weight', amount=sparsity)
        
        actual_sparsity = (model.weight == 0).float().mean()
        assert abs(actual_sparsity.item() - sparsity) < 0.1
```

---

## Tests d'Intégration

### Tests de Workflows Complets

```python
class IntegrationTests(unittest.TestCase):
    """
    Tests d'intégration pour workflows complets
    """
    
    def test_full_compression_pipeline(self):
        """Test pipeline compression complet"""
        # Setup
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        # Entraîner (simplifié)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        for _ in range(10):
            optimizer.zero_grad()
            output = model(torch.randn(32, 10))
            loss = criterion(output, torch.randn(32, 5))
            loss.backward()
            optimizer.step()
        
        baseline_acc = self.evaluate_model(model)
        
        # Compression
        from torch.nn.utils import prune
        prune.l1_unstructured(model, name='weight', amount=0.5)
        
        compressed_acc = self.evaluate_model(model)
        
        # Vérifier dégradation acceptable
        acc_drop = baseline_acc - compressed_acc
        self.assertLess(acc_drop, 0.1)  # Moins de 10% de drop
    
    def evaluate_model(self, model):
        """Évalue modèle (simplifié)"""
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(100, 10))
            # Calculer accuracy (simplifié)
            return 0.95  # Placeholder
```

---

## Couverture de Tests

### Mesure et Amélioration

```python
"""
Outils pour couverture tests:

1. coverage.py
   - Mesure couverture code
   - Rapport détaillé
   - Identification code non testé

2. pytest-cov
   - Intégration avec pytest
   - Rapport HTML
   - CI/CD integration

Usage:
    pip install pytest-cov
    pytest --cov=project --cov-report=html
"""

class TestCoverage:
    """
    Gestion couverture de tests
    """
    
    def __init__(self):
        self.coverage_goals = {
            'minimum': 70,  # Pourcent minimum
            'target': 80,   # Objectif
            'critical': 95  # Pour code critique
        }
    
    def generate_coverage_report(self):
        """Génère rapport couverture"""
        commands = """
# Avec pytest-cov
pytest --cov=src --cov-report=html --cov-report=term

# Avec coverage.py
coverage run -m pytest tests/
coverage report
coverage html  # Génère HTML report

# Voir rapport
open htmlcov/index.html
"""
        return commands
```

---

## Documentation Interactive

### Exemples et Notebooks

```python
class InteractiveDocumentation:
    """
    Documentation interactive avec exemples
    """
    
    def create_example_script(self):
        """Crée script d'exemple"""
        example = """
# Example: Using Tensor Compression

import torch
import torch.nn as nn
from project import compress_model

# Créer modèle
model = nn.Sequential(
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 10)
)

# Compresser
compressed = compress_model(model, method='pruning', sparsity=0.5)

# Utiliser
input_data = torch.randn(32, 100)
output = compressed(input_data)

print(f"Output shape: {output.shape}")
"""
        return example
    
    def create_jupyter_notebook(self):
        """Structure notebook tutorial"""
        notebook_structure = """
# Notebook Tutorial Structure

## 1. Setup and Imports
```python
import project
import torch
```

## 2. Basic Usage
```python
# Exemple simple
```

## 3. Advanced Features
```python
# Exemples avancés
```

## 4. Tips and Best Practices
# Markdown avec conseils
"""
        return notebook_structure
```

---

## Exercices

### Exercice 27.3.1
Créez README complet pour un projet (vrai ou fictif) avec toutes sections.

### Exercice 27.3.2
Écrivez docstrings complètes pour fonctions/modules suivant standards.

### Exercice 27.3.3
Créez suite de tests unitaires pour module de compression avec pytest.

### Exercice 27.3.4
Mesurez couverture de tests et identifiez code non testé.

---

## Points Clés à Retenir

> 📌 **README est première impression projet - doit être clair et complet**

> 📌 **Documentation API avec docstrings facilite utilisation**

> 📌 **Tests unitaires garantissent fonctionnalité correcte**

> 📌 **Tests d'intégration vérifient workflows complets**

> 📌 **Couverture de tests identifie code non testé**

> 📌 **Exemples interactifs aident adoption**

---

*Section précédente : [27.2 Git](./27_02_Git.md) | Section suivante : [27.4 Code Review](./27_04_Code_Review.md)*

