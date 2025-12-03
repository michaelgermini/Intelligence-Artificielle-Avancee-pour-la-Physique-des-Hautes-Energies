# 22.5 Bonnes Pratiques de Code

---

## Introduction

Écrire du code maintenable, lisible, et efficace est essentiel pour le développement de deep learning professionnel. Cette section présente les bonnes pratiques pour organiser, documenter, tester, et optimiser le code Python pour le deep learning.

---

## Structure de Projet

### Organisation Recommandée

```python
"""
Structure de projet recommandée:

project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   └── my_model.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── validate.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── visualization.py
├── configs/
│   └── config.yaml
├── tests/
│   └── test_models.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── setup.py
└── README.md
"""
```

---

## Documentation

### Docstrings

```python
def train_model(model, dataloader, criterion, optimizer, n_epochs=10):
    """
    Entraîne un modèle de deep learning.
    
    Args:
        model (nn.Module): Modèle PyTorch à entraîner.
        dataloader (DataLoader): DataLoader pour données d'entraînement.
        criterion (nn.Module): Fonction de loss.
        optimizer (torch.optim.Optimizer): Optimiseur.
        n_epochs (int): Nombre d'époques. Default: 10.
    
    Returns:
        list: Liste des losses par époque.
    
    Example:
        >>> model = SimpleNN()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> losses = train_model(model, loader, nn.CrossEntropyLoss(), 
        ...                      torch.optim.Adam(model.parameters()))
    """
    losses = []
    for epoch in range(n_epochs):
        # ... code ...
        pass
    return losses
```

---

## Type Hints

### Annotations de Types

```python
from typing import List, Tuple, Optional, Dict
import torch
from torch import nn

def process_batch(
    data: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Traite un batch de données.
    
    Returns:
        Tuple contenant (predictions, loss).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = data.to(device)
    labels = labels.to(device)
    
    predictions = model(data)
    loss = nn.functional.cross_entropy(predictions, labels)
    
    return predictions, loss
```

---

## Configuration Management

### Configuration Files

```python
# config.yaml
"""
model:
  input_dim: 10
  hidden_dim: 128
  output_dim: 3
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  n_epochs: 100
  optimizer: adam

data:
  train_path: "data/train.pt"
  val_path: "data/val.pt"
"""

# config.py
import yaml
from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_dim: int = 10
    hidden_dim: int = 128
    output_dim: int = 3
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 100
    optimizer: str = 'adam'

def load_config(config_path: str) -> Tuple[ModelConfig, TrainingConfig]:
    """Charge configuration depuis fichier YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = ModelConfig(**config['model'])
    training_config = TrainingConfig(**config['training'])
    
    return model_config, training_config
```

---

## Logging

### Utilisation de logging

```python
import logging
from pathlib import Path

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Utilisation
logger.info("Starting training")
logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
logger.warning("Learning rate might be too high")
logger.error("Training failed")
```

---

## Tests Unitaires

### pytest

```python
# tests/test_models.py
import pytest
import torch
from models import SimpleNN

def test_model_forward():
    """Test forward pass du modèle"""
    model = SimpleNN(input_dim=10, output_dim=3)
    x = torch.randn(32, 10)
    output = model(x)
    
    assert output.shape == (32, 3)
    assert not torch.isnan(output).any()

def test_model_gradient():
    """Test que gradients sont calculés"""
    model = SimpleNN(input_dim=10, output_dim=3)
    x = torch.randn(32, 10)
    output = model(x)
    loss = output.mean()
    loss.backward()
    
    # Vérifier gradients existent
    for param in model.parameters():
        assert param.grad is not None
```

---

## Error Handling

### Gestion d'Erreurs

```python
def safe_train(model, dataloader, device):
    """Entraînement avec gestion d'erreurs"""
    try:
        model.train()
        for batch_idx, (data, labels) in enumerate(dataloader):
            try:
                data = data.to(device)
                labels = labels.to(device)
                
                # Forward
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at batch {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Sauvegarder checkpoint
        save_checkpoint(model, optimizer, epoch)
        raise
```

---

## Performance

### Optimisations

```python
# 1. Utiliser torch.no_grad() pour inference
with torch.no_grad():
    predictions = model(X_test)

# 2. Désactiver gradient computation quand pas nécessaire
torch.set_grad_enabled(False)
predictions = model(X_test)
torch.set_grad_enabled(True)

# 3. Utiliser DataLoader avec num_workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# 4. Pin memory pour GPU
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

# 5. Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Version Control

### Git Best Practices

```python
"""
.gitignore pour projet deep learning

# Modèles et checkpoints
*.pth
*.pt
*.h5
models/
checkpoints/
weights/

# Données
data/
datasets/
*.h5
*.hdf5

# Logs
logs/
*.log
tensorboard/

# Python
__pycache__/
*.pyc
venv/
env/
"""
```

---

## Exercices

### Exercice 22.5.1
Organisez un projet deep learning avec structure de dossiers recommandée.

### Exercice 22.5.2
Ajoutez docstrings et type hints à vos fonctions.

### Exercice 22.5.3
Créez tests unitaires pour vos modèles avec pytest.

### Exercice 22.5.4
Configurez logging et utilisez-le pendant entraînement.

---

## Points Clés à Retenir

> 📌 **Structure claire de projet facilite collaboration et maintenance**

> 📌 **Documentation (docstrings) rend code compréhensible**

> 📌 **Type hints améliore lisibilité et détection d'erreurs**

> 📌 **Configuration externalisée permet expériences reproductibles**

> 📌 **Logging permet traçabilité pendant entraînement**

> 📌 **Tests unitaires garantissent correct fonctionnement**

> 📌 **Gestion d'erreurs robuste prévient crashes**

> 📌 **Optimisations (no_grad, mixed precision) améliorent performance**

---

*Section précédente : [22.4 TensorFlow](./22_04_TensorFlow.md)*

