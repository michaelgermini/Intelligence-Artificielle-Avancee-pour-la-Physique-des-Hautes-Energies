# 22.3.2 Modules et Optimizers

---

## Introduction

PyTorch fournit des **modules** (`nn.Module`) pour construire des réseaux de neurones et des **optimiseurs** pour entraîner ces modèles. Cette section présente comment créer des modèles avec `torch.nn` et les entraîner avec différents optimiseurs.

---

## Modules de Base

### nn.Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    """Réseau de neurones simple"""
    
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2):
        super().__init__()
        
        # Couches
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utilisation
model = SimpleNN(input_dim=20, hidden_dim=128, output_dim=3)
x = torch.randn(32, 20)  # Batch de 32 échantillons
output = model(x)
print(f"Output shape: {output.shape}")  # (32, 3)
```

---

## Couches Communes

### Couches Linéaires et Activations

```python
# Couches linéaires
linear = nn.Linear(in_features=10, out_features=5)

# Activations
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
gelu = nn.GELU()
leaky_relu = nn.LeakyReLU(0.2)

# Ou utiliser fonctionnelles (sans état)
x = torch.randn(5, 10)
y = F.relu(x)
y = F.sigmoid(x)

# Dropout
dropout = nn.Dropout(0.3)
x_dropped = dropout(x)  # Training mode
dropout.eval()  # Eval mode (pas de dropout)
x_no_drop = dropout(x)

# Batch Normalization
bn = nn.BatchNorm1d(10)
x_norm = bn(x)
```

---

## Modules Séquentiels

### nn.Sequential

```python
# Méthode 1: Passer modules en arguments
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Méthode 2: OrderedDict (pour noms)
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 64)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(64, 32)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(32, 2))
]))

# Utilisation
x = torch.randn(32, 10)
output = model(x)
```

---

## Loss Functions

### Fonctions de Coût

```python
# Classification
criterion_ce = nn.CrossEntropyLoss()
criterion_bce = nn.BCELoss()
criterion_bce_logits = nn.BCEWithLogitsLoss()  # Plus stable

# Régression
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()
criterion_huber = nn.HuberLoss()

# Exemple
predictions = torch.randn(32, 10)  # Logits
targets = torch.randint(0, 10, (32,))  # Classes
loss = criterion_ce(predictions, targets)

# Avec probabilités
probs = torch.softmax(predictions, dim=1)
target_probs = torch.zeros(32, 10)
target_probs.scatter_(1, targets.unsqueeze(1), 1.0)
loss_ce_manual = -(target_probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
```

---

## Optimizers

### Optimiseurs Disponibles

```python
model = SimpleNN()

# SGD
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (le plus populaire)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

# AdamW (avec weight decay corrigé)
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer_rmsprop = torch.optim.RMSprop(model.parameters(), lr=0.001)

# Adagrad
optimizer_adagrad = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

---

## Boucle d'Entraînement

### Exemple Complet

```python
# Données simulées
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 3, (1000,))

# Modèle
model = SimpleNN(input_dim=10, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement
model.train()
n_epochs = 10

for epoch in range(n_epochs):
    # Forward
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward
    optimizer.zero_grad()  # Réinitialiser gradients
    loss.backward()        # Calculer gradients
    optimizer.step()       # Mettre à jour paramètres
    
    # Afficher progression
    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
```

---

## Learning Rate Scheduling

### Ajustement du Learning Rate

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# StepLR: réduit LR périodiquement
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ExponentialLR: décroissance exponentielle
scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# ReduceLROnPlateau: réduit si plateau
scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Utilisation dans boucle
for epoch in range(n_epochs):
    # ... entraînement ...
    scheduler.step()  # Pour StepLR/ExponentialLR
    # ou
    scheduler_plateau.step(loss)  # Pour ReduceLROnPlateau
```

---

## Mode Train/Eval

### Switching Modes

```python
model = SimpleNN()

# Training mode (par défaut)
model.train()
# Active: dropout, batch norm en mode training

# Evaluation mode
model.eval()
# Désactive: dropout, batch norm en mode eval

# Utilisation
with torch.no_grad():  # Pas besoin gradients en eval
    model.eval()
    predictions = model(X_test)
    # Pas de backward nécessaire
```

---

## Exercices

### Exercice 22.3.2.1
Créez un réseau avec 3 couches cachées et entraînez-le sur données de classification binaire.

### Exercice 22.3.2.2
Comparez performances de différents optimiseurs (SGD, Adam, RMSprop) sur même modèle.

### Exercice 22.3.2.3
Implémentez un learning rate scheduler qui réduit le LR quand validation loss stagne.

### Exercice 22.3.2.4
Créez un modèle avec batch normalization et dropout, et observez différence entre train/eval modes.

---

## Points Clés à Retenir

> 📌 **nn.Module est classe de base pour tous modèles PyTorch**

> 📌 **nn.Sequential permet créer modèles simplement**

> 📌 **Optimizer.zero_grad() avant backward() pour éviter accumulation gradients**

> 📌 **model.train() et model.eval() contrôlent comportement dropout/batch norm**

> 📌 **Learning rate scheduling améliore convergence**

> 📌 **Différents optimiseurs ont différents trade-offs (SGD vs Adam)**

---

*Section précédente : [22.3.1 Tenseurs et Autograd](./22_03_01_Tenseurs_Autograd.md) | Section suivante : [22.3.3 DataLoaders](./22_03_03_DataLoaders.md)*

