# 22.3.3 DataLoaders et Datasets

---

## Introduction

PyTorch fournit `torch.utils.data` pour gérer efficacement les données pendant l'entraînement. Les `Dataset` encapsulent les données et les `DataLoader` permettent de charger les données par batches avec multiprocessing.

---

## Dataset Personnalisé

### Implémentation

```python
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    """Dataset personnalisé"""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Appliquer transform si disponible
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# Exemple utilisation
data = torch.randn(1000, 10)
labels = torch.randint(0, 3, (1000,))

dataset = CustomDataset(data, labels)
print(f"Dataset size: {len(dataset)}")
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}, Label: {label}")
```

---

## DataLoader

### Chargement par Batches

```python
# Créer DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Multiprocessing
    pin_memory=True  # Pour GPU (transfer plus rapide)
)

# Utilisation dans boucle d'entraînement
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape {data.shape}, labels shape {labels.shape}")
    # ... traitement du batch ...
    
    if batch_idx == 2:  # Juste 3 batches pour exemple
        break
```

---

## Transforms

### Transformations de Données

```python
from torchvision import transforms
import torch

# Transforms pour images (exemple)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transforms pour tenseurs
class TensorTransform:
    """Transform personnalisé pour tenseurs"""
    
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_std
        return tensor + noise

# Utilisation
transform = TensorTransform(noise_std=0.05)
dataset_transformed = CustomDataset(data, labels, transform=transform)
```

---

## Exemple Complet

### Entraînement avec DataLoader

```python
# Préparer données
X = torch.randn(1000, 10)
y = torch.randint(0, 3, (1000,))

# Split train/val
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Créer datasets
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Créer dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Modèle
from torch import nn
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement
n_epochs = 5
for epoch in range(n_epochs):
    # Training
    model.train()
    train_loss = 0
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}")
```

---

## Exercices

### Exercice 22.3.3.1
Créez un Dataset pour charger des fichiers depuis disque et un DataLoader avec multiprocessing.

### Exercice 22.3.3.2
Implémentez des transforms personnalisés pour normaliser et ajouter du bruit aux données.

### Exercice 22.3.3.3
Créez un DataLoader avec différents batch sizes et comparez temps de chargement.

---

## Points Clés à Retenir

> 📌 **Dataset encapsule données, __getitem__ retourne un échantillon**

> 📌 **DataLoader gère batching, shuffling, et multiprocessing**

> 📌 **Transforms permettent preprocessing/augmentation données**

> 📌 **pin_memory=True accélère transfer CPU→GPU**

> 📌 **num_workers permet parallélisation chargement données**

---

*Section précédente : [22.3.2 Modules et Optimizers](./22_03_02_Modules_Optimizers.md) | Section suivante : [22.4 TensorFlow](./22_04_TensorFlow.md)*

