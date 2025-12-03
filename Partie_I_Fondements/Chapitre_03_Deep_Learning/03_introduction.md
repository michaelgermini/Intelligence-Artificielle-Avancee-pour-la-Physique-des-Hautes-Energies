# Chapitre 3 : Deep Learning - Architectures et Principes

---

## Introduction

Le **Deep Learning** a révolutionné l'intelligence artificielle et trouve des applications cruciales en physique des hautes énergies. Ce chapitre présente les architectures fondamentales des réseaux de neurones, les principes d'optimisation, et les techniques de régularisation essentielles pour la compression de modèles.

---

## Objectifs d'Apprentissage

À la fin de ce chapitre, vous serez capable de :

- Comprendre l'architecture des réseaux de neurones modernes
- Implémenter et entraîner des CNN, RNN et Transformers
- Maîtriser les techniques d'optimisation et de régularisation
- Identifier les opportunités de compression dans chaque architecture

---

## Plan du Chapitre

1. [Réseaux de Neurones Feedforward](./03_01_Feedforward.md)
2. [Réseaux Convolutionnels (CNN)](./03_02_CNN.md)
3. [Réseaux Récurrents et Transformers](./03_03_RNN_Transformers.md)
4. [Fonctions de Perte et Optimisation](./03_04_Optimisation.md)
5. [Régularisation et Généralisation](./03_05_Regularisation.md)

---

## Contexte Historique

```
┌─────────────────────────────────────────────────────────────────┐
│              Évolution du Deep Learning                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1943  │ McCulloch & Pitts : Neurone artificiel                │
│  1958  │ Rosenblatt : Perceptron                               │
│  1986  │ Rumelhart et al. : Backpropagation                    │
│  1989  │ LeCun : CNN pour reconnaissance de chiffres           │
│  1997  │ Hochreiter & Schmidhuber : LSTM                       │
│  2006  │ Hinton : Deep Belief Networks                         │
│  2012  │ Krizhevsky : AlexNet (révolution ImageNet)            │
│  2014  │ Goodfellow : GANs                                      │
│  2015  │ He : ResNet (réseaux très profonds)                   │
│  2017  │ Vaswani : Transformer (Attention is All You Need)     │
│  2018  │ BERT, GPT : Modèles de langage pré-entraînés          │
│  2020+ │ Scaling laws, Foundation models                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Le Neurone Artificiel

### Modèle Mathématique

Un neurone artificiel calcule :

$$y = \sigma\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

où $\sigma$ est une fonction d'activation non-linéaire.

```python
import numpy as np
import torch
import torch.nn as nn

class Neuron:
    """
    Implémentation d'un neurone artificiel
    """
    
    def __init__(self, n_inputs, activation='relu'):
        # Initialisation Xavier/Glorot
        self.weights = np.random.randn(n_inputs) / np.sqrt(n_inputs)
        self.bias = 0.0
        self.activation = activation
        
    def forward(self, x):
        """Propagation avant"""
        z = np.dot(self.weights, x) + self.bias
        return self._activate(z)
    
    def _activate(self, z):
        """Applique la fonction d'activation"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Activation inconnue: {self.activation}")

# Démonstration
neuron = Neuron(5, activation='relu')
x = np.array([1.0, -0.5, 0.3, 0.8, -0.2])
output = neuron.forward(x)
print(f"Entrée: {x}")
print(f"Sortie: {output:.4f}")
```

### Fonctions d'Activation

```python
import matplotlib.pyplot as plt

def plot_activations():
    """Visualise les fonctions d'activation courantes"""
    x = np.linspace(-5, 5, 1000)
    
    activations = {
        'Sigmoid': (1 / (1 + np.exp(-x)), 'Sortie ∈ (0, 1)'),
        'Tanh': (np.tanh(x), 'Sortie ∈ (-1, 1)'),
        'ReLU': (np.maximum(0, x), 'Sortie ∈ [0, ∞)'),
        'Leaky ReLU': (np.where(x > 0, x, 0.01 * x), 'Évite les neurones morts'),
        'GELU': (x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))), 
                 'Utilisé dans Transformers'),
        'Swish': (x / (1 + np.exp(-x)), 'Auto-gated')
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for ax, (name, (y, desc)) in zip(axes, activations.items()):
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 5)
        ax.set_title(f'{name}\n{desc}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Propriétés des activations pour la compression
activation_properties = {
    'ReLU': {
        'sparsity': 'Haute (50% des neurones inactifs en moyenne)',
        'quantization': 'Excellente (sortie non-négative)',
        'pruning': 'Facile (neurones morts identifiables)'
    },
    'Sigmoid': {
        'sparsity': 'Faible',
        'quantization': 'Bonne (sortie bornée)',
        'pruning': 'Modérée'
    },
    'GELU': {
        'sparsity': 'Modérée',
        'quantization': 'Plus difficile (lisse)',
        'pruning': 'Modérée'
    }
}
```

---

## Architectures de Base

### Réseaux Fully-Connected (MLP)

```python
class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    
    Architecture: Input → [Hidden]* → Output
    """
    
    def __init__(self, layer_sizes, activation='relu', dropout=0.0):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Crée les couches
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                nn.Linear(layer_sizes[i], layer_sizes[i+1])
            )
            if i < len(layer_sizes) - 2:  # Pas de dropout sur la dernière couche
                self.dropouts.append(nn.Dropout(dropout))
        
        # Activation
        self.activation = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }[activation]
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # Dernière couche sans activation
        x = self.layers[-1](x)
        return x
    
    def count_parameters(self):
        """Compte le nombre de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def layer_analysis(self):
        """Analyse de chaque couche"""
        analysis = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                analysis.append({
                    'layer': i,
                    'type': 'Linear',
                    'input_dim': layer.in_features,
                    'output_dim': layer.out_features,
                    'parameters': layer.weight.numel() + layer.bias.numel(),
                    'weight_shape': tuple(layer.weight.shape)
                })
        return analysis

# Exemple
mlp = MLP([784, 512, 256, 128, 10], activation='relu', dropout=0.2)
print(f"Nombre total de paramètres: {mlp.count_parameters():,}")

print("\nAnalyse des couches:")
for info in mlp.layer_analysis():
    print(f"  Couche {info['layer']}: {info['input_dim']} → {info['output_dim']} "
          f"({info['parameters']:,} params)")
```

---

## Pourquoi la Profondeur ?

### Avantages des Réseaux Profonds

```python
def depth_vs_width_analysis():
    """
    Compare réseaux profonds vs larges
    """
    # Même nombre de paramètres, architectures différentes
    
    # Réseau large et peu profond
    wide_shallow = MLP([100, 1000, 10])
    
    # Réseau étroit et profond
    narrow_deep = MLP([100, 100, 100, 100, 100, 100, 100, 100, 100, 10])
    
    print("Comparaison profondeur vs largeur:")
    print(f"  Large/Shallow: {wide_shallow.count_parameters():,} params, 2 couches")
    print(f"  Narrow/Deep: {narrow_deep.count_parameters():,} params, 9 couches")
    
    # Le réseau profond peut apprendre des représentations hiérarchiques
    # mais est plus difficile à entraîner (vanishing gradients)

depth_vs_width_analysis()
```

### Représentations Hiérarchiques

```
┌─────────────────────────────────────────────────────────────────┐
│           Hiérarchie des Représentations                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Couche 1: Features bas niveau (bords, textures)               │
│      ↓                                                          │
│  Couche 2: Combinaisons simples (coins, formes simples)        │
│      ↓                                                          │
│  Couche 3: Motifs (parties d'objets)                           │
│      ↓                                                          │
│  Couche N: Concepts abstraits (objets, catégories)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implications pour la Compression

### Redondance dans les Réseaux

```python
def analyze_redundancy(model):
    """
    Analyse la redondance dans un modèle
    """
    results = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            W = param.data.numpy()
            
            # SVD pour analyser le rang effectif
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            
            # Rang effectif (99% de l'énergie)
            cumsum = np.cumsum(S**2) / np.sum(S**2)
            effective_rank = np.searchsorted(cumsum, 0.99) + 1
            
            # Sparsité
            sparsity = np.mean(np.abs(W) < 0.01)
            
            results[name] = {
                'shape': W.shape,
                'full_rank': min(W.shape),
                'effective_rank': effective_rank,
                'rank_ratio': effective_rank / min(W.shape),
                'sparsity': sparsity,
                'condition_number': S[0] / S[-1] if S[-1] > 1e-10 else np.inf
            }
    
    return results

# Analyse d'un MLP entraîné
mlp = MLP([256, 512, 256, 64, 10])

# Simulation d'entraînement (les poids réels seraient différents)
print("Analyse de redondance (poids aléatoires):")
redundancy = analyze_redundancy(mlp)
for name, info in redundancy.items():
    print(f"\n{name}:")
    print(f"  Shape: {info['shape']}")
    print(f"  Rang effectif: {info['effective_rank']} / {info['full_rank']} "
          f"({info['rank_ratio']:.1%})")
    print(f"  Sparsité: {info['sparsity']:.1%}")
```

### Opportunités de Compression par Architecture

| Architecture | Technique Principale | Ratio Typique |
|--------------|---------------------|---------------|
| MLP | Low-rank factorization | 2-10x |
| CNN | Filter pruning | 2-5x |
| Transformer | Attention pruning + quantization | 4-8x |
| RNN/LSTM | Structured pruning | 2-4x |

---

## Framework PyTorch : Rappels Essentiels

```python
# Structure de base d'un entraînement PyTorch
def training_loop(model, train_loader, optimizer, criterion, device, epochs=10):
    """
    Boucle d'entraînement standard
    """
    model.to(device)
    model.train()
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistiques
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
    
    return history

# Exemple d'utilisation
"""
model = MLP([784, 256, 128, 10])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

history = training_loop(model, train_loader, optimizer, criterion, 'cuda')
"""
```

---

## Exercices Préliminaires

### Exercice 3.0.1
Calculez le nombre de paramètres d'un MLP avec architecture [1000, 500, 200, 50, 10].

### Exercice 3.0.2
Un réseau a 10 millions de paramètres en float32. Quelle est sa taille en mémoire ? En int8 ?

### Exercice 3.0.3
Implémentez une fonction qui calcule le nombre de FLOPs pour un forward pass d'un MLP.

---

## Points Clés à Retenir

> 📌 **Les réseaux profonds apprennent des représentations hiérarchiques**

> 📌 **La redondance dans les poids permet la compression**

> 📌 **Chaque architecture a ses opportunités de compression spécifiques**

> 📌 **L'activation ReLU favorise la sparsité et facilite la compression**

---

*Commençons par la première section : [3.1 Réseaux de Neurones Feedforward](./03_01_Feedforward.md)*

