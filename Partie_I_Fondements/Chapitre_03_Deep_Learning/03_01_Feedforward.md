# 3.1 Réseaux de Neurones Feedforward

---

## Introduction

Les **réseaux feedforward** (ou perceptrons multicouches - MLP) sont l'architecture fondamentale du deep learning. L'information circule dans une seule direction, de l'entrée vers la sortie, sans boucles de rétroaction.

---

## Architecture

### Structure Générale

```
┌─────────────────────────────────────────────────────────────────┐
│                    Réseau Feedforward                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Entrée      Couche 1      Couche 2      ...      Sortie       │
│    x          h₁            h₂                      y          │
│                                                                 │
│   (○)         (○)           (○)                    (○)         │
│   (○) ──────► (○) ────────► (○) ────► ... ──────► (○)         │
│   (○)         (○)           (○)                    (○)         │
│   (○)         (○)           (○)                                │
│                                                                 │
│  d₀ dims     d₁ dims       d₂ dims              dₙ dims        │
│                                                                 │
│  Paramètres par couche: Wᵢ ∈ ℝ^(dᵢ × dᵢ₋₁), bᵢ ∈ ℝ^dᵢ         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Formulation Mathématique

Pour un réseau à $L$ couches :

$$\mathbf{h}^{(0)} = \mathbf{x}$$
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{h}^{(l)} = \sigma(\mathbf{z}^{(l)})$$
$$\mathbf{y} = \mathbf{h}^{(L)}$$

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedforwardNetwork(nn.Module):
    """
    Réseau feedforward avec configuration flexible
    """
    
    def __init__(self, 
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activation='relu',
                 output_activation=None,
                 dropout=0.0,
                 batch_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Construction des couches
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Couche linéaire
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optionnel)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout (optionnel)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids
        self._init_weights()
        
    def _get_activation(self, name):
        """Retourne la fonction d'activation"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=-1)
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        """Initialisation Xavier/Glorot"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x):
        return self.network(x)
    
    def get_layer_outputs(self, x):
        """Retourne les activations de chaque couche"""
        outputs = [x]
        current = x
        
        for layer in self.network:
            current = layer(current)
            if isinstance(layer, nn.Linear):
                outputs.append(current.clone())
        
        return outputs

# Exemple d'utilisation
model = FeedforwardNetwork(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=10,
    activation='relu',
    dropout=0.2,
    batch_norm=True
)

print(model)
print(f"\nNombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Propagation Avant (Forward Pass)

### Calcul Efficace

```python
def forward_pass_analysis(model, x):
    """
    Analyse détaillée du forward pass
    """
    print("Analyse du Forward Pass")
    print("=" * 60)
    
    current = x
    total_flops = 0
    total_memory = 0
    
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Dimensions
            batch_size = current.shape[0]
            in_features = module.in_features
            out_features = module.out_features
            
            # FLOPs: multiplication matricielle + addition biais
            # y = Wx + b : 2 * batch * in * out FLOPs
            flops = 2 * batch_size * in_features * out_features
            total_flops += flops
            
            # Mémoire pour les activations
            memory = batch_size * out_features * 4  # float32
            total_memory += memory
            
            print(f"\nCouche {layer_idx} ({name}):")
            print(f"  Input: {tuple(current.shape)}")
            print(f"  Weight: {tuple(module.weight.shape)}")
            print(f"  Output: ({batch_size}, {out_features})")
            print(f"  FLOPs: {flops:,}")
            print(f"  Mémoire activations: {memory / 1024:.2f} KB")
            
            current = module(current)
            layer_idx += 1
            
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid)):
            # Activations: ~1 FLOP par élément
            flops = current.numel()
            total_flops += flops
            current = module(current)
    
    print(f"\n{'=' * 60}")
    print(f"Total FLOPs: {total_flops:,}")
    print(f"Total mémoire activations: {total_memory / 1024:.2f} KB")
    
    return total_flops, total_memory

# Test
x = torch.randn(32, 784)  # Batch de 32
forward_pass_analysis(model, x)
```

---

## Rétropropagation (Backpropagation)

### Algorithme

La rétropropagation calcule les gradients via la règle de la chaîne :

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial \mathbf{W}^{(l)}}$$

```python
class ManualBackprop:
    """
    Implémentation manuelle de la rétropropagation
    pour comprendre le mécanisme
    """
    
    def __init__(self, layer_sizes):
        self.n_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        
        # Initialisation
        for i in range(self.n_layers):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) / np.sqrt(layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(W)
            self.biases.append(b)
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, x):
        """Forward pass avec cache pour backprop"""
        self.cache = {'a': [x], 'z': []}
        
        a = x
        for i in range(self.n_layers):
            z = self.weights[i] @ a + self.biases[i]
            self.cache['z'].append(z)
            
            if i < self.n_layers - 1:  # ReLU sauf dernière couche
                a = self.relu(z)
            else:
                a = z  # Sortie linéaire
            
            self.cache['a'].append(a)
        
        return a
    
    def backward(self, y_true):
        """Backward pass"""
        m = y_true.shape[1]  # Taille du batch
        
        gradients = {'dW': [], 'db': []}
        
        # Gradient de la loss (MSE)
        dz = self.cache['a'][-1] - y_true
        
        for i in range(self.n_layers - 1, -1, -1):
            # Gradient des poids et biais
            dW = (1/m) * dz @ self.cache['a'][i].T
            db = (1/m) * np.sum(dz, axis=1, keepdims=True)
            
            gradients['dW'].insert(0, dW)
            gradients['db'].insert(0, db)
            
            if i > 0:
                # Propagation du gradient
                da = self.weights[i].T @ dz
                dz = da * self.relu_derivative(self.cache['z'][i-1])
        
        return gradients
    
    def update(self, gradients, learning_rate):
        """Mise à jour des poids"""
        for i in range(self.n_layers):
            self.weights[i] -= learning_rate * gradients['dW'][i]
            self.biases[i] -= learning_rate * gradients['db'][i]

# Démonstration
net = ManualBackprop([10, 20, 15, 5])
x = np.random.randn(10, 32)  # 32 exemples
y = np.random.randn(5, 32)

# Forward
output = net.forward(x)
print(f"Output shape: {output.shape}")

# Backward
grads = net.backward(y)
print(f"Gradient W[0] shape: {grads['dW'][0].shape}")

# Update
net.update(grads, learning_rate=0.01)
```

---

## Analyse de la Complexité

### Paramètres et FLOPs

```python
def complexity_analysis(layer_sizes, batch_size=1):
    """
    Analyse complète de la complexité d'un MLP
    """
    n_layers = len(layer_sizes) - 1
    
    # Paramètres
    total_params = 0
    params_per_layer = []
    
    for i in range(n_layers):
        weights = layer_sizes[i] * layer_sizes[i+1]
        biases = layer_sizes[i+1]
        layer_params = weights + biases
        params_per_layer.append(layer_params)
        total_params += layer_params
    
    # FLOPs (forward pass)
    total_flops_forward = 0
    for i in range(n_layers):
        # Multiplication: 2 * batch * in * out (multiply-add)
        flops = 2 * batch_size * layer_sizes[i] * layer_sizes[i+1]
        total_flops_forward += flops
    
    # FLOPs (backward pass) ≈ 2x forward
    total_flops_backward = 2 * total_flops_forward
    
    # Mémoire
    # Poids
    weight_memory = total_params * 4  # float32
    
    # Activations (pour backprop)
    activation_memory = sum(layer_sizes[1:]) * batch_size * 4
    
    # Gradients
    gradient_memory = total_params * 4
    
    print("Analyse de Complexité")
    print("=" * 50)
    print(f"Architecture: {layer_sizes}")
    print(f"\nParamètres:")
    for i, p in enumerate(params_per_layer):
        print(f"  Couche {i}: {p:,}")
    print(f"  Total: {total_params:,}")
    
    print(f"\nFLOPs (batch={batch_size}):")
    print(f"  Forward: {total_flops_forward:,}")
    print(f"  Backward: {total_flops_backward:,}")
    print(f"  Total/step: {total_flops_forward + total_flops_backward:,}")
    
    print(f"\nMémoire:")
    print(f"  Poids: {weight_memory / 1024**2:.2f} MB")
    print(f"  Activations: {activation_memory / 1024:.2f} KB")
    print(f"  Gradients: {gradient_memory / 1024**2:.2f} MB")
    
    return {
        'params': total_params,
        'flops_forward': total_flops_forward,
        'memory_weights': weight_memory,
        'memory_activations': activation_memory
    }

# Analyse pour différentes architectures
print("\n--- Petit réseau ---")
complexity_analysis([784, 128, 64, 10], batch_size=32)

print("\n--- Réseau moyen ---")
complexity_analysis([784, 512, 256, 128, 10], batch_size=32)

print("\n--- Grand réseau ---")
complexity_analysis([784, 1024, 1024, 512, 256, 10], batch_size=32)
```

---

## Problèmes d'Entraînement

### Vanishing/Exploding Gradients

```python
def gradient_flow_analysis(model, x, y):
    """
    Analyse le flux des gradients dans le réseau
    """
    model.train()
    
    # Forward
    output = model(x)
    loss = F.mse_loss(output, y)
    
    # Backward
    loss.backward()
    
    # Analyse des gradients
    print("Analyse du flux des gradients")
    print("=" * 50)
    
    gradient_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms.append((name, grad_norm))
            print(f"{name}: grad_norm = {grad_norm:.6f}")
    
    # Détection de problèmes
    norms = [n for _, n in gradient_norms]
    if max(norms) / (min(norms) + 1e-10) > 1000:
        print("\n⚠️ Attention: Grande disparité dans les normes de gradient")
        print("   Possible vanishing/exploding gradient")
    
    return gradient_norms

# Test avec réseau profond
deep_model = FeedforwardNetwork(
    input_dim=100,
    hidden_dims=[100] * 10,  # 10 couches cachées
    output_dim=10,
    activation='relu'
)

x = torch.randn(32, 100)
y = torch.randn(32, 10)

gradient_flow_analysis(deep_model, x, y)
```

### Solutions

```python
class ResidualBlock(nn.Module):
    """
    Bloc résiduel pour faciliter le flux des gradients
    """
    
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = dim
        
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)  # Connexion résiduelle


class ResidualMLP(nn.Module):
    """
    MLP avec connexions résiduelles
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=5):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

# Comparaison
print("Réseau standard profond:")
standard = FeedforwardNetwork(100, [100]*10, 10)
gradient_flow_analysis(standard, x, y)

print("\n\nRéseau résiduel:")
residual = ResidualMLP(100, 100, 10, n_blocks=10)
gradient_flow_analysis(residual, x, y)
```

---

## Applications en Physique des Particules

### Classification de Particules

```python
class ParticleClassifier(nn.Module):
    """
    Classificateur de particules basé sur des features de haut niveau
    """
    
    def __init__(self, n_features=20, n_classes=6):
        super().__init__()
        
        self.network = nn.Sequential(
            # Couche d'entrée
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Couches cachées
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Sortie
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

# Features typiques pour l'identification de particules
feature_names = [
    'pt', 'eta', 'phi',           # Cinématique
    'dE_dx',                       # Perte d'énergie
    'E_over_p',                    # Rapport E/p
    'n_hits_pixel', 'n_hits_sct',  # Hits dans le tracker
    'track_chi2',                  # Qualité de trace
    'calo_E_em', 'calo_E_had',     # Énergie calorimétrique
    'shower_width', 'shower_depth', # Forme de gerbe
    'isolation_track', 'isolation_calo',  # Isolation
    'vertex_significance',         # Distance au vertex
    # ... autres features
]

# Classes de particules
particle_classes = ['electron', 'muon', 'photon', 'pion', 'kaon', 'proton']

# Création et analyse du modèle
classifier = ParticleClassifier(n_features=len(feature_names), n_classes=len(particle_classes))
print(f"Classificateur de particules: {sum(p.numel() for p in classifier.parameters()):,} paramètres")
```

---

## Exercices

### Exercice 3.1.1
Implémentez un MLP avec skip connections (connexions sautant une couche) et comparez sa performance avec un MLP standard.

### Exercice 3.1.2
Calculez analytiquement le gradient de la loss MSE par rapport aux poids de la première couche d'un réseau à 2 couches.

### Exercice 3.1.3
Créez une fonction qui visualise la distribution des activations à chaque couche d'un réseau pour détecter les problèmes de saturation.

---

## Points Clés à Retenir

> 📌 **Les MLP sont des approximateurs universels mais nécessitent une profondeur suffisante**

> 📌 **La rétropropagation calcule efficacement tous les gradients en O(n) opérations**

> 📌 **Les connexions résiduelles facilitent l'entraînement des réseaux profonds**

> 📌 **La complexité est dominée par les multiplications matricielles**

---

## Références

1. Goodfellow, I. et al. "Deep Learning." Chapter 6: Deep Feedforward Networks
2. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR, 2016
3. Glorot, X., Bengio, Y. "Understanding the difficulty of training deep feedforward neural networks." AISTATS, 2010

---

*Section suivante : [3.2 Réseaux Convolutionnels (CNN)](./03_02_CNN.md)*

