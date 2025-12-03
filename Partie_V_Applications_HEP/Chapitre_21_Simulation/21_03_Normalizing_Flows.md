# 21.3 Normalizing Flows

---

## Introduction

Les **Normalizing Flows** sont des modèles génératifs qui apprennent une transformation inversible entre une distribution simple (gaussienne) et la distribution complexe des données. Leur avantage principal est de fournir une densité explicite, permettant échantillonnage exact et évaluation de probabilités, ce qui est particulièrement utile pour la simulation en physique des hautes énergies.

Cette section présente les principes des normalizing flows, leur application à la génération d'événements HEP, et les architectures spécialisées développées.

---

## Principe des Normalizing Flows

### Transformation Inversible

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class NormalizingFlowPrinciple:
    """
    Principes de base des normalizing flows
    """
    
    def __init__(self):
        self.concepts = {
            'transformation': {
                'description': 'Transformation inversible f: X → Z',
                'base_distribution': 'Distribution simple (gaussienne)',
                'target_distribution': 'Distribution complexe des données'
            },
            'change_of_variables': {
                'formula': 'p_X(x) = p_Z(f(x)) |det(∂f/∂x)|',
                'importance': 'Permet calcul densité explicite'
            },
            'composition': {
                'description': 'Composition de transformations simples',
                'formula': 'f = f_K ○ f_{K-1} ○ ... ○ f_1',
                'benefit': 'Flexibilité avec transformations simples'
            }
        }
    
    def display_principles(self):
        """Affiche les principes"""
        print("\n" + "="*70)
        print("Principes des Normalizing Flows")
        print("="*70)
        
        for concept, info in self.concepts.items():
            print(f"\n{concept.replace('_', ' ').title()}:")
            if isinstance(info, dict):
                for key, value in info.items():
                    print(f"  {key}: {value}")

principle = NormalizingFlowPrinciple()
principle.display_principles()
```

---

## Couche Affine Couplée (Affine Coupling)

### Architecture de Base

```python
class AffineCouplingLayer(nn.Module):
    """
    Couche affine couplée
    
    Transformation simple et inversible
    """
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        
        self.dim = dim
        self.split_dim = dim // 2
        
        # Network pour calculer scale et shift
        self.network = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * 2)  # scale et shift
        )
    
    def forward(self, x):
        """
        Forward: x → z
        
        Args:
            x: (batch, dim)
        Returns:
            z: (batch, dim)
            log_det: log déterminant jacobien
        """
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Calculer scale et shift
        params = self.network(x1)
        log_scale, shift = torch.split(params, self.split_dim, dim=1)
        
        # Transformation
        z1 = x1
        z2 = x2 * torch.exp(log_scale) + shift
        
        z = torch.cat([z1, z2], dim=1)
        
        # Log déterminant jacobien
        log_det = log_scale.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z):
        """
        Inverse: z → x
        """
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
        
        # Calculer scale et shift
        params = self.network(z1)
        log_scale, shift = torch.split(params, self.split_dim, dim=1)
        
        # Transformation inverse
        x1 = z1
        x2 = (z2 - shift) * torch.exp(-log_scale)
        
        x = torch.cat([x1, x2], dim=1)
        
        return x

# Test couche affine
coupling = AffineCouplingLayer(dim=10, hidden_dim=32)

x = torch.randn(5, 10)
z, log_det = coupling(x)
x_reconstructed = coupling.inverse(z)

print(f"\nAffine Coupling Layer:")
print(f"  Erreur reconstruction: {(x - x_reconstructed).abs().max().item():.6f}")
print(f"  Log det jacobien moyen: {log_det.mean().item():.4f}")
```

---

## Real NVP (Non-volume Preserving)

### Flow avec Permutations

```python
class RealNVPFlow(nn.Module):
    """
    Real NVP Flow
    
    Composition de couches de couplage avec permutations
    """
    
    def __init__(self, dim, n_layers=4, hidden_dim=64):
        super().__init__()
        
        self.dim = dim
        self.n_layers = n_layers
        
        # Couches de couplage
        self.coupling_layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim) for _ in range(n_layers)
        ])
        
        # Permutations aléatoires (fixées)
        self.permutations = []
        for i in range(n_layers):
            perm = torch.randperm(dim)
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def permute(self, x, perm):
        """Applique permutation"""
        return x[:, perm]
    
    def inverse_permute(self, x, perm):
        """Applique permutation inverse"""
        inv_perm = torch.argsort(perm)
        return x[:, inv_perm]
    
    def forward(self, x):
        """
        Forward: x → z
        
        Returns:
            z: (batch, dim)
            log_det_total: log déterminant total
        """
        log_det_total = torch.zeros(x.size(0))
        z = x
        
        for i, coupling in enumerate(self.coupling_layers):
            # Permutation
            perm = getattr(self, f'perm_{i}')
            z = self.permute(z, perm)
            
            # Couplage
            z, log_det = coupling(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def inverse(self, z):
        """Inverse: z → x"""
        x = z
        
        # Inverse dans ordre inverse
        for i in reversed(range(self.n_layers)):
            coupling = self.coupling_layers[i]
            perm = getattr(self, f'perm_{i}')
            
            # Inverse couplage
            x = coupling.inverse(x)
            
            # Inverse permutation
            x = self.inverse_permute(x, perm)
        
        return x
    
    def log_prob(self, x):
        """
        Calcule log-probabilité p(x)
        
        p(x) = p_z(f(x)) + log|det(∂f/∂x)|
        """
        z, log_det = self.forward(x)
        
        # Probabilité dans espace latent (gaussien standard)
        log_prob_z = -0.5 * (z**2).sum(dim=1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        # Log prob dans espace original
        log_prob_x = log_prob_z + log_det
        
        return log_prob_x
    
    def sample(self, n_samples=1000):
        """
        Échantillonne depuis la distribution
        
        z ~ N(0,1) → x = f^{-1}(z)
        """
        # Échantillonner depuis distribution de base
        z = torch.randn(n_samples, self.dim)
        
        # Transformer
        x = self.inverse(z)
        
        return x

# Créer Real NVP flow
flow = RealNVPFlow(dim=20, n_layers=6, hidden_dim=64)

print(f"\nReal NVP Flow:")
print(f"  Dimensions: {flow.dim}")
print(f"  Nombre de couches: {flow.n_layers}")
print(f"  Paramètres: {sum(p.numel() for p in flow.parameters()):,}")

# Test
x_test = torch.randn(10, 20)
z, log_det = flow(x_test)
x_reconstructed = flow.inverse(z)

log_prob = flow.log_prob(x_test)
samples = flow.sample(n_samples=100)

print(f"  Erreur reconstruction: {(x_test - x_reconstructed).abs().max().item():.6f}")
print(f"  Log prob moyen: {log_prob.mean().item():.4f}")
print(f"  Échantillons générés: {samples.shape}")
```

---

## Neural Spline Flows

### Flows avec Splines

```python
class NeuralSplineCoupling(nn.Module):
    """
    Couche de couplage avec splines rationnelles quadratiques
    
    Plus flexible que transformation affine
    """
    
    def __init__(self, dim, hidden_dim=64, n_bins=8):
        super().__init__()
        
        self.dim = dim
        self.split_dim = dim // 2
        self.n_bins = n_bins
        
        # Network pour calculer paramètres spline
        # Pour chaque output: widths, heights, derivatives (n_bins + 1 valeurs chacun)
        self.network = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * (n_bins * 3 + 1))
        )
    
    def forward(self, x):
        """
        Forward avec splines (simplifié)
        
        En pratique: implémentation complète de RQ-spline
        """
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Calculer paramètres spline
        params = self.network(x1)
        # Simplifié: utiliser transformation affine approximée
        # En pratique: implémenter vraie spline
        
        # Approximation avec affine
        log_scale = params[:, :(self.dim - self.split_dim)]
        shift = params[:, (self.dim - self.split_dim):2*(self.dim - self.split_dim)]
        
        z1 = x1
        z2 = x2 * torch.exp(log_scale) + shift
        
        z = torch.cat([z1, z2], dim=1)
        log_det = log_scale.sum(dim=1)
        
        return z, log_det

# Note: Implémentation complète de spline flows est complexe
# Ici: structure de base
```

---

## Application aux Événements HEP

### Flow pour Génération d'Événements

```python
class HEPEventFlow(nn.Module):
    """
    Normalizing Flow pour génération d'événements HEP
    """
    
    def __init__(self, event_dim=50, n_layers=8, hidden_dim=128):
        super().__init__()
        
        self.event_dim = event_dim
        
        # Normalizing flow
        self.flow = RealNVPFlow(dim=event_dim, n_layers=n_layers, hidden_dim=hidden_dim)
        
        # Normalisation des données (important pour flows)
        self.register_buffer('data_mean', torch.zeros(event_dim))
        self.register_buffer('data_std', torch.ones(event_dim))
    
    def fit_normalization(self, data):
        """Ajuste normalisation aux données"""
        self.data_mean = data.mean(dim=0)
        self.data_std = data.std(dim=0) + 1e-6  # Éviter division par zéro
    
    def normalize(self, x):
        """Normalise données"""
        return (x - self.data_mean) / self.data_std
    
    def denormalize(self, x):
        """Dénormalise données"""
        return x * self.data_std + self.data_mean
    
    def forward(self, x):
        """Forward avec normalisation"""
        x_norm = self.normalize(x)
        z, log_det = self.flow(x_norm)
        return z, log_det
    
    def inverse(self, z):
        """Inverse avec dénormalisation"""
        x_norm = self.flow.inverse(z)
        x = self.denormalize(x_norm)
        return x
    
    def log_prob(self, x):
        """Log probabilité avec normalisation"""
        x_norm = self.normalize(x)
        log_prob_norm = self.flow.log_prob(x_norm)
        
        # Ajuster pour changement de variables (normalisation)
        log_det_norm = -self.data_std.log().sum()
        
        return log_prob_norm + log_det_norm
    
    def sample(self, n_samples=1000):
        """Échantillonne événements"""
        z = torch.randn(n_samples, self.event_dim)
        x = self.inverse(z)
        return x

hep_flow = HEPEventFlow(event_dim=50, n_layers=8)

print(f"\nHEP Event Flow:")
print(f"  Dimensions événement: {hep_flow.event_dim}")
print(f"  Paramètres: {sum(p.numel() for p in hep_flow.parameters()):,}")

# Simuler ajustement
training_data = torch.randn(10000, 50)
hep_flow.fit_normalization(training_data)

# Générer événements
generated_events = hep_flow.sample(n_samples=1000)

print(f"  Événements générés: {generated_events.shape}")
print(f"  Moyenne générée: {generated_events.mean(dim=0)[:5]}")
print(f"  Std générée: {generated_events.std(dim=0)[:5]}")
```

---

## Flows Conditionnels

### Génération Conditionnée

```python
class ConditionalRealNVP(nn.Module):
    """
    Real NVP conditionnel
    
    Génère événements conditionnés sur paramètres
    """
    
    def __init__(self, event_dim=50, condition_dim=5, n_layers=6, hidden_dim=128):
        super().__init__()
        
        self.event_dim = event_dim
        self.condition_dim = condition_dim
        
        # Couches de couplage conditionnelles
        self.coupling_layers = nn.ModuleList()
        self.permutations = []
        
        for i in range(n_layers):
            # Network inclut condition
            network = nn.Sequential(
                nn.Linear(event_dim // 2 + condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, (event_dim - event_dim // 2) * 2)
            )
            
            coupling = ConditionalAffineCoupling(network, event_dim)
            self.coupling_layers.append(coupling)
            
            perm = torch.randperm(event_dim)
            self.register_buffer(f'perm_{i}', perm)
            self.permutations.append(perm)
    
    def forward(self, x, condition):
        """Forward conditionnel"""
        log_det_total = torch.zeros(x.size(0))
        z = x
        
        for i, coupling in enumerate(self.coupling_layers):
            perm = getattr(self, f'perm_{i}')
            z = z[:, perm]
            
            z, log_det = coupling(z, condition)
            log_det_total += log_det
        
        return z, log_det_total
    
    def inverse(self, z, condition):
        """Inverse conditionnel"""
        x = z
        
        for i in reversed(range(len(self.coupling_layers))):
            coupling = self.coupling_layers[i]
            perm = getattr(self, f'perm_{i}')
            
            x = coupling.inverse(x, condition)
            inv_perm = torch.argsort(perm)
            x = x[:, inv_perm]
        
        return x
    
    def sample(self, condition, n_samples=1000):
        """Échantillonne conditionnel"""
        z = torch.randn(n_samples, self.event_dim)
        x = self.inverse(z, condition)
        return x

class ConditionalAffineCoupling(nn.Module):
    """Couplage affine conditionnel"""
    
    def __init__(self, network, dim):
        super().__init__()
        self.network = network
        self.dim = dim
        self.split_dim = dim // 2
    
    def forward(self, x, condition):
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        # Concaténer avec condition
        input_concat = torch.cat([x1, condition], dim=1)
        params = self.network(input_concat)
        log_scale, shift = torch.split(params, self.split_dim, dim=1)
        
        z1 = x1
        z2 = x2 * torch.exp(log_scale) + shift
        
        z = torch.cat([z1, z2], dim=1)
        log_det = log_scale.sum(dim=1)
        
        return z, log_det
    
    def inverse(self, z, condition):
        z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
        
        input_concat = torch.cat([z1, condition], dim=1)
        params = self.network(input_concat)
        log_scale, shift = torch.split(params, self.split_dim, dim=1)
        
        x1 = z1
        x2 = (z2 - shift) * torch.exp(-log_scale)
        
        x = torch.cat([x1, x2], dim=1)
        return x

cond_flow = ConditionalRealNVP(event_dim=50, condition_dim=5, n_layers=6)

print(f"\nConditional Flow:")
print(f"  Génère événements selon condition (énergie, processus, etc.)")
```

---

## Entraînement

### Loss et Optimisation

```python
class FlowTraining:
    """
    Entraînement d'un normalizing flow
    """
    
    def __init__(self, flow, lr=0.001):
        self.flow = flow
        self.optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    
    def train_step(self, data):
        """
        Une étape d'entraînement
        
        Loss = -log_prob (negative log-likelihood)
        """
        self.optimizer.zero_grad()
        
        # Calculer log probabilité
        log_prob = self.flow.log_prob(data)
        
        # Loss = negative log-likelihood
        loss = -log_prob.mean()
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'avg_log_prob': log_prob.mean().item()
        }
    
    def train(self, data_loader, n_epochs=50):
        """Entraînement complet"""
        losses = []
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch in data_loader:
                result = self.train_step(batch)
                epoch_losses.append(result['loss'])
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
        
        return losses

# Exemple entraînement
flow_trainer = FlowTraining(flow, lr=0.001)

# Simuler données
train_data = torch.randn(1000, 20)

# Une étape
step_result = flow_trainer.train_step(train_data)
print(f"\nEntraînement Flow:")
print(f"  Loss: {step_result['loss']:.4f}")
print(f"  Log prob moyen: {step_result['avg_log_prob']:.4f}")
```

---

## Exercices

### Exercice 21.3.1
Implémentez une couche de couplage affine complète et testez son inverse.

### Exercice 21.3.2
Créez un Real NVP flow pour apprendre une distribution 2D complexe (ex: deux gaussiennes).

### Exercice 21.3.3
Entraînez un normalizing flow sur données d'événements HEP simulées et comparez distributions générées vs réelles.

### Exercice 21.3.4
Implémentez un flow conditionnel qui génère événements selon l'énergie du centre de masse.

---

## Points Clés à Retenir

> 📌 **Les normalizing flows apprennent transformation inversible vers distribution simple**

> 📌 **Ils fournissent densité explicite (contrairement GANs)**

> 📌 **L'échantillonnage est exact (pas d'approximation)**

> 📌 **Real NVP et Neural Spline Flows sont architectures populaires**

> 📌 **Les flows conditionnels permettent génération selon paramètres**

> 📌 **L'entraînement maximise likelihood (plus stable que GANs)**

---

*Section précédente : [21.2 GANs](./21_02_GANs.md) | Section suivante : [21.4 Compression](./21_04_Compression.md)*

