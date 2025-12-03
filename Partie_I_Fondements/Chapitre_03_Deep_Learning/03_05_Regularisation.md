# 3.5 Régularisation et Généralisation

---

## Introduction

La **régularisation** est l'ensemble des techniques qui empêchent le sur-apprentissage (overfitting) et améliorent la **généralisation** du modèle à de nouvelles données. Ces techniques sont essentielles pour obtenir des modèles robustes et sont intimement liées à la compression.

---

## Le Problème du Sur-apprentissage

### Diagnostic

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def diagnose_overfitting(train_losses, val_losses):
    """
    Diagnostique le sur-apprentissage à partir des courbes de loss
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Calcul du gap
    gap = np.array(val_losses) - np.array(train_losses)
    
    # Détection du point de divergence
    divergence_point = None
    for i in range(1, len(val_losses)):
        if val_losses[i] > val_losses[i-1] and train_losses[i] < train_losses[i-1]:
            divergence_point = i
            break
    
    diagnosis = {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'gap': gap[-1],
        'max_gap': np.max(gap),
        'divergence_epoch': divergence_point,
        'is_overfitting': gap[-1] > 0.1 * train_losses[-1]
    }
    
    return diagnosis

# Exemple de courbes typiques
epochs = np.arange(100)
train_loss = 2 * np.exp(-0.05 * epochs) + 0.1
val_loss = 2 * np.exp(-0.03 * epochs) + 0.2 + 0.005 * epochs  # Commence à remonter

diagnosis = diagnose_overfitting(train_loss.tolist(), val_loss.tolist())
print("Diagnostic:")
for key, value in diagnosis.items():
    print(f"  {key}: {value}")
```

### Compromis Biais-Variance

```
┌─────────────────────────────────────────────────────────────────┐
│                    Compromis Biais-Variance                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Erreur totale = Biais² + Variance + Bruit irréductible        │
│                                                                 │
│  Modèle simple:                                                 │
│    ✓ Faible variance (stable)                                  │
│    ✗ Fort biais (sous-apprentissage)                           │
│                                                                 │
│  Modèle complexe:                                               │
│    ✗ Forte variance (instable)                                 │
│    ✓ Faible biais (peut capturer des patterns complexes)       │
│                                                                 │
│  Régularisation: Réduit la variance au prix d'un peu de biais  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Régularisation des Poids

### L2 Regularization (Weight Decay)

```python
class L2Regularization:
    """
    Régularisation L2: pénalise les grands poids
    
    Loss_total = Loss_data + λ * ||W||²
    """
    
    @staticmethod
    def compute_penalty(model, lambda_reg):
        """Calcule la pénalité L2"""
        l2_penalty = 0
        for param in model.parameters():
            l2_penalty += torch.sum(param ** 2)
        return lambda_reg * l2_penalty
    
    @staticmethod
    def apply_weight_decay(optimizer, weight_decay):
        """
        Applique le weight decay directement dans l'optimiseur
        
        Équivalent à L2 pour SGD, mais différent pour Adam
        """
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.data.add_(param.data, alpha=-weight_decay * group['lr'])

# Exemple d'utilisation
model = nn.Linear(100, 10)
criterion = nn.CrossEntropyLoss()

# Méthode 1: Pénalité explicite
x = torch.randn(32, 100)
y = torch.randint(0, 10, (32,))

output = model(x)
data_loss = criterion(output, y)
l2_loss = L2Regularization.compute_penalty(model, lambda_reg=0.01)
total_loss = data_loss + l2_loss

print(f"Data loss: {data_loss.item():.4f}")
print(f"L2 penalty: {l2_loss.item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")

# Méthode 2: Via l'optimiseur (recommandé)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
```

### L1 Regularization (Sparsity)

```python
class L1Regularization:
    """
    Régularisation L1: encourage la sparsité des poids
    
    Loss_total = Loss_data + λ * ||W||₁
    """
    
    @staticmethod
    def compute_penalty(model, lambda_reg):
        """Calcule la pénalité L1"""
        l1_penalty = 0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        return lambda_reg * l1_penalty
    
    @staticmethod
    def count_near_zero(model, threshold=1e-3):
        """Compte les poids proches de zéro"""
        total = 0
        near_zero = 0
        for param in model.parameters():
            total += param.numel()
            near_zero += (torch.abs(param) < threshold).sum().item()
        return near_zero, total, near_zero / total

# Comparaison L1 vs L2
class RegularizedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Entraînement avec L1 vs L2
def train_with_regularization(reg_type='l2', lambda_reg=0.01, epochs=100):
    model = RegularizedMLP(100, 50, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        if reg_type == 'l1':
            loss += L1Regularization.compute_penalty(model, lambda_reg)
        elif reg_type == 'l2':
            loss += L2Regularization.compute_penalty(model, lambda_reg)
        
        loss.backward()
        optimizer.step()
    
    # Analyse de la sparsité
    near_zero, total, sparsity = L1Regularization.count_near_zero(model)
    return model, sparsity

print("Comparaison L1 vs L2:")
_, sparsity_l1 = train_with_regularization('l1', 0.001)
_, sparsity_l2 = train_with_regularization('l2', 0.001)
_, sparsity_none = train_with_regularization(None, 0)

print(f"  Sans régularisation: {sparsity_none:.2%} sparsité")
print(f"  L2: {sparsity_l2:.2%} sparsité")
print(f"  L1: {sparsity_l1:.2%} sparsité")
```

---

## Dropout

### Implémentation et Analyse

```python
class DropoutAnalysis:
    """
    Analyse du dropout comme régularisation
    """
    
    @staticmethod
    def manual_dropout(x, p=0.5, training=True):
        """
        Implémentation manuelle du dropout
        
        Pendant l'entraînement: met à zéro avec probabilité p, scale par 1/(1-p)
        Pendant l'inférence: identité
        """
        if not training or p == 0:
            return x
        
        # Masque binomial
        mask = (torch.rand_like(x) > p).float()
        
        # Scale pour maintenir l'espérance
        return x * mask / (1 - p)
    
    @staticmethod
    def analyze_dropout_effect(model, x, n_samples=100):
        """
        Analyse la variance introduite par le dropout
        """
        model.train()  # Active le dropout
        
        outputs = []
        for _ in range(n_samples):
            with torch.no_grad():
                out = model(x)
                outputs.append(out)
        
        outputs = torch.stack(outputs)
        
        mean_output = outputs.mean(dim=0)
        std_output = outputs.std(dim=0)
        
        return {
            'mean': mean_output,
            'std': std_output,
            'coefficient_of_variation': (std_output / (mean_output.abs() + 1e-10)).mean().item()
        }

class MLPWithDropout(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

# Test
model = MLPWithDropout(100, 64, 10, dropout_rate=0.5)
x = torch.randn(1, 100)

analysis = DropoutAnalysis.analyze_dropout_effect(model, x)
print(f"Coefficient de variation avec dropout: {analysis['coefficient_of_variation']:.4f}")
```

### Variantes de Dropout

```python
class DropoutVariants(nn.Module):
    """
    Différentes variantes de dropout
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Dropout standard
        self.dropout = nn.Dropout(0.5)
        
        # Dropout 2D (pour CNN) - drop des feature maps entières
        self.dropout2d = nn.Dropout2d(0.5)
        
        # Alpha Dropout (pour SELU)
        self.alpha_dropout = nn.AlphaDropout(0.5)
        
        # DropConnect (drop des poids, pas des activations)
        # Implémenté manuellement
        
    def dropconnect(self, x, weight, p=0.5, training=True):
        """
        DropConnect: applique dropout aux poids
        """
        if not training:
            return F.linear(x, weight)
        
        mask = (torch.rand_like(weight) > p).float()
        masked_weight = weight * mask / (1 - p)
        return F.linear(x, masked_weight)


class SpatialDropout(nn.Module):
    """
    Spatial Dropout pour séquences/CNN
    
    Drop des canaux entiers plutôt que des éléments individuels
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        # x: (batch, channels, ...) 
        if not self.training or self.p == 0:
            return x
        
        # Crée un masque par canal
        shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
        mask = (torch.rand(shape, device=x.device) > self.p).float()
        
        return x * mask / (1 - self.p)
```

---

## Batch Normalization

```python
class BatchNormAnalysis:
    """
    Analyse de la Batch Normalization comme régularisation
    """
    
    @staticmethod
    def manual_batch_norm(x, gamma, beta, eps=1e-5, training=True, 
                          running_mean=None, running_var=None, momentum=0.1):
        """
        Implémentation manuelle de BatchNorm
        """
        if training:
            # Statistiques du batch
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Mise à jour des running statistics
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(mean * momentum)
                running_var.mul_(1 - momentum).add_(var * momentum)
        else:
            # Utilise les running statistics
            mean = running_mean
            var = running_var
        
        # Normalisation
        x_norm = (x - mean) / torch.sqrt(var + eps)
        
        # Scale et shift
        return gamma * x_norm + beta
    
    @staticmethod
    def analyze_bn_effect(model, data_loader):
        """
        Analyse l'effet de BatchNorm sur les activations
        """
        activations_before = []
        activations_after = []
        
        # Hook pour capturer les activations
        def hook_before(module, input, output):
            activations_before.append(input[0].detach())
        
        def hook_after(module, input, output):
            activations_after.append(output.detach())
        
        # Trouve les couches BN et ajoute les hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.register_forward_hook(hook_before)
                module.register_forward_hook(hook_after)
        
        # Forward pass
        model.eval()
        for x, _ in data_loader:
            _ = model(x)
            break
        
        # Analyse
        if activations_before and activations_after:
            before = activations_before[0]
            after = activations_after[0]
            
            return {
                'mean_before': before.mean().item(),
                'std_before': before.std().item(),
                'mean_after': after.mean().item(),
                'std_after': after.std().item()
            }
        return None
```

---

## Data Augmentation

```python
class DataAugmentation:
    """
    Techniques d'augmentation de données
    """
    
    @staticmethod
    def mixup(x1, y1, x2, y2, alpha=0.2):
        """
        Mixup: interpolation linéaire entre exemples
        
        x_mix = λ*x1 + (1-λ)*x2
        y_mix = λ*y1 + (1-λ)*y2
        """
        lam = np.random.beta(alpha, alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix, y_mix, lam
    
    @staticmethod
    def cutout(x, n_holes=1, length=8):
        """
        Cutout: masque des régions carrées de l'image
        """
        h, w = x.shape[-2:]
        mask = torch.ones_like(x)
        
        for _ in range(n_holes):
            y = np.random.randint(h)
            x_pos = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x_pos - length // 2, 0, w)
            x2 = np.clip(x_pos + length // 2, 0, w)
            
            mask[..., y1:y2, x1:x2] = 0
        
        return x * mask
    
    @staticmethod
    def cutmix(x1, y1, x2, y2, alpha=1.0):
        """
        CutMix: coupe et colle des régions entre images
        """
        lam = np.random.beta(alpha, alpha)
        
        h, w = x1.shape[-2:]
        
        # Taille de la région à couper
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        # Position aléatoire
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        
        y1_pos = np.clip(cy - cut_h // 2, 0, h)
        y2_pos = np.clip(cy + cut_h // 2, 0, h)
        x1_pos = np.clip(cx - cut_w // 2, 0, w)
        x2_pos = np.clip(cx + cut_w // 2, 0, w)
        
        # Mixage
        x_mix = x1.clone()
        x_mix[..., y1_pos:y2_pos, x1_pos:x2_pos] = x2[..., y1_pos:y2_pos, x1_pos:x2_pos]
        
        # Ajuste lambda selon la région réellement coupée
        lam = 1 - ((y2_pos - y1_pos) * (x2_pos - x1_pos) / (h * w))
        
        return x_mix, lam * y1 + (1 - lam) * y2, lam


class PhysicsAugmentation:
    """
    Augmentation spécifique à la physique des particules
    """
    
    @staticmethod
    def rotate_jet(jet_constituents, angle=None):
        """
        Rotation aléatoire d'un jet dans le plan (η, φ)
        """
        if angle is None:
            angle = np.random.uniform(0, 2 * np.pi)
        
        # jet_constituents: (n_particles, features) où features inclut [pt, eta, phi, ...]
        eta = jet_constituents[:, 1]
        phi = jet_constituents[:, 2]
        
        # Rotation
        new_phi = phi + angle
        new_phi = torch.where(new_phi > np.pi, new_phi - 2*np.pi, new_phi)
        new_phi = torch.where(new_phi < -np.pi, new_phi + 2*np.pi, new_phi)
        
        result = jet_constituents.clone()
        result[:, 2] = new_phi
        
        return result
    
    @staticmethod
    def smear_momentum(particles, resolution=0.1):
        """
        Smearing gaussien du momentum (simule la résolution du détecteur)
        """
        pt = particles[:, 0]
        smeared_pt = pt * (1 + resolution * torch.randn_like(pt))
        
        result = particles.clone()
        result[:, 0] = smeared_pt
        
        return result
```

---

## Early Stopping

```python
class EarlyStopping:
    """
    Early stopping pour éviter le sur-apprentissage
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min', restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.should_stop = False
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_model_state)
                return True
        
        return False

# Utilisation
"""
early_stopping = EarlyStopping(patience=10, mode='min')

for epoch in range(1000):
    train_loss = train_epoch()
    val_loss = validate()
    
    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
"""
```

---

## Connexion avec la Compression

```python
class RegularizationForCompression:
    """
    Régularisations qui facilitent la compression
    """
    
    @staticmethod
    def group_lasso(model, groups, lambda_reg=0.01):
        """
        Group Lasso: encourage la sparsité de groupes de poids
        
        Utile pour le pruning structuré (filtres entiers, neurones)
        """
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Groupe par filtre (pour CNN) ou par neurone (pour FC)
                if param.dim() == 4:  # Conv
                    group_norms = param.view(param.size(0), -1).norm(dim=1)
                elif param.dim() == 2:  # Linear
                    group_norms = param.norm(dim=1)
                else:
                    continue
                
                penalty += group_norms.sum()
        
        return lambda_reg * penalty
    
    @staticmethod
    def low_rank_regularization(model, lambda_reg=0.01):
        """
        Régularisation nucléaire: encourage les matrices de rang faible
        """
        penalty = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                # Norme nucléaire = somme des valeurs singulières
                U, S, V = torch.svd(param)
                penalty += S.sum()
        
        return lambda_reg * penalty
    
    @staticmethod
    def entropy_regularization(activations, lambda_reg=0.01):
        """
        Régularisation par entropie: encourage des activations "décidées"
        
        Facilite la quantification
        """
        # Normalise les activations
        probs = torch.softmax(activations, dim=-1)
        
        # Entropie
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        
        # On veut minimiser l'entropie (décisions nettes)
        return lambda_reg * entropy

# Exemple d'entraînement avec régularisation pour compression
def train_for_compression(model, data_loader, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            
            output = model(x)
            
            # Loss principale
            loss = criterion(output, y)
            
            # Régularisations pour compression
            loss += RegularizationForCompression.group_lasso(model, None, 0.001)
            
            loss.backward()
            optimizer.step()
    
    return model
```

---

## Exercices

### Exercice 3.5.1
Comparez l'effet de L1 et L2 sur la distribution des poids après entraînement. Visualisez les histogrammes.

### Exercice 3.5.2
Implémentez le "scheduled dropout" où le taux de dropout diminue au cours de l'entraînement.

### Exercice 3.5.3
Créez une fonction d'augmentation de données spécifique aux images de calorimètres.

---

## Points Clés à Retenir

> 📌 **La régularisation contrôle la complexité effective du modèle**

> 📌 **L1 encourage la sparsité, L2 encourage les petits poids uniformes**

> 📌 **Le dropout peut être vu comme un ensemble implicite de modèles**

> 📌 **Les régularisations structurées (group lasso) facilitent le pruning**

---

## Références

1. Srivastava, N. et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR, 2014
2. Ioffe, S., Szegedy, C. "Batch Normalization." ICML, 2015
3. Zhang, H. et al. "mixup: Beyond Empirical Risk Minimization." ICLR, 2018
4. Wen, W. et al. "Learning Structured Sparsity in Deep Neural Networks." NeurIPS, 2016

---

*Partie suivante : [Partie II - Réseaux de Tenseurs](../../Partie_II_Reseaux_Tenseurs/Chapitre_04_Introduction_Tenseurs/04_introduction.md)*

