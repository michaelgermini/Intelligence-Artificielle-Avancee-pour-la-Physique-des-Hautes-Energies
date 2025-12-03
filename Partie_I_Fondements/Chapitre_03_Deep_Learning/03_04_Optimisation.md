# 3.4 Fonctions de Perte et Optimisation

---

## Introduction

L'entraînement des réseaux de neurones consiste à minimiser une **fonction de perte** qui mesure l'écart entre les prédictions et les cibles. Le choix de la fonction de perte et de l'algorithme d'optimisation est crucial pour la convergence et la performance finale.

---

## Fonctions de Perte

### Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossFunctions:
    """
    Collection de fonctions de perte pour la classification
    """
    
    @staticmethod
    def cross_entropy(logits, targets):
        """
        Cross-entropy loss pour classification multi-classe
        
        L = -sum_c y_c * log(p_c)
        """
        return F.cross_entropy(logits, targets)
    
    @staticmethod
    def binary_cross_entropy(logits, targets):
        """
        Binary cross-entropy pour classification binaire
        
        L = -[y * log(p) + (1-y) * log(1-p)]
        """
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    
    @staticmethod
    def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
        """
        Focal loss pour données déséquilibrées
        
        FL = -α(1-p)^γ * log(p)
        
        Réduit le poids des exemples faciles
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** gamma
        
        # Alpha weighting
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        return (focal_weight * ce_loss).mean()
    
    @staticmethod
    def label_smoothing_loss(logits, targets, smoothing=0.1):
        """
        Cross-entropy avec label smoothing
        
        Remplace one-hot par distribution lissée
        """
        n_classes = logits.size(-1)
        
        # Smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss

# Démonstration
logits = torch.randn(32, 10)  # Batch de 32, 10 classes
targets = torch.randint(0, 10, (32,))

print("Comparaison des losses:")
print(f"  Cross-entropy: {LossFunctions.cross_entropy(logits, targets):.4f}")
print(f"  Focal loss: {LossFunctions.focal_loss(logits, targets):.4f}")
print(f"  Label smoothing: {LossFunctions.label_smoothing_loss(logits, targets):.4f}")
```

### Régression

```python
class RegressionLosses:
    """
    Fonctions de perte pour la régression
    """
    
    @staticmethod
    def mse(predictions, targets):
        """
        Mean Squared Error
        L = (1/n) * sum((y - ŷ)²)
        """
        return F.mse_loss(predictions, targets)
    
    @staticmethod
    def mae(predictions, targets):
        """
        Mean Absolute Error (L1)
        L = (1/n) * sum(|y - ŷ|)
        """
        return F.l1_loss(predictions, targets)
    
    @staticmethod
    def huber(predictions, targets, delta=1.0):
        """
        Huber loss (Smooth L1)
        Combine MSE et MAE: quadratique près de 0, linéaire loin
        """
        return F.smooth_l1_loss(predictions, targets, beta=delta)
    
    @staticmethod
    def quantile_loss(predictions, targets, quantile=0.5):
        """
        Quantile loss pour régression quantile
        Permet de prédire différents percentiles
        """
        errors = targets - predictions
        loss = torch.max(quantile * errors, (quantile - 1) * errors)
        return loss.mean()

# Comparaison sur données avec outliers
predictions = torch.randn(100)
targets = predictions + 0.1 * torch.randn(100)
targets[0] = 100  # Outlier

print("\nComparaison des losses de régression (avec outlier):")
print(f"  MSE: {RegressionLosses.mse(predictions, targets):.4f}")
print(f"  MAE: {RegressionLosses.mae(predictions, targets):.4f}")
print(f"  Huber: {RegressionLosses.huber(predictions, targets):.4f}")
```

### Losses pour Physique des Particules

```python
class PhysicsLosses:
    """
    Fonctions de perte spécifiques à la physique des particules
    """
    
    @staticmethod
    def weighted_cross_entropy(logits, targets, weights):
        """
        Cross-entropy pondérée par événement
        
        Utile quand les événements ont des poids différents
        (ex: poids Monte Carlo)
        """
        ce = F.cross_entropy(logits, targets, reduction='none')
        return (weights * ce).sum() / weights.sum()
    
    @staticmethod
    def binned_likelihood_loss(predicted_yields, observed_counts):
        """
        Negative log-likelihood pour histogrammes
        
        Utilisé pour les fits de distributions
        L = sum_i [μ_i - n_i * log(μ_i)]
        """
        # Évite log(0)
        predicted_yields = torch.clamp(predicted_yields, min=1e-10)
        
        # Poisson NLL
        nll = predicted_yields - observed_counts * torch.log(predicted_yields)
        return nll.sum()
    
    @staticmethod
    def adversarial_loss(classifier_output, nuisance_output, lambda_adv=1.0):
        """
        Loss adversariale pour décorrélation
        
        Entraîne le classifieur à être indépendant d'une variable de nuisance
        """
        # Classification loss
        # (à compléter avec les targets)
        
        # Adversarial loss: maximise l'entropie de la prédiction de nuisance
        entropy = -(nuisance_output * torch.log(nuisance_output + 1e-10)).sum(dim=-1)
        
        return -lambda_adv * entropy.mean()
```

---

## Algorithmes d'Optimisation

### Gradient Descent et Variantes

```python
class OptimizersExplained:
    """
    Implémentation pédagogique des optimiseurs
    """
    
    @staticmethod
    def sgd_step(params, grads, lr):
        """
        Stochastic Gradient Descent
        θ = θ - lr * ∇L
        """
        return [p - lr * g for p, g in zip(params, grads)]
    
    @staticmethod
    def momentum_step(params, grads, velocities, lr, momentum=0.9):
        """
        SGD avec momentum
        v = momentum * v + lr * ∇L
        θ = θ - v
        """
        new_velocities = []
        new_params = []
        
        for p, g, v in zip(params, grads, velocities):
            v_new = momentum * v + lr * g
            p_new = p - v_new
            new_velocities.append(v_new)
            new_params.append(p_new)
        
        return new_params, new_velocities
    
    @staticmethod
    def adam_step(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam optimizer
        
        Combine momentum et RMSprop avec correction de biais
        """
        new_m = []
        new_v = []
        new_params = []
        
        for p, g, m_i, v_i in zip(params, grads, m, v):
            # Mise à jour des moments
            m_new = beta1 * m_i + (1 - beta1) * g
            v_new = beta2 * v_i + (1 - beta2) * g**2
            
            # Correction de biais
            m_hat = m_new / (1 - beta1**t)
            v_hat = v_new / (1 - beta2**t)
            
            # Mise à jour des paramètres
            p_new = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
            
            new_m.append(m_new)
            new_v.append(v_new)
            new_params.append(p_new)
        
        return new_params, new_m, new_v


class AdamW(torch.optim.Optimizer):
    """
    AdamW: Adam avec weight decay découplé
    
    Meilleur pour la généralisation que Adam standard
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay découplé (avant la mise à jour Adam)
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                state = self.state[p]
                
                # Initialisation
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Mise à jour des moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Correction de biais
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                
                # Mise à jour
                denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
```

### Comparaison des Optimiseurs

```python
def compare_optimizers(model_fn, data_loader, n_epochs=10):
    """
    Compare différents optimiseurs sur le même problème
    """
    optimizers = {
        'SGD': lambda p: torch.optim.SGD(p, lr=0.01),
        'SGD+Momentum': lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
        'Adam': lambda p: torch.optim.Adam(p, lr=0.001),
        'AdamW': lambda p: torch.optim.AdamW(p, lr=0.001, weight_decay=0.01),
    }
    
    results = {}
    
    for name, opt_fn in optimizers.items():
        model = model_fn()
        optimizer = opt_fn(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            for batch_x, batch_y in data_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(data_loader))
        
        results[name] = losses
        print(f"{name}: Final loss = {losses[-1]:.4f}")
    
    return results
```

---

## Learning Rate Scheduling

```python
class LRSchedulers:
    """
    Stratégies de scheduling du learning rate
    """
    
    @staticmethod
    def step_decay(initial_lr, epoch, drop_factor=0.5, drop_every=10):
        """Décroissance par paliers"""
        return initial_lr * (drop_factor ** (epoch // drop_every))
    
    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """Décroissance exponentielle"""
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs):
        """Cosine annealing"""
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    
    @staticmethod
    def warmup_cosine(initial_lr, epoch, warmup_epochs, total_epochs):
        """Warmup linéaire puis cosine decay"""
        if epoch < warmup_epochs:
            return initial_lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    @staticmethod
    def one_cycle(initial_lr, epoch, total_epochs, max_lr_factor=10):
        """
        One-cycle policy (Smith, 2018)
        
        Monte jusqu'à max_lr puis descend
        """
        mid_epoch = total_epochs // 2
        
        if epoch < mid_epoch:
            # Phase montante
            return initial_lr + (initial_lr * max_lr_factor - initial_lr) * epoch / mid_epoch
        else:
            # Phase descendante
            progress = (epoch - mid_epoch) / (total_epochs - mid_epoch)
            return initial_lr * max_lr_factor * (1 - progress)

# Visualisation
import matplotlib.pyplot as plt

def plot_lr_schedules():
    epochs = np.arange(100)
    initial_lr = 0.1
    
    schedules = {
        'Step Decay': [LRSchedulers.step_decay(initial_lr, e) for e in epochs],
        'Exponential': [LRSchedulers.exponential_decay(initial_lr, e) for e in epochs],
        'Cosine': [LRSchedulers.cosine_annealing(initial_lr, e, 100) for e in epochs],
        'Warmup+Cosine': [LRSchedulers.warmup_cosine(initial_lr, e, 10, 100) for e in epochs],
        'One-Cycle': [LRSchedulers.one_cycle(initial_lr, e, 100) for e in epochs],
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, lrs in schedules.items():
        ax.plot(epochs, lrs, label=name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedules')
    ax.legend()
    ax.grid(True)
    
    return fig

# Utilisation avec PyTorch
"""
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()
"""
```

---

## Gradient Clipping et Stabilité

```python
def gradient_analysis(model, loss):
    """
    Analyse les gradients pour détecter les problèmes
    """
    loss.backward()
    
    grad_norms = []
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            grad_norm = grad.norm().item()
            grad_norms.append(grad_norm)
            
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item()
            }
    
    total_norm = np.sqrt(sum(n**2 for n in grad_norms))
    
    return {
        'total_norm': total_norm,
        'layer_stats': grad_stats,
        'has_issues': any(s['has_nan'] or s['has_inf'] for s in grad_stats.values())
    }


def apply_gradient_clipping(model, max_norm=1.0, norm_type=2):
    """
    Applique le gradient clipping
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)


class GradientAccumulation:
    """
    Accumulation de gradients pour simuler des batch plus grands
    """
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def backward_step(self, loss):
        """
        Accumule les gradients
        """
        # Normalise la loss par le nombre d'étapes
        (loss / self.accumulation_steps).backward()
        
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            # Applique les gradients accumulés
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True  # Mise à jour effectuée
        
        return False  # Gradients accumulés
```

---

## Entraînement Complet

```python
class Trainer:
    """
    Classe d'entraînement complète avec toutes les bonnes pratiques
    """
    
    def __init__(self, model, optimizer, criterion, scheduler=None, 
                 device='cuda', grad_clip=None, mixed_precision=False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        return total_loss / len(val_loader), correct / total
    
    def fit(self, train_loader, val_loader, epochs):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}, "
                  f"LR={current_lr:.6f}")
            
            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
        
        return self.history
```

---

## Exercices

### Exercice 3.4.1
Implémentez une fonction de perte personnalisée qui combine cross-entropy et une pénalité sur la norme des poids.

### Exercice 3.4.2
Comparez Adam et SGD+Momentum sur un problème de classification avec différents learning rates.

### Exercice 3.4.3
Implémentez le learning rate finder (Smith, 2017) qui trouve automatiquement un bon learning rate.

---

## Points Clés à Retenir

> 📌 **Le choix de la loss dépend du problème: CE pour classification, MSE/Huber pour régression**

> 📌 **Adam converge plus vite mais AdamW généralise souvent mieux**

> 📌 **Le warmup est crucial pour les Transformers**

> 📌 **Le gradient clipping stabilise l'entraînement des réseaux profonds**

---

## Références

1. Kingma, D., Ba, J. "Adam: A Method for Stochastic Optimization." ICLR, 2015
2. Loshchilov, I., Hutter, F. "Decoupled Weight Decay Regularization." ICLR, 2019
3. Smith, L. "Cyclical Learning Rates for Training Neural Networks." WACV, 2017
4. Goyal, P. et al. "Accurate, Large Minibatch SGD." arXiv, 2017

---

*Section suivante : [3.5 Régularisation et Généralisation](./03_05_Regularisation.md)*

