# 10.5 Self-distillation

---

## Introduction

La **self-distillation** est une variante où le modèle apprend de ses propres prédictions précédentes ou d'une version antérieure de lui-même. C'est utile pour améliorer un modèle sans avoir besoin d'un teacher séparé.

---

## Principe

### Types de Self-distillation

1. **Temporal** : Le modèle à l'epoch $t$ apprend du modèle à l'epoch $t-1$
2. **Progressive** : Différentes versions du modèle (checkpoints) servent de teachers
3. **Ensemble** : Un ensemble de modèles sert de teacher pour chaque membre

---

## Self-distillation Temporelle

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSelfDistillation(nn.Module):
    """
    Self-distillation temporelle
    Le modèle apprend de ses prédictions précédentes
    """
    
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, current_logits, previous_logits, labels, alpha=0.5):
        """
        Args:
            current_logits: Logits de l'étape actuelle
            previous_logits: Logits de l'étape précédente (teacher)
            labels: Labels ground truth
            alpha: Poids de la self-distillation
        """
        # Soft labels depuis les prédictions précédentes
        prev_probs = F.softmax(previous_logits / self.temperature, dim=1)
        curr_log_probs = F.log_softmax(current_logits / self.temperature, dim=1)
        
        # Distillation
        soft_loss = self.kl_loss(curr_log_probs, prev_probs) * (self.temperature ** 2)
        
        # Classification standard
        hard_loss = self.ce_loss(current_logits, labels)
        
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return {
            'total': total_loss,
            'soft': soft_loss,
            'hard': hard_loss
        }

# Training avec self-distillation temporelle
def train_with_temporal_self_distillation(model, train_loader, epochs=50, alpha=0.3):
    """
    Entraînement avec self-distillation temporelle
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TemporalSelfDistillation(temperature=4.0)
    ce_loss = nn.CrossEntropyLoss()
    
    previous_logits = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            current_logits = model(data)
            
            if previous_logits is not None and previous_logits.size(0) == current_logits.size(0):
                # Self-distillation
                losses = loss_fn(current_logits, previous_logits, labels, alpha=alpha)
                loss = losses['total']
            else:
                # Premier batch ou tailles différentes: classification standard
                loss = ce_loss(current_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Sauvegarde pour la prochaine itération
            previous_logits = current_logits.detach()
        
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}')
```

---

## Self-distillation Progressive

```python
class ProgressiveSelfDistillation:
    """
    Self-distillation progressive
    Utilise des checkpoints précédents comme teachers
    """
    
    def __init__(self, model, checkpoint_interval=10):
        self.model = model
        self.checkpoint_interval = checkpoint_interval
        self.teacher_checkpoints = []
    
    def save_checkpoint(self, epoch, model_state_dict):
        """Sauvegarde un checkpoint pour servir de teacher"""
        if epoch % self.checkpoint_interval == 0:
            self.teacher_checkpoints.append({
                'epoch': epoch,
                'state_dict': model_state_dict.copy()
            })
            # Garde seulement les N derniers checkpoints
            if len(self.teacher_checkpoints) > 3:
                self.teacher_checkpoints.pop(0)
    
    def get_teacher_logits(self, data, epoch):
        """
        Récupère les logits d'un teacher (checkpoint précédent)
        """
        if not self.teacher_checkpoints:
            return None
        
        # Utilise le checkpoint le plus récent avant epoch actuel
        teacher_epoch = None
        for checkpoint in reversed(self.teacher_checkpoints):
            if checkpoint['epoch'] < epoch:
                teacher_epoch = checkpoint['epoch']
                teacher_state = checkpoint['state_dict']
                break
        
        if teacher_epoch is None:
            return None
        
        # Charge temporairement le teacher
        teacher_model = type(self.model)()
        teacher_model.load_state_dict(teacher_state)
        teacher_model.eval()
        
        with torch.no_grad():
            teacher_logits = teacher_model(data)
        
        return teacher_logits

def train_with_progressive_self_distillation(model, train_loader, epochs=100):
    """
    Entraînement avec self-distillation progressive
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = TemporalSelfDistillation(temperature=4.0)
    ce_loss = nn.CrossEntropyLoss()
    
    progressive_distill = ProgressiveSelfDistillation(model, checkpoint_interval=10)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            
            current_logits = model(data)
            
            # Récupère les logits du teacher (checkpoint précédent)
            teacher_logits = progressive_distill.get_teacher_logits(data, epoch)
            
            if teacher_logits is not None:
                # Self-distillation
                losses = loss_fn(current_logits, teacher_logits, labels, alpha=0.3)
                loss = losses['total']
            else:
                # Pas encore de teacher: classification standard
                loss = ce_loss(current_logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Sauvegarde checkpoint
        progressive_distill.save_checkpoint(epoch, model.state_dict())
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}')
```

---

## Self-distillation Ensemble

```python
class EnsembleSelfDistillation:
    """
    Self-distillation avec ensemble
    Chaque membre apprend de l'ensemble
    """
    
    def __init__(self, models, temperature=4.0):
        """
        Args:
            models: Liste de modèles (ensemble)
        """
        self.models = models
        self.temperature = temperature
    
    def get_ensemble_logits(self, data):
        """
        Calcule les logits de l'ensemble (moyenne)
        """
        all_logits = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(data)
                all_logits.append(logits)
        
        # Moyenne des logits
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        return ensemble_logits
    
    def train_member(self, member_idx, train_loader, epochs=10, alpha=0.5):
        """
        Entraîne un membre de l'ensemble avec self-distillation
        """
        model = self.models[member_idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = TemporalSelfDistillation(temperature=self.temperature)
        ce_loss = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            model.train()
            
            for data, labels in train_loader:
                optimizer.zero_grad()
                
                # Logits du membre actuel
                member_logits = model(data)
                
                # Logits de l'ensemble (sans le membre actuel)
                # En pratique, exclure le membre actuel de la moyenne
                ensemble_logits = self.get_ensemble_logits(data)
                
                # Self-distillation
                losses = loss_fn(member_logits, ensemble_logits, labels, alpha=alpha)
                losses['total'].backward()
                
                optimizer.step()

# Exemple
models = [SimpleStudent() for _ in range(3)]
ensemble_distill = EnsembleSelfDistillation(models)

# Entraîne chaque membre
for i in range(len(models)):
    # train_loader défini ailleurs
    # ensemble_distill.train_member(i, train_loader, epochs=10)
    pass
```

---

## Avantages de la Self-distillation

### Avantages

- **Pas besoin de teacher externe** : Le modèle s'améliore lui-même
- **Progressive refinement** : Amélioration continue
- **Simplicité** : Pas de gestion de deux modèles

### Limitations

- **Convergence** : Peut converger plus lentement
- **Stabilité** : Risque d'oscillations si alpha trop élevé

---

## Exercices

### Exercice 10.5.1
Expérimentez avec différentes valeurs d'alpha pour la self-distillation temporelle.

### Exercice 10.5.2
Implémentez une variante qui utilise une moyenne mobile des prédictions précédentes comme teacher.

---

## Points Clés à Retenir

> 📌 **Self-distillation permet d'améliorer un modèle sans teacher externe**

> 📌 **Temporelle: apprend des prédictions précédentes**

> 📌 **Progressive: utilise des checkpoints précédents**

> 📌 **Ensemble: chaque membre apprend de l'ensemble**

> 📌 **Alpha doit être ajusté pour éviter les oscillations**

---

*Section suivante : [10.6 Combinaison avec Compression Tensorielle](./10_06_Tensor_Compression.md)*

