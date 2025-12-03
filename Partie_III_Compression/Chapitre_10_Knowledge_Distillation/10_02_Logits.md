# 10.2 Distillation des Logits

---

## Introduction

La **distillation des logits** est la forme la plus simple et la plus utilisée de knowledge distillation. Elle transfère les connaissances via les sorties finales (logits) du teacher.

---

## Principe

### Logits et Soft Labels

Les logits sont les sorties non normalisées avant softmax. Les soft labels sont obtenues en appliquant softmax avec température :

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

où $T$ est la température et $z_i$ sont les logits.

---

## Implémentation Complète

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitsDistillationLoss(nn.Module):
    """
    Loss de distillation basée uniquement sur les logits
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: (batch, num_classes)
            teacher_logits: (batch, num_classes)
            labels: (batch,)
        """
        # Soft labels avec température
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Loss soft: KL divergence
        soft_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Loss hard: Cross-entropy
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combinaison
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return {
            'total': total_loss,
            'soft': soft_loss,
            'hard': hard_loss,
            'teacher_probs': teacher_probs,
            'student_probs': F.softmax(student_logits, dim=1)
        }

# Exemple
loss_fn = LogitsDistillationLoss(temperature=4.0, alpha=0.7)

student_logits = torch.randn(32, 10)
teacher_logits = torch.randn(32, 10) * 3
labels = torch.randint(0, 10, (32,))

losses = loss_fn(student_logits, teacher_logits, labels)
print(f"Logits Distillation:")
print(f"  Total: {losses['total'].item():.4f}")
print(f"  Soft: {losses['soft'].item():.4f}")
print(f"  Hard: {losses['hard'].item():.4f}")
```

---

## Variantes de la Loss

### Variante 1 : Focal Distillation

```python
class FocalDistillationLoss(nn.Module):
    """
    Distillation avec focal loss pour se concentrer sur les exemples difficiles
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, student_logits, teacher_logits, labels):
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - teacher_probs.max(dim=1)[0]) ** self.gamma
        
        # Weighted KL divergence
        kl_per_sample = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs)).sum(dim=1)
        soft_loss = (focal_weight * kl_per_sample).mean() * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### Variante 2 : Attention Distillation

```python
class AttentionLogitsLoss(nn.Module):
    """
    Distillation avec attention sur les classes importantes
    """
    
    def __init__(self, temperature=4.0, alpha=0.7, top_k=5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.top_k = top_k
    
    def forward(self, student_logits, teacher_logits, labels):
        # Top-k classes du teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        top_k_probs, top_k_indices = teacher_probs.topk(self.top_k, dim=1)
        
        # Masque d'attention
        attention_mask = torch.zeros_like(teacher_probs)
        attention_mask.scatter_(1, top_k_indices, 1.0)
        
        # Loss avec attention
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        masked_kl = (teacher_probs * (torch.log(teacher_probs + 1e-10) - student_log_probs) * attention_mask).sum(dim=1)
        soft_loss = masked_kl.mean() * (self.temperature ** 2)
        
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

---

## Analyse de la Distillation Logits

### Visualisation des Distributions

```python
def visualize_distillation_distributions(teacher_logits, student_logits, temperature=4.0):
    """
    Visualise les distributions teacher vs student
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.softmax(student_logits / temperature, dim=1)
    student_probs_hard = F.softmax(student_logits, dim=1)
    
    # Pour un exemple
    idx = 0
    
    print(f"Exemple {idx}:")
    print(f"  Teacher (T={temperature}): {teacher_probs[idx].tolist()}")
    print(f"  Student (T={temperature}): {student_probs[idx].tolist()}")
    print(f"  Student (T=1.0): {student_probs_hard[idx].tolist()}")
    print(f"  Hard label: {labels[idx].item()}")
```

---

## Training avec Logits Distillation

```python
def train_with_logits_distillation(teacher, student, train_loader, val_loader,
                                   temperature=4.0, alpha=0.7, epochs=50):
    """
    Entraînement complet avec distillation de logits
    """
    teacher.eval()
    student.train()
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_fn = LogitsDistillationLoss(temperature, alpha)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            student_logits = student(data)
            
            # Loss
            losses = loss_fn(student_logits, teacher_logits, labels)
            losses['total'].backward()
            
            optimizer.step()
            total_loss += losses['total'].item()
        
        # Validation
        student.eval()
        val_acc = evaluate(student, val_loader)
        student.train()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), 'best_logits_distilled.pt')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
                  f'Acc={val_acc:.2f}%')
    
    return student

def evaluate(model, data_loader):
    """Évalue un modèle"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total
```

---

## Exercices

### Exercice 10.2.1
Expérimentez avec différentes températures (1, 2, 4, 8, 16) et mesurez l'impact sur les performances.

### Exercice 10.2.2
Implémentez une variante de distillation qui utilise uniquement les top-k classes du teacher.

---

## Points Clés à Retenir

> 📌 **La distillation de logits est la méthode la plus simple et efficace**

> 📌 **La température est cruciale pour révéler les relations entre classes**

> 📌 **La combinaison soft + hard loss fonctionne mieux que l'une ou l'autre seule**

> 📌 **Les variantes (focal, attention) peuvent améliorer dans certains cas**

---

*Section suivante : [10.3 Feature-based Distillation](./10_03_Features.md)*

