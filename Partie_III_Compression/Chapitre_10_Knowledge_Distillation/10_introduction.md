# Chapitre 10 : Knowledge Distillation

---

## Introduction

La **Knowledge Distillation** (distillation de connaissances) transfère les connaissances d'un grand modèle "teacher" vers un petit modèle "student". Cette technique permet de créer des modèles compactes tout en préservant une grande partie des performances.

---

## Plan du Chapitre

1. [Principes de la Distillation](./10_01_Principes.md)
2. [Distillation des Logits](./10_02_Logits.md)
3. [Feature-based Distillation](./10_03_Features.md)
4. [Relation-based Distillation](./10_04_Relations.md)
5. [Self-distillation](./10_05_Self_Distillation.md)
6. [Combinaison avec Compression Tensorielle](./10_06_Tensor_Compression.md)

---

## Principe Fondamental

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Distillation                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │   Teacher    │         │   Student    │                    │
│  │  (grand)     │────────▶│   (petit)    │                    │
│  │              │  soft   │              │                    │
│  │  Complexité  │  labels │  Simplicité  │                    │
│  │  Précision ↑ │         │  Vitesse ↑   │                    │
│  └──────────────┘         └──────────────┘                    │
│                                                                 │
│  Loss = α·L_hard(teacher, student) + β·L_soft(teacher, student)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Distillation Standard (Logits)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KnowledgeDistillationLoss(nn.Module):
    """
    Loss de distillation de connaissances standard
    
    Combine la loss hard (labels réels) et soft (prédictions teacher)
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Poids de la loss soft
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Logits du student (non normalisés)
            teacher_logits: Logits du teacher (non normalisés)
            labels: Labels ground truth
        """
        # Soft labels: probabilités "adoucies" du teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Loss soft: KL divergence entre distributions adoucies
        soft_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Loss hard: Cross-entropy avec les labels réels
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combinaison
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss

# Exemple d'utilisation
temperature = 4.0
alpha = 0.7

loss_fn = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)

# Simule des logits
student_logits = torch.randn(32, 10)
teacher_logits = torch.randn(32, 10) * 2  # Teacher plus confiant
labels = torch.randint(0, 10, (32,))

total_loss, soft_loss, hard_loss = loss_fn(student_logits, teacher_logits, labels)

print("Knowledge Distillation Loss:")
print(f"  Total: {total_loss.item():.4f}")
print(f"  Soft (KL): {soft_loss.item():.4f}")
print(f"  Hard (CE): {hard_loss.item():.4f}")
```

---

## Feature-based Distillation

```python
class FeatureDistillationLoss(nn.Module):
    """
    Distillation basée sur les features intermédiaires
    
    Force le student à apprendre des représentations similaires au teacher
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            student_features: Features du student (liste de tenseurs)
            teacher_features: Features du teacher (liste de tenseurs)
            student_logits: Logits finaux du student
            labels: Labels ground truth
        """
        # Loss feature: MSE ou cosine similarity
        feature_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Aligne les dimensions si nécessaire
            if s_feat.shape != t_feat.shape:
                # Adaptation de la taille (ex: pooling)
                if s_feat.shape[-1] < t_feat.shape[-1]:
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[-2:])
                else:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
            
            # MSE loss
            feature_loss += F.mse_loss(s_feat, t_feat)
        
        feature_loss = feature_loss / len(student_features)
        
        # Loss classification standard
        ce_loss = F.cross_entropy(student_logits, labels)
        
        total_loss = self.alpha * feature_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, feature_loss, ce_loss

# Adaptation layer pour aligner les dimensions
class AdaptationLayer(nn.Module):
    """
    Couche d'adaptation pour aligner les features student/teacher
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels != out_channels:
            self.adapt = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.adapt = nn.Identity()
        
    def forward(self, x):
        return self.adapt(x)
```

---

## Attention Transfer

```python
class AttentionTransfer(nn.Module):
    """
    Transfert d'attention: le student apprend à prêter attention aux mêmes régions
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def attention_map(self, features):
        """
        Génère une carte d'attention depuis les features
        """
        # Attention = norme spatiale des features
        attention = torch.pow(features, 2).mean(dim=1, keepdim=True)
        # Normalisation spatiale
        attention = F.normalize(attention.view(attention.size(0), -1), p=2, dim=1)
        return attention.view(attention.size(0), 1, *features.shape[-2:])
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            student_features: Liste de feature maps du student
            teacher_features: Liste de feature maps du teacher
        """
        attention_loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_att = self.attention_map(s_feat)
            t_att = self.attention_map(t_feat)
            
            # Aligne les dimensions
            if s_att.shape != t_att.shape:
                t_att = F.interpolate(t_att, size=s_att.shape[-2:], mode='bilinear')
            
            attention_loss += F.mse_loss(s_att, t_att)
        
        attention_loss = attention_loss / len(student_features)
        
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * attention_loss + (1 - self.alpha) * ce_loss
        
        return total_loss

# Exemple
student_feats = [torch.randn(4, 64, 32, 32), torch.randn(4, 128, 16, 16)]
teacher_feats = [torch.randn(4, 256, 32, 32), torch.randn(4, 512, 16, 16)]

att_transfer = AttentionTransfer()
loss = att_transfer(student_feats, teacher_feats, 
                   torch.randn(4, 10), torch.randint(0, 10, (4,)))
print(f"Attention Transfer Loss: {loss.item():.4f}")
```

---

## Relation-based Distillation

```python
class RelationDistillation(nn.Module):
    """
    Distillation basée sur les relations entre exemples
    
    Préserve la structure relationnelle plutôt que les valeurs absolues
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def pairwise_relations(self, features):
        """
        Calcule les relations par paires entre exemples du batch
        """
        # Normalise les features
        features = F.normalize(features, p=2, dim=-1)
        
        # Matrice de similarité cosine
        relations = torch.matmul(features, features.t())
        
        return relations
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            features: (batch, features) - représentations à comparer
        """
        # Relations par paires
        s_relations = self.pairwise_relations(student_features)
        t_relations = self.pairwise_relations(teacher_features)
        
        # Loss: MSE des relations
        relation_loss = F.mse_loss(s_relations, t_relations)
        
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * relation_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, relation_loss, ce_loss

# Test
batch_size = 32
student_feat = torch.randn(batch_size, 128)
teacher_feat = torch.randn(batch_size, 256)

rel_distill = RelationDistillation()
loss, rel_loss, ce_loss = rel_distill(
    student_feat, teacher_feat,
    torch.randn(batch_size, 10),
    torch.randint(0, 10, (batch_size,))
)

print("Relation-based Distillation:")
print(f"  Total loss: {loss.item():.4f}")
print(f"  Relation loss: {rel_loss.item():.4f}")
print(f"  CE loss: {ce_loss.item():.4f}")
```

---

## Self-Distillation

```python
class SelfDistillation(nn.Module):
    """
    Self-distillation: le modèle apprend de ses propres prédictions précédentes
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
        """
        # Soft labels depuis les prédictions précédentes
        prev_probs = F.softmax(previous_logits / self.temperature, dim=-1)
        curr_log_probs = F.log_softmax(current_logits / self.temperature, dim=-1)
        
        # Distillation
        soft_loss = self.kl_loss(curr_log_probs, prev_probs) * (self.temperature ** 2)
        
        # Classification standard
        hard_loss = self.ce_loss(current_logits, labels)
        
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return total_loss

# Application: progressive distillation pendant l'entraînement
def progressive_self_distillation(model, dataloader, epochs=10):
    """
    Entraînement avec self-distillation progressive
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SelfDistillation()
    
    previous_logits = None
    
    for epoch in range(epochs):
        for batch_idx, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            current_logits = model(data)
            
            if previous_logits is not None:
                loss = loss_fn(current_logits, previous_logits, labels, alpha=0.3)
            else:
                loss = F.cross_entropy(current_logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Sauvegarde pour la prochaine itération
            previous_logits = current_logits.detach()
```

---

## Distillation avec Compression Tensorielle

```python
class TensorDistillation:
    """
    Distillation combinée avec compression par réseaux de tenseurs
    """
    
    @staticmethod
    def compress_teacher_with_tensor_train(teacher_model, rank=32):
        """
        Compresse le teacher en format Tensor Train avant distillation
        """
        from tensor_compression import convert_to_tensor_train
        
        compressed_teacher = convert_to_tensor_train(teacher_model, rank=rank)
        return compressed_teacher
    
    @staticmethod
    def distill_to_compressed_student(teacher_model, student_model, 
                                     train_loader, epochs=50):
        """
        Distille un teacher vers un student déjà compressé (Tensor Train)
        """
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
        loss_fn = KnowledgeDistillationLoss(temperature=4.0, alpha=0.7)
        
        teacher_model.eval()
        student_model.train()
        
        for epoch in range(epochs):
            for data, labels in train_loader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = teacher_model(data)
                
                student_logits = student_model(data)
                
                loss, _, _ = loss_fn(student_logits, teacher_logits, labels)
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Exemple d'utilisation
"""
# Teacher: modèle large
teacher = LargeModel()

# Student: modèle compressé (Tensor Train)
student = CompressedModel(rank=64)

# Distillation
TensorDistillation.distill_to_compressed_student(
    teacher, student, train_loader
)
"""
```

---

## Comparaison des Méthodes

```python
def compare_distillation_methods():
    """
    Compare différentes méthodes de distillation
    """
    methods = {
        'Logits only': {
            'components': ['soft labels', 'hard labels'],
            'complexity': 'Low',
            'performance': 'Good',
            'speed': 'Fast'
        },
        'Feature-based': {
            'components': ['features', 'logits'],
            'complexity': 'Medium',
            'performance': 'Better',
            'speed': 'Medium'
        },
        'Attention Transfer': {
            'components': ['attention maps', 'logits'],
            'complexity': 'Medium',
            'performance': 'Better',
            'speed': 'Medium'
        },
        'Relation-based': {
            'components': ['pairwise relations', 'logits'],
            'complexity': 'High',
            'performance': 'Best',
            'speed': 'Slow'
        }
    }
    
    print("Comparaison des méthodes de distillation:")
    print(f"{'Méthode':<20} | {'Complexité':<12} | {'Performance':<12} | {'Vitesse':<10}")
    print("-" * 60)
    for name, info in methods.items():
        print(f"{name:<20} | {info['complexity']:<12} | {info['performance']:<12} | {info['speed']:<10}")

compare_distillation_methods()
```

---

## Exercices

### Exercice 10.1
Implémentez une distillation multi-teacher où plusieurs teachers sont combinés pour guider le student.

### Exercice 10.2
Comparez les performances d'un student entraîné avec et sans distillation pour différentes tailles de student.

### Exercice 10.3
Créez un pipeline combinant distillation, pruning et quantification pour maximiser la compression.

---

## Points Clés à Retenir

> 📌 **La distillation utilise les prédictions "soft" du teacher comme supervision supplémentaire**

> 📌 **La température élevée révèle les relations entre classes cachées dans les hard labels**

> 📌 **La distillation de features préserve les représentations intermédiaires**

> 📌 **La combinaison distillation + compression peut multiplier les gains**

---

*Chapitre suivant : [Chapitre 11 - Approximations de Rang Faible](../Chapitre_11_Low_Rank/11_introduction.md)*

