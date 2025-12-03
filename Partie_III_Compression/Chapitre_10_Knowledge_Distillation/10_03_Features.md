# 10.3 Feature-based Distillation

---

## Introduction

La **feature-based distillation** transfère les connaissances via les représentations intermédiaires (features) plutôt que les sorties finales. Cela permet de guider le student à apprendre des représentations similaires au teacher.

---

## Principe

### Motivation

Les features intermédiaires contiennent de l'information riche sur la représentation des données. Forcer le student à avoir des features similaires améliore souvent les performances.

---

## Implémentation de Base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    """
    Loss de distillation basée sur les features
    """
    
    def __init__(self, alpha=0.5, criterion='mse'):
        """
        Args:
            alpha: Poids de la loss feature vs classification
            criterion: 'mse' ou 'cosine'
        """
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            student_features: Liste de tenseurs de features
            teacher_features: Liste de tenseurs de features correspondants
            student_logits: Logits finaux du student
            labels: Labels ground truth
        """
        feature_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Aligne les dimensions si nécessaire
            s_feat_aligned, t_feat_aligned = self._align_features(s_feat, t_feat)
            
            # Loss feature
            if self.criterion == 'mse':
                feature_loss += F.mse_loss(s_feat_aligned, t_feat_aligned)
            elif self.criterion == 'cosine':
                # Cosine similarity loss
                s_feat_flat = s_feat_aligned.view(s_feat_aligned.size(0), -1)
                t_feat_flat = t_feat_aligned.view(t_feat_aligned.size(0), -1)
                s_feat_norm = F.normalize(s_feat_flat, p=2, dim=1)
                t_feat_norm = F.normalize(t_feat_flat, p=2, dim=1)
                feature_loss += (1 - (s_feat_norm * t_feat_norm).sum(dim=1)).mean()
        
        feature_loss = feature_loss / len(student_features)
        
        # Loss classification
        ce_loss = F.cross_entropy(student_logits, labels)
        
        total_loss = self.alpha * feature_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total': total_loss,
            'feature': feature_loss,
            'classification': ce_loss
        }
    
    def _align_features(self, s_feat, t_feat):
        """Aligne les dimensions des features"""
        if s_feat.shape == t_feat.shape:
            return s_feat, t_feat
        
        # Pour Conv2d features: pooling pour aligner spatialement
        if s_feat.dim() == 4 and t_feat.dim() == 4:
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                # Interpolation spatiale
                if s_feat.shape[-2:] < t_feat.shape[-2:]:
                    # Student plus petit: downsample teacher
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[-2:])
                else:
                    # Teacher plus petit: downsample student
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[-2:])
            
            # Aligne les canaux
            if s_feat.shape[1] != t_feat.shape[1]:
                # Utilise une couche d'adaptation (sera définie ailleurs)
                pass
        
        return s_feat, t_feat
```

---

## Couches d'Adaptation

```python
class AdaptationLayer(nn.Module):
    """
    Couche d'adaptation pour aligner les dimensions student/teacher
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        if in_channels != out_channels:
            if kernel_size == 1:
                self.adapt = nn.Conv2d(in_channels, out_channels, 1)
            else:
                self.adapt = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                     padding=kernel_size//2)
        else:
            self.adapt = nn.Identity()
        
        # Initialisation
        if hasattr(self.adapt, 'weight'):
            nn.init.xavier_uniform_(self.adapt.weight)
    
    def forward(self, x):
        return self.adapt(x)

# Exemple
adapt_layer = AdaptationLayer(256, 128)
x = torch.randn(32, 256, 32, 32)
x_adapted = adapt_layer(x)  # Shape: (32, 128, 32, 32)
```

---

## Student avec Feature Extraction

```python
class StudentWithFeatureExtraction(nn.Module):
    """
    Student qui expose les features intermédiaires pour distillation
    """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x, return_features=False):
        # Features à différentes profondeurs
        feat1 = F.relu(self.conv1(x))  # Feature 1
        feat2 = F.relu(self.conv2(feat1))  # Feature 2
        feat3 = F.relu(self.conv3(feat2))  # Feature 3
        
        # Classification
        pooled = self.pool(feat3)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.fc(pooled)
        
        if return_features:
            return logits, [feat1, feat2, feat3]
        return logits
```

---

## Training avec Feature Distillation

```python
def train_with_feature_distillation(teacher, student, train_loader, val_loader,
                                    feature_layers=['conv2', 'conv4'], epochs=50):
    """
    Entraînement avec feature-based distillation
    """
    teacher.eval()
    student.train()
    
    # Crée les couches d'adaptation
    adaptation_layers = nn.ModuleDict()
    for i, layer_name in enumerate(feature_layers):
        # Détermine les dimensions (simplifié)
        adapt = AdaptationLayer(teacher_channels[i], student_channels[i])
        adaptation_layers[layer_name] = adapt
    
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(adaptation_layers.parameters()),
        lr=1e-3
    )
    
    loss_fn = FeatureDistillationLoss(alpha=0.5, criterion='mse')
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward teacher
            with torch.no_grad():
                teacher_logits, teacher_features = teacher(data, return_features=True)
            
            # Forward student
            student_logits, student_features = student(data, return_features=True)
            
            # Adapte les features
            adapted_teacher_features = []
            for i, (s_feat, t_feat, layer_name) in enumerate(
                zip(student_features, teacher_features, feature_layers)
            ):
                if layer_name in adaptation_layers:
                    t_feat_adapted = adaptation_layers[layer_name](t_feat)
                    adapted_teacher_features.append(t_feat_adapted)
                else:
                    adapted_teacher_features.append(t_feat)
            
            # Loss
            losses = loss_fn(student_features, adapted_teacher_features, 
                           student_logits, labels)
            losses['total'].backward()
            
            optimizer.step()
            total_loss += losses['total'].item()
        
        # Validation
        student.eval()
        val_acc = evaluate(student, val_loader)
        student.train()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
                  f'Acc={val_acc:.2f}%')
```

---

## Attention Transfer

```python
class AttentionTransfer(nn.Module):
    """
    Transfert d'attention: le student apprend les cartes d'attention du teacher
    """
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    
    def attention_map(self, features):
        """
        Génère une carte d'attention depuis les features
        
        Attention = moyenne spatiale du carré des features
        """
        # Attention par canal
        attention = torch.pow(features, 2).mean(dim=1, keepdim=True)
        
        # Normalisation spatiale
        attention_flat = attention.view(attention.size(0), -1)
        attention_norm = F.normalize(attention_flat, p=2, dim=1)
        attention = attention_norm.view_as(attention)
        
        return attention
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        attention_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_att = self.attention_map(s_feat)
            t_att = self.attention_map(t_feat)
            
            # Aligne spatialement
            if s_att.shape != t_att.shape:
                t_att = F.interpolate(t_att, size=s_att.shape[-2:], 
                                    mode='bilinear', align_corners=False)
            
            attention_loss += F.mse_loss(s_att, t_att)
        
        attention_loss = attention_loss / len(student_features)
        
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * attention_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total': total_loss,
            'attention': attention_loss,
            'classification': ce_loss
        }
```

---

## Exercices

### Exercice 10.3.1
Implémentez une variante qui utilise plusieurs niveaux de features avec des poids différents.

### Exercice 10.3.2
Comparez feature-based vs logits-only distillation sur un réseau profond.

---

## Points Clés à Retenir

> 📌 **Feature-based distillation guide les représentations intermédiaires**

> 📌 **Les couches d'adaptation sont nécessaires pour aligner les dimensions**

> 📌 **Attention transfer est efficace pour les réseaux convolutionnels**

> 📌 **La combinaison de plusieurs niveaux de features améliore souvent les résultats**

---

*Section suivante : [10.4 Relation-based Distillation](./10_04_Relations.md)*

