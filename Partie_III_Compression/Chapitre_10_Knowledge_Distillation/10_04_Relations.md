# 10.4 Relation-based Distillation

---

## Introduction

La **relation-based distillation** préserve les relations structurelles entre exemples plutôt que les valeurs absolues. Cela permet de capturer des patterns plus complexes dans les données.

---

## Principe

### Idée Fondamentale

Au lieu de forcer le student à avoir des features similaires au teacher, on préserve les **relations** entre les exemples du batch. Cela est plus robuste et capture mieux la structure des données.

---

## Implémentation de Base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationDistillationLoss(nn.Module):
    """
    Loss basée sur les relations par paires
    """
    
    def __init__(self, alpha=0.5, metric='cosine'):
        """
        Args:
            alpha: Poids de la loss relation vs classification
            metric: 'cosine', 'euclidean', ou 'dot_product'
        """
        super().__init__()
        self.alpha = alpha
        self.metric = metric
    
    def compute_pairwise_relations(self, features):
        """
        Calcule les relations par paires
        
        Returns:
            matrice de relations (batch, batch)
        """
        if self.metric == 'cosine':
            # Similarité cosine
            features_norm = F.normalize(features, p=2, dim=-1)
            relations = torch.matmul(features_norm, features_norm.t())
        elif self.metric == 'euclidean':
            # Distance euclidienne (convertie en similarité)
            pairwise_dist = torch.cdist(features, features, p=2)
            relations = 1.0 / (1.0 + pairwise_dist)
        elif self.metric == 'dot_product':
            # Produit scalaire
            relations = torch.matmul(features, features.t())
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return relations
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        """
        Args:
            student_features: (batch, feature_dim)
            teacher_features: (batch, feature_dim)
        """
        # Relations par paires
        s_relations = self.compute_pairwise_relations(student_features)
        t_relations = self.compute_pairwise_relations(teacher_features)
        
        # Loss: MSE des relations
        relation_loss = F.mse_loss(s_relations, t_relations)
        
        # Loss classification
        ce_loss = F.cross_entropy(student_logits, labels)
        
        total_loss = self.alpha * relation_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total': total_loss,
            'relation': relation_loss,
            'classification': ce_loss,
            'student_relations': s_relations,
            'teacher_relations': t_relations
        }

# Exemple
batch_size = 32
student_feat = torch.randn(batch_size, 128)
teacher_feat = torch.randn(batch_size, 256)

rel_loss_fn = RelationDistillationLoss(alpha=0.5, metric='cosine')
losses = rel_loss_fn(
    student_feat,
    teacher_feat,
    torch.randn(batch_size, 10),
    torch.randint(0, 10, (batch_size,))
)

print(f"Relation-based Loss:")
print(f"  Total: {losses['total'].item():.4f}")
print(f"  Relation: {losses['relation'].item():.4f}")
```

---

## Variantes Avancées

### Triplet Relations

```python
class TripletRelationLoss(nn.Module):
    """
    Préserve les relations triplet (anchor, positive, negative)
    """
    
    def __init__(self, alpha=0.5, margin=1.0):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
    
    def compute_triplet_relations(self, features, labels):
        """
        Calcule les relations triplet
        """
        # Trouve les triplets
        # (Simplifié - en pratique, nécessite stratégie de sélection)
        
        # Pour chaque anchor, trouve positive (même classe) et negative (classe différente)
        relations = []
        
        for i in range(features.size(0)):
            anchor = features[i]
            anchor_label = labels[i]
            
            # Positives (même classe)
            positives = features[labels == anchor_label]
            if positives.size(0) > 1:  # Au moins l'anchor lui-même
                positive = positives[torch.randint(0, positives.size(0), (1,))]
            else:
                continue
            
            # Negatives (classe différente)
            negatives = features[labels != anchor_label]
            if negatives.size(0) > 0:
                negative = negatives[torch.randint(0, negatives.size(0), (1,))]
            else:
                continue
            
            # Distance anchor-positive et anchor-negative
            dist_pos = F.pairwise_distance(anchor.unsqueeze(0), positive)
            dist_neg = F.pairwise_distance(anchor.unsqueeze(0), negative)
            
            relations.append({
                'anchor': i,
                'positive': positive,
                'negative': negative,
                'dist_pos': dist_pos,
                'dist_neg': dist_neg
            })
        
        return relations
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        # Triplet relations pour teacher et student
        s_triplets = self.compute_triplet_relations(student_features, labels)
        t_triplets = self.compute_triplet_relations(teacher_features, labels)
        
        # Loss: préserve les ratios de distances
        relation_loss = 0.0
        for s_trip, t_trip in zip(s_triplets, t_triplets):
            s_ratio = s_trip['dist_pos'] / (s_trip['dist_neg'] + 1e-8)
            t_ratio = t_trip['dist_pos'] / (t_trip['dist_neg'] + 1e-8)
            relation_loss += F.mse_loss(s_ratio, t_ratio)
        
        relation_loss = relation_loss / len(s_triplets) if s_triplets else 0.0
        
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * relation_loss + (1 - self.alpha) * ce_loss
        
        return {'total': total_loss, 'relation': relation_loss}
```

---

## Relation Knowledge Distillation (RKD)

```python
class RKDLoss(nn.Module):
    """
    Relation Knowledge Distillation
    Préserve les distances et angles entre paires d'exemples
    """
    
    def __init__(self, alpha=0.5, distance_weight=25.0, angle_weight=50.0):
        super().__init__()
        self.alpha = alpha
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
    
    def compute_pairwise_distance(self, features):
        """Distance euclidienne par paires"""
        batch_size = features.size(0)
        # Distance matrix
        dist_matrix = torch.cdist(features, features, p=2)
        return dist_matrix
    
    def compute_angle(self, features):
        """
        Calcule les angles entre triplets de points
        """
        batch_size = features.size(0)
        
        # Sélectionne 3 points aléatoires (simplifié)
        # En pratique, utilise tous les triplets possibles
        
        angles = []
        for i in range(min(10, batch_size)):  # Limite pour efficacité
            for j in range(i+1, min(10, batch_size)):
                for k in range(j+1, min(10, batch_size)):
                    vec1 = features[i] - features[j]
                    vec2 = features[k] - features[j]
                    
                    # Angle entre vec1 et vec2
                    cos_angle = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                    angle = torch.acos(torch.clamp(cos_angle, -1+1e-7, 1-1e-7))
                    
                    angles.append(angle)
        
        return torch.stack(angles) if angles else torch.tensor([])
    
    def forward(self, student_features, teacher_features, student_logits, labels):
        # Distances
        s_dist = self.compute_pairwise_distance(student_features)
        t_dist = self.compute_pairwise_distance(teacher_features)
        
        # Normalise les distances
        s_dist_mean = s_dist.mean()
        t_dist_mean = t_dist.mean()
        
        s_dist_norm = s_dist / (s_dist_mean + 1e-8)
        t_dist_norm = t_dist / (t_dist_mean + 1e-8)
        
        distance_loss = F.mse_loss(s_dist_norm, t_dist_norm) * self.distance_weight
        
        # Angles
        s_angles = self.compute_angle(student_features)
        t_angles = self.compute_angle(teacher_features)
        
        if s_angles.numel() > 0 and t_angles.numel() > 0:
            angle_loss = F.mse_loss(s_angles, t_angles) * self.angle_weight
        else:
            angle_loss = torch.tensor(0.0, device=student_features.device)
        
        relation_loss = distance_loss + angle_loss
        
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = self.alpha * relation_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total': total_loss,
            'relation': relation_loss,
            'distance': distance_loss,
            'angle': angle_loss,
            'classification': ce_loss
        }
```

---

## Exercices

### Exercice 10.4.1
Implémentez une variante qui préserve les relations entre groupes de classes plutôt que des paires individuelles.

### Exercice 10.4.2
Comparez relation-based vs feature-based distillation sur différents datasets.

---

## Points Clés à Retenir

> 📌 **Relation-based distillation préserve la structure plutôt que les valeurs absolues**

> 📌 **Plus robuste aux différences de dimension entre teacher et student**

> 📌 **RKD (distance + angle) capture mieux la géométrie des features**

> 📌 **Coûteux computationnellement (O(batch²) ou O(batch³))**

---

*Section suivante : [10.5 Self-distillation](./10_05_Self_Distillation.md)*

