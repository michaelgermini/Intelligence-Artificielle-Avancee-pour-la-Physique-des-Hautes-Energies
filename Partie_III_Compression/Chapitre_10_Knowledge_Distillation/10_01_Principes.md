# 10.1 Principes de la Distillation

---

## Introduction

La **Knowledge Distillation** (distillation de connaissances) est une technique de compression qui transfère les connaissances d'un modèle "teacher" (grand et performant) vers un modèle "student" (plus petit et rapide). Cette section présente les principes fondamentaux.

---

## Motivation

### Pourquoi Distiller ?

- **Compression** : Réduire la taille et la complexité du modèle
- **Vitesse** : Accélérer l'inférence
- **Déploiement** : Faciliter le déploiement sur hardware contraint
- **Qualité** : Maintenir de bonnes performances malgré la compression

---

## Principe Fondamental

### Idée Clé

Le teacher apprend non seulement les **hard labels** (classes correctes) mais aussi des **soft labels** (distributions de probabilités) qui contiennent plus d'information :

```
Hard label: [0, 0, 1, 0, 0]  (classe 3)
Soft label: [0.05, 0.1, 0.7, 0.1, 0.05]  (classe 3 avec incertitude)
```

Les soft labels révèlent les relations entre classes.

---

## Architecture Standard

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillationSetup:
    """
    Configuration standard de distillation
    """
    
    def __init__(self, teacher_model, student_model):
        """
        Args:
            teacher_model: Modèle teacher (grand, pré-entraîné)
            student_model: Modèle student (petit, à entraîner)
        """
        self.teacher = teacher_model
        self.student = student_model
        
        # Teacher en mode eval (pas de gradients)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def forward_pass(self, x):
        """
        Forward pass sur teacher et student
        """
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        
        return teacher_logits, student_logits

# Exemple
class SimpleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

class SimpleStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

teacher = SimpleTeacher()
student = SimpleStudent()
distill_setup = KnowledgeDistillationSetup(teacher, student)
```

---

## Fonction de Perte de Base

### Combinaison Hard + Soft

```python
class BasicDistillationLoss(nn.Module):
    """
    Loss de distillation de base
    
    L = α·L_soft + (1-α)·L_hard
    """
    
    def __init__(self, temperature=4.0, alpha=0.7):
        """
        Args:
            temperature: Température pour adoucir les distributions
            alpha: Poids de la loss soft (vs hard)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Args:
            student_logits: Logits du student
            teacher_logits: Logits du teacher
            labels: Labels ground truth
        """
        # Soft labels: distributions adoucies
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # Loss soft: KL divergence entre distributions adoucies
        soft_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Loss hard: Cross-entropy avec labels réels
        hard_loss = self.ce_loss(student_logits, labels)
        
        # Combinaison
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return {
            'total': total_loss,
            'soft': soft_loss,
            'hard': hard_loss
        }

# Test
loss_fn = BasicDistillationLoss(temperature=4.0, alpha=0.7)

student_logits = torch.randn(32, 10)
teacher_logits = torch.randn(32, 10) * 3  # Teacher plus confiant
labels = torch.randint(0, 10, (32,))

losses = loss_fn(student_logits, teacher_logits, labels)
print(f"Distillation Loss:")
print(f"  Total: {losses['total'].item():.4f}")
print(f"  Soft (KL): {losses['soft'].item():.4f}")
print(f"  Hard (CE): {losses['hard'].item():.4f}")
```

---

## Rôle de la Température

### Pourquoi la Température ?

La température "adoucit" la distribution :

```python
def demonstrate_temperature_effect():
    """
    Montre l'effet de la température sur les soft labels
    """
    logits = torch.tensor([[1.0, 2.0, 7.0, 2.0, 1.0]])  # Classe 2 très probable
    
    temperatures = [1.0, 2.0, 4.0, 8.0]
    
    print("Effet de la Température:")
    print(f"Logits originaux: {logits[0].tolist()}")
    print()
    
    for T in temperatures:
        probs = F.softmax(logits / T, dim=1)
        print(f"T={T}: {probs[0].tolist()}")
        print(f"  Entropie: {-(probs * torch.log(probs + 1e-10)).sum().item():.4f}")

demonstrate_temperature_effect()
```

**Température élevée** :
- Distributions plus uniformes
- Plus d'information sur les relations entre classes
- Meilleure pour la distillation

---

## Processus d'Entraînement

```python
class DistillationTrainer:
    """
    Entraîneur pour distillation
    """
    
    def __init__(self, teacher, student, train_loader, val_loader,
                 temperature=4.0, alpha=0.7, lr=1e-3):
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.loss_fn = BasicDistillationLoss(temperature, alpha)
        self.optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        
        # Teacher en mode eval
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def train_epoch(self):
        """Un epoch d'entraînement"""
        self.student.train()
        total_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Forward
            with torch.no_grad():
                teacher_logits = self.teacher(data)
            
            student_logits = self.student(data)
            
            # Loss
            losses = self.loss_fn(student_logits, teacher_logits, labels)
            losses['total'].backward()
            
            self.optimizer.step()
            total_loss += losses['total'].item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                outputs = self.student(data)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def train(self, epochs=50):
        """Entraînement complet"""
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_acc = self.validate()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Loss={train_loss:.4f}, Acc={val_acc:.2f}%')
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.student.state_dict(), 'best_student.pt')
        
        return self.student
```

---

## Comparaison Student vs Baseline

```python
def compare_student_vs_baseline(student_distilled, student_baseline, test_loader):
    """
    Compare un student entraîné avec et sans distillation
    """
    def evaluate(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = model(data)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        return 100.0 * correct / total
    
    acc_distilled = evaluate(student_distilled)
    acc_baseline = evaluate(student_baseline)
    
    print("Comparaison Student:")
    print(f"  Avec distillation: {acc_distilled:.2f}%")
    print(f"  Sans distillation: {acc_baseline:.2f}%")
    print(f"  Amélioration: {acc_distilled - acc_baseline:.2f}%")
    
    return {
        'distilled': acc_distilled,
        'baseline': acc_baseline,
        'improvement': acc_distilled - acc_baseline
    }
```

---

## Facteurs Clés de Succès

### 1. Température Optimale

```python
def find_optimal_temperature(teacher, student, val_loader, temp_range=[1, 2, 4, 8, 16]):
    """
    Trouve la température optimale pour la distillation
    """
    best_temp = 1.0
    best_acc = 0.0
    
    for temp in temp_range:
        loss_fn = BasicDistillationLoss(temperature=temp, alpha=0.7)
        
        # Test rapide
        student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader[:10]:  # Sous-ensemble
                teacher_logits = teacher(data)
                student_logits = student(data)
                
                # Pas de training, juste évaluation
                # (Simplifié - en pratique, besoin d'entraîner)
        
        # En pratique, entraîne avec cette température et évalue
        # acc = train_and_evaluate(teacher, student, temp, val_loader)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_temp = temp
    
    return best_temp
```

### 2. Ratio Alpha Optimal

```python
def find_optimal_alpha(teacher, student, val_loader, alpha_range=[0.3, 0.5, 0.7, 0.9]):
    """
    Trouve le ratio alpha optimal (soft vs hard loss)
    """
    # Même principe que pour température
    # Teste différents alpha et garde le meilleur
    best_alpha = 0.7
    return best_alpha
```

---

## Avantages et Limitations

### Avantages

- **Compression efficace** : Student beaucoup plus petit
- **Performance** : Meilleure que training classique pour student petit
- **Flexibilité** : Différentes architectures teacher/student

### Limitations

- **Nécessite un teacher** : Doit être entraîné au préalable
- **Coût computationnel** : Forward pass sur teacher pendant training
- **Complexité** : Hyperparamètres (température, alpha) à ajuster

---

## Exercices

### Exercice 10.1.1
Implémentez une fonction qui trouve automatiquement la température optimale via validation.

### Exercice 10.1.2
Comparez les performances d'un student avec différents ratios alpha (0.3, 0.5, 0.7, 0.9).

### Exercice 10.1.3
Analysez comment la taille du student affecte le gain de la distillation.

---

## Points Clés à Retenir

> 📌 **La distillation utilise les soft labels du teacher comme supervision supplémentaire**

> 📌 **La température élevée révèle les relations entre classes**

> 📌 **La combinaison soft + hard loss (α) est cruciale**

> 📌 **La distillation améliore surtout les petits students**

> 📌 **Les hyperparamètres (T, α) doivent être ajustés selon le problème**

---

*Section suivante : [10.2 Distillation des Logits](./10_02_Logits.md)*

