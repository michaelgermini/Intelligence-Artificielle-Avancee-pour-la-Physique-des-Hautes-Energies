# 25.3 Fine-tuning Post-Compression

---

## Introduction

Le **fine-tuning post-compression** est crucial pour récupérer la performance perdue lors de la compression. Cette section présente les stratégies de fine-tuning efficaces, incluant learning rates adaptatifs, progressive unfreezing, et techniques de stabilisation.

---

## Stratégies de Fine-tuning

### Approches de Base

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FinetuningStrategy:
    """
    Stratégies de fine-tuning pour modèles compressés
    """
    
    def __init__(self, compressed_model, train_loader, val_loader):
        self.model = compressed_model
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def standard_finetuning(self, epochs=20, lr=0.001, weight_decay=1e-4):
        """
        Fine-tuning standard
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping pour stabilité
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            val_acc = self.evaluate()
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_finetuned.pth')
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(self.train_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
        
        return best_val_acc
    
    def evaluate(self):
        """Évalue modèle"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return 100 * correct / total
```

---

## Progressive Unfreezing

### Déverrouillage Progressif

```python
class ProgressiveUnfreezing(FinetuningStrategy):
    """
    Fine-tuning avec déverrouillage progressif des couches
    """
    
    def progressive_unfreeze(self, epochs_per_stage=5, lr=0.001):
        """
        Déverrouille couches progressivement (dernières d'abord)
        """
        # Grouper couches par profondeur
        layers = list(self.model.children())
        n_layers = len(layers)
        
        best_val_acc = 0.0
        
        # Stages: de dernières à premières couches
        for stage in range(n_layers):
            # Geler toutes couches
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Déverrouiller dernières (n_layers - stage) couches
            layers_to_train = layers[stage:]
            for layer in layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = True
            
            print(f"\nStage {stage+1}: Training last {n_layers - stage} layers")
            
            # Fine-tuning pour ce stage
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr
            )
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs_per_stage):
                self.model.train()
                for data, targets in self.train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
                val_acc = self.evaluate()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            # Réduire LR pour prochain stage
            lr *= 0.5
        
        return best_val_acc
```

---

## Learning Rate Scheduling

### Schedulers Avancés

```python
class AdvancedFinetuning(FinetuningStrategy):
    """
    Fine-tuning avec schedulers avancés
    """
    
    def cosine_annealing_finetuning(self, epochs=30, lr_max=0.001, lr_min=1e-6):
        """
        Fine-tuning avec cosine annealing
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr_max)
        criterion = nn.CrossEntropyLoss()
        
        # Cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr_min
        )
        
        for epoch in range(epochs):
            self.model.train()
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            val_acc = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Val Acc: {val_acc:.2f}%")
        
        return val_acc
    
    def warmup_finetuning(self, epochs=30, warmup_epochs=5, lr=0.001):
        """
        Fine-tuning avec warmup
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 ** ((epoch - warmup_epochs) // 10)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        
        for epoch in range(epochs):
            self.model.train()
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            val_acc = self.evaluate()
        
        return val_acc
```

---

## Knowledge Distillation lors du Fine-tuning

### Distillation Continue

```python
class DistillationFinetuning(FinetuningStrategy):
    """
    Fine-tuning avec distillation du modèle original
    """
    
    def __init__(self, compressed_model, teacher_model, train_loader, val_loader):
        super().__init__(compressed_model, train_loader, val_loader)
        self.teacher = teacher_model
        self.teacher.eval()  # Teacher en mode eval
    
    def distill_finetuning(self, epochs=30, alpha=0.5, temperature=4.0, lr=0.001):
        """
        Fine-tuning avec distillation
        
        Args:
            alpha: Poids loss distillation vs classification
            temperature: Température pour softmax
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for data, targets in self.train_loader:
                optimizer.zero_grad()
                
                # Prédictions étudiant (modèle compressé)
                student_logits = self.model(data)
                
                # Prédictions enseignant (modèle original)
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                
                # Loss classification
                loss_cls = nn.CrossEntropyLoss()(student_logits, targets)
                
                # Loss distillation
                loss_distill = nn.KLDivLoss(reduction='batchmean')(
                    nn.functional.log_softmax(student_logits / temperature, dim=1),
                    nn.functional.softmax(teacher_logits / temperature, dim=1)
                ) * (temperature ** 2)
                
                # Loss combinée
                loss = alpha * loss_cls + (1 - alpha) * loss_distill
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            val_acc = self.evaluate()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.train_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
        
        return best_val_acc
```

---

## Exercices

### Exercice 25.3.1
Implémentez fine-tuning standard et comparez avec et sans gradient clipping.

### Exercice 25.3.2
Testez progressive unfreezing vs fine-tuning complet sur modèle compressé.

### Exercice 25.3.3
Comparez différents schedulers (cosine, step, plateau) pour fine-tuning.

### Exercice 25.3.4
Implémentez fine-tuning avec distillation et analysez impact sur performance finale.

---

## Points Clés à Retenir

> 📌 **Fine-tuning est essentiel pour récupérer performance post-compression**

> 📌 **Learning rate scheduling améliore convergence**

> 📌 **Progressive unfreezing peut stabiliser fine-tuning**

> 📌 **Distillation aide modèles compressés à apprendre du teacher**

> 📌 **Gradient clipping stabilise entraînement modèles compressés**

> 📌 **Warmup peut améliorer convergence initiale**

---

*Section précédente : [25.2 Hyperparamètres](./25_02_Hyperparametres.md) | Section suivante : [25.4 Validation](./25_04_Validation.md)*

