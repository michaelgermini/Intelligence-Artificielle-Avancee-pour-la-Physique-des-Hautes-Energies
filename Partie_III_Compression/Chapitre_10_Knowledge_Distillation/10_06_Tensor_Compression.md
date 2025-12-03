# 10.6 Combinaison avec Compression Tensorielle

---

## Introduction

La **combinaison de la distillation avec la compression tensorielle** permet d'obtenir des modèles extrêmement compacts tout en préservant de bonnes performances. Cette section explore les différentes stratégies d'intégration.

---

## Stratégies de Combinaison

### Approche 1 : Distillation puis Compression

```python
import torch
import torch.nn as nn

class DistillThenCompress:
    """
    Stratégie: Distiller d'abord, puis compresser
    """
    
    def __init__(self, teacher, student_architecture):
        self.teacher = teacher
        self.student_architecture = student_architecture
    
    def execute(self, train_loader, compress_method='tensor_train', rank=32):
        """
        1. Distille le teacher vers un student standard
        2. Compresse le student avec Tensor Train
        """
        # Étape 1: Distillation
        student_standard = self.student_architecture()
        
        # Distillation
        from knowledge_distillation import train_with_logits_distillation
        student_distilled = train_with_logits_distillation(
            self.teacher, student_standard, train_loader, epochs=50
        )
        
        # Étape 2: Compression tensorielle
        if compress_method == 'tensor_train':
            from tensor_compression import convert_to_tensor_train
            student_compressed = convert_to_tensor_train(
                student_distilled, rank=rank
            )
        
        return student_compressed
```

### Approche 2 : Compression puis Distillation

```python
class CompressThenDistill:
    """
    Stratégie: Compresser d'abord, puis distiller
    """
    
    def __init__(self, teacher):
        self.teacher = teacher
    
    def execute(self, train_loader, compress_method='tensor_train', rank=32):
        """
        1. Compresse le teacher
        2. Distille vers un student compressé
        """
        # Étape 1: Compression du teacher
        if compress_method == 'tensor_train':
            from tensor_compression import convert_to_tensor_train
            teacher_compressed = convert_to_tensor_train(
                self.teacher, rank=rank
            )
        else:
            teacher_compressed = self.teacher
        
        # Étape 2: Student compressé (même structure)
        student_compressed = type(teacher_compressed)()  # Même architecture
        
        # Distillation
        from knowledge_distillation import train_with_logits_distillation
        student_final = train_with_logits_distillation(
            teacher_compressed, student_compressed, train_loader, epochs=50
        )
        
        return student_final
```

### Approche 3 : Distillation avec Student Compressé

```python
class DistillToCompressedStudent:
    """
    Distillation directe vers un student déjà en format compressé
    """
    
    def __init__(self, teacher, student_tt_config):
        """
        Args:
            teacher: Modèle teacher (standard)
            student_tt_config: Configuration pour student Tensor Train
                {'input_dims': ..., 'output_dims': ..., 'tt_rank': ...}
        """
        self.teacher = teacher
        self.student_config = student_tt_config
    
    def create_compressed_student(self):
        """Crée un student en format Tensor Train"""
        from tensor_compression import TensorizedLinear
        
        student_layers = []
        for layer_config in self.student_config['layers']:
            layer = TensorizedLinear(
                input_dims=layer_config['input_dims'],
                output_dims=layer_config['output_dims'],
                tt_rank=layer_config['tt_rank']
            )
            student_layers.append(layer)
            student_layers.append(nn.ReLU())
        
        return nn.Sequential(*student_layers)
    
    def execute(self, train_loader, epochs=50):
        """Distille vers student compressé"""
        student_compressed = self.create_compressed_student()
        
        from knowledge_distillation import train_with_logits_distillation
        student_trained = train_with_logits_distillation(
            self.teacher, student_compressed, train_loader, epochs=epochs
        )
        
        return student_trained
```

---

## Distillation Multi-stage

```python
class MultiStageDistillation:
    """
    Distillation en plusieurs étapes avec compression progressive
    """
    
    def __init__(self, teacher, stages):
        """
        Args:
            stages: Liste de configurations [(rank1, alpha1), (rank2, alpha2), ...]
        """
        self.teacher = teacher
        self.stages = stages
    
    def execute(self, train_loader):
        """
        Distillation progressive avec compression croissante
        """
        current_teacher = self.teacher
        
        for stage_idx, (rank, alpha) in enumerate(self.stages):
            print(f"Stage {stage_idx+1}: Rank={rank}, Alpha={alpha}")
            
            # Crée student pour ce stage
            from tensor_compression import create_tt_student
            student = create_tt_student(rank=rank)
            
            # Distille
            from knowledge_distillation import train_with_logits_distillation
            student = train_with_logits_distillation(
                current_teacher, student, train_loader, epochs=20
            )
            
            # Le student devient le teacher pour la prochaine étape
            current_teacher = student
        
        return current_teacher
```

---

## Distillation avec QAT Combinée

```python
class DistillWithQuantization:
    """
    Combine distillation et quantization-aware training
    """
    
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
    
    def execute(self, train_loader, epochs=50, quantize_student=True, n_bits=8):
        """
        Distille avec student quantifié
        """
        # Convertit student en QAT si demandé
        if quantize_student:
            from quantization import convert_to_qat
            student_qat = convert_to_qat(self.student, n_bits=n_bits)
        else:
            student_qat = self.student
        
        # Distillation avec fake quantization
        teacher.eval()
        student_qat.train()
        
        optimizer = torch.optim.Adam(student_qat.parameters(), lr=1e-3)
        loss_fn = LogitsDistillationLoss(temperature=4.0, alpha=0.7)
        
        for epoch in range(epochs):
            for data, labels in train_loader:
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_logits = self.teacher(data)
                
                student_logits = student_qat(data)
                
                losses = loss_fn(student_logits, teacher_logits, labels)
                losses['total'].backward()
                
                optimizer.step()
        
        return student_qat
```

---

## Analyse Comparative

```python
def compare_compression_strategies(teacher, train_loader, val_loader):
    """
    Compare différentes stratégies de compression + distillation
    """
    results = {}
    
    # Baseline: Student standard sans distillation
    student_baseline = SimpleStudent()
    train_standard(student_baseline, train_loader)
    results['baseline'] = evaluate(student_baseline, val_loader)
    
    # Distillation seule
    student_distilled = distill(teacher, SimpleStudent(), train_loader)
    results['distilled'] = evaluate(student_distilled, val_loader)
    
    # Compression seule (Tensor Train)
    student_compressed = compress_with_tt(teacher, rank=32)
    results['compressed'] = evaluate(student_compressed, val_loader)
    
    # Distillation + Compression
    student_combined = distill_then_compress(teacher, train_loader, rank=32)
    results['combined'] = evaluate(student_combined, val_loader)
    
    print("Comparaison des stratégies:")
    for strategy, acc in results.items():
        print(f"  {strategy}: {acc:.2f}%")
    
    return results
```

---

## Exercices

### Exercice 10.6.1
Implémentez une stratégie qui distille progressivement en réduisant le rang TT à chaque étape.

### Exercice 10.6.2
Comparez les trois approches (distill→compress, compress→distill, distill-to-compressed) sur un modèle réel.

---

## Points Clés à Retenir

> 📌 **Distillation + compression tensorielle peut multiplier les gains**

> 📌 **L'ordre (distill→compress vs compress→distill) affecte les résultats**

> 📌 **Distillation vers student compressé est souvent la meilleure stratégie**

> 📌 **Multi-stage distillation permet compression progressive**

> 📌 **La combinaison avec QAT peut encore améliorer la compression**

---

*Chapitre suivant : [Chapitre 11 - Approximations de Rang Faible](../Chapitre_11_Low_Rank/11_introduction.md)*

