# 22.3.1 Tenseurs et Autograd

---

## Introduction

Les **tenseurs** sont la structure de données fondamentale de PyTorch, similaires aux tableaux NumPy mais avec support GPU et automatic differentiation. Cette section couvre la manipulation de tenseurs et le système d'autograd pour calculer des gradients automatiquement.

---

## Création de Tenseurs

### Initialisation

```python
import torch
import numpy as np

# Création depuis liste
t1 = torch.tensor([1, 2, 3, 4])

# Création depuis NumPy
arr = np.array([1, 2, 3])
t2 = torch.from_numpy(arr)

# Création de tenseurs spéciaux
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
rand = torch.randn(5, 3)  # Distribution normale
uniform = torch.rand(2, 2)  # Distribution uniforme [0, 1]

# Avec dtype spécifique
t_float = torch.tensor([1, 2, 3], dtype=torch.float32)
t_int = torch.tensor([1, 2, 3], dtype=torch.int64)

# Sur GPU (si disponible)
if torch.cuda.is_available():
    t_gpu = torch.randn(3, 3).cuda()
    # ou
    t_gpu = torch.randn(3, 3).to('cuda')
```

---

## Opérations sur Tenseurs

### Opérations Mathématiques

```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Opérations élément par élément
sum_t = a + b
diff_t = a - b
prod_t = a * b
div_t = a / b

# Multiplication matricielle
matmul = torch.matmul(a, b)  # ou a @ b

# Fonctions mathématiques
sin_t = torch.sin(a)
exp_t = torch.exp(a)
log_t = torch.log(a + 1e-8)  # Éviter log(0)

# Agrégations
total_sum = torch.sum(a)
mean_val = torch.mean(a)
std_val = torch.std(a)
max_val, max_idx = torch.max(a, dim=1)  # Max le long de l'axe 1
```

---

## Propriétés et Manipulation

### Shape, Device, dtype

```python
t = torch.randn(2, 3, 4)

print(f"Shape: {t.shape}")           # torch.Size([2, 3, 4])
print(f"Size: {t.size()}")           # torch.Size([2, 3, 4])
print(f"Number of elements: {t.numel()}")  # 24
print(f"Device: {t.device}")         # cpu ou cuda:0
print(f"dtype: {t.dtype}")           # torch.float32

# Reshape
reshaped = t.reshape(6, 4)           # (6, 4)
reshaped = t.view(6, 4)              # Vue (même mémoire)

# Transpose
t_T = t.transpose(0, 2)              # Échange dimensions 0 et 2
t_T = t.permute(2, 1, 0)             # Réordonne dimensions

# Squeeze/Unsqueeze
t_1d = torch.randn(5)
t_2d = t_1d.unsqueeze(0)             # (1, 5)
t_back = t_2d.squeeze(0)             # (5,)

# Concaténation
a = torch.randn(2, 3)
b = torch.randn(2, 3)
cat_vertical = torch.cat([a, b], dim=0)    # (4, 3)
cat_horizontal = torch.cat([a, b], dim=1)  # (2, 6)
```

---

## Automatic Differentiation (Autograd)

### Concept

```python
# Pour calculer gradients, besoin requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Fonction
z = x ** 2 + y ** 2

# Calculer gradients
z.backward()

# Accéder gradients
print(f"dz/dx = {x.grad}")  # 4.0 (2*x avec x=2)
print(f"dz/dy = {y.grad}")  # 6.0 (2*y avec y=3)
```

---

## Gradients pour Tenseurs

### Exemple Complet

```python
# Tenseur avec gradients
x = torch.randn(3, 4, requires_grad=True)

# Opérations
y = x * 2
z = y.sum()

# Backward
z.backward()

# Gradients
print(f"x.grad shape: {x.grad.shape}")  # (3, 4)
print(f"Gradients:\n{x.grad}")

# Note: gradients s'accumulent par défaut
# Pour réinitialiser: x.grad.zero_()
```

---

## Gradient Accumulation et Zero

### Gestion des Gradients

```python
# Exemple: entraînement avec gradient accumulation
model_params = torch.randn(3, 4, requires_grad=True)

# Première itération
loss1 = (model_params ** 2).sum()
loss1.backward()
print(f"After first backward: {model_params.grad}")

# Deuxième itération (gradients s'accumulent)
loss2 = (model_params * 2).sum()
loss2.backward()
print(f"After second backward (accumulated): {model_params.grad}")

# Réinitialiser gradients
model_params.grad.zero_()
print(f"After zero: {model_params.grad}")

# Nouveau calcul
loss3 = (model_params ** 2).sum()
loss3.backward()
print(f"After third backward: {model_params.grad}")
```

---

## Détacher et No Gradient

### Contrôle du Gradient Computation

```python
x = torch.randn(3, 4, requires_grad=True)
y = x * 2

# Détacher: crée tenseur sans gradient tracking
y_detached = y.detach()  # Nouveau tenseur, pas de gradient

# No gradient context: désactive gradient computation temporairement
with torch.no_grad():
    z = x * 3  # Pas de gradient tracking pour z
    w = y * 2  # Pas de gradient tracking pour w

# Gradient check: désactive pour certains tenseurs
x1 = torch.randn(3, 4, requires_grad=True)
x2 = torch.randn(3, 4, requires_grad=False)

z = x1 + x2
loss = z.sum()
loss.backward()  # Seulement x1.grad sera calculé
```

---

## Exemple: Gradient Descent Manuel

### Implémentation Simple

```python
# Minimiser f(x) = x^2 avec gradient descent
x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1

for epoch in range(10):
    # Forward
    loss = x ** 2
    
    # Backward
    loss.backward()
    
    # Mise à jour (en no_grad car pas besoin de gradient pour update)
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()
    
    print(f"Epoch {epoch}: x = {x.item():.4f}, loss = {loss.item():.4f}")
```

---

## Conversion NumPy

### Interopérabilité

```python
# PyTorch → NumPy
t = torch.randn(3, 4)
arr = t.numpy()  # Si sur CPU
# ou
arr = t.cpu().numpy()  # Si sur GPU

# NumPy → PyTorch
arr = np.array([1, 2, 3])
t = torch.from_numpy(arr)  # Partage mémoire

# Attention: modification affecte les deux
arr[0] = 999
print(f"Tensor also changed: {t[0]}")  # 999

# Pour copie indépendante
t_copy = torch.tensor(arr)  # Copie
```

---

## Exercices

### Exercice 22.3.1.1
Créez un tenseur 3×3 avec requires_grad=True, calculez la trace de sa matrice au carré, et affichez les gradients.

### Exercice 22.3.1.2
Implémentez une fonction qui calcule le gradient de f(x,y) = x^2 + y^2 + xy et vérifiez manuellement.

### Exercice 22.3.1.3
Créez un tenseur sur GPU, effectuez des opérations, et convertissez le résultat en NumPy.

### Exercice 22.3.1.4
Implémentez gradient descent pour trouver le minimum de f(x) = x^4 - 3x^2 + x.

---

## Points Clés à Retenir

> 📌 **Les tenseurs PyTorch sont similaires à NumPy mais avec support GPU**

> 📌 **requires_grad=True active automatic differentiation**

> 📌 **backward() calcule gradients pour tous tenseurs avec requires_grad=True**

> 📌 **Les gradients s'accumulent: utiliser zero_() pour réinitialiser**

> 📌 **torch.no_grad() désactive gradient tracking pour performance**

> 📌 **Détacher ou utiliser no_grad() est important pour éviter accumulation mémoire**

---

*Section précédente : [22.3 PyTorch](./22_03_PyTorch.md) | Section suivante : [22.3.2 Modules et Optimizers](./22_03_02_Modules_Optimizers.md)*

