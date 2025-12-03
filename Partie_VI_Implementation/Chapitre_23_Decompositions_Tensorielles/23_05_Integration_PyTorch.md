# 23.5 Intégration avec PyTorch

---

## Introduction

L'intégration des décompositions tensorielles avec PyTorch permet d'utiliser les réseaux de tenseurs dans des modèles de deep learning avec entraînement end-to-end. Cette section présente comment créer des couches tensorielles compatibles PyTorch, les intégrer dans des modèles, et les entraîner efficacement.

---

## Couche Linéaire Tensorielle

### Implémentation avec CP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CPLinear(nn.Module):
    """
    Couche linéaire avec contrainte CP
    
    Remplace W (M×N) par décomposition CP
    """
    
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Facteurs CP
        # Factoriser M et N (ex: M=m1*m2, N=n1*n2)
        self.m1, self.m2 = self._factorize(in_features)
        self.n1, self.n2 = self._factorize(out_features)
        
        # Facteurs CP (4 facteurs pour 2D→2D)
        self.factor1 = nn.Parameter(torch.randn(self.m1, rank))
        self.factor2 = nn.Parameter(torch.randn(self.m2, rank))
        self.factor3 = nn.Parameter(torch.randn(self.n1, rank))
        self.factor4 = nn.Parameter(torch.randn(self.n2, rank))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _factorize(self, n):
        """Factorise n en deux facteurs"""
        # Trouver facteurs proches de sqrt(n)
        sqrt_n = int(np.sqrt(n))
        for i in range(sqrt_n, 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n
    
    def _init_weights(self):
        """Initialise poids"""
        nn.init.normal_(self.factor1, std=0.1)
        nn.init.normal_(self.factor2, std=0.1)
        nn.init.normal_(self.factor3, std=0.1)
        nn.init.normal_(self.factor4, std=0.1)
    
    def get_weight_matrix(self):
        """Reconstruit matrice de poids depuis facteurs CP"""
        # Reconstruire depuis CP
        # W = Σᵣ (f1[:,r] ○ f2[:,r]) @ (f3[:,r] ○ f4[:,r]).T
        
        W = torch.zeros(self.in_features, self.out_features)
        
        for r in range(self.rank):
            # Premier mode
            f1_expanded = self.factor1[:, r].unsqueeze(1)  # (m1, 1)
            f2_expanded = self.factor2[:, r].unsqueeze(0)  # (1, m2)
            mode1 = (f1_expanded @ f2_expanded).flatten()  # (m1*m2,)
            
            # Deuxième mode
            f3_expanded = self.factor3[:, r].unsqueeze(1)  # (n1, 1)
            f4_expanded = self.factor4[:, r].unsqueeze(0)  # (1, n2)
            mode2 = (f3_expanded @ f4_expanded).flatten()  # (n1*n2,)
            
            # Produit externe
            W += torch.outer(mode1, mode2)
        
        return W
    
    def forward(self, x):
        """Forward pass"""
        # Méthode efficace: contraction directe sans reconstruire matrice
        batch_size = x.shape[0]
        
        # Reshape input
        x_reshaped = x.view(batch_size, self.m1, self.m2)
        
        # Contraction avec facteurs
        # x @ W = x @ (Σᵣ f1⊗f2 @ f3⊗f4.T)
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        for r in range(self.rank):
            # Contracter avec facteurs input
            temp = torch.tensordot(x_reshaped, self.factor1[:, r], dims=([1], [0]))
            temp = torch.tensordot(temp, self.factor2[:, r], dims=([1], [0]))
            
            # Contracter avec facteurs output
            temp = temp @ self.factor3[:, r].unsqueeze(1)  # (batch, n1)
            temp = temp @ self.factor4[:, r].unsqueeze(0)  # (batch, n2)
            
            output += temp.flatten(start_dim=1)
        
        if self.bias is not None:
            output += self.bias
        
        return output

# Test
cp_linear = CPLinear(in_features=100, out_features=50, rank=5)
x = torch.randn(32, 100)
output = cp_linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Compter paramètres
original_params = 100 * 50  # 5000
cp_params = sum(p.numel() for p in cp_linear.parameters())
print(f"Paramètres originaux: {original_params}")
print(f"Paramètres CP: {cp_params}")
print(f"Compression: {original_params / cp_params:.2f}×")
```

---

## Couche avec Tensor Train

### TT-Linear Layer

```python
class TTLinear(nn.Module):
    """
    Couche linéaire avec contrainte Tensor Train
    """
    
    def __init__(self, in_features, out_features, tt_rank=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tt_rank = tt_rank
        
        # Factoriser dimensions
        in_factors = self._factorize_dims(in_features)
        out_factors = self._factorize_dims(out_features)
        
        self.n_modes_in = len(in_factors)
        self.n_modes_out = len(out_factors)
        
        # Cores TT
        self.tt_cores = nn.ModuleList()
        
        # Premier core: (1, in_dim1, tt_rank)
        self.tt_cores.append(
            nn.Parameter(torch.randn(1, in_factors[0], tt_rank))
        )
        
        # Cores intermédiaires: (tt_rank, in_dimi, tt_rank)
        for i in range(1, self.n_modes_in):
            self.tt_cores.append(
                nn.Parameter(torch.randn(tt_rank, in_factors[i], tt_rank))
            )
        
        # Core de transition: (tt_rank, tt_rank) pour connecter in/out
        self.tt_cores.append(
            nn.Parameter(torch.randn(tt_rank, tt_rank))
        )
        
        # Cores output: (tt_rank, out_dimi, tt_rank)
        for i in range(self.n_modes_out - 1):
            self.tt_cores.append(
                nn.Parameter(torch.randn(tt_rank, out_factors[i], tt_rank))
            )
        
        # Dernier core: (tt_rank, out_dim_last, 1)
        self.tt_cores.append(
            nn.Parameter(torch.randn(tt_rank, out_factors[-1], 1))
        )
        
        self._init_weights()
    
    def _factorize_dims(self, n):
        """Factorise en plusieurs facteurs"""
        factors = []
        remaining = n
        while remaining > 1:
            factor = 2
            while remaining % factor != 0 and factor * factor <= remaining:
                factor += 1
            if remaining % factor == 0:
                factors.append(factor)
                remaining //= factor
            else:
                factors.append(remaining)
                break
        return factors if factors else [n]
    
    def _init_weights(self):
        """Initialise poids"""
        for core in self.tt_cores:
            nn.init.normal_(core, std=0.1)
    
    def forward(self, x):
        """Forward pass avec TT"""
        batch_size = x.shape[0]
        
        # Reshape input
        x_reshaped = x.view(batch_size, *[self.tt_cores[i].shape[1] 
                                          for i in range(self.n_modes_in)])
        
        # Contracter avec cores input
        result = x_reshaped
        for i in range(self.n_modes_in):
            result = torch.tensordot(result, self.tt_cores[i], dims=([1], [1]))
            # Reshape pour prochaine contraction
        
        # Transition
        result = torch.tensordot(result, self.tt_cores[self.n_modes_in], dims=([-1], [0]))
        
        # Contracter avec cores output
        for i in range(self.n_modes_out):
            idx = self.n_modes_in + 1 + i
            result = torch.tensordot(result, self.tt_cores[idx], dims=([-1], [0]))
        
        output = result.flatten(start_dim=1)
        return output

# Test
tt_linear = TTLinear(in_features=100, out_features=50, tt_rank=5)
x = torch.randn(32, 100)
output = tt_linear(x)
print(f"TT Linear output shape: {output.shape}")
```

---

## Modèle Complet avec Couches Tensorielles

### Exemple d'Architecture

```python
class TensorNeuralNetwork(nn.Module):
    """
    Réseau de neurones avec couches tensorielles
    """
    
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=10, 
                 rank=5, use_tt=False):
        super().__init__()
        
        if use_tt:
            self.fc1 = TTLinear(input_dim, hidden_dim, tt_rank=rank)
            self.fc2 = TTLinear(hidden_dim, hidden_dim // 2, tt_rank=rank)
            self.fc3 = TTLinear(hidden_dim // 2, output_dim, tt_rank=rank)
        else:
            self.fc1 = CPLinear(input_dim, hidden_dim, rank=rank)
            self.fc2 = CPLinear(hidden_dim, hidden_dim // 2, rank=rank)
            self.fc3 = CPLinear(hidden_dim // 2, output_dim, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Comparaison
model_cp = TensorNeuralNetwork(use_tt=False, rank=5)
model_tt = TensorNeuralNetwork(use_tt=True, rank=5)

x = torch.randn(32, 100)
output_cp = model_cp(x)
output_tt = model_tt(x)

print(f"CP model parameters: {sum(p.numel() for p in model_cp.parameters())}")
print(f"TT model parameters: {sum(p.numel() for p in model_tt.parameters())}")
```

---

## Entraînement End-to-End

### Pipeline Complet

```python
def train_tensor_model(model, train_loader, n_epochs=10, lr=0.001):
    """Entraîne modèle avec couches tensorielles"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Gradient clipping (important pour stabilité)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
```

---

## Exercices

### Exercice 23.5.1
Créez couche CPLinear et intégrez-la dans réseau simple pour classification.

### Exercice 23.5.2
Implémentez TTLinear et comparez compression vs CPLinear.

### Exercice 23.5.3
Entraînez modèle avec couches tensorielles et comparez performance vs modèle standard.

### Exercice 23.5.4
Analysez impact de rank sur compression et performance du modèle.

---

## Points Clés à Retenir

> 📌 **Les couches tensorielles peuvent remplacer couches linéaires standards**

> 📌 **L'intégration PyTorch permet entraînement end-to-end avec gradients**

> 📌 **Compression significative possible avec faible perte performance**

> 📌 **Gradient clipping peut améliorer stabilité entraînement**

> 📌 **TT et CP ont différents trade-offs (compression, vitesse, flexibilité)**

---

*Section précédente : [23.4 Optimisation](./23_04_Optimisation.md)*

