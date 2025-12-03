# 9.5 Quantification Binaire et Ternaire

---

## Introduction

La **quantification binaire** (1 bit) et **ternaire** (2 bits) représentent l'extrême de la compression. Les poids deviennent +1/-1 (binaire) ou +1/0/-1 (ternaire), permettant des implémentations hardware très efficaces.

---

## Quantification Binaire

### Principe

Pour une valeur $w$, la quantification binaire est :

$$w_b = \text{sign}(w) \times \alpha$$

où $\alpha$ est un facteur d'échelle et $\text{sign}(w) = +1$ si $w \geq 0$, $-1$ sinon.

### Implémentation

```python
class BinaryQuantizer:
    """
    Quantificateur binaire
    """
    
    @staticmethod
    def quantize(x, alpha=None):
        """
        Quantifie en binaire
        
        Args:
            x: tenseur à quantifier
            alpha: facteur d'échelle (si None, calculé automatiquement)
        """
        # Sign binaire
        sign = torch.sign(x)
        sign[sign == 0] = 1  # 0 -> +1
        
        # Alpha: moyenne des valeurs absolues
        if alpha is None:
            alpha = x.abs().mean()
        
        return sign * alpha
    
    @staticmethod
    def binarize_weights(weight):
        """Binarise les poids"""
        alpha = weight.abs().mean()
        return BinaryQuantizer.quantize(weight, alpha)

# Exemple
W = torch.randn(128, 256) * 0.1
W_binary = BinaryQuantizer.binarize_weights(W)

print(f"Binaire: {W_binary.unique()}")  # Devrait être [-alpha, +alpha]
print(f"Compression: {W.numel() * 32} bits → {W.numel() * 1 + 32} bits")
```

---

## Réseaux Binaires (BNN)

```python
class BinaryLinear(nn.Module):
    """
    Couche linéaire binaire
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Poids réels (seront binarisés)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # Binarise les poids
        W_binary = BinaryQuantizer.binarize_weights(self.weight)
        
        # Forward
        return F.linear(x, W_binary, self.bias)

# Exemple
binary_linear = BinaryLinear(784, 256)
x = torch.randn(32, 784)
y = binary_linear(x)
```

---

## Quantification Ternaire

### Principe

Les poids ternaires sont dans $\{-1, 0, +1\}$ :

$$w_t = \begin{cases}
+1 \times \alpha & \text{si } w > \Delta \\
0 & \text{si } |w| \leq \Delta \\
-1 \times \alpha & \text{si } w < -\Delta
\end{cases}$$

où $\Delta$ est un seuil et $\alpha$ un facteur d'échelle.

### Implémentation

```python
class TernaryQuantizer:
    """
    Quantificateur ternaire
    """
    
    @staticmethod
    def quantize(x, delta=None, alpha=None):
        """
        Quantifie en ternaire
        
        Args:
            x: tenseur à quantifier
            delta: seuil (si None, calculé comme 0.7 * mean(abs(x)))
            alpha: facteur d'échelle (si None, calculé automatiquement)
        """
        if delta is None:
            delta = 0.7 * x.abs().mean()
        
        # Ternarisation
        w_ternary = torch.zeros_like(x)
        w_ternary[x > delta] = 1.0
        w_ternary[x < -delta] = -1.0
        
        # Alpha: moyenne des valeurs absolues des poids non-nuls
        if alpha is None:
            abs_vals = x.abs()
            mask = abs_vals > delta
            if mask.any():
                alpha = abs_vals[mask].mean()
            else:
                alpha = 1.0
        
        return w_ternary * alpha
    
    @staticmethod
    def ternarize_weights(weight, delta=None):
        """Ternarise les poids"""
        return TernaryQuantizer.quantize(weight, delta=delta)

# Exemple
W = torch.randn(128, 256) * 0.1
W_ternary = TernaryQuantizer.ternarize_weights(W)

print(f"Ternaire unique values: {W_ternary.unique()}")
print(f"Compression: {W.numel() * 32} bits → {W.numel() * 2 + 32} bits (2 bits/value)")
```

---

## Entraînement de Réseaux Binaires/Ternaires

```python
class BinaryStraightThrough(torch.autograd.Function):
    """
    Straight-Through Estimator pour binarisation
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        # Binarise
        sign = torch.sign(x)
        sign[sign == 0] = 1
        return sign * alpha
    
    @staticmethod
    def backward(ctx, grad_output):
        # Gradient passe tel quel (Straight-Through)
        return grad_output, None

class TernaryStraightThrough(torch.autograd.Function):
    """
    Straight-Through Estimator pour ternarisation
    """
    
    @staticmethod
    def forward(ctx, x, delta, alpha):
        # Ternarise
        w_t = torch.zeros_like(x)
        w_t[x > delta] = 1.0
        w_t[x < -delta] = -1.0
        return w_t * alpha
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

# Utilisation dans une couche
class BinaryLinearTrainable(nn.Module):
    """Couche binaire avec entraînement"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        W_binary = BinaryStraightThrough.apply(self.weight, self.alpha)
        return F.linear(x, W_binary)
```

---

## Avantages et Limitations

### Avantages

- **Compression extrême** : 32x pour binaire, 16x pour ternaire
- **Vitesse hardware** : Opérations XNOR/POPCOUNT très rapides
- **Efficacité énergétique** : Très faible consommation

### Limitations

- **Perte de précision** : Souvent significative
- **Difficulté d'entraînement** : Gradients instables
- **Domaine d'application limité** : Fonctionne mieux pour certaines tâches

---

## Exercices

### Exercice 9.5.1
Implémentez un réseau binaire complet et comparez sa performance avec un réseau standard.

### Exercice 9.5.2
Expérimentez avec différentes méthodes de calcul d'alpha et delta pour la ternarisation.

---

## Points Clés à Retenir

> 📌 **Binaire: 1 bit (signe), compression 32x**

> 📌 **Ternaire: 2 bits (-1,0,+1), compression 16x**

> 📌 **Straight-Through Estimator nécessaire pour l'entraînement**

> 📌 **Très efficace sur hardware mais perte de précision importante**

---

*Section suivante : [9.6 Quantification pour Réseaux de Tenseurs](./09_06_TN_Quantization.md)*

