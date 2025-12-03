# Chapitre 6 : Réseaux de Tenseurs en Physique Quantique

---

## Introduction

Les **réseaux de tenseurs** ont été initialement développés en physique quantique pour représenter efficacement les états quantiques. Ces techniques se sont révélées extrêmement puissantes et ont trouvé des applications naturelles en machine learning.

---

## Plan du Chapitre

1. [États Produits Matriciels (MPS)](./06_01_MPS.md)
2. [États Projetés par Paires Entrelacées (PEPS)](./06_02_PEPS.md)
3. [MERA (Multi-scale Entanglement Renormalization Ansatz)](./06_03_MERA.md)
4. [Applications en Mécanique Quantique](./06_04_Mecanique_Quantique.md)
5. [Connexions avec l'Apprentissage Automatique](./06_05_ML_Connexions.md)

---

## Motivation : Problème de la Dimension Exponentielle

En mécanique quantique, l'état d'un système à $n$ particules nécessite :

$$2^n \text{ coefficients} \quad \text{(pour des spins 1/2)}$$

Pour $n=100$, cela représente $2^{100} \approx 10^{30}$ nombres !

Les réseaux de tenseurs permettent de représenter ces états avec un nombre polynomial de paramètres.

---

## États MPS (Matrix Product States)

### Définition

Un MPS représente un état quantique comme :

$$|\psi\rangle = \sum_{i_1,\ldots,i_n} \text{Tr}(A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n}) |i_1 i_2 \cdots i_n\rangle$$

où chaque $A^{[k]}_{i_k}$ est une matrice.

```python
import numpy as np

class MPSState:
    """
    Représente un état quantique en format MPS
    """
    
    def __init__(self, local_dims, bond_dims):
        """
        Args:
            local_dims: dimensions locales (d₁, d₂, ..., dₙ)
            bond_dims: dimensions de liaison (χ₁, χ₂, ..., χₙ₋₁)
        """
        self.n_sites = len(local_dims)
        self.local_dims = local_dims
        self.bond_dims = bond_dims
        
        # Initialise les matrices
        self.tensors = []
        for i in range(self.n_sites):
            bond_left = 1 if i == 0 else bond_dims[i-1]
            bond_right = 1 if i == self.n_sites-1 else bond_dims[i]
            physical = local_dims[i]
            
            # Matrices aléatoires normalisées
            tensor = np.random.randn(bond_left, physical, bond_right)
            tensor = tensor / np.linalg.norm(tensor)
            self.tensors.append(tensor)
    
    def contract_to_full_state(self):
        """
        Contracte le MPS pour obtenir l'état complet (coûteux!)
        """
        result = self.tensors[0]  # Shape: (1, d₁, χ₁)
        
        for i in range(1, self.n_sites):
            # Contracte avec le tenseur suivant
            # result: (..., χ_{i-1})
            # tensor: (χ_{i-1}, d_i, χ_i)
            result = np.tensordot(result, self.tensors[i], axes=([-1], [0]))
        
        # Squeeze les dimensions de liaison aux bords
        return result.squeeze()
    
    def compute_norm(self):
        """Calcule la norme de l'état"""
        # Pour un MPS normalisé, la norme devrait être ~1
        state = self.contract_to_full_state()
        return np.linalg.norm(state.flatten())
    
    def count_parameters(self):
        """Compte le nombre de paramètres"""
        total = 0
        for i, tensor in enumerate(self.tensors):
            total += tensor.size
        return total
    
    def full_state_size(self):
        """Taille de l'état complet (non compressé)"""
        return np.prod(self.local_dims)

# Exemple: 10 spins 1/2
mps = MPSState(local_dims=[2]*10, bond_dims=[4]*9)

print("État MPS:")
print(f"  Nombre de sites: {mps.n_sites}")
print(f"  Paramètres MPS: {mps.count_parameters():,}")
print(f"  Taille état complet: {mps.full_state_size():,}")
print(f"  Compression: {mps.full_state_size() / mps.count_parameters():.1f}x")
print(f"  Norme: {mps.compute_norm():.4f}")
```

---

## Évolution Temporelle avec MPS

```python
class MPSEvolution:
    """
    Évolution temporelle d'un état MPS
    """
    
    @staticmethod
    def apply_operator_local(mps, operator, site):
        """
        Applique un opérateur local au site 'site'
        
        operator: matrice (d, d) agissant sur l'espace local
        """
        tensor = mps.tensors[site]  # (χ_left, d, χ_right)
        
        # Contracte l'opérateur avec le tenseur
        # Nouvelle forme: (χ_left, d, χ_right)
        new_tensor = np.tensordot(tensor, operator, axes=([1], [1]))
        new_tensor = np.moveaxis(new_tensor, -1, 1)
        
        mps.tensors[site] = new_tensor
        return mps
    
    @staticmethod
    def apply_two_site_operator(mps, operator, sites):
        """
        Applique un opérateur à deux sites
        
        operator: tenseur (d₁, d₂, d₁', d₂')
        """
        i, j = sites
        
        # Contracte les deux tenseurs
        left_tensor = mps.tensors[i]  # (χ_{i-1}, d_i, χ_i)
        right_tensor = mps.tensors[j]  # (χ_{j-1}, d_j, χ_j)
        
        # Fusionne temporairement
        merged = np.tensordot(left_tensor, right_tensor, axes=([-1], [0]))
        # Shape: (χ_{i-1}, d_i, χ_i, d_j, χ_j)
        
        # Applique l'opérateur
        # (Complexe, nécessite reshape approprié)
        
        # Décompose avec SVD pour maintenir le format MPS
        
        return mps

# Exemple: rotation d'un spin
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Rotation autour de l'axe X
theta = np.pi / 4
rotation = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * pauli_x

mps_test = MPSState([2]*5, [4]*4)
MPSEvolution.apply_operator_local(mps_test, rotation, site=2)
```

---

## PEPS (Projected Entangled Pair States)

### Introduction

Les **PEPS** généralisent les MPS aux dimensions supérieures (2D, 3D).

```python
class PEPSState:
    """
    PEPS pour systèmes 2D
    
    Chaque site a des connexions avec ses voisins (haut, bas, gauche, droite)
    """
    
    def __init__(self, lattice_shape, physical_dim, bond_dim):
        """
        Args:
            lattice_shape: (Lx, Ly) dimensions du réseau 2D
            physical_dim: dimension de l'espace local
            bond_dim: dimension des liens
        """
        self.Lx, self.Ly = lattice_shape
        self.physical_dim = physical_dim
        self.bond_dim = bond_dim
        
        # Tenseur par site: (bond_up, bond_down, bond_left, bond_right, physical)
        self.tensors = {}
        
        for x in range(self.Lx):
            for y in range(self.Ly):
                # Dimensions des liens (1 aux bords)
                bond_up = 1 if y == 0 else bond_dim
                bond_down = 1 if y == self.Ly-1 else bond_dim
                bond_left = 1 if x == 0 else bond_dim
                bond_right = 1 if x == self.Lx-1 else bond_dim
                
                tensor = np.random.randn(
                    bond_up, bond_down, bond_left, bond_right, physical_dim
                )
                self.tensors[(x, y)] = tensor
    
    def count_parameters(self):
        """Compte les paramètres"""
        return sum(t.size for t in self.tensors.values())
    
    def full_state_size(self):
        """Taille de l'état complet"""
        return self.physical_dim ** (self.Lx * self.Ly)

# Exemple: réseau 4×4
peps = PEPSState(lattice_shape=(4, 4), physical_dim=2, bond_dim=3)

print("État PEPS:")
print(f"  Réseau: {peps.Lx}×{peps.Ly}")
print(f"  Paramètres: {peps.count_parameters():,}")
print(f"  État complet: {peps.full_state_size():,}")
print(f"  Compression: {peps.full_state_size() / peps.count_parameters():.2e}x")
```

---

## MERA (Multi-scale Entanglement Renormalization Ansatz)

### Principe

MERA utilise une structure hiérarchique pour capturer l'intrication à toutes les échelles.

```python
class MERAState:
    """
    MERA: structure hiérarchique pour l'intrication multi-échelle
    """
    
    def __init__(self, n_sites, bond_dim, n_layers):
        """
        Args:
            n_sites: nombre de sites physiques
            bond_dim: dimension de liaison
            n_layers: nombre de couches de rénormalisation
        """
        self.n_sites = n_sites
        self.bond_dim = bond_dim
        self.n_layers = n_layers
        
        # Disentanglers et isometries pour chaque couche
        self.disentanglers = []
        self.isometries = []
        
        current_sites = n_sites
        
        for layer in range(n_layers):
            # Disentanglers: unitaires sur paires de sites
            n_pairs = current_sites // 2
            layer_disentanglers = []
            for _ in range(n_pairs):
                # Unitaire (bond_dim², bond_dim²)
                U = self._random_unitary(bond_dim ** 2)
                layer_disentanglers.append(U)
            self.disentanglers.append(layer_disentanglers)
            
            # Isometries: projection vers couche supérieure
            layer_isometries = []
            for _ in range(n_pairs):
                # Isométrie (bond_dim, bond_dim, bond_dim)
                V = self._random_isometry(bond_dim, bond_dim, bond_dim)
                layer_isometries.append(V)
            self.isometries.append(layer_isometries)
            
            current_sites = current_sites // 2
    
    @staticmethod
    def _random_unitary(n):
        """Génère une matrice unitaire aléatoire"""
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        Q, R = np.linalg.qr(A)
        return Q
    
    @staticmethod
    def _random_isometry(n_in1, n_in2, n_out):
        """Génère une isométrie aléatoire"""
        # V: (n_in1, n_in2, n_out) tel que V†V = I
        V = np.random.randn(n_in1, n_in2, n_out) + 1j * np.random.randn(n_in1, n_in2, n_out)
        # Normalisation (approximation)
        V = V / np.linalg.norm(V)
        return V

# Exemple
mera = MERAState(n_sites=8, bond_dim=2, n_layers=3)
print(f"MERA: {mera.n_sites} sites, {mera.n_layers} couches")
```

---

## Connexions avec le Machine Learning

### Analogies

```
┌─────────────────────────────────────────────────────────────────┐
│              Analogies Physique Quantique ↔ ML                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Physique Quantique          │  Machine Learning              │
│  ─────────────────────────────────────────────────────────────  │
│  État quantique |ψ⟩          │  Vecteur de features           │
│  Intrication (entanglement)   │  Corrélations complexes        │
│  Réseau MPS/PEPS             │  Architecture Tensor Train     │
│  Évolution temporelle         │  Forward pass                  │
│  Réduction de dimension      │  Compression de modèle         │
│  Ansatz variationnel         │  Approximateur universel       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Utilisation en Deep Learning

```python
def mps_as_neural_network(input_dim, output_dim, hidden_dims, bond_dim):
    """
    Utilise un MPS comme couche de réseau de neurones
    
    L'état quantique devient le vecteur de features
    """
    n_sites = input_dim
    
    # MPS avec dimensions locales = dimensions d'entrée
    mps = MPSState(
        local_dims=hidden_dims,
        bond_dims=[bond_dim] * (len(hidden_dims) - 1)
    )
    
    # Pour l'inférence:
    # 1. Encode l'input dans les indices physiques
    # 2. Contracte le MPS
    # 3. Lit la sortie
    
    return mps

# Application: classification avec MPS
class MPSClassifier:
    """
    Classificateur utilisant un MPS
    """
    
    def __init__(self, input_dim, n_classes, bond_dim):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.bond_dim = bond_dim
        
        # MPS avec dimension locale = input_dim
        self.mps = MPSState(
            local_dims=[input_dim],
            bond_dims=[]
        )
        
        # Couche de sortie
        self.classifier = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        x: (batch, input_dim)
        """
        # Encode l'input dans le MPS (simplifié)
        # Contracte le MPS
        # Classification
        return self.classifier(x)
```

---

## Exercices

### Exercice 6.1
Implémentez l'application d'un opérateur à deux sites sur un MPS avec décomposition SVD pour maintenir le format.

### Exercice 6.2
Comparez le nombre de paramètres d'un MPS vs état complet pour 20 spins 1/2 avec différents rangs de liaison.

### Exercice 6.3
Créez une fonction qui convertit un état quantique complet (petit) en format MPS via SVD.

---

## Points Clés à Retenir

> 📌 **Les MPS évitent la malédiction de la dimensionnalité pour les systèmes 1D**

> 📌 **Les PEPS généralisent aux dimensions supérieures mais sont plus complexes**

> 📌 **MERA capture l'intrication à toutes les échelles**

> 📌 **Les techniques de physique quantique inspirent directement le ML moderne**

---

*Section suivante : [6.1 États Produits Matriciels (MPS)](./06_01_MPS.md)*

