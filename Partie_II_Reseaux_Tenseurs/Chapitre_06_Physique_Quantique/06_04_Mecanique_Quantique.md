# 6.4 Applications en Mécanique Quantique

---

## Introduction

Les réseaux de tenseurs sont des outils puissants pour résoudre des problèmes en mécanique quantique. Cette section présente les applications principales : simulation de systèmes quantiques, calcul de l'énergie du fondamental, dynamique temporelle, et systèmes quantiques à plusieurs corps.

---

## Simulation de Systèmes Quantiques

### Problème Fondamental

La simulation d'un système quantique à $n$ particules nécessite de stocker :
- $d^n$ coefficients pour l'état
- $d^{2n}$ éléments pour les opérateurs

Les réseaux de tenseurs réduisent cette complexité.

---

## Calcul de l'Énergie du Fondamental

### Ground State Energy

Pour un hamiltonien $H$, l'énergie du fondamental est :

$$E_0 = \min_{|\psi\rangle} \frac{\langle\psi|H|\psi\rangle}{\langle\psi|\psi\rangle}$$

Les réseaux de tenseurs permettent de trouver l'état fondamental approximatif.

### Exemple : Modèle d'Ising Transverse

```python
import numpy as np

class IsingModel:
    """
    Modèle d'Ising 1D avec champ transverse
    
    H = -J Σ σᵢᶻ σᵢ₊₁ᶻ - h Σ σᵢˣ
    """
    
    def __init__(self, n_sites, J=1.0, h=1.0):
        self.n_sites = n_sites
        self.J = J  # Coupling
        self.h = h  # Champ transverse
        
        # Matrices de Pauli
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        
        # Construit le hamiltonien sous forme MPO (Matrix Product Operator)
        self.mpo = self._construct_mpo()
    
    def _construct_mpo(self):
        """
        Construit le hamiltonien en format MPO
        
        MPO: opérateur sous forme produit matriciel
        """
        # Pour chaque site, on a:
        # W^[i] avec indices (bond_left, bond_right, physical_in, physical_out)
        # (Simplifié - construction complète MPO est complexe)
        
        return None
    
    def compute_energy_mps(self, mps):
        """
        Calcule ⟨ψ|H|ψ⟩ pour un état MPS
        
        Utilise la forme MPO du hamiltonien
        """
        # Contracte MPS† H MPS
        # (Simplifié)
        return 0.0

# Exemple
ising = IsingModel(n_sites=10, J=1.0, h=1.0)
print(f"Ising Model: {ising.n_sites} sites, J={ising.J}, h={ising.h}")
```

### DMRG (Density Matrix Renormalization Group)

```python
class SimpleDMRG:
    """
    Implémentation simplifiée de DMRG pour trouver le fondamental
    """
    
    def __init__(self, hamiltonian_mpo, initial_mps, max_bond_dim=10):
        self.hamiltonian = hamiltonian_mpo
        self.mps = initial_mps
        self.max_bond_dim = max_bond_dim
    
    def optimize_two_sites(self, sites):
        """
        Optimise deux sites adjacents
        
        Pour sites (i, i+1):
        1. Fusionne les tenseurs
        2. Minimise l'énergie localement
        3. Décompose via SVD
        """
        i, j = sites
        assert j == i + 1, "Sites doivent être adjacents"
        
        # Fusionne
        left_tensor = self.mps.tensors[i]
        right_tensor = self.mps.tensors[j]
        merged = np.tensordot(left_tensor, right_tensor, axes=([2], [0]))
        
        # Minimise l'énergie (simplifié - nécessite construction de l'hamiltonien effectif)
        # ...
        
        # SVD pour décomposer
        # ...
        
        return self.mps
    
    def sweep(self, direction='left_to_right'):
        """
        Effectue un sweep DMRG
        
        Optimise tous les sites dans une direction
        """
        if direction == 'left_to_right':
            for i in range(self.mps.n_sites - 1):
                self.optimize_two_sites((i, i+1))
        else:
            for i in range(self.mps.n_sites - 2, -1, -1):
                self.optimize_two_sites((i, i+1))
    
    def run(self, num_sweeps=10):
        """
        Exécute plusieurs sweeps DMRG
        """
        energies = []
        
        for sweep in range(num_sweeps):
            # Sweep gauche-droite
            self.sweep('left_to_right')
            
            # Sweep droite-gauche
            self.sweep('right_to_left')
            
            # Calcule l'énergie
            energy = self.compute_energy()
            energies.append(energy)
        
        return energies
    
    def compute_energy(self):
        """Calcule l'énergie actuelle"""
        # (Simplifié)
        return 0.0
```

---

## Dynamique Temporelle

### Évolution Temporelle avec MPS

L'évolution temporelle suit :

$$|\psi(t)\rangle = e^{-iHt}|\psi(0)\rangle$$

Problème : $e^{-iHt}$ est difficile à appliquer directement sur un MPS.

### TDVP (Time-Dependent Variational Principle)

```python
class TDVP:
    """
    Time-Dependent Variational Principle pour évolution MPS
    
    Projette l'évolution sur la variété MPS
    """
    
    def __init__(self, hamiltonian, mps, dt=0.01):
        self.hamiltonian = hamiltonian
        self.mps = mps
        self.dt = dt
    
    def evolve_step(self):
        """
        Un pas d'évolution temporelle
        
        Résout: d|ψ⟩/dt = -i H |ψ⟩
        sur la variété MPS
        """
        # TDVP: résout des équations différentielles pour chaque tenseur
        # (Simplifié - implémentation complète est complexe)
        pass
    
    def evolve(self, total_time):
        """
        Évolue l'état sur total_time
        """
        n_steps = int(total_time / self.dt)
        
        for step in range(n_steps):
            self.evolve_step()
        
        return self.mps
```

### TEBD (Time-Evolving Block Decimation)

```python
class TEBD:
    """
    Time-Evolving Block Decimation
    
    Décompose l'évolution en petits pas locaux
    """
    
    def __init__(self, hamiltonian, mps, dt=0.01):
        self.hamiltonian = hamiltonian
        self.mps = mps
        self.dt = dt
    
    def trotter_decomposition(self):
        """
        Décompose e^{-iHΔt} ≈ ∏ e^{-iH_i Δt}
        
        Où H = Σ H_i (somme d'opérateurs locaux)
        """
        # Pour Ising: H = Σ H_{i,i+1}
        # e^{-iHΔt} ≈ ∏ e^{-iH_{i,i+1} Δt}
        pass
    
    def evolve_step(self):
        """
        Applique un pas Trotter
        """
        # 1. Applique les opérateurs impairs: e^{-iH_{1,2}Δt}, e^{-iH_{3,4}Δt}, ...
        # 2. Applique les opérateurs pairs: e^{-iH_{2,3}Δt}, e^{-iH_{4,5}Δt}, ...
        pass
```

---

## Systèmes à Plusieurs Corps

### États de Bell

```python
def create_bell_state():
    """
    Crée l'état de Bell |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
    
    Représentable avec MPS bond_dim = 1 (état produit... non!)
    En fait, nécessite bond_dim = 2
    """
    # État: |00⟩ + |11⟩ (non normalisé)
    state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    
    # Convertit en MPS
    from Chapitre_06_Physique_Quantique.MPS import state_to_mps
    mps_tensors, bond_dims = state_to_mps(state, local_dims=[2, 2])
    
    print("État de Bell en MPS:")
    print(f"  Bond dims: {bond_dims}")
    print(f"  Intrication maximale → bond_dim = 2")
    
    return mps_tensors, bond_dims

create_bell_state()
```

### États GHZ

```python
def create_ghz_state(n_qubits):
    """
    Crée l'état GHZ: (|00...0⟩ + |11...1⟩) / √2
    
    Nécessite bond_dim = 2 (peu d'intrication mais longue distance)
    """
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0  # |00...0⟩
    state[-1] = 1.0  # |11...1⟩
    state = state / np.sqrt(2)
    
    # Convertit en MPS
    mps_tensors, bond_dims = state_to_mps(state, local_dims=[2]*n_qubits)
    
    print(f"État GHZ ({n_qubits} qubits) en MPS:")
    print(f"  Bond dims: {bond_dims}")
    print(f"  Bond_dim max = 2 (intrication longue distance)")
    
    return mps_tensors, bond_dims

create_ghz_state(5)
```

### États W

```python
def create_w_state(n_qubits):
    """
    Crée l'état W: (|10...0⟩ + |01...0⟩ + ... + |00...1⟩) / √n
    
    Nécessite bond_dim croissant avec n
    """
    state = np.zeros(2**n_qubits, dtype=complex)
    
    # Tous les états avec exactement un 1
    for i in range(n_qubits):
        idx = 2**i
        state[idx] = 1.0
    
    state = state / np.sqrt(n_qubits)
    
    # Convertit en MPS
    mps_tensors, bond_dims = state_to_mps(state, local_dims=[2]*n_qubits)
    
    print(f"État W ({n_qubits} qubits) en MPS:")
    print(f"  Bond dims: {bond_dims}")
    print(f"  Bond_dim croît avec n (intrication complexe)")
    
    return mps_tensors, bond_dims

create_w_state(4)
```

---

## Propriétés Quantiques

### Mesure de l'Intrication

```python
def compute_entanglement_entropy(mps, cut_site):
    """
    Calcule l'entropie d'intrication de Von Neumann
    
    S = -Tr(ρ_A log ρ_A)
    
    où ρ_A est la matrice de densité réduite d'une partition
    """
    # Met en forme canonique mixte avec centre en cut_site
    mps_mixed = mixed_canonical_form(mps.copy(), cut_site)
    
    # À partir de la forme mixte, la matrice de densité réduite
    # est directement donnée par le tenseur au centre
    # (Simplifié)
    
    # Calcule les valeurs propres
    # eigenvalues = ...
    # entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
    
    return 0.0

def von_neumann_entropy(rho):
    """Calcule S(ρ) = -Tr(ρ log ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Évite log(0)
    return -np.sum(eigenvalues * np.log2(eigenvalues))
```

### Corrélations

```python
def compute_correlation(mps, operator1, operator2, sites):
    """
    Calcule ⟨O₁(site1) O₂(site2)⟩
    """
    i, j = sites
    
    # Applique les opérateurs
    mps_copy = mps.copy()
    apply_local_operator(mps_copy, operator1, i)
    apply_local_operator(mps_copy, operator2, j)
    
    # Calcule le produit scalaire
    correlation = mps.compute_overlap(mps_copy)
    
    return correlation

# Exemple: corrélation spin-spin
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)

mps_test = MPSState([2]*10, [4]*9)
corr = compute_correlation(mps_test, pauli_z, pauli_z, sites=(2, 5))
print(f"Corrélation σᶻ(2) σᶻ(5): {corr:.4f}")
```

---

## Systèmes Critiques

### Point Critique

Au point critique d'une transition de phase :
- Corrélations décroissent lentement (power-law)
- Intrication croît logarithmiquement avec la taille
- MERA est particulièrement adapté

### Scaling de l'Intrication

```python
def entanglement_scaling():
    """
    Étudie le scaling de l'intrication avec la taille du système
    """
    sizes = [4, 8, 16, 32]
    entropies = []
    
    for n in sizes:
        # Crée un état critique (ex: Ising à h = J)
        # (Simplifié)
        entropy = np.log2(n)  # Scaling logarithmique typique
        entropies.append(entropy)
    
    print("Scaling de l'entropie d'intrication:")
    for n, S in zip(sizes, entropies):
        print(f"  n={n:2d}: S={S:.3f}")
    
    print("\n  Régime critique: S ~ log(n)")
    print("  Régime gappé: S ~ constante")

entanglement_scaling()
```

---

## Applications Spécifiques

### Modèle de Heisenberg

```python
class HeisenbergModel:
    """
    Modèle de Heisenberg: H = J Σ Sᵢ · Sᵢ₊₁
    """
    
    def __init__(self, n_sites, J=1.0):
        self.n_sites = n_sites
        self.J = J
        
        # Matrices de spin S = (1/2) σ
        self.Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        self.Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def compute_energy_mps(self, mps):
        """Calcule l'énergie pour un MPS"""
        # Contracte MPS† H MPS
        # (Simplifié)
        return 0.0
```

### Modèles Frustrés

```python
def frustrated_system():
    """
    Systèmes frustrés nécessitent PEPS plutôt que MPS
    """
    # Exemple: modèle J1-J2 sur réseau 2D
    # Frustration → intrication 2D importante
    pass
```

---

## Exercices

### Exercice 6.4.1
Implémentez le calcul de l'énergie du fondamental pour le modèle d'Ising avec DMRG.

### Exercice 6.4.2
Calculez l'évolution temporelle d'un état de Bell avec TEBD.

### Exercice 6.4.3
Mesurez l'entropie d'intrication d'un MPS pour différentes partitions.

---

## Points Clés à Retenir

> 📌 **Les réseaux de tenseurs permettent de simuler des systèmes quantiques à plusieurs corps**

> 📌 **DMRG est la méthode standard pour trouver le fondamental en 1D**

> 📌 **TDVP et TEBD permettent l'évolution temporelle d'états MPS**

> 📌 **L'entropie d'intrication mesure l'intrication quantique**

> 📌 **Les systèmes critiques ont un scaling logarithmique de l'intrication**

---

*Section suivante : [6.5 Connexions avec l'Apprentissage Automatique](./06_05_ML_Connexions.md)*

