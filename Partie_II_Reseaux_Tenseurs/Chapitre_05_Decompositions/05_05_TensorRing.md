# 5.5 Tensor Ring Decomposition

---

## Introduction

Le **Tensor Ring** (TR) est une généralisation du Tensor Train où les rangs aux bords sont libres (pas forcés à 1). Cela donne plus de flexibilité mais avec des contraintes différentes.

---

## Définition

Pour un tenseur $\mathcal{T} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$, la décomposition TR est :

$$\mathcal{T}[i_1, \ldots, i_N] = \text{Tr}\left(G_1[i_1] \cdot G_2[i_2] \cdots G_N[i_N]\right)$$

où $\text{Tr}$ est la trace matricielle, et les $G_k[i_k]$ sont des matrices de taille $(R, R)$ (rangs circulaires).

---

## Différence avec TT

```python
class TensorRing:
    """
    Représentation Tensor Ring
    """
    
    def __init__(self, cores, rank):
        """
        Args:
            cores: Liste de tenseurs [G₁, ..., Gₙ]
                   Gₖ.shape = (R, i_k, R) - rangs circulaires
            rank: Rang TR (identique pour tous les cores)
        """
        self.cores = cores
        self.rank = rank
        self.n_modes = len(cores)
        self.local_dims = [core.shape[1] for core in cores]
    
    def reconstruct(self):
        """
        Reconstruit via contraction circulaire + trace
        """
        # Contracte tous les cores
        result = self.cores[0]
        for core in self.cores[1:]:
            result = np.tensordot(result, core, axes=([-1], [0]))
        
        # Trace sur les dimensions de liaison (première et dernière)
        result = np.trace(result, axis1=0, axis2=-1)
        
        return result
    
    def count_parameters(self):
        """Paramètres TR"""
        return sum(core.size for core in self.cores)

# Comparaison TT vs TR
tt_cores = [
    np.random.randn(1, 5, 4),   # TT: r₀=1, r₁=4
    np.random.randn(4, 6, 3),   # TT: r₁=4, r₂=3
    np.random.randn(3, 7, 1)    # TT: r₂=3, r₃=1
]

tr_cores = [
    np.random.randn(3, 5, 3),   # TR: R=3 pour tous
    np.random.randn(3, 6, 3),
    np.random.randn(3, 7, 3)
]

print("Comparaison TT vs TR:")
print(f"  TT ranks: {[1, 4, 3, 1]}")
print(f"  TR rank: {3} (constant)")
print(f"  TT params: {sum(c.size for c in tt_cores):,}")
print(f"  TR params: {sum(c.size for c in tr_cores):,}")
```

---

## Avantages du TR

- **Symétrie circulaire** : Pas de contrainte aux bords
- **Flexibilité** : Rang constant simplifie certaines opérations
- **Meilleure compression** : Parfois meilleur que TT pour certains tenseurs

---

## Conversion TT ↔ TR

```python
def tt_to_tr(tt_cores):
    """
    Convertit TT en TR
    
    Augmente les rangs aux bords pour créer la circularité
    """
    # Prend le maximum des rangs aux bords
    rank = max(tt_cores[0].shape[0], tt_cores[-1].shape[-1])
    
    tr_cores = []
    for i, core in enumerate(tt_cores):
        r_left, i_dim, r_right = core.shape
        
        # Pad pour avoir rangs circulaires
        if i == 0:
            # Premier core: pad à gauche
            new_core = np.zeros((rank, i_dim, r_right))
            new_core[-r_left:, :, :] = core
        elif i == len(tt_cores) - 1:
            # Dernier core: pad à droite et connecte avec le premier
            new_core = np.zeros((r_left, i_dim, rank))
            new_core[:, :, :r_right] = core
        else:
            # Cores intermédiaires: pas de changement
            new_core = core
        
        tr_cores.append(new_core)
    
    return tr_cores, rank
```

---

## Points Clés à Retenir

> 📌 **TR généralise TT avec des rangs circulaires**

> 📌 **La symétrie circulaire peut donner une meilleure compression**

> 📌 **TR simplifie certaines opérations grâce au rang constant**

---

*Section suivante : [5.6 Comparaison et Choix de Décomposition](./05_06_Comparaison.md)*

