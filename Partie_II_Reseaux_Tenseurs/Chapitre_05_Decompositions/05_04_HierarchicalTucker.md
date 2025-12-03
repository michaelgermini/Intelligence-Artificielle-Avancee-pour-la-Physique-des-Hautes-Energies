# 5.4 Hierarchical Tucker (HT)

---

## Introduction

Le **Hierarchical Tucker** (HT) utilise une structure arborescente pour organiser la décomposition, offrant un compromis entre flexibilité et efficacité computationnelle.

---

## Structure Hiérarchique

Le HT organise les modes du tenseur selon un arbre binaire :

```
┌─────────────────────────────────────────────────────────────────┐
│                    Structure HT                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         Root                                    │
│                          (1,2,3,4)                             │
│                       /           \                             │
│                 (1,2)              (3,4)                        │
│                /     \            /     \                       │
│             (1)       (2)       (3)      (4)                   │
│                                                                 │
│  Chaque nœud représente un groupe de modes                     │
│  Les feuilles sont les modes individuels                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Définition

```python
import numpy as np

class HierarchicalTucker:
    """
    Décomposition Hierarchical Tucker
    """
    
    def __init__(self, tensor, tree_structure, ranks):
        """
        Args:
            tensor: Tenseur à décomposer
            tree_structure: Arbre binaire définissant le regroupement
            ranks: Rangs pour chaque nœud de l'arbre
        """
        self.tensor = np.array(tensor)
        self.tree = tree_structure
        self.ranks = ranks
        self.shape = tensor.shape
        
        # Facteurs à chaque nœud
        self.factors = {}
        
    def build_tree(self, modes):
        """
        Construit un arbre binaire équilibré pour les modes
        """
        if len(modes) == 1:
            return {'mode': modes[0], 'children': None}
        
        # Divise en deux
        mid = len(modes) // 2
        left = modes[:mid]
        right = modes[mid:]
        
        return {
            'left': self.build_tree(left),
            'right': self.build_tree(right),
            'modes': modes
        }

# Exemple: arbre pour 4 modes
ht = HierarchicalTucker(None, None, None)
tree = ht.build_tree([0, 1, 2, 3])
print("Arbre HT pour 4 modes:")
print(f"  Structure: {tree}")
```

---

## Algorithme de Décomposition

```python
def hierarchical_tucker_decomposition(tensor, tree, ranks):
    """
    Décomposition HT récursive
    
    Pour chaque nœud de l'arbre:
    1. Si feuille: factorise selon le mode
    2. Si nœud interne: combine les facteurs des enfants
    """
    n_modes = tensor.ndim
    
    # Décomposition bottom-up
    factors = {}
    
    def decompose_node(node, tensor_slice):
        """
        Décompose récursivement un nœud
        """
        if node.get('mode') is not None:
            # Feuille: SVD selon le mode
            mode = node['mode']
            tensor_mat = unfold_tensor(tensor_slice, mode)
            U, S, Vt = np.linalg.svd(tensor_mat, full_matrices=False)
            
            rank = ranks.get(mode, min(U.shape[1], 10))
            factors[node['mode']] = U[:, :rank]
            
            return Vt[:rank, :]
        
        else:
            # Nœud interne: décompose les enfants
            left_result = decompose_node(node['left'], tensor_slice)
            right_result = decompose_node(node['right'], tensor_slice)
            
            # Combine les résultats
            # (Simplification)
            return combine_factors(left_result, right_result)
    
    decompose_node(tree, tensor)
    return factors

def combine_factors(left_factor, right_factor):
    """Combine les facteurs de deux enfants"""
    # Logique de combinaison pour HT
    # (Simplifié)
    pass
```

---

## Avantages du HT

- **Structure hiérarchique** : Permet une organisation naturelle
- **Compression efficace** : Bon compromis CP/Tucker
- **Opérations efficaces** : Structure arborescente facilite les calculs

---

## Comparaison avec Autres Décompositions

```python
def compare_decompositions(tensor, ranks):
    """
    Compare CP, Tucker et HT
    """
    # CP
    cp_factors, _, cp_error = cp_als(tensor, ranks[0], max_iter=20)
    cp_params = sum(f.size for f in cp_factors)
    
    # Tucker
    tucker_core, tucker_factors = hosvd(tensor, ranks)
    tucker_params = tucker_core.size + sum(f.size for f in tucker_factors)
    
    # HT (approximation)
    ht_params = estimate_ht_params(tensor.shape, ranks)
    
    print("Comparaison des décompositions:")
    print(f"  CP: {cp_params:,} params, erreur: {cp_error:.4f}")
    print(f"  Tucker: {tucker_params:,} params")
    print(f"  HT: {ht_params:,} params (estimé)")
```

---

## Points Clés à Retenir

> 📌 **HT utilise une structure arborescente pour organiser la décomposition**

> 📌 **HT offre un bon compromis entre flexibilité et efficacité**

> 📌 **La structure hiérarchique facilite certaines opérations**

---

*Section suivante : [5.5 Tensor Ring Decomposition](./05_05_TensorRing.md)*

