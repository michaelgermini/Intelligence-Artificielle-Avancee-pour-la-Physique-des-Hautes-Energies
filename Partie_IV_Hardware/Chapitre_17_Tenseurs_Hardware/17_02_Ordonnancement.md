# 17.2 Ordonnancement Optimal des Contractions

---

## Introduction

L'**ordonnancement des contractions** est crucial pour l'efficacité des réseaux de tenseurs. Pour un réseau avec N tenseurs, il existe O(2^N) ordonnancements possibles, et le choix de l'ordonnancement peut changer la complexité computationnelle et l'utilisation mémoire de plusieurs ordres de grandeur.

Cette section présente les algorithmes pour trouver l'ordonnancement optimal, incluant la recherche exhaustive, les heuristiques gloutonnes, et les méthodes d'optimisation dynamique.

---

## Problème d'Ordonnancement

### Définition

```python
import numpy as np
from typing import List, Tuple, Dict
import itertools

class ContractionScheduling:
    """
    Ordonnancement de contractions tensorielles
    """
    
    def __init__(self):
        self.problem_definition = """
        Problème d'ordonnancement:
        
        Étant donné un ensemble de tenseurs {T₁, T₂, ..., Tₙ},
        trouver un ordre de contractions qui minimise:
        
        1. Complexité computationnelle totale
        2. Mémoire maximale utilisée
        3. Latence totale
        
        Contraintes:
        - Chaque tenseur doit être contracté exactement une fois
        - Indices contractés doivent correspondre
        """
    
    def contraction_tree_example(self):
        """
        Exemple d'arbre de contraction
        """
        # Exemple: 4 tenseurs A, B, C, D
        # Contractions possibles: ((A*B)*C)*D ou (A*B)*(C*D) ou A*((B*C)*D)
        
        trees = {
            'left_associative': {
                'order': '((A*B)*C)*D',
                'description': 'Association à gauche',
                'memory_peak': 'Peut être élevée si résultat croît rapidement'
            },
            'right_associative': {
                'order': 'A*(B*(C*D))',
                'description': 'Association à droite',
                'memory_peak': 'Peut différer selon tailles'
            },
            'balanced': {
                'order': '(A*B)*(C*D)',
                'description': 'Arbre équilibré',
                'memory_peak': 'Souvent meilleur compromis'
            }
        }
        
        return trees

scheduling = ContractionScheduling()
print(scheduling.problem_definition)
```

---

## Complexité des Contractions

### Analyse de Complexité

```python
class ContractionComplexity:
    """
    Analyse de complexité des contractions
    """
    
    def compute_contraction_complexity(self, shape_A: Tuple, shape_B: Tuple, 
                                      contracted_dims_A: List[int], 
                                      contracted_dims_B: List[int]) -> Dict:
        """
        Calcule la complexité d'une contraction
        
        Returns:
            dict avec complexité computationnelle et mémoire
        """
        # Dimensions libres de A
        free_dims_A = [i for i in range(len(shape_A)) if i not in contracted_dims_A]
        free_dims_B = [i for i in range(len(shape_B)) if i not in contracted_dims_B]
        
        # Taille des dimensions contractées
        contracted_size_A = np.prod([shape_A[i] for i in contracted_dims_A])
        contracted_size_B = np.prod([shape_B[i] for i in contracted_dims_B])
        
        # Vérifier compatibilité
        if contracted_size_A != contracted_size_B:
            raise ValueError("Dimensions contractées incompatibles")
        
        # Taille des dimensions libres
        free_size_A = np.prod([shape_A[i] for i in free_dims_A])
        free_size_B = np.prod([shape_B[i] for i in free_dims_B])
        
        # Shape du résultat
        result_shape = tuple([shape_A[i] for i in free_dims_A] + 
                           [shape_B[i] for i in free_dims_B])
        result_size = np.prod(result_shape)
        
        # Complexité computationnelle: O(free_A × free_B × contracted)
        computation_ops = free_size_A * free_size_B * contracted_size_A
        
        # Mémoire: tenseurs d'input + résultat
        memory_input = np.prod(shape_A) + np.prod(shape_B)
        memory_output = result_size
        memory_peak = memory_input + memory_output  # Pendant contraction
        
        return {
            'computation_ops': computation_ops,
            'memory_input': memory_input,
            'memory_output': memory_output,
            'memory_peak': memory_peak,
            'result_shape': result_shape
        }
    
    def compare_contraction_orders(self, tensors: List[Tuple], contraction_orders: List[List[Tuple]]):
        """
        Compare différents ordres de contractions
        
        Args:
            tensors: Liste de shapes des tenseurs initiaux
            contraction_orders: Liste d'ordres (chaque ordre = liste de paires (i, j))
        """
        results = []
        
        for order_idx, order in enumerate(contraction_orders):
            total_ops = 0
            peak_memory = 0
            current_tensors = list(tensors)
            
            for step, (i, j) in enumerate(order):
                # Simplifié: suppose contraction standard sur dernières/premières dims
                shape_A = current_tensors[i]
                shape_B = current_tensors[j]
                
                # Approximation: contracter dernière dim de A avec première de B
                contracted_dims_A = [len(shape_A) - 1] if len(shape_A) > 0 else []
                contracted_dims_B = [0] if len(shape_B) > 0 else []
                
                complexity = self.compute_contraction_complexity(
                    shape_A, shape_B, contracted_dims_A, contracted_dims_B
                )
                
                total_ops += complexity['computation_ops']
                peak_memory = max(peak_memory, complexity['memory_peak'])
                
                # Mettre à jour: remplacer i et j par résultat
                new_shape = complexity['result_shape']
                current_tensors = ([current_tensors[k] for k in range(len(current_tensors)) 
                                  if k not in [i, j]] + [new_shape])
            
            results.append({
                'order': order,
                'total_ops': total_ops,
                'peak_memory': peak_memory
            })
        
        return results

# Exemple
complexity_analyzer = ContractionComplexity()

# Exemple: 3 tenseurs
tensors = [(10, 20), (20, 30), (30, 40)]

# Ordre 1: ((A*B)*C)
order1 = [(0, 1), (0, 1)]  # (A*B), puis résultat*C
# Ordre 2: (A*(B*C))
order2 = [(1, 2), (0, 1)]  # (B*C), puis A*résultat

results = complexity_analyzer.compare_contraction_orders(tensors, [order1, order2])

print("\n" + "="*70)
print("Comparaison d'Ordonnancements")
print("="*70)
for i, result in enumerate(results):
    print(f"\nOrdre {i+1}:")
    print(f"  Opérations totales: {result['total_ops']:,}")
    print(f"  Mémoire peak: {result['peak_memory']:,} éléments")
```

---

## Algorithmes d'Ordonnancement

### Recherche Exhaustive (Petit N)

```python
class ExhaustiveScheduler:
    """
    Recherche exhaustive de l'ordonnancement optimal
    """
    
    def __init__(self, max_tensors=8):
        """
        Args:
            max_tensors: Nombre maximum de tenseurs pour recherche exhaustive
        """
        self.max_tensors = max_tensors
    
    def generate_all_orders(self, n_tensors: int) -> List[List[Tuple]]:
        """
        Génère tous les ordres possibles de contractions
        
        Pour N tenseurs, il y a (2N-2)! / (N-1)! ordres possibles
        """
        if n_tensors > self.max_tensors:
            raise ValueError(f"Trop de tenseurs pour recherche exhaustive (max {self.max_tensors})")
        
        # Génération récursive de tous les arbres binaires
        orders = []
        
        def generate_recursive(remaining: List[int], current_order: List[Tuple]):
            if len(remaining) == 1:
                orders.append(current_order.copy())
                return
            
            # Essayer toutes les paires possibles
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    # Nouvelle paire
                    new_pair = (remaining[i], remaining[j])
                    
                    # Nouveaux indices (remplacer i, j par nouveau)
                    new_remaining = ([remaining[k] for k in range(len(remaining)) 
                                    if k not in [i, j]] + [len(current_order)])
                    
                    # Nouvel ordre
                    new_order = current_order + [new_pair]
                    
                    generate_recursive(new_remaining, new_order)
        
        initial_indices = list(range(n_tensors))
        generate_recursive(initial_indices, [])
        
        return orders
    
    def find_optimal_order(self, tensors: List[Tuple], 
                          objective='computation') -> Dict:
        """
        Trouve l'ordonnancement optimal par recherche exhaustive
        
        Args:
            tensors: Liste de shapes des tenseurs
            objective: 'computation', 'memory', ou 'combined'
        """
        n = len(tensors)
        
        if n > self.max_tensors:
            raise ValueError(f"Trop de tenseurs: {n} (max {self.max_tensors})")
        
        # Générer tous les ordres
        all_orders = self.generate_all_orders(n)
        
        # Évaluer chaque ordre
        complexity = ContractionComplexity()
        best_order = None
        best_score = float('inf')
        
        for order in all_orders:
            results = complexity.compare_contraction_orders(tensors, [order])
            result = results[0]
            
            # Score selon objectif
            if objective == 'computation':
                score = result['total_ops']
            elif objective == 'memory':
                score = result['peak_memory']
            else:  # combined
                # Normaliser et combiner (exemple)
                score = (result['total_ops'] / 1e6) + (result['peak_memory'] / 1e6)
            
            if score < best_score:
                best_score = score
                best_order = order
        
        return {
            'optimal_order': best_order,
            'score': best_score,
            'total_orders_evaluated': len(all_orders)
        }

# Exemple petit
exhaustive = ExhaustiveScheduler(max_tensors=5)

tensors_small = [(10, 20), (20, 30), (30, 40), (40, 50)]
optimal = exhaustive.find_optimal_order(tensors_small, objective='computation')

print(f"\nOrdre optimal trouvé:")
print(f"  Ordre: {optimal['optimal_order']}")
print(f"  Score: {optimal['score']:.2e}")
print(f"  Ordres évalués: {optimal['total_orders_evaluated']}")
```

---

## Heuristiques Gloutonnes

### Greedy Algorithms

```python
class GreedyScheduler:
    """
    Heuristiques gloutonnes pour ordonnancement
    """
    
    def greedy_min_complexity(self, tensors: List[Tuple]) -> List[Tuple]:
        """
        Algorithme glouton: choisit contraction de complexité minimale à chaque étape
        """
        complexity = ContractionComplexity()
        order = []
        current_tensors = list(tensors)
        current_indices = list(range(len(tensors)))
        
        while len(current_tensors) > 1:
            best_i, best_j = None, None
            best_complexity = float('inf')
            
            # Tester toutes les paires possibles
            for i in range(len(current_tensors)):
                for j in range(i + 1, len(current_tensors)):
                    shape_A = current_tensors[i]
                    shape_B = current_tensors[j]
                    
                    # Approximation: contracter dernières/premières dims
                    contracted_dims_A = [len(shape_A) - 1] if len(shape_A) > 0 else []
                    contracted_dims_B = [0] if len(shape_B) > 0 else []
                    
                    comp = complexity.compute_contraction_complexity(
                        shape_A, shape_B, contracted_dims_A, contracted_dims_B
                    )
                    
                    if comp['computation_ops'] < best_complexity:
                        best_complexity = comp['computation_ops']
                        best_i, best_j = i, j
            
            # Effectuer contraction
            order.append((current_indices[best_i], current_indices[best_j]))
            
            # Mettre à jour
            shape_A = current_tensors[best_i]
            shape_B = current_tensors[best_j]
            result_shape = complexity.compute_contraction_complexity(
                shape_A, shape_B, [len(shape_A)-1], [0]
            )['result_shape']
            
            # Remplacer i et j par résultat
            new_tensors = ([current_tensors[k] for k in range(len(current_tensors)) 
                          if k not in [best_i, best_j]] + [result_shape])
            new_indices = ([current_indices[k] for k in range(len(current_indices)) 
                          if k not in [best_i, best_j]] + [max(current_indices) + 1])
            
            current_tensors = new_tensors
            current_indices = new_indices
        
        return order
    
    def greedy_min_memory(self, tensors: List[Tuple]) -> List[Tuple]:
        """
        Algorithme glouton: minimise mémoire peak à chaque étape
        """
        complexity = ContractionComplexity()
        order = []
        current_tensors = list(tensors)
        current_indices = list(range(len(tensors)))
        
        while len(current_tensors) > 1:
            best_i, best_j = None, None
            best_memory = float('inf')
            
            for i in range(len(current_tensors)):
                for j in range(i + 1, len(current_tensors)):
                    shape_A = current_tensors[i]
                    shape_B = current_tensors[j]
                    
                    contracted_dims_A = [len(shape_A) - 1] if len(shape_A) > 0 else []
                    contracted_dims_B = [0] if len(shape_B) > 0 else []
                    
                    comp = complexity.compute_contraction_complexity(
                        shape_A, shape_B, contracted_dims_A, contracted_dims_B
                    )
                    
                    if comp['memory_peak'] < best_memory:
                        best_memory = comp['memory_peak']
                        best_i, best_j = i, j
            
            # Effectuer contraction (même logique que précédent)
            order.append((current_indices[best_i], current_indices[best_j]))
            
            # Mise à jour (simplifiée)
            result_shape = complexity.compute_contraction_complexity(
                current_tensors[best_i], current_tensors[best_j],
                [len(current_tensors[best_i])-1], [0]
            )['result_shape']
            
            current_tensors = ([current_tensors[k] for k in range(len(current_tensors)) 
                              if k not in [best_i, best_j]] + [result_shape])
            current_indices = ([current_indices[k] for k in range(len(current_indices)) 
                              if k not in [best_i, best_j]] + [max(current_indices) + 1])
        
        return order

# Test heuristique
greedy = GreedyScheduler()

tensors = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]

order_comp = greedy.greedy_min_complexity(tensors)
order_mem = greedy.greedy_min_memory(tensors)

print("\n" + "="*70)
print("Heuristiques Gloutonnes")
print("="*70)
print(f"\nOrdre (min complexité): {order_comp}")
print(f"Ordre (min mémoire): {order_mem}")
```

---

## Programmation Dynamique

### Optimal Substructure

```python
class DynamicProgrammingScheduler:
    """
    Ordonnancement optimal avec programmation dynamique
    """
    
    def __init__(self):
        self.memo = {}  # Cache pour résultats
    
    def dp_optimal_order(self, tensor_shapes: List[Tuple], 
                        objective='computation') -> Dict:
        """
        Trouve ordre optimal avec programmation dynamique
        
        Optimal substructure: ordre optimal pour sous-ensemble = partie de ordre global optimal
        """
        n = len(tensor_shapes)
        
        # DP state: (mask, remaining_tensors)
        # mask: bits indiquant quels tenseurs restent
        
        def dp_recursive(mask: int, remaining: List[int]) -> Tuple[List[Tuple], float]:
            """
            Retourne (ordre, score) optimal pour tenseurs restants
            """
            if len(remaining) == 1:
                return [], 0.0
            
            # Vérifier cache
            cache_key = (mask, tuple(remaining))
            if cache_key in self.memo:
                return self.memo[cache_key]
            
            complexity = ContractionComplexity()
            best_order = None
            best_score = float('inf')
            
            # Essayer toutes les paires
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    idx_i, idx_j = remaining[i], remaining[j]
                    shape_A = tensor_shapes[idx_i]
                    shape_B = tensor_shapes[idx_j]
                    
                    # Calculer complexité de cette contraction
                    contracted_dims_A = [len(shape_A) - 1] if len(shape_A) > 0 else []
                    contracted_dims_B = [0] if len(shape_B) > 0 else []
                    
                    comp = complexity.compute_contraction_complexity(
                        shape_A, shape_B, contracted_dims_A, contracted_dims_B
                    )
                    
                    # Score de cette étape
                    if objective == 'computation':
                        step_score = comp['computation_ops']
                    elif objective == 'memory':
                        step_score = comp['memory_peak']
                    else:
                        step_score = (comp['computation_ops'] / 1e6) + (comp['memory_peak'] / 1e6)
                    
                    # Nouveaux tenseurs restants
                    new_remaining = ([remaining[k] for k in range(len(remaining)) 
                                    if k not in [i, j]] + [n + len(remaining)])  # nouveau index
                    new_mask = mask & ~((1 << idx_i) | (1 << idx_j))
                    
                    # Récursion
                    sub_order, sub_score = dp_recursive(new_mask, new_remaining)
                    
                    total_score = step_score + sub_score
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_order = [(idx_i, idx_j)] + sub_order
            
            self.memo[cache_key] = (best_order, best_score)
            return best_order, best_score
        
        # Appel initial
        initial_mask = (1 << n) - 1  # Tous les bits à 1
        initial_remaining = list(range(n))
        
        optimal_order, optimal_score = dp_recursive(initial_mask, initial_remaining)
        
        return {
            'optimal_order': optimal_order,
            'optimal_score': optimal_score
        }

# Exemple DP (petit pour éviter explosion combinatoire)
dp_scheduler = DynamicProgrammingScheduler()

tensors_dp = [(10, 20), (20, 30), (30, 40), (40, 50)]

result_dp = dp_scheduler.dp_optimal_order(tensors_dp, objective='computation')

print("\n" + "="*70)
print("Programmation Dynamique")
print("="*70)
print(f"Ordre optimal: {result_dp['optimal_order']}")
print(f"Score optimal: {result_dp['optimal_score']:.2e}")
```

---

## Approximations et Bornes

### Bornes Théoriques

```python
class ContractionBounds:
    """
    Bornes théoriques sur complexité de contractions
    """
    
    def treewidth_bound(self, contraction_graph):
        """
        Borne basée sur treewidth du graphe de contraction
        
        La complexité est bornée par exp(treewidth)
        """
        # En pratique, calculer treewidth est difficile (NP-hard)
        # Mais donne borne théorique
        
        return {
            'bound': 'O(exp(treewidth))',
            'significance': 'Complexité minimale possible pour ce graphe'
        }
    
    def rank_bound(self, tensor_decomposition):
        """
        Borne basée sur rang tensoriel
        
        La complexité est liée aux rangs des décompositions
        """
        return {
            'bound': 'O(rank^n) pour certains réseaux',
            'significance': 'Structure du réseau limite complexité'
        }

bounds = ContractionBounds()
```

---

## Applications Pratiques

### Optimisation pour Réseaux Spécifiques

```python
class SpecificNetworkOptimization:
    """
    Optimisations pour réseaux de tenseurs spécifiques
    """
    
    def mps_optimal_order(self, bond_dimensions: List[int]):
        """
        Ordonnancement optimal pour MPS (Matrix Product State)
        
        MPS: chaîne linéaire, ordre naturel de gauche à droite (ou droite à gauche)
        """
        n = len(bond_dimensions) - 1  # n tenseurs
        
        # Ordre séquentiel est souvent optimal pour MPS
        order = [(i, i+1) for i in range(n-1)]
        
        return {
            'order': order,
            'complexity': 'O(n * d^3) où d = max bond dimension',
            'memory': 'O(d^2)'
        }
    
    def peps_heuristic_order(self, grid_shape: Tuple[int, int], bond_dims: Dict):
        """
        Heuristique pour PEPS (grille 2D)
        
        Plus complexe: contraction exacte est exponentielle
        Utiliser approximations (boundary MPS, etc.)
        """
        return {
            'strategy': 'Boundary contraction avec MPS approximation',
            'complexity': 'O(n^2 * d^4) au lieu de O(d^10)',
            'trade_off': 'Approximation vs exactitude'
        }

# Exemple MPS
specific = SpecificNetworkOptimization()

mps_bonds = [10, 20, 30, 20, 10]  # 4 tenseurs avec ces bond dimensions
mps_order = specific.mps_optimal_order(mps_bonds)

print("\n" + "="*70)
print("Optimisation pour MPS")
print("="*70)
print(f"Ordre optimal: {mps_order['order']}")
print(f"Complexité: {mps_order['complexity']}")
print(f"Mémoire: {mps_order['memory']}")
```

---

## Exercices

### Exercice 17.2.1
Trouvez l'ordonnancement optimal pour contracter 5 tenseurs avec shapes données en minimisant la complexité computationnelle.

### Exercice 17.2.2
Implémentez une heuristique gloutonne qui minimise la mémoire peak et comparez avec minimisation de complexité.

### Exercice 17.2.3
Analysez la complexité de différents ordres de contraction pour un réseau MPS à 10 tenseurs.

### Exercice 17.2.4
Développez une méthode d'approximation pour ordonnancement de PEPS qui trouve un compromis temps/mémoire.

---

## Points Clés à Retenir

> 📌 **L'ordonnancement optimal peut réduire complexité et mémoire de plusieurs ordres de grandeur**

> 📌 **Recherche exhaustive est possible seulement pour petit nombre de tenseurs (N < 10)**

> 📌 **Heuristiques gloutonnes sont rapides mais peuvent être sous-optimales**

> 📌 **Programmation dynamique trouve optimal mais coût exponentiel en nombre de tenseurs**

> 📌 **Pour réseaux spécifiques (MPS, TT), ordres optimaux sont connus**

> 📌 **Les approximations sont nécessaires pour grands réseaux (PEPS, MERA)**

---

*Section précédente : [17.1 Implémentation Efficace des Contractions](./17_01_Contractions.md) | Section suivante : [17.3 Mapping sur Architectures Parallèles](./17_03_Mapping.md)*

