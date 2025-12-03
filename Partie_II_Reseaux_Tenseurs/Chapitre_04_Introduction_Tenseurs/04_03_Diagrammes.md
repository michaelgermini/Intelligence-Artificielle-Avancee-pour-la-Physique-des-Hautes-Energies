# 4.3 Diagrammes Tensoriels

---

## Introduction

Les **diagrammes tensoriels** (tensor network diagrams) offrent une représentation visuelle intuitive des tenseurs et de leurs opérations. Cette notation graphique, issue de la physique quantique, simplifie grandement la visualisation et le raisonnement sur des structures tensorielles complexes.

---

## Principe Fondamental

### Éléments de Base

```
┌─────────────────────────────────────────────────────────────────┐
│              Éléments d'un Diagramme Tensoriel                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • Nœud (Vertex) = Tenseur                                      │
│    ──○──  (matrice M_ij)                                        │
│                                                                 │
│  • Patte (Leg/Edge) = Indice                                    │
│    ───  représente un indice du tenseur                         │
│                                                                 │
│  • Connexion = Contraction                                      │
│    ───○───○───  (deux tenseurs connectés)                      │
│                                                                 │
│  • Patte libre = Indice libre (non contracté)                  │
│    ──○───  (vecteur avec un indice libre)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Notation selon l'Ordre

```python
def visualize_tensor_diagrams():
    """
    Visualise les diagrammes pour différents ordres de tenseurs
    """
    diagrams = {
        'Ordre 0 (Scalaire)': '    ○    ',
        'Ordre 1 (Vecteur)': '   ──○    ',
        'Ordre 2 (Matrice)': '  ──○──   ',
        'Ordre 3': '     │\n    ──○──\n     │',
        'Ordre 4': '     │ │\n    ──○──\n     │ │',
    }
    
    print("Diagrammes Tensoriels par Ordre:")
    print("=" * 50)
    for name, diagram in diagrams.items():
        print(f"\n{name}:")
        print(diagram)

visualize_tensor_diagrams()
```

---

## Opérations de Base

### Produit Tensoriel (Outer Product)

```python
def outer_product_diagram():
    """
    Produit tensoriel u ⊗ v
    
    Diagramme:
      u───  v───
         │   │
         └─○─┘
           │
    Résultat: tenseur d'ordre 2
    """
    u = np.array([1, 2, 3])
    v = np.array([4, 5])
    
    # Produit tensoriel
    result = np.outer(u, v)
    
    print("Produit Tensoriel:")
    print("  Diagramme: u───○─── v")
    print(f"  u shape: {u.shape}")
    print(f"  v shape: {v.shape}")
    print(f"  u ⊗ v shape: {result.shape}")
    print(f"  Résultat:\n{result}")

outer_product_diagram()
```

### Produit Matriciel

```python
def matrix_product_diagram():
    """
    Produit matriciel A × B = C
    
    Diagramme:
      A───○───B
      │       │
      └───C───┘
    
    Notation: C_ik = Σ_j A_ij B_jk
    """
    A = np.random.randn(3, 4)
    B = np.random.randn(4, 5)
    C = A @ B
    
    print("Produit Matriciel:")
    print("  Diagramme: A───○───B = C")
    print(f"  A: {A.shape}, B: {B.shape} → C: {C.shape}")
    print(f"  Contraction sur l'indice j (dimension {A.shape[1]})")

matrix_product_diagram()
```

### Trace

```python
def trace_diagram():
    """
    Trace d'une matrice: Tr(M) = Σ_i M_ii
    
    Diagramme:
         ┌─┐
      ───┤M├───
         └─┘
    (pattes connectées en boucle)
    """
    M = np.random.randn(5, 5)
    trace = np.trace(M)
    
    print("Trace:")
    print("  Diagramme: M avec pattes connectées en boucle")
    print(f"  Tr(M) = {trace:.4f}")
    print(f"  Vérification: Σ M_ii = {np.sum(np.diag(M)):.4f}")

trace_diagram()
```

---

## Diagrammes Complexes

### Contraction Multiple

```python
def multiple_contraction_diagram():
    """
    Contraction de plusieurs tenseurs
    
    Exemple: T_ijk × M_jl × N_klm = R_ilm
    
    Diagramme:
         ┌─T─┐
         │ │ │
      ───┴─┼─┴───
          │
         ┌┴─┐
         │M │
      ───┴─┬───
          │
         ┌┴─┐
         │N │
      ───┴──┴───
    """
    T = np.random.randn(3, 4, 5)
    M = np.random.randn(4, 6)
    N = np.random.randn(5, 6, 7)
    
    # Contraction: T_ijk, M_jl, N_klm → R_ilm
    # Contracte j entre T et M
    # Contracte k entre T et N
    # Contracte l entre M et N
    
    # Utilise einsum
    R = np.einsum('ijk,jl,klm->ilm', T, M, N)
    
    print("Contraction Multiple:")
    print(f"  T: {T.shape}, M: {M.shape}, N: {N.shape}")
    print(f"  Résultat R: {R.shape}")
    print(f"  Contractions: j (T↔M), k (T↔N), l (M↔N)")

multiple_contraction_diagram()
```

---

## Implémentation d'un Système de Diagrammes

```python
class TensorDiagram:
    """
    Système complet pour représenter et évaluer des diagrammes tensoriels
    """
    
    def __init__(self):
        self.tensors = {}  # name -> (tensor, labels)
        self.contractions = []  # list of (name1, labels1, name2, labels2, common_labels)
        self.output_labels = []  # labels du tenseur de sortie
    
    def add_tensor(self, name, tensor, labels):
        """
        Ajoute un tenseur avec ses labels d'indices
        
        Args:
            name: Nom du tenseur
            tensor: Array NumPy
            labels: List de strings, une par dimension
                   Ex: ['i', 'j'] pour une matrice M_ij
        """
        assert tensor.ndim == len(labels), \
            f"Nombre de labels ({len(labels)}) doit égaler l'ordre ({tensor.ndim})"
        
        assert len(set(labels)) == len(labels), \
            f"Labels doivent être uniques: {labels}"
        
        self.tensors[name] = (tensor, labels)
    
    def contract(self, name1, name2, common_labels):
        """
        Spécifie une contraction entre deux tenseurs
        
        Args:
            name1, name2: Noms des tenseurs
            common_labels: Liste des labels communs à contracter
        """
        tensor1, labels1 = self.tensors[name1]
        tensor2, labels2 = self.tensors[name2]
        
        # Vérifie que les labels communs existent
        for label in common_labels:
            assert label in labels1 and label in labels2, \
                f"Label {label} doit être dans les deux tenseurs"
        
        self.contractions.append((name1, name2, common_labels))
    
    def evaluate(self):
        """
        Évalue le diagramme en contractant tous les tenseurs
        
        Utilise np.einsum pour une évaluation efficace
        """
        if len(self.tensors) == 0:
            return None
        
        if len(self.tensors) == 1:
            name = list(self.tensors.keys())[0]
            tensor, labels = self.tensors[name]
            return tensor
        
        # Construit l'expression einsum
        # Format: 'ij,jk->ik' pour A_ij @ B_jk = C_ik
        
        # Collecte tous les labels uniques
        all_labels = set()
        for tensor, labels in self.tensors.values():
            all_labels.update(labels)
        
        # Détermine les labels libres (non contractés)
        contracted_labels = set()
        for _, _, common in self.contractions:
            contracted_labels.update(common)
        
        free_labels = sorted(all_labels - contracted_labels)
        
        # Construit la chaîne einsum
        input_expr = ','.join(
            ''.join(labels) for _, labels in self.tensors.values()
        )
        output_expr = ''.join(free_labels)
        einsum_expr = f'{input_expr}->{output_expr}'
        
        # Évalue
        tensors = [tensor for tensor, _ in self.tensors.values()]
        result = np.einsum(einsum_expr, *tensors)
        
        return result
    
    def visualize(self):
        """
        Visualise le diagramme en ASCII
        """
        print("Diagramme Tensoriel:")
        print("-" * 50)
        
        for name, (tensor, labels) in self.tensors.items():
            print(f"  {name}: shape {tensor.shape}, labels {labels}")
        
        for name1, name2, common in self.contractions:
            print(f"  Contraction: {name1} <--{common}--> {name2}")

# Exemple: Produit matriciel
diagram = TensorDiagram()
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

diagram.add_tensor('A', A, ['i', 'j'])
diagram.add_tensor('B', B, ['j', 'k'])
diagram.contract('A', 'B', ['j'])

diagram.visualize()
result = diagram.evaluate()
print(f"\nRésultat: {result.shape}")
print(f"Vérification avec @: {(A @ B).shape}")
```

---

## Diagrammes dans les Réseaux de Neurones

### Couche Dense

```python
def dense_layer_diagram():
    """
    Couche dense: y = W @ x + b
    
    Diagramme:
      x───○───W
      │       │
      └───y───┘
          ↑
          b (addition)
    """
    batch_size = 32
    in_features = 256
    out_features = 128
    
    x = np.random.randn(batch_size, in_features)
    W = np.random.randn(in_features, out_features)
    b = np.random.randn(out_features)
    
    # Forward: y = x @ W + b
    y = x @ W + b
    
    print("Couche Dense:")
    print(f"  x: (batch={batch_size}, in={in_features})")
    print(f"  W: (in={in_features}, out={out_features})")
    print(f"  Contraction sur 'in' → y: (batch={batch_size}, out={out_features})")

dense_layer_diagram()
```

### Couche Convolutionnelle

```python
def conv_layer_diagram():
    """
    Convolution 2D
    
    Diagramme:
      Input: (batch, channels, height, width)
      Kernel: (out_ch, in_ch, kH, kW)
      
      Input───○───Kernel
       │ │     │ │
       └─┴─────┴─┘
         │
        Output
    """
    batch, in_ch, H, W = 8, 3, 32, 32
    out_ch, kH, kW = 16, 3, 3
    
    input_tensor = np.random.randn(batch, in_ch, H, W)
    kernel = np.random.randn(out_ch, in_ch, kH, kW)
    
    print("Couche Convolutionnelle:")
    print(f"  Input: (batch={batch}, in_ch={in_ch}, H={H}, W={W})")
    print(f"  Kernel: (out_ch={out_ch}, in_ch={in_ch}, kH={kH}, kW={kW})")
    print(f"  Contractions: in_ch, kH, kW")
    print(f"  Output: (batch={batch}, out_ch={out_ch}, H'={H-kH+1}, W'={W-kW+1})")

conv_layer_diagram()
```

### Attention Mechanism

```python
def attention_diagram():
    """
    Attention: Attention(Q, K, V) = softmax(QK^T / √d) V
    
    Diagramme:
      Q───○───K^T
      │       │
      └───S───┘
          │
      ────○───V
          │
        Output
    """
    batch, seq_len, d_model = 4, 128, 512
    
    Q = np.random.randn(batch, seq_len, d_model)
    K = np.random.randn(batch, seq_len, d_model)
    V = np.random.randn(batch, seq_len, d_model)
    
    # Attention
    scores = np.einsum('bsd,btd->bst', Q, K) / np.sqrt(d_model)
    attention_weights = softmax(scores, axis=-1)
    output = np.einsum('bst,btd->bsd', attention_weights, V)
    
    print("Attention Mechanism:")
    print(f"  Q, K, V: (batch={batch}, seq={seq_len}, d={d_model})")
    print(f"  Scores S: (batch={batch}, seq_q={seq_len}, seq_k={seq_len})")
    print(f"  Contractions: Q×K^T sur 'd', puis S×V sur 'seq_k'")

def softmax(x, axis=-1):
    """Softmax simplifié"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attention_diagram()
```

---

## Optimisation de Diagrammes

### Réduction de Complexité

```python
def optimal_contraction_order(diagram):
    """
    Trouve l'ordre optimal de contraction pour minimiser la complexité
    
    Complexité d'une contraction = produit de toutes les dimensions
    """
    # Algorithme simplifié: essaie différents ordres et garde le meilleur
    # En pratique, utilise un algorithme de programmation dynamique
    
    best_cost = float('inf')
    best_order = None
    
    # Génère toutes les permutations possibles
    from itertools import permutations
    
    tensor_names = list(diagram.tensors.keys())
    
    if len(tensor_names) <= 2:
        return tensor_names  # Un seul ordre possible
    
    for order in permutations(tensor_names[1:], len(tensor_names) - 1):
        order = [tensor_names[0]] + list(order)
        
        # Calcule le coût de contraction dans cet ordre
        cost = compute_contraction_cost(diagram, order)
        
        if cost < best_cost:
            best_cost = cost
            best_order = order
    
    return best_order, best_cost

def compute_contraction_cost(diagram, order):
    """Calcule le coût total pour un ordre de contraction donné"""
    # Implémentation simplifiée
    # En pratique, calcule la taille des tenseurs intermédiaires
    cost = 0
    current_tensors = {name: diagram.tensors[name][0] 
                      for name in order}
    
    # Contracte dans l'ordre donné
    for i in range(len(order) - 1):
        name1, name2 = order[i], order[i+1]
        tensor1 = current_tensors[name1]
        tensor2 = current_tensors[name2]
        
        # Coût = produit de toutes les dimensions
        cost += tensor1.size * tensor2.size
        
        # Simule la contraction (simplifié)
        # En pratique, calcule la taille exacte du résultat
    
    return cost
```

---

## Visualisation avec ASCII Art

```python
def draw_tensor_diagram_ascii(diagram):
    """
    Dessine un diagramme tensoriel en ASCII
    """
    lines = []
    
    # Dessine les tenseurs
    for i, (name, (tensor, labels)) in enumerate(diagram.tensors.items()):
        order = tensor.ndim
        node = '○'
        
        if order == 0:
            lines.append(f"    {name}: {node}")
        elif order == 1:
            lines.append(f"    {name}: ──{node}──")
        elif order == 2:
            lines.append(f"    {name}: ──{node}──")
        else:
            # Dessine plusieurs pattes
            top_pattes = '─' * (order // 2) + node
            bottom_pattes = '─' * ((order - 1) // 2)
            lines.append(f"    {name}: {top_pattes}──")
            if order > 2:
                lines.append(f"         {'│ ' * ((order - 1) // 2)}")
    
    # Dessine les contractions
    for name1, name2, common in diagram.contractions:
        lines.append(f"    Contraction: {name1} ←{common}→ {name2}")
    
    return '\n'.join(lines)

# Exemple
diagram = TensorDiagram()
diagram.add_tensor('A', np.random.randn(3, 4), ['i', 'j'])
diagram.add_tensor('B', np.random.randn(4, 5), ['j', 'k'])
diagram.contract('A', 'B', ['j'])

print(draw_tensor_diagram_ascii(diagram))
```

---

## Applications Avancées

### Réseaux de Tenseurs (Tensor Networks)

```python
def tensor_network_diagram():
    """
    Exemple de réseau de tenseurs: MPS (Matrix Product State)
    
    Diagramme:
      ──○───○───○───○───
      │ │   │ │   │ │   │ │
      
    Chaque ○ est un tenseur 3D connecté à ses voisins
    """
    # MPS pour un tenseur d'ordre 4
    cores = [
        np.random.randn(1, 5, 3),   # Core 1: (r0, d1, r1)
        np.random.randn(3, 6, 4),   # Core 2: (r1, d2, r2)
        np.random.randn(4, 7, 2),   # Core 3: (r2, d3, r3)
        np.random.randn(2, 8, 1)    # Core 4: (r3, d4, r4)
    ]
    
    print("Matrix Product State (MPS):")
    print("  Diagramme: ──○───○───○───○───")
    for i, core in enumerate(cores):
        print(f"  Core {i+1}: shape {core.shape}")
    
    # Reconstruction
    result = cores[0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    result = result.squeeze()
    
    print(f"  Tenseur reconstruit: shape {result.shape}")

tensor_network_diagram()
```

---

## Exercices

### Exercice 4.3.1
Dessinez le diagramme tensoriel pour une convolution 3D avec batch.

### Exercice 4.3.2
Implémentez une fonction qui convertit un diagramme tensoriel en expression einsum.

### Exercice 4.3.3
Visualisez le diagramme d'un Transformer complet (multi-head attention + feed-forward).

---

## Points Clés à Retenir

> 📌 **Les diagrammes tensoriels visualisent les tenseurs et leurs contractions**

> 📌 **Un nœud = tenseur, une patte = indice, connexion = contraction**

> 📌 **La notation diagrammatique simplifie le raisonnement sur les structures complexes**

> 📌 **Les diagrammes sont essentiels pour comprendre les réseaux de tenseurs**

---

*Section suivante : [4.4 Complexité Computationnelle](./04_04_Complexite.md)*

