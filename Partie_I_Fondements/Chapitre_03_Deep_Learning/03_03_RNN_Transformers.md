# 3.3 Réseaux Récurrents et Transformers

---

## Introduction

Les **réseaux récurrents** (RNN) et les **Transformers** sont conçus pour traiter des données séquentielles. Alors que les RNN traitent les séquences élément par élément avec une mémoire interne, les Transformers utilisent des mécanismes d'attention pour capturer les dépendances à longue portée.

---

## Réseaux Récurrents (RNN)

### Architecture de Base

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    """
    RNN simple (vanilla RNN)
    
    h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Poids
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        """
        x: (batch, seq_len, input_size)
        h: (batch, hidden_size) - état caché initial
        """
        batch_size, seq_len, _ = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h))
            outputs.append(h)
        
        # Stack outputs: (batch, seq_len, hidden_size)
        outputs = torch.stack(outputs, dim=1)
        
        # Sortie finale
        y = self.W_hy(outputs)
        
        return y, h

# Test
rnn = SimpleRNN(input_size=10, hidden_size=32, output_size=5)
x = torch.randn(4, 20, 10)  # Batch=4, Seq=20, Features=10
y, h_final = rnn(x)
print(f"Input: {x.shape} → Output: {y.shape}, Hidden: {h_final.shape}")
```

### LSTM (Long Short-Term Memory)

```python
class LSTMCell(nn.Module):
    """
    Cellule LSTM manuelle pour comprendre le mécanisme
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Gates combinées pour efficacité
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        
    def forward(self, x, state=None):
        """
        x: (batch, input_size)
        state: (h, c) chacun (batch, hidden_size)
        """
        batch_size = x.size(0)
        
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = state
        
        # Concatène input et hidden
        combined = torch.cat([x, h], dim=1)
        
        # Calcule les 4 gates en une fois
        gates = self.gates(combined)
        
        # Split en 4 parties
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Mise à jour de l'état
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


class LSTM(nn.Module):
    """
    LSTM complet pour séquences
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Utilise le LSTM de PyTorch pour l'efficacité
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x, state=None):
        """
        x: (batch, seq_len, input_size)
        """
        output, (h_n, c_n) = self.lstm(x, state)
        return output, (h_n, c_n)

# Comparaison des paramètres
input_size, hidden_size = 64, 128

rnn_simple = SimpleRNN(input_size, hidden_size, 10)
lstm = LSTM(input_size, hidden_size)

print(f"RNN simple: {sum(p.numel() for p in rnn_simple.parameters()):,} params")
print(f"LSTM: {sum(p.numel() for p in lstm.parameters()):,} params")
print(f"Ratio LSTM/RNN: {sum(p.numel() for p in lstm.parameters()) / sum(p.numel() for p in rnn_simple.parameters()):.2f}x")
```

---

## Mécanisme d'Attention

### Attention de Base

```python
class Attention(nn.Module):
    """
    Mécanisme d'attention scaled dot-product
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_q, d_model)
        key: (batch, seq_k, d_model)
        value: (batch, seq_k, d_model)
        mask: (batch, seq_q, seq_k) - optionnel
        """
        # Scores d'attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax pour obtenir les poids
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Weighted sum des values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Permet au modèle d'apprendre différents types de relations
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projections linéaires
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Projections linéaires
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape pour multi-head: (batch, n_heads, seq, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Appliquer aux values
        context = torch.matmul(attention, V)
        
        # Concat heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Projection finale
        output = self.W_o(context)
        
        return output, attention

# Test
mha = MultiHeadAttention(d_model=256, n_heads=8)
x = torch.randn(4, 20, 256)  # Batch=4, Seq=20, d_model=256
out, attn = mha(x, x, x)
print(f"MHA Output: {out.shape}, Attention: {attn.shape}")
print(f"MHA params: {sum(p.numel() for p in mha.parameters()):,}")
```

---

## Architecture Transformer

### Bloc Transformer

```python
class TransformerBlock(nn.Module):
    """
    Bloc Transformer: Attention + FFN avec connexions résiduelles
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention avec connexion résiduelle
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN avec connexion résiduelle
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class Transformer(nn.Module):
    """
    Transformer encoder complet
    """
    
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, 
                 max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        return x

# Configuration type BERT-base
transformer = Transformer(
    vocab_size=30000,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    n_layers=12
)

print(f"Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
```

---

## Applications en Physique des Particules

### Transformer pour Jets

```python
class ParticleTransformer(nn.Module):
    """
    Transformer pour classification de jets
    
    Traite les constituants du jet comme une séquence
    """
    
    def __init__(self, n_features=4, d_model=128, n_heads=4, 
                 n_layers=4, n_classes=5, max_particles=100):
        super().__init__()
        
        # Projection des features de particules
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Embedding de position (optionnel pour les jets)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_particles, d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes)
        )
        
        # Token [CLS] pour la classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, particles, mask=None):
        """
        particles: (batch, n_particles, n_features)
                   features = [pt, eta, phi, mass] ou similaire
        mask: (batch, n_particles) - 1 pour particules valides, 0 pour padding
        """
        batch_size, n_particles, _ = particles.shape
        
        # Projection
        x = self.input_proj(particles)
        
        # Ajoute position embedding
        x = x + self.pos_embedding[:, :n_particles, :]
        
        # Ajoute CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Ajuste le mask pour le CLS token
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer
        for block in self.blocks:
            x = block(x, mask)
        
        # Classification basée sur le CLS token
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits

# Test
particle_transformer = ParticleTransformer(
    n_features=4,
    d_model=128,
    n_heads=4,
    n_layers=4,
    n_classes=5
)

# Simule un batch de jets avec 50 constituants
particles = torch.randn(8, 50, 4)  # (batch, particles, features)
mask = torch.ones(8, 50)  # Toutes les particules sont valides

output = particle_transformer(particles, mask)
print(f"Output shape: {output.shape}")
print(f"ParticleTransformer params: {sum(p.numel() for p in particle_transformer.parameters()):,}")
```

### Attention pour Tracking

```python
class TrackingTransformer(nn.Module):
    """
    Transformer pour la reconstruction de traces
    
    Prédit les associations entre hits
    """
    
    def __init__(self, hit_features=3, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        
        self.hit_encoder = nn.Linear(hit_features, d_model)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 2)
            for _ in range(n_layers)
        ])
        
        # Prédiction d'association (hit_i connecté à hit_j ?)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hits):
        """
        hits: (batch, n_hits, hit_features)
               features = [x, y, z] ou [r, phi, z]
        
        Returns:
            edge_scores: (batch, n_hits, n_hits) probabilité de connexion
        """
        batch_size, n_hits, _ = hits.shape
        
        # Encode les hits
        x = self.hit_encoder(hits)
        
        # Transformer
        for block in self.encoder_blocks:
            x = block(x)
        
        # Prédiction des edges (toutes les paires)
        # Expand pour toutes les paires (i, j)
        x_i = x.unsqueeze(2).expand(-1, -1, n_hits, -1)
        x_j = x.unsqueeze(1).expand(-1, n_hits, -1, -1)
        
        # Concatène les représentations
        pair_features = torch.cat([x_i, x_j], dim=-1)
        
        # Prédiction
        edge_scores = self.edge_predictor(pair_features).squeeze(-1)
        
        return edge_scores

tracking_transformer = TrackingTransformer()
hits = torch.randn(4, 100, 3)  # 100 hits avec coordonnées (x, y, z)
edge_scores = tracking_transformer(hits)
print(f"Edge scores shape: {edge_scores.shape}")
```

---

## Compression des Transformers

### Analyse de la Complexité

```python
def transformer_complexity(d_model, n_heads, d_ff, n_layers, seq_len, batch_size=1):
    """
    Analyse la complexité d'un Transformer
    """
    # Paramètres par bloc
    # Attention: 4 * d_model² (Q, K, V, O projections)
    attn_params = 4 * d_model * d_model
    
    # FFN: d_model * d_ff + d_ff * d_model
    ffn_params = 2 * d_model * d_ff
    
    # LayerNorm: 2 * d_model (scale + shift) × 2
    norm_params = 4 * d_model
    
    params_per_block = attn_params + ffn_params + norm_params
    total_params = n_layers * params_per_block
    
    # FLOPs par bloc
    # Attention: O(seq_len² * d_model)
    attn_flops = 4 * seq_len * d_model * d_model  # Projections
    attn_flops += 2 * seq_len * seq_len * d_model  # QK^T et attention @ V
    
    # FFN: O(seq_len * d_model * d_ff)
    ffn_flops = 2 * seq_len * d_model * d_ff + 2 * seq_len * d_ff * d_model
    
    flops_per_block = attn_flops + ffn_flops
    total_flops = n_layers * flops_per_block * batch_size
    
    # Mémoire pour les activations (attention matrix)
    attn_memory = n_heads * seq_len * seq_len * 4  # float32
    
    print("Analyse de complexité Transformer")
    print("=" * 50)
    print(f"Configuration: d={d_model}, h={n_heads}, ff={d_ff}, L={n_layers}")
    print(f"Séquence: {seq_len}, Batch: {batch_size}")
    print(f"\nParamètres:")
    print(f"  Par bloc: {params_per_block:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nFLOPs:")
    print(f"  Attention: {attn_flops:,}")
    print(f"  FFN: {ffn_flops:,}")
    print(f"  Total: {total_flops:,}")
    print(f"\nMémoire attention (par bloc): {attn_memory / 1024**2:.2f} MB")
    
    return total_params, total_flops

# BERT-base
transformer_complexity(d_model=768, n_heads=12, d_ff=3072, n_layers=12, seq_len=512)
```

### Techniques de Compression

```python
class EfficientAttention(nn.Module):
    """
    Attention linéaire (approximation)
    
    Réduit la complexité de O(n²) à O(n)
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Feature map pour linéarisation
        self.feature_map = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.ELU(),
        )
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Applique feature map
        Q = self.feature_map(Q) + 1  # +1 pour assurer positivité
        K = self.feature_map(K) + 1
        
        # Attention linéaire: (Q @ K^T) @ V = Q @ (K^T @ V)
        # Calcule K^T @ V d'abord: O(n * d²)
        KV = torch.einsum('bshd,bshm->bhdm', K, V)
        
        # Puis Q @ (K^T @ V): O(n * d²)
        output = torch.einsum('bshd,bhdm->bshm', Q, KV)
        
        # Normalisation
        K_sum = K.sum(dim=1, keepdim=True)
        normalizer = torch.einsum('bshd,bthd->bsh', Q, K_sum)
        output = output / (normalizer.unsqueeze(-1) + 1e-6)
        
        output = output.reshape(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, None

# Comparaison
d_model, n_heads, seq_len = 256, 8, 1000

standard_attn = MultiHeadAttention(d_model, n_heads)
efficient_attn = EfficientAttention(d_model, n_heads)

x = torch.randn(4, seq_len, d_model)

import time

# Standard attention
start = time.time()
for _ in range(10):
    _ = standard_attn(x, x, x)
standard_time = time.time() - start

# Efficient attention
start = time.time()
for _ in range(10):
    _ = efficient_attn(x, x, x)
efficient_time = time.time() - start

print(f"Temps attention standard: {standard_time:.3f}s")
print(f"Temps attention efficace: {efficient_time:.3f}s")
print(f"Accélération: {standard_time / efficient_time:.2f}x")
```

---

## Exercices

### Exercice 3.3.1
Calculez la complexité mémoire de l'attention pour une séquence de 4096 tokens avec 12 heads et d_model=768.

### Exercice 3.3.2
Implémentez une version "sparse attention" qui ne calcule l'attention que pour les k plus proches voisins.

### Exercice 3.3.3
Créez un Transformer pour prédire les propriétés des jets (masse, pt) à partir de leurs constituants.

---

## Points Clés à Retenir

> 📌 **Les LSTM résolvent le problème du vanishing gradient des RNN simples**

> 📌 **L'attention permet de capturer des dépendances à longue portée en O(n²)**

> 📌 **Les Transformers dominent le NLP mais sont coûteux en mémoire**

> 📌 **L'attention linéaire réduit la complexité à O(n) mais perd en expressivité**

---

## Références

1. Hochreiter, S., Schmidhuber, J. "Long Short-Term Memory." Neural Computation, 1997
2. Vaswani, A. et al. "Attention Is All You Need." NeurIPS, 2017
3. Qu, H. et al. "ParticleNet: Jet Tagging via Particle Clouds." Phys. Rev. D, 2020
4. Katharopoulos, A. et al. "Transformers are RNNs." ICML, 2020

---

*Section suivante : [3.4 Fonctions de Perte et Optimisation](./03_04_Optimisation.md)*

