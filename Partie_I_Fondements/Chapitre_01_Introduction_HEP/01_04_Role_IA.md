# 1.4 Rôle de l'Intelligence Artificielle dans la Recherche Fondamentale

---

## Introduction

L'intelligence artificielle est devenue un pilier incontournable de la physique des hautes énergies. Des premiers réseaux de neurones utilisés dans les années 1990 aux architectures deep learning modernes, l'IA a transformé chaque aspect de l'analyse des données au LHC.

---

## Historique de l'IA en Physique des Particules

### Chronologie

```
┌─────────────────────────────────────────────────────────────────┐
│              Évolution de l'IA en HEP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1990s  │ Premiers MLP pour identification de particules        │
│         │ Réseaux à 1-2 couches cachées                        │
│         │                                                       │
│  2000s  │ BDT (Boosted Decision Trees) dominent                │
│         │ TMVA devient le standard                              │
│         │                                                       │
│  2012   │ Révolution deep learning (ImageNet)                  │
│         │ Début de l'adoption en HEP                           │
│         │                                                       │
│  2015+  │ CNN pour images de calorimètres                      │
│         │ RNN pour séquences de hits                           │
│         │                                                       │
│  2018+  │ Graph Neural Networks pour détecteurs                │
│         │ Attention mechanisms                                  │
│         │                                                       │
│  2020+  │ Transformers, Foundation Models                      │
│         │ IA générative pour simulation                        │
│         │ Déploiement FPGA avec hls4ml                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Exemple : Évolution du B-Tagging

```python
# Évolution des performances de b-tagging au fil des années
btag_evolution = {
    '1990s - Cuts': {
        'method': 'Coupures séquentielles',
        'b_efficiency': 0.50,
        'light_rejection': 10,
        'variables': 3
    },
    '2000s - BDT': {
        'method': 'Boosted Decision Trees',
        'b_efficiency': 0.70,
        'light_rejection': 100,
        'variables': 20
    },
    '2015 - DNN': {
        'method': 'Deep Neural Network',
        'b_efficiency': 0.77,
        'light_rejection': 300,
        'variables': 50
    },
    '2020 - GNN': {
        'method': 'Graph Neural Network',
        'b_efficiency': 0.82,
        'light_rejection': 1000,
        'variables': 'Tous les constituants'
    }
}

for era, perf in btag_evolution.items():
    print(f"\n{era}")
    print(f"  Méthode: {perf['method']}")
    print(f"  Efficacité b: {perf['b_efficiency']:.0%}")
    print(f"  Rejet light: {perf['light_rejection']}x")
```

---

## Applications Principales de l'IA en HEP

### 1. Identification de Particules

```python
import torch
import torch.nn as nn

class ParticleIdentifier(nn.Module):
    """
    Réseau de classification de particules
    
    Classes: électron, muon, photon, pion, kaon, proton
    """
    
    def __init__(self, input_features=32, n_classes=6):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

# Features typiques pour l'identification
particle_features = [
    'p', 'pt', 'eta', 'phi',           # Cinématique
    'dE/dx',                            # Perte d'énergie
    'E/p',                              # Rapport énergie/impulsion
    'shower_shape_1', 'shower_shape_2', # Forme de gerbe
    'track_chi2', 'n_hits',             # Qualité de trace
    'isolation',                         # Isolation
    # ... autres variables
]
```

### 2. Reconstruction de Jets

```python
class JetClassifier(nn.Module):
    """
    Classification de jets : quark vs gluon, b vs light, etc.
    Utilise les constituants du jet comme entrée
    """
    
    def __init__(self, n_constituents=50, n_features=4):
        super().__init__()
        
        # Encodeur par constituant
        self.constituent_encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Agrégation (attention simplifiée)
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # binaire : b vs light
        )
        
    def forward(self, constituents):
        """
        Args:
            constituents: [batch, n_constituents, n_features]
                         features = [pt, eta, phi, mass] par constituant
        """
        # Encode chaque constituant
        encoded = self.constituent_encoder(constituents)  # [B, N, 64]
        
        # Calcul des poids d'attention
        attn_weights = torch.softmax(self.attention(encoded), dim=1)  # [B, N, 1]
        
        # Agrégation pondérée
        jet_repr = (encoded * attn_weights).sum(dim=1)  # [B, 64]
        
        # Classification
        return self.classifier(jet_repr)
```

### 3. Reconstruction de Traces (Tracking)

```python
class TrackingGNN(nn.Module):
    """
    Graph Neural Network pour la reconstruction de traces
    
    Les hits du détecteur forment les nœuds
    Les arêtes potentielles connectent des hits compatibles
    """
    
    def __init__(self, node_features=3, hidden_dim=64):
        super().__init__()
        
        # Encodeur de nœuds (hits)
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        
        # Message passing layers
        self.edge_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.node_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classificateur d'arêtes (cette arête est-elle vraie ?)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: [n_nodes, node_features] - coordonnées des hits
            edge_index: [2, n_edges] - paires de nœuds connectés
        """
        # Encode les nœuds
        x = self.node_encoder(node_features)
        
        # Message passing
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_messages = self.edge_network(edge_features)
        
        # Agrégation des messages
        aggregated = scatter_mean(edge_messages, dst, dim=0)
        x = self.node_network(torch.cat([x, aggregated], dim=-1))
        
        # Classification des arêtes
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_scores = self.edge_classifier(edge_features)
        
        return edge_scores
```

### 4. Simulation Rapide (Fast Simulation)

```python
class CaloGAN(nn.Module):
    """
    GAN pour la simulation rapide de calorimètres
    Génère des images de gerbes électromagnétiques
    """
    
    def __init__(self, latent_dim=64, energy_dim=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Générateur conditionnel (conditionné par l'énergie)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + energy_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 30 * 30 * 30),  # Calorimètre 3D
            nn.Sigmoid()
        )
        
        # Discriminateur
        self.discriminator = nn.Sequential(
            nn.Linear(30 * 30 * 30 + energy_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def generate(self, energy, n_samples=1):
        """Génère des gerbes de calorimètre"""
        z = torch.randn(n_samples, self.latent_dim)
        condition = energy.view(-1, 1).expand(n_samples, 1)
        
        input_vec = torch.cat([z, condition], dim=-1)
        shower = self.generator(input_vec)
        
        return shower.view(n_samples, 30, 30, 30)
```

---

## Avantages de l'IA en HEP

### 1. Performance Supérieure

```python
# Comparaison de performance typique
comparison = {
    'Tâche': ['B-tagging', 'Tau ID', 'Pile-up mitigation', 'Tracking'],
    'Méthode classique': ['BDT', 'Cuts', 'PUPPI', 'Kalman'],
    'Méthode IA': ['GNN', 'DNN', 'PUPPI-ML', 'GNN'],
    'Amélioration': ['+40%', '+25%', '+15%', '+30%']
}

import pandas as pd
df = pd.DataFrame(comparison)
print(df.to_string(index=False))
```

### 2. Automatisation

- **Feature engineering automatique** : Les réseaux profonds apprennent les représentations
- **Optimisation end-to-end** : Entraînement différentiable de bout en bout
- **Adaptation** : Fine-tuning pour nouvelles conditions

### 3. Scalabilité

```python
# L'IA scale mieux avec la complexité
def classical_complexity(pile_up):
    """Complexité algorithmique classique"""
    return pile_up ** 2  # Souvent quadratique

def ml_complexity(pile_up):
    """Complexité avec ML bien conçu"""
    return pile_up * np.log(pile_up)  # Quasi-linéaire

pile_ups = [50, 100, 150, 200]
print("Pile-up | Classique | ML")
for pu in pile_ups:
    print(f"{pu:7} | {classical_complexity(pu):9} | {ml_complexity(pu):.0f}")
```

---

## Défis et Limitations

### 1. Interprétabilité

```python
class InterpretableClassifier(nn.Module):
    """
    Classificateur avec mécanisme d'attention interprétable
    """
    
    def __init__(self, n_features):
        super().__init__()
        
        # Réseau d'attention sur les features
        self.feature_attention = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Linear(n_features, 2)
        
    def forward(self, x):
        # Calcul des importances de features
        attention = self.feature_attention(x)
        
        # Pondération des features
        weighted_x = x * attention
        
        return self.classifier(weighted_x), attention
    
    def explain(self, x, feature_names):
        """Explique la prédiction"""
        _, attention = self.forward(x)
        
        # Top features
        top_idx = attention.argsort(descending=True)[:5]
        
        explanation = []
        for idx in top_idx:
            explanation.append({
                'feature': feature_names[idx],
                'importance': attention[idx].item()
            })
        
        return explanation
```

### 2. Incertitudes

```python
class BayesianClassifier(nn.Module):
    """
    Classificateur bayésien avec estimation d'incertitude
    """
    
    def __init__(self, n_features, n_classes, dropout_rate=0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, n_classes)
        )
        
    def forward(self, x, n_samples=1):
        if n_samples == 1:
            return self.network(x)
        
        # Monte Carlo Dropout
        self.train()  # Active dropout
        predictions = torch.stack([
            self.network(x) for _ in range(n_samples)
        ])
        
        return predictions
    
    def predict_with_uncertainty(self, x, n_samples=100):
        predictions = self.forward(x, n_samples)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
```

### 3. Robustesse

```python
def check_robustness(model, x, epsilon=0.01):
    """
    Vérifie la robustesse aux perturbations
    """
    original_pred = model(x)
    
    # Perturbation aléatoire
    noise = torch.randn_like(x) * epsilon
    perturbed_pred = model(x + noise)
    
    # Mesure de stabilité
    stability = (original_pred.argmax() == perturbed_pred.argmax()).float().mean()
    
    return stability
```

---

## Tendances Actuelles

### 1. Foundation Models pour la Physique

```python
class PhysicsFoundationModel:
    """
    Concept de modèle fondation pré-entraîné pour la physique
    """
    
    def __init__(self):
        self.encoder = TransformerEncoder(...)  # Pré-entraîné
        self.task_heads = {}
        
    def add_task_head(self, task_name, head):
        """Ajoute une tête pour une tâche spécifique"""
        self.task_heads[task_name] = head
        
    def fine_tune(self, task_name, data, labels):
        """Fine-tune pour une tâche spécifique"""
        # Gèle l'encodeur, entraîne seulement la tête
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Entraîne la tête de tâche
        train_head(self.task_heads[task_name], data, labels)
```

### 2. IA Générative pour la Simulation

- **Diffusion Models** : Génération de haute qualité
- **Normalizing Flows** : Densités exactes
- **Score-based Models** : Flexibilité architecturale

### 3. Quantum Machine Learning

```python
# Concept de circuit quantique variationnel
class QuantumClassifier:
    """
    Classificateur hybride classique-quantique (conceptuel)
    """
    
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = initialize_quantum_params(n_qubits, n_layers)
        
    def quantum_circuit(self, x, params):
        """Exécute le circuit quantique"""
        # Encoding des données
        state = encode_data(x, self.n_qubits)
        
        # Couches variationnelles
        for layer in range(self.n_layers):
            state = apply_variational_layer(state, params[layer])
            
        # Mesure
        return measure(state)
```

---

## Impact sur la Découverte Scientifique

### Découverte du Boson de Higgs (2012)

L'IA a joué un rôle crucial :

1. **Sélection des événements** : BDT pour le canal H→ττ
2. **Reconstruction de masse** : Amélioration de la résolution
3. **Séparation signal/bruit** : Facteur 2-3 d'amélioration

### Recherche de Nouvelle Physique

```python
class NewPhysicsSearch:
    """
    Framework de recherche de nouvelle physique avec IA
    """
    
    def __init__(self):
        self.anomaly_detector = AutoEncoder(...)
        self.classifier = None  # Entraîné si signal trouvé
        
    def blind_search(self, data):
        """Recherche aveugle d'anomalies"""
        # Entraîne sur données de contrôle
        self.anomaly_detector.fit(control_region_data)
        
        # Cherche des anomalies dans la région signal
        anomaly_scores = self.anomaly_detector.score(data)
        
        return anomaly_scores
    
    def supervised_search(self, data, signal_model):
        """Recherche supervisée avec modèle de signal"""
        # Génère du signal simulé
        signal_mc = generate_signal(signal_model)
        
        # Entraîne classificateur
        self.classifier = train_classifier(data, signal_mc)
        
        return self.classifier.predict(data)
```

---

## Exercices

### Exercice 1.4.1
Implémentez un simple réseau de neurones pour classifier des jets en deux catégories (quark vs gluon) en utilisant 5 variables : multiplicité, largeur, pt_D, LHA, et masse.

### Exercice 1.4.2
Calculez le nombre de paramètres d'un réseau fully-connected avec architecture [32, 64, 32, 2] (entrée, couches cachées, sortie).

### Exercice 1.4.3
Un modèle de b-tagging améliore le rejet des jets légers de 100 à 300 à efficacité fixe de 70%. Si le rapport signal/bruit initial est de 1:1000, quel est le nouveau rapport après application du tagger ?

---

## Points Clés à Retenir

> 📌 **L'IA a révolutionné chaque aspect de l'analyse HEP depuis 2012**

> 📌 **Les GNN sont particulièrement adaptés à la structure des données de détecteur**

> 📌 **L'interprétabilité et la quantification des incertitudes restent des défis**

> 📌 **Les tendances actuelles incluent les foundation models et l'IA générative**

---

## Références

1. Guest, D. et al. "Deep Learning and its Application to LHC Physics." Ann. Rev. Nucl. Part. Sci. 68 (2018)
2. Albertsson, K. et al. "Machine Learning in High Energy Physics Community White Paper." J. Phys. Conf. Ser. 1085 (2018)
3. Feickert, M., Nachman, B. "A Living Review of Machine Learning for Particle Physics." arXiv:2102.02770

---

*Chapitre suivant : [Chapitre 2 - Fondements Mathématiques : Algèbre Linéaire Avancée](../Chapitre_02_Algebre_Lineaire/02_introduction.md)*

