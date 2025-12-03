# 19.1 Reconstruction de Traces

---

## Introduction

La **reconstruction de traces** (tracking) consiste à identifier les trajectoires des particules chargées à partir des hits dans les détecteurs de traces (trackers). C'est l'une des tâches les plus critiques en physique des hautes énergies, car la précision des traces affecte toutes les analyses ultérieures.

Les techniques de machine learning, en particulier les **Graph Neural Networks (GNN)**, révolutionnent cette tâche en permettant de traiter directement la structure de graphe des hits.

---

## Problème de Reconstruction de Traces

### Définition

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple

class TrackReconstructionProblem:
    """
    Définition du problème de reconstruction de traces
    """
    
    def __init__(self):
        self.problem_definition = """
        Problème:
        - Input: Ensemble de hits (x, y, z, layer)
        - Output: Ensemble de traces (séquences de hits)
        
        Contraintes:
        - Une trace = hits compatibles spatialement et temporellement
        - Plusieurs traces peuvent partager des hits (overlapping)
        - Bruit: hits faux (pas de trace associée)
        
        Défis:
        - Combinatoire: O(n!) pour n hits
        - Ambiguïté: hits compatibles avec plusieurs traces
        - Efficacité: ne pas perdre de traces vraies
        - Pureté: éviter traces fausses
        """
    
    def simulate_detector_hits(self, n_tracks=5, hits_per_track=10, noise_hits=20):
        """
        Simule des hits de détecteur
        
        Returns:
            hits: array (n_hits, 4) [x, y, z, layer]
            track_labels: list of lists (hit indices par trace)
        """
        hits = []
        track_labels = []
        hit_idx = 0
        
        # Générer traces vraies
        for track_id in range(n_tracks):
            track_hits = []
            
            # Paramètres de trace (droite dans l'espace)
            origin = np.random.randn(3) * 0.1  # Origine proche de (0,0,0)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)  # Normalisé
            
            # Hits le long de la trace
            for layer in range(hits_per_track):
                t = layer * 0.5  # Paramètre le long de la trace
                hit_pos = origin + t * direction
                
                # Ajouter bruit de mesure
                noise = np.random.randn(3) * 0.01
                hit_pos += noise
                
                hits.append([hit_pos[0], hit_pos[1], hit_pos[2], layer])
                track_hits.append(hit_idx)
                hit_idx += 1
            
            track_labels.append(track_hits)
        
        # Ajouter hits de bruit
        for _ in range(noise_hits):
            noise_pos = np.random.randn(3) * 1.0
            noise_layer = np.random.randint(0, hits_per_track)
            hits.append([noise_pos[0], noise_pos[1], noise_pos[2], noise_layer])
            hit_idx += 1
        
        return np.array(hits), track_labels

problem = TrackReconstructionProblem()
print(problem.problem_definition)

# Simuler données
hits, track_labels = problem.simulate_detector_hits()
print(f"\nSimulation:")
print(f"  Nombre de hits: {len(hits)}")
print(f"  Nombre de traces vraies: {len(track_labels)}")
print(f"  Hits par trace: {len(track_labels[0])}")
```

---

## Algorithmes Classiques

### Kalman Filter

```python
class KalmanFilterTracker:
    """
    Tracking avec Kalman Filter (méthode classique)
    """
    
    def __init__(self):
        self.process_noise = 0.01
        self.measurement_noise = 0.01
    
    def predict(self, state, covariance):
        """
        Prédiction du prochain état
        
        Args:
            state: [x, y, z, vx, vy, vz] (position + vitesse)
            covariance: Matrice de covariance
        """
        # Modèle: mouvement rectiligne
        F = np.eye(6)  # Matrice de transition
        F[0, 3] = 1.0  # x += vx
        F[1, 4] = 1.0  # y += vy
        F[2, 5] = 1.0  # z += vz
        
        Q = np.eye(6) * self.process_noise  # Bruit de processus
        
        predicted_state = F @ state
        predicted_cov = F @ covariance @ F.T + Q
        
        return predicted_state, predicted_cov
    
    def update(self, state, covariance, measurement):
        """
        Mise à jour avec nouvelle mesure
        
        Args:
            measurement: [x, y, z] position mesurée
        """
        # Matrice d'observation (on observe position seulement)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        
        R = np.eye(3) * self.measurement_noise  # Bruit de mesure
        
        # Innovation
        y = measurement - H @ state
        
        # Covariance de l'innovation
        S = H @ covariance @ H.T + R
        
        # Gain de Kalman
        K = covariance @ H.T @ np.linalg.inv(S)
        
        # Mise à jour
        updated_state = state + K @ y
        updated_cov = (np.eye(6) - K @ H) @ covariance
        
        return updated_state, updated_cov
    
    def track_hits(self, hits: np.ndarray, seed_hits: List[int]):
        """
        Reconstruit trace en commençant depuis seed hits
        
        Args:
            hits: (n_hits, 4) [x, y, z, layer]
            seed_hits: Indices de hits pour initialiser trace
        """
        if len(seed_hits) < 2:
            return []
        
        # Initialiser état depuis seed
        hit1 = hits[seed_hits[0]]
        hit2 = hits[seed_hits[1]]
        
        position = hit1[:3]
        velocity = (hit2[:3] - hit1[:3]) / max(hit2[3] - hit1[3], 1)
        
        state = np.concatenate([position, velocity])
        covariance = np.eye(6) * 0.1
        
        track = [seed_hits[0], seed_hits[1]]
        current_layer = int(hit2[3])
        
        # Extend track layer par layer
        for next_layer in range(current_layer + 1, int(hits[:, 3].max()) + 1):
            # Prédire prochain hit
            state, covariance = self.predict(state, covariance)
            
            # Trouver hits dans cette couche
            layer_hits = np.where(hits[:, 3] == next_layer)[0]
            
            if len(layer_hits) == 0:
                continue
            
            # Mesurer distance aux hits
            predicted_pos = state[:3]
            distances = [np.linalg.norm(hits[i][:3] - predicted_pos) for i in layer_hits]
            
            # Choisir hit le plus proche (si compatible)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 0.1:  # Seuil de compatibilité
                hit_idx = layer_hits[closest_idx]
                measurement = hits[hit_idx][:3]
                
                # Mise à jour Kalman
                state, covariance = self.update(state, covariance, measurement)
                track.append(hit_idx)
            else:
                break  # Pas de hit compatible
        
        return track

# Test Kalman Filter
kalman_tracker = KalmanFilterTracker()
hits, track_labels = problem.simulate_detector_hits(n_tracks=3, hits_per_track=8)

# Reconstruire une trace
seed = [0, 1]  # Premiers hits d'une trace
reconstructed = kalman_tracker.track_hits(hits, seed)

print(f"\nKalman Filter Tracking:")
print(f"  Hits dans seed: {len(seed)}")
print(f"  Hits reconstruits: {len(reconstructed)}")
```

---

## Graph Neural Networks pour Tracking

### Architecture GNN

```python
class TrackReconstructionGNN(nn.Module):
    """
    Reconstruction de traces avec Graph Neural Network
    """
    
    def __init__(self, hit_features=4, hidden_dim=64, n_layers=4):
        """
        Args:
            hit_features: Dimension des features par hit (x, y, z, layer)
            hidden_dim: Dimension des embeddings
            n_layers: Nombre de couches de message passing
        """
        super().__init__()
        
        # Encodeur initial des hits
        self.hit_encoder = nn.Sequential(
            nn.Linear(hit_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Couches de message passing
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(n_layers)
        ])
        
        # Classificateur d'arêtes (probabilité que deux hits soient sur même trace)
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hits: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            hits: (n_hits, hit_features) features des hits
            edge_index: (2, n_edges) indices des paires de hits connectés
        """
        # Encode hits
        x = self.hit_encoder(hits)
        
        # Message passing
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index)
        
        # Prédire probabilité d'arête
        src, dst = edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=-1)
        edge_scores = self.edge_classifier(edge_features).squeeze()
        
        return edge_scores
    
    def build_edge_index(self, hits: torch.Tensor, max_distance: float = 0.5):
        """
        Construit graphe en connectant hits proches
        
        Args:
            hits: (n_hits, hit_features)
            max_distance: Distance maximale pour connecter hits
        """
        n_hits = hits.shape[0]
        edges = []
        
        # Connecter hits dans couches adjacentes ou proches
        for i in range(n_hits):
            for j in range(i + 1, n_hits):
                hit_i = hits[i]
                hit_j = hits[j]
                
                # Distance spatiale
                pos_i = hit_i[:3]
                pos_j = hit_j[:3]
                distance = torch.norm(pos_i - pos_j)
                
                # Différence de layer
                layer_diff = abs(hit_i[3] - hit_j[3])
                
                # Connecter si proches et couches adjacentes
                if distance < max_distance and layer_diff <= 2:
                    edges.append([i, j])
        
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

class MessagePassingLayer(nn.Module):
    """Couche de message passing pour GNN"""
    
    def __init__(self, dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU()
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: (n_nodes, dim) embeddings des nœuds
            edge_index: (2, n_edges)
        """
        src, dst = edge_index
        
        # Messages: concaténation source + destination
        messages = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.message_net(messages)
        
        # Agrégation: somme des messages entrants
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)
        
        # Mise à jour: combine état actuel + agrégation
        updated = torch.cat([x, aggregated], dim=-1)
        updated = self.update_net(updated)
        
        return updated

# Créer modèle GNN
gnn_model = TrackReconstructionGNN(hit_features=4, hidden_dim=64, n_layers=4)

# Exemple d'utilisation
hits_tensor = torch.tensor(hits, dtype=torch.float32)
edge_index = gnn_model.build_edge_index(hits_tensor, max_distance=0.5)

print(f"\nGNN Model:")
print(f"  Hits: {hits_tensor.shape[0]}")
print(f"  Arêtes: {edge_index.shape[1]}")

# Forward pass
with torch.no_grad():
    edge_scores = gnn_model(hits_tensor, edge_index)
    print(f"  Scores d'arêtes: mean={edge_scores.mean():.3f}, std={edge_scores.std():.3f}")
```

---

## Reconstruction de Traces depuis Prédictions GNN

### Extraction de Traces

```python
class TrackExtractor:
    """
    Extrait traces depuis prédictions GNN
    """
    
    def __init__(self, edge_threshold=0.5):
        """
        Args:
            edge_threshold: Seuil pour considérer arête comme vraie
        """
        self.edge_threshold = edge_threshold
    
    def extract_tracks(self, hits: np.ndarray, edge_index: torch.Tensor,
                      edge_scores: torch.Tensor) -> List[List[int]]:
        """
        Extrait traces depuis graphe d'arêtes
        
        Args:
            hits: (n_hits, 4) positions des hits
            edge_index: (2, n_edges) indices des arêtes
            edge_scores: (n_edges,) probabilités des arêtes
        
        Returns:
            List de traces (chaque trace = liste d'indices de hits)
        """
        # Filtrer arêtes au-dessus du seuil
        valid_edges = edge_scores > self.edge_threshold
        valid_edge_index = edge_index[:, valid_edges]
        
        # Construire graphe (non dirigé)
        from collections import defaultdict
        graph = defaultdict(list)
        
        src, dst = valid_edge_index
        for i in range(len(src)):
            s, d = src[i].item(), dst[i].item()
            graph[s].append(d)
            graph[d].append(s)
        
        # Extraire chemins (traces) avec DFS
        visited = set()
        tracks = []
        
        def dfs(node, current_track):
            """Depth-first search pour trouver trace"""
            visited.add(node)
            current_track.append(node)
            
            # Trier voisins par layer pour maintenir ordre
            neighbors = graph[node]
            neighbors = sorted(neighbors, key=lambda n: hits[n][3])
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Vérifier compatibilité (couches consécutives)
                    current_layer = int(hits[node][3])
                    neighbor_layer = int(hits[neighbor][3])
                    
                    if abs(neighbor_layer - current_layer) <= 2:
                        dfs(neighbor, current_track)
        
        # Trouver tous les chemins
        for node in graph:
            if node not in visited:
                track = []
                dfs(node, track)
                if len(track) >= 3:  # Au moins 3 hits pour une trace valide
                    # Trier par layer
                    track = sorted(track, key=lambda i: hits[i][3])
                    tracks.append(track)
        
        return tracks
    
    def filter_duplicate_tracks(self, tracks: List[List[int]], 
                               hits: np.ndarray) -> List[List[int]]:
        """
        Filtre traces dupliquées ou qui se chevauchent beaucoup
        """
        filtered = []
        used_hits = set()
        
        # Trier traces par longueur (longues d'abord)
        sorted_tracks = sorted(tracks, key=len, reverse=True)
        
        for track in sorted_tracks:
            # Calculer overlap avec traces déjà ajoutées
            track_hits = set(track)
            overlap = len(track_hits & used_hits) / len(track_hits)
            
            if overlap < 0.5:  # Moins de 50% d'overlap
                filtered.append(track)
                used_hits.update(track_hits)
        
        return filtered

# Utiliser extracteur
extractor = TrackExtractor(edge_threshold=0.7)

# Simuler reconstruction
hits_sim, true_tracks = problem.simulate_detector_hits(n_tracks=5, hits_per_track=10)
hits_tensor = torch.tensor(hits_sim, dtype=torch.float32)

# Construire graphe et prédire
edge_index = gnn_model.build_edge_index(hits_tensor)
with torch.no_grad():
    edge_scores = gnn_model(hits_tensor, edge_index)

# Extraire traces
reconstructed_tracks = extractor.extract_tracks(hits_sim, edge_index, edge_scores)
filtered_tracks = extractor.filter_duplicate_tracks(reconstructed_tracks, hits_sim)

print(f"\nReconstruction de Traces:")
print(f"  Traces vraies: {len(true_tracks)}")
print(f"  Traces reconstruites: {len(reconstructed_tracks)}")
print(f"  Traces filtrées: {len(filtered_tracks)}")
```

---

## Métriques de Performance

### Évaluation

```python
class TrackReconstructionMetrics:
    """
    Métriques pour évaluer reconstruction de traces
    """
    
    def compute_efficiency(self, true_tracks: List[List[int]],
                          reconstructed_tracks: List[List[int]],
                          hits: np.ndarray) -> Dict:
        """
        Calcule efficacité de reconstruction
        
        Efficacité = fraction de traces vraies retrouvées
        """
        n_true = len(true_tracks)
        
        # Pour chaque trace vraie, chercher meilleure correspondance
        matched = 0
        match_details = []
        
        for true_track in true_tracks:
            best_match = None
            best_overlap = 0
            
            for rec_track in reconstructed_tracks:
                true_set = set(true_track)
                rec_set = set(rec_track)
                
                # Calculer overlap (IoU)
                overlap = len(true_set & rec_set) / len(true_set | rec_set) if len(true_set | rec_set) > 0 else 0
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = rec_track
            
            # Considérer matché si overlap > 0.5
            if best_overlap > 0.5:
                matched += 1
                match_details.append({
                    'true_track': true_track,
                    'matched_track': best_match,
                    'overlap': best_overlap
                })
        
        efficiency = matched / n_true if n_true > 0 else 0
        
        return {
            'efficiency': efficiency,
            'n_true': n_true,
            'n_matched': matched,
            'match_details': match_details
        }
    
    def compute_fake_rate(self, true_tracks: List[List[int]],
                         reconstructed_tracks: List[List[int]]) -> Dict:
        """
        Calcule taux de traces fausses (fakes)
        
        Fake = trace reconstruite qui ne correspond à aucune trace vraie
        """
        n_reconstructed = len(reconstructed_tracks)
        
        # Trouver traces reconstruites qui ne matchent aucune vraie trace
        fake_count = 0
        fake_tracks = []
        
        for rec_track in reconstructed_tracks:
            max_overlap = 0
            
            for true_track in true_tracks:
                true_set = set(true_track)
                rec_set = set(rec_track)
                overlap = len(true_set & rec_set) / len(true_set | rec_set) if len(true_set | rec_set) > 0 else 0
                max_overlap = max(max_overlap, overlap)
            
            if max_overlap < 0.5:  # Pas de bonne correspondance
                fake_count += 1
                fake_tracks.append(rec_track)
        
        fake_rate = fake_count / n_reconstructed if n_reconstructed > 0 else 0
        
        return {
            'fake_rate': fake_rate,
            'n_fake': fake_count,
            'n_reconstructed': n_reconstructed,
            'fake_tracks': fake_tracks
        }
    
    def compute_track_quality(self, true_track: List[int],
                             reconstructed_track: List[int],
                             hits: np.ndarray) -> Dict:
        """
        Calcule qualité d'une trace reconstruite
        
        Métriques: résolution pT, résolution impact parameter, etc.
        """
        if len(reconstructed_track) < 2:
            return {'quality': 0.0}
        
        # Calculer paramètres de trace vraie
        true_hits = hits[true_track]
        true_direction = true_hits[-1][:3] - true_hits[0][:3]
        true_direction /= np.linalg.norm(true_direction)
        
        # Calculer paramètres de trace reconstruite
        rec_hits = hits[reconstructed_track]
        rec_direction = rec_hits[-1][:3] - rec_hits[0][:3]
        rec_direction /= np.linalg.norm(rec_direction) if np.linalg.norm(rec_direction) > 0 else 1
        
        # Angle entre directions
        cos_angle = np.dot(true_direction, rec_direction)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Nombre de hits en commun
        true_set = set(true_track)
        rec_set = set(reconstructed_track)
        common_hits = len(true_set & rec_set)
        total_hits = len(true_set | rec_set)
        
        # Score de qualité combiné
        angle_score = 1.0 - (angle / np.pi)  # 1 si angle=0, 0 si angle=π
        hit_score = common_hits / total_hits if total_hits > 0 else 0
        quality = (angle_score + hit_score) / 2
        
        return {
            'quality': quality,
            'angle_rad': angle,
            'common_hits': common_hits,
            'total_hits': total_hits
        }

# Évaluer performance
metrics = TrackReconstructionMetrics()

efficiency = metrics.compute_efficiency(true_tracks, filtered_tracks, hits_sim)
fake_rate = metrics.compute_fake_rate(true_tracks, filtered_tracks)

print(f"\n" + "="*70)
print("Métriques de Performance")
print("="*70)
print(f"Efficacité: {efficiency['efficiency']:.2%} ({efficiency['n_matched']}/{efficiency['n_true']})")
print(f"Taux de fakes: {fake_rate['fake_rate']:.2%} ({fake_rate['n_fake']}/{fake_rate['n_reconstructed']})")
```

---

## Exercices

### Exercice 19.1.1
Implémentez un algorithme de Kalman Filter pour tracker des traces hélicoïdales (particules dans champ magnétique).

### Exercice 19.1.2
Entraînez un modèle GNN pour reconstruction de traces et évaluez l'impact du nombre de couches de message passing sur les performances.

### Exercice 19.1.3
Développez un système de track extraction qui gère les ambiguïtés (hits partagés entre plusieurs traces).

### Exercice 19.1.4
Comparez l'efficacité et le taux de fakes d'un tracker GNN vs Kalman Filter classique sur un dataset simulé.

---

## Points Clés à Retenir

> 📌 **La reconstruction de traces est un problème de graphe: hits = nœuds, traces = chemins**

> 📌 **Les GNN sont particulièrement adaptés grâce à leur capacité à traiter structures de graphe**

> 📌 **Le Kalman Filter reste efficace pour extension de traces depuis seeds**

> 📌 **L'extraction de traces depuis graphe prédit nécessite algorithmes de chemin (DFS, etc.)**

> 📌 **L'efficacité et le taux de fakes sont les métriques principales**

> 📌 **La gestion des ambiguïtés (hits partagés) est un défi majeur**

---

*Section précédente : [19.0 Introduction](./19_introduction.md) | Section suivante : [19.2 Identification de Jets](./19_02_Jets.md)*

