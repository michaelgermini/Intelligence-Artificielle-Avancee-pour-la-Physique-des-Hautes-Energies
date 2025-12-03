# 20.3 Méthodes Non Supervisées

---

## Introduction

Les **méthodes non supervisées** sont essentielles pour la détection d'anomalies en physique des hautes énergies, car elles ne nécessitent pas de labels de signal (qui sont par définition inconnus pour la nouvelle physique). Ces méthodes apprennent directement depuis les données pour identifier des patterns et outliers.

Cette section présente diverses méthodes non supervisées utilisées pour la détection d'anomalies, incluant les méthodes basées sur densité, clustering, et isolation.

---

## Types de Méthodes Non Supervisées

### Classification

```python
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor

class UnsupervisedMethods:
    """
    Vue d'ensemble des méthodes non supervisées
    """
    
    def __init__(self):
        self.methods = {
            'density_based': {
                'examples': ['DBSCAN', 'LOF (Local Outlier Factor)'],
                'principle': 'Anomalies = points dans régions de faible densité',
                'advantages': ['Détecte clusters de formes arbitraires'],
                'disadvantages': ['Sensible à paramètres', 'Coût computationnel']
            },
            'isolation_based': {
                'examples': ['Isolation Forest', 'Extended Isolation Forest'],
                'principle': 'Anomalies = faciles à isoler',
                'advantages': ['Rapide', 'Bien pour haute dimension'],
                'disadvantages': ['Moins précis pour clusters d\'anomalies']
            },
            'distance_based': {
                'examples': ['k-NN distance', 'Average k-NN distance'],
                'principle': 'Anomalies = points loin de leurs voisins',
                'advantages': ['Simple', 'Intuitif'],
                'disadvantages': ['Sensible curse of dimensionality']
            },
            'clustering_based': {
                'examples': ['k-means outliers', 'Hierarchical clustering'],
                'principle': 'Anomalies = points loin des clusters',
                'advantages': ['Interprétable'],
                'disadvantages': ['Nécessite nombre clusters']
            },
            'neural_network_based': {
                'examples': ['Autoencoders', 'VAE', 'GAN discriminators'],
                'principle': 'Anomalies = mal reconstruites/générées',
                'advantages': ['Capte patterns complexes'],
                'disadvantages': ['Requiert entraînement', 'Black box']
            }
        }
    
    def display_methods(self):
        """Affiche les méthodes"""
        print("\n" + "="*70)
        print("Méthodes Non Supervisées pour Détection d'Anomalies")
        print("="*70)
        
        for method_type, info in self.methods.items():
            print(f"\n{method_type.replace('_', ' ').title()}:")
            print(f"  Exemples: {', '.join(info['examples'])}")
            print(f"  Principe: {info['principle']}")
            print(f"  Avantages:")
            for adv in info['advantages']:
                print(f"    + {adv}")
            print(f"  Inconvénients:")
            for disadv in info['disadvantages']:
                print(f"    - {disadv}")

unsupervised = UnsupervisedMethods()
unsupervised.display_methods()
```

---

## Isolation Forest

### Principe et Implémentation

```python
class IsolationForestAnomalyDetection:
    """
    Isolation Forest pour détection d'anomalies
    
    Principe: Anomalies sont faciles à isoler (peu de splits nécessaires)
    """
    
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1):
        """
        Args:
            n_estimators: Nombre d'arbres
            max_samples: Nombre échantillons par arbre
            contamination: Fraction attendue d'anomalies
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=42
        )
    
    def fit(self, X):
        """Entraîne sur données background"""
        self.model.fit(X)
        return self
    
    def predict_anomalies(self, X):
        """
        Prédit anomalies
        
        Returns:
            predictions: 1 = normal, -1 = anomalie
            scores: Score d'anomalie (plus négatif = plus anormal)
        """
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        return {
            'predictions': predictions,
            'scores': scores,
            'anomaly_indices': np.where(predictions == -1)[0]
        }
    
    def compute_anomaly_scores(self, X):
        """Retourne seulement les scores"""
        return self.model.score_samples(X)

class ExtendedIsolationForest:
    """
    Extended Isolation Forest
    
    Amélioration qui utilise hyperplanes de dimension quelconque
    """
    
    def __init__(self, n_estimators=100, extension_level=1):
        """
        Args:
            extension_level: Niveau d'extension (dimension hyperplanes)
        """
        # En pratique: utiliser bibliothèque spécialisée
        # Ici: simulation avec Isolation Forest standard
        self.base_model = IsolationForest(n_estimators=n_estimators)
        self.extension_level = extension_level
    
    def fit(self, X):
        """Entraîne modèle"""
        self.base_model.fit(X)
        return self
    
    def predict(self, X):
        """Prédit anomalies"""
        return self.base_model.predict(X)

# Test Isolation Forest
iso_forest = IsolationForestAnomalyDetection(n_estimators=100, contamination=0.05)

# Simuler données
background_data = np.random.randn(10000, 10)
anomaly_data = np.random.randn(100, 10) * 2 + 5
all_data = np.vstack([background_data, anomaly_data])

iso_forest.fit(background_data)
results = iso_forest.predict_anomalies(all_data)

print(f"\nIsolation Forest:")
print(f"  Anomalies détectées: {len(results['anomaly_indices'])}")
print(f"  Score moyen background: {results['scores'][:10000].mean():.4f}")
print(f"  Score moyen anomalies: {results['scores'][10000:].mean():.4f}")
```

---

## Local Outlier Factor (LOF)

### Détection Basée sur Densité Locale

```python
class LocalOutlierFactorDetection:
    """
    Local Outlier Factor pour détection d'anomalies
    
    Compare densité locale d'un point avec densité de ses voisins
    """
    
    def __init__(self, n_neighbors=20, contamination=0.1):
        """
        Args:
            n_neighbors: Nombre de voisins à considérer
            contamination: Fraction attendue d'anomalies
        """
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        self.n_neighbors = n_neighbors
    
    def fit(self, X):
        """Entraîne sur données background"""
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """
        Prédit anomalies
        
        Note: LOF nécessite refit pour nouvelles données en mode novelty=False
        """
        predictions = self.model.fit_predict(X)
        scores = -self.model.negative_outlier_factor_  # Convert to positive (higher = more anomalous)
        
        return {
            'predictions': predictions,
            'scores': scores,
            'anomaly_indices': np.where(predictions == -1)[0]
        }

lof = LocalOutlierFactorDetection(n_neighbors=20, contamination=0.05)
lof_results = lof.predict(all_data)

print(f"\nLocal Outlier Factor:")
print(f"  Anomalies détectées: {len(lof_results['anomaly_indices'])}")
print(f"  Score moyen background: {lof_results['scores'][:10000].mean():.4f}")
```

---

## DBSCAN pour Détection d'Anomalies

### Clustering avec Points de Bruit

```python
class DBSCANAnomalyDetection:
    """
    DBSCAN: points non assignés à clusters = anomalies
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        Args:
            eps: Distance maximale entre voisins
            min_samples: Nombre minimal de points pour former cluster
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.eps = eps
        self.min_samples = min_samples
    
    def fit_predict(self, X):
        """
        Clustering: -1 = bruit (anomalies)
        """
        labels = self.model.fit_predict(X)
        
        # Anomalies = labels = -1
        anomaly_indices = np.where(labels == -1)[0]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'labels': labels,
            'anomaly_indices': anomaly_indices,
            'n_anomalies': len(anomaly_indices),
            'n_clusters': n_clusters
        }

dbscan = DBSCANAnomalyDetection(eps=0.3, min_samples=10)
dbscan_results = dbscan.fit_predict(all_data[:5000])  # DBSCAN peut être lent

print(f"\nDBSCAN:")
print(f"  Clusters trouvés: {dbscan_results['n_clusters']}")
print(f"  Anomalies (bruit): {dbscan_results['n_anomalies']}")
```

---

## Méthodes Basées sur Distance

### k-NN Distance

```python
class KNNAnomalyDetection:
    """
    Détection d'anomalies basée sur distance k-NN
    """
    
    def __init__(self, k=5):
        """
        Args:
            k: Nombre de voisins
        """
        self.k = k
    
    def compute_knn_distances(self, X_train, X_test):
        """
        Calcule distances aux k plus proches voisins
        
        Anomalies = grandes distances aux k-NN
        """
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=self.k + 1)  # +1 car inclut point lui-même
        nn.fit(X_train)
        
        distances, indices = nn.kneighbors(X_test)
        
        # Prendre distances aux k voisins (exclure point lui-même)
        knn_distances = distances[:, 1:]  # Exclure distance à soi (indice 0)
        avg_knn_distance = knn_distances.mean(axis=1)
        
        return {
            'knn_distances': knn_distances,
            'avg_knn_distance': avg_knn_distance,
            'max_knn_distance': knn_distances.max(axis=1)
        }
    
    def detect_anomalies(self, X_train, X_test, threshold_percentile=95):
        """
        Détecte anomalies avec seuil sur distance k-NN
        """
        knn_results = self.compute_knn_distances(X_train, X_test)
        
        # Seuil depuis données d'entraînement
        train_distances = self.compute_knn_distances(X_train, X_train)
        threshold = np.percentile(train_distances['avg_knn_distance'], threshold_percentile)
        
        # Détecter anomalies
        anomaly_mask = knn_results['avg_knn_distance'] > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return {
            'anomaly_indices': anomaly_indices,
            'scores': knn_results['avg_knn_distance'],
            'threshold': threshold
        }

knn_detector = KNNAnomalyDetection(k=5)
knn_results = knn_detector.detect_anomalies(background_data[:5000], all_data, threshold_percentile=95)

print(f"\nk-NN Anomaly Detection:")
print(f"  Seuil: {knn_results['threshold']:.4f}")
print(f"  Anomalies détectées: {len(knn_results['anomaly_indices'])}")
```

---

## One-Class SVM

### Support Vector Machine pour Détection

```python
class OneClassSVMDetection:
    """
    One-Class SVM pour détection d'anomalies
    
    Apprend frontière autour des données normales
    """
    
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Args:
            nu: Limite supérieure fraction d'outliers
            kernel: Type de kernel ('rbf', 'linear', 'poly')
            gamma: Paramètre kernel RBF
        """
        from sklearn.svm import OneClassSVM
        
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.nu = nu
    
    def fit(self, X):
        """Entraîne sur données background"""
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """
        Prédit: 1 = normal, -1 = anomalie
        """
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        return {
            'predictions': predictions,
            'scores': scores,
            'anomaly_indices': np.where(predictions == -1)[0]
        }

oc_svm = OneClassSVMDetection(nu=0.05, kernel='rbf')
oc_svm.fit(background_data[:5000])
svm_results = oc_svm.predict(all_data)

print(f"\nOne-Class SVM:")
print(f"  Anomalies détectées: {len(svm_results['anomaly_indices'])}")
print(f"  Score moyen background: {svm_results['scores'][:10000].mean():.4f}")
```

---

## Comparaison des Méthodes

### Benchmark

```python
class UnsupervisedMethodComparison:
    """
    Compare différentes méthodes non supervisées
    """
    
    def compare_methods(self, X_train, X_test, true_labels=None):
        """
        Compare performances de différentes méthodes
        """
        results = {}
        
        # Isolation Forest
        iso_forest = IsolationForestAnomalyDetection(contamination=0.05)
        iso_forest.fit(X_train)
        iso_results = iso_forest.predict_anomalies(X_test)
        results['Isolation Forest'] = {
            'n_anomalies': len(iso_results['anomaly_indices']),
            'scores': iso_results['scores']
        }
        
        # LOF
        lof = LocalOutlierFactorDetection(contamination=0.05)
        lof_results = lof.predict(X_test)  # Note: fit inclus
        results['LOF'] = {
            'n_anomalies': len(lof_results['anomaly_indices']),
            'scores': lof_results['scores']
        }
        
        # k-NN
        knn = KNNAnomalyDetection(k=5)
        knn_results = knn.detect_anomalies(X_train, X_test)
        results['k-NN'] = {
            'n_anomalies': len(knn_results['anomaly_indices']),
            'scores': knn_results['scores']
        }
        
        # One-Class SVM
        oc_svm = OneClassSVMDetection(nu=0.05)
        oc_svm.fit(X_train)
        svm_results = oc_svm.predict(X_test)
        results['One-Class SVM'] = {
            'n_anomalies': len(svm_results['anomaly_indices']),
            'scores': svm_results['scores']
        }
        
        # Évaluer si true_labels disponibles
        if true_labels is not None:
            for method_name, result in results.items():
                predictions = np.zeros(len(X_test))
                predictions[result['anomaly_indices']] = 1
                
                # Métriques (si binaire)
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average='binary', zero_division=0
                )
                
                result['precision'] = precision
                result['recall'] = recall
                result['f1'] = f1
        
        return results
    
    def display_comparison(self, results):
        """Affiche comparaison"""
        print("\n" + "="*70)
        print("Comparaison des Méthodes Non Supervisées")
        print("="*70)
        
        print(f"\n{'Méthode':<20} {'Anomalies':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for method, result in results.items():
            anomalies = result['n_anomalies']
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1 = result.get('f1', 0)
            
            print(f"{method:<20} {anomalies:<15} {precision:<11.3f} {recall:<11.3f} {f1:<11.3f}")

# Comparer méthodes
comparison = UnsupervisedMethodComparison()

# Simuler labels (derniers 100 = anomalies)
true_labels = np.zeros(len(all_data))
true_labels[-100:] = 1

comp_results = comparison.compare_methods(
    background_data[:5000], all_data, true_labels=true_labels
)
comparison.display_comparison(comp_results)
```

---

## Exercices

### Exercice 20.3.1
Comparez Isolation Forest, LOF, et k-NN sur un dataset simulé avec différentes distributions d'anomalies.

### Exercice 20.3.2
Analysez l'impact des hyperparamètres (n_neighbors, eps, contamination) sur les performances.

### Exercice 20.3.3
Implémentez une méthode de détection d'anomalies basée sur clustering hiérarchique.

### Exercice 20.3.4
Développez un système qui combine plusieurs méthodes non supervisées avec voting ou stacking.

---

## Points Clés à Retenir

> 📌 **Les méthodes non supervisées ne nécessitent pas labels de signal**

> 📌 **Isolation Forest est rapide et efficace pour haute dimension**

> 📌 **LOF détecte anomalies locales en comparant densités**

> 📌 **DBSCAN identifie anomalies comme points de bruit (non-clustered)**

> 📌 **k-NN distance est simple mais sensible à curse of dimensionality**

> 📌 **La combinaison de méthodes peut améliorer robustesse**

---

*Section précédente : [20.2 Autoencoders](./20_02_Autoencoders.md) | Section suivante : [20.4 Réseaux de Tenseurs](./20_04_Tenseurs.md)*

