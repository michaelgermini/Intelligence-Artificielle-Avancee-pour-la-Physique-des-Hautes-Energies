# Chapitre 18 : Systèmes de Trigger et Acquisition de Données

---

## Introduction

Les systèmes de **trigger** (déclenchement) sont le premier niveau de filtrage des données au LHC. Ils doivent décider en quelques microsecondes quels événements méritent d'être conservés parmi les 40 millions de croisements de faisceaux par seconde.

---

## Plan du Chapitre

1. [Architecture du Système de Trigger du LHC](./18_01_Architecture.md)
2. [Level-1 Trigger et Contraintes Temps Réel](./18_02_L1_Trigger.md)
3. [High-Level Trigger (HLT)](./18_03_HLT.md)
4. [Intégration de l'IA dans les Triggers](./18_04_IA_Trigger.md)
5. [Latence et Requirements de Performance](./18_05_Performance.md)

---

## Architecture du Trigger ATLAS/CMS

```
┌─────────────────────────────────────────────────────────────────┐
│                 Système de Trigger (CMS Run 3)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Collisions                                                     │
│      │                                                          │
│      │ 40 MHz (25 ns entre croisements)                        │
│      ▼                                                          │
│  ┌─────────────────────────────────────────┐                   │
│  │         Level-1 Trigger (L1)            │                   │
│  │  • Hardware (FPGA, ASIC)                │                   │
│  │  • Latence: 4 μs                        │                   │
│  │  • Calorimètres + Muons                 │                   │
│  │  • Granularité réduite                  │                   │
│  └────────────────┬────────────────────────┘                   │
│                   │                                             │
│                   │ ~100 kHz (réduction ×400)                  │
│                   ▼                                             │
│  ┌─────────────────────────────────────────┐                   │
│  │       High-Level Trigger (HLT)          │                   │
│  │  • Software (CPU/GPU farm)              │                   │
│  │  • Latence: ~300 ms                     │                   │
│  │  • Reconstruction complète              │                   │
│  │  • Algorithmes ML complexes             │                   │
│  └────────────────┬────────────────────────┘                   │
│                   │                                             │
│                   │ ~1-5 kHz (réduction ×100)                  │
│                   ▼                                             │
│              Stockage                                           │
│              (~1 GB/s)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Level-1 Trigger avec ML

```python
import numpy as np
import torch
import torch.nn as nn

class L1TriggerNN(nn.Module):
    """
    Réseau de neurones pour le trigger L1
    
    Contraintes strictes:
    - Latence: < 100 ns contribution
    - Ressources: < 1000 DSP slices
    - Précision: 6-8 bits
    """
    
    def __init__(self, n_inputs=16, n_hidden=32, n_outputs=1):
        super().__init__()
        
        # Architecture minimale pour FPGA
        self.fc1 = nn.Linear(n_inputs, n_hidden, bias=True)
        self.fc2 = nn.Linear(n_hidden, n_outputs, bias=True)
        
        # Activation compatible FPGA (ReLU ou LUT-based)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def count_operations(self):
        """Compte les opérations (MACs)"""
        macs = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                macs += module.in_features * module.out_features
        return macs
    
    def estimate_latency_ns(self, clock_freq_mhz=200):
        """
        Estime la latence en nanosecondes
        
        Hypothèse: fully pipelined, 1 MAC par cycle par DSP
        """
        # Latence = profondeur du pipeline
        n_layers = sum(1 for m in self.modules() if isinstance(m, nn.Linear))
        clock_period_ns = 1000 / clock_freq_mhz
        
        # Approximation: quelques cycles par couche
        cycles_per_layer = 3  # Multiply + accumulate + activation
        latency_ns = n_layers * cycles_per_layer * clock_period_ns
        
        return latency_ns

# Création et analyse
l1_model = L1TriggerNN(n_inputs=16, n_hidden=32, n_outputs=1)
print(f"MACs: {l1_model.count_operations()}")
print(f"Latence estimée: {l1_model.estimate_latency_ns():.1f} ns")
print(f"Paramètres: {sum(p.numel() for p in l1_model.parameters())}")
```

---

## Algorithmes de Trigger Classiques vs ML

```python
class TriggerAlgorithms:
    """
    Comparaison des algorithmes de trigger
    """
    
    @staticmethod
    def sliding_window_trigger(calo_towers, threshold=20.0):
        """
        Trigger classique par fenêtre glissante
        
        Cherche des clusters d'énergie dans le calorimètre
        """
        n_eta, n_phi = calo_towers.shape
        candidates = []
        
        # Fenêtre 2×2
        for i in range(n_eta - 1):
            for j in range(n_phi - 1):
                et_sum = (calo_towers[i, j] + calo_towers[i+1, j] +
                         calo_towers[i, j+1] + calo_towers[i+1, j+1])
                
                if et_sum > threshold:
                    candidates.append({
                        'eta_idx': i,
                        'phi_idx': j,
                        'et': et_sum
                    })
        
        return candidates
    
    @staticmethod
    def ml_trigger(features, model, threshold=0.5):
        """
        Trigger basé sur ML
        
        Prend une décision basée sur des features de haut niveau
        """
        with torch.no_grad():
            score = model(torch.tensor(features, dtype=torch.float32))
            decision = score > threshold
        
        return decision.item(), score.item()
    
    @staticmethod
    def compare_efficiency(classical_decisions, ml_decisions, true_labels):
        """
        Compare l'efficacité des deux approches
        """
        classical_decisions = np.array(classical_decisions)
        ml_decisions = np.array(ml_decisions)
        true_labels = np.array(true_labels)
        
        # Signal efficiency (recall)
        signal_mask = true_labels == 1
        classical_signal_eff = classical_decisions[signal_mask].mean()
        ml_signal_eff = ml_decisions[signal_mask].mean()
        
        # Background rejection
        bkg_mask = true_labels == 0
        classical_bkg_rej = 1 - classical_decisions[bkg_mask].mean()
        ml_bkg_rej = 1 - ml_decisions[bkg_mask].mean()
        
        return {
            'classical': {
                'signal_efficiency': classical_signal_eff,
                'background_rejection': classical_bkg_rej
            },
            'ml': {
                'signal_efficiency': ml_signal_eff,
                'background_rejection': ml_bkg_rej
            }
        }

# Simulation
np.random.seed(42)
n_events = 10000

# Génère des features simulées
signal_features = np.random.randn(n_events // 2, 16) + 1
bkg_features = np.random.randn(n_events // 2, 16)
all_features = np.vstack([signal_features, bkg_features])
labels = np.array([1] * (n_events // 2) + [0] * (n_events // 2))

# Shuffle
idx = np.random.permutation(n_events)
all_features = all_features[idx]
labels = labels[idx]

print("Comparaison des triggers:")
print("(Simulation avec features aléatoires)")
```

---

## High-Level Trigger avec Deep Learning

```python
class HLTModel(nn.Module):
    """
    Modèle pour le High-Level Trigger
    
    Plus de temps disponible (~300ms) permet des modèles complexes
    """
    
    def __init__(self, n_features=100, n_classes=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        self.classifier = nn.Linear(64, n_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
    
    def predict_with_threshold(self, x, thresholds):
        """
        Prédiction avec seuils par classe
        
        Permet d'ajuster le compromis efficacité/pureté par canal
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Applique les seuils
        decisions = probs > torch.tensor(thresholds)
        
        return decisions, probs


class HLTTriggerMenu:
    """
    Menu de trigger HLT
    
    Définit les différents chemins de trigger et leurs critères
    """
    
    def __init__(self):
        self.paths = {}
        
    def add_path(self, name, model, threshold, prescale=1):
        """
        Ajoute un chemin de trigger
        
        prescale: fraction d'événements à garder (1 = tous)
        """
        self.paths[name] = {
            'model': model,
            'threshold': threshold,
            'prescale': prescale
        }
    
    def evaluate(self, event_features):
        """
        Évalue tous les chemins pour un événement
        """
        results = {}
        
        for name, path in self.paths.items():
            model = path['model']
            threshold = path['threshold']
            
            with torch.no_grad():
                score = model(torch.tensor(event_features, dtype=torch.float32).unsqueeze(0))
                passed = score.item() > threshold
            
            # Applique le prescale
            if passed and np.random.random() > path['prescale']:
                passed = False
            
            results[name] = {
                'passed': passed,
                'score': score.item()
            }
        
        # L'événement passe si au moins un chemin est satisfait
        event_passed = any(r['passed'] for r in results.values())
        
        return event_passed, results

# Exemple de menu
menu = HLTTriggerMenu()

# Différents modèles pour différents canaux physiques
electron_model = HLTModel(n_features=50, n_classes=1)
muon_model = HLTModel(n_features=50, n_classes=1)
jet_model = HLTModel(n_features=50, n_classes=1)

menu.add_path('HLT_Electron_25', electron_model, threshold=0.9, prescale=1)
menu.add_path('HLT_Muon_20', muon_model, threshold=0.85, prescale=1)
menu.add_path('HLT_Jet_100', jet_model, threshold=0.7, prescale=0.1)  # Prescaled

print("Menu HLT configuré:")
for name, path in menu.paths.items():
    print(f"  {name}: threshold={path['threshold']}, prescale={path['prescale']}")
```

---

## Optimisation pour le Trigger

```python
class TriggerOptimization:
    """
    Outils d'optimisation pour les systèmes de trigger
    """
    
    @staticmethod
    def optimize_thresholds(model, val_data, val_labels, 
                           target_rate, signal_class=1):
        """
        Trouve le seuil optimal pour un taux de trigger cible
        """
        with torch.no_grad():
            scores = model(torch.tensor(val_data, dtype=torch.float32))
            if scores.dim() > 1:
                scores = scores[:, signal_class]
        
        scores = scores.numpy()
        
        # Cherche le seuil pour le taux cible
        thresholds = np.linspace(0, 1, 100)
        
        best_threshold = 0
        best_efficiency = 0
        
        for thresh in thresholds:
            decisions = scores > thresh
            rate = decisions.mean()
            
            if rate <= target_rate:
                # Calcule l'efficacité signal
                signal_mask = val_labels == signal_class
                efficiency = decisions[signal_mask].mean()
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_threshold = thresh
        
        return best_threshold, best_efficiency
    
    @staticmethod
    def rate_vs_efficiency_curve(model, val_data, val_labels, signal_class=1):
        """
        Génère la courbe taux vs efficacité (ROC-like)
        """
        with torch.no_grad():
            scores = model(torch.tensor(val_data, dtype=torch.float32))
            if scores.dim() > 1:
                scores = scores[:, signal_class]
        
        scores = scores.numpy()
        signal_mask = val_labels == signal_class
        
        thresholds = np.linspace(0, 1, 100)
        rates = []
        efficiencies = []
        
        for thresh in thresholds:
            decisions = scores > thresh
            rates.append(decisions.mean())
            efficiencies.append(decisions[signal_mask].mean())
        
        return np.array(rates), np.array(efficiencies), thresholds
    
    @staticmethod
    def latency_budget_allocation(total_latency_us, components):
        """
        Alloue le budget de latence entre composants
        
        components: dict {name: (min_latency, max_latency, importance)}
        """
        # Algorithme glouton: alloue d'abord aux plus importants
        sorted_components = sorted(
            components.items(), 
            key=lambda x: x[1][2],  # Par importance
            reverse=True
        )
        
        allocation = {}
        remaining = total_latency_us
        
        for name, (min_lat, max_lat, importance) in sorted_components:
            # Alloue le minimum nécessaire
            allocation[name] = min_lat
            remaining -= min_lat
        
        # Distribue le reste proportionnellement à l'importance
        total_importance = sum(c[2] for c in components.values())
        
        for name, (min_lat, max_lat, importance) in sorted_components:
            extra = remaining * (importance / total_importance)
            allocation[name] = min(allocation[name] + extra, max_lat)
        
        return allocation

# Exemple d'allocation de latence
components = {
    'calorimeter_processing': (0.5, 1.0, 3),
    'muon_processing': (0.3, 0.8, 2),
    'global_decision': (0.2, 0.5, 1),
    'ml_inference': (0.1, 0.3, 4)
}

allocation = TriggerOptimization.latency_budget_allocation(2.5, components)
print("Allocation de latence (μs):")
for name, latency in allocation.items():
    print(f"  {name}: {latency:.2f}")
```

---

## Exercices

### Exercice 18.1
Concevez un réseau de neurones pour le trigger L1 avec moins de 500 paramètres et une latence < 50 ns.

### Exercice 18.2
Implémentez un système de trigger à deux niveaux avec un modèle léger (L1) et un modèle complexe (HLT).

### Exercice 18.3
Optimisez les seuils d'un menu de trigger pour maximiser l'efficacité signal tout en respectant un budget de bande passante.

---

## Points Clés à Retenir

> 📌 **Le L1 trigger doit décider en ~4 μs avec du hardware dédié**

> 📌 **Le HLT dispose de ~300 ms et peut utiliser des algorithmes ML complexes**

> 📌 **L'efficacité signal vs le taux de trigger est le compromis fondamental**

> 📌 **La quantification et le pruning sont essentiels pour le L1**

---

*Chapitre suivant : [Chapitre 19 - Reconstruction et Identification de Particules](../Chapitre_19_Reconstruction/19_introduction.md)*

