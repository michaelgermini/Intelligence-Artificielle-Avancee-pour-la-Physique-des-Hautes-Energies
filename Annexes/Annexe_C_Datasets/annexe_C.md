# Annexe C : Datasets et Benchmarks

---

## Introduction

Cette annexe présente les datasets et benchmarks utilisés pour évaluer les modèles et techniques présentés dans ce livre. Nous couvrons les datasets de physique des particules, les benchmarks de compression de modèles, et les métriques d'évaluation standard. Chaque section inclut des descriptions détaillées, des exemples d'utilisation, et des informations sur la structure des données.

---

## Plan de l'Annexe

1. [C.1 Datasets de Physique des Particules](#c1-datasets-de-physique-des-particules)
2. [C.2 Benchmarks de Compression](#c2-benchmarks-de-compression)
3. [C.3 Métriques d'Évaluation](#c3-métriques-dévaluation)

---

## C.1 Datasets de Physique des Particules

### CERN Open Data

#### Description

**Lien** : http://opendata.cern.ch/

Le portail CERN Open Data fournit accès public aux données du LHC. Les données sont formatées en ROOT et incluent événements réels de collisions proton-proton.

#### Structure des Données

Les fichiers ROOT contiennent des **arbres** (TTree) avec branches pour différentes collections de particules et quantités reconstruites.

```python
import uproot
import numpy as np
import awkward as ak

# Ouvrir fichier ROOT
file = uproot.open("Run2012B_DoubleMuParked.root")

# Lister les arbres disponibles
print("Arbres disponibles:", file.keys())

# Accéder à l'arbre Events
tree = file["Events"]

# Lister les branches disponibles
print("\nBranches disponibles:")
for branch in tree.keys():
    print(f"  {branch}")

# Lire une branche spécifique
muons_pt = tree["Muon_pt"].array()

# Lire plusieurs branches
data = tree.arrays(["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"])
print(f"\nNombre d'événements: {len(data)}")
print(f"Structure données: {data.type}")
```

#### Types de Données Disponibles

**Collections de Particules** :
- `Muon_*` : Propriétés muons
- `Electron_*` : Propriétés électrons
- `Jet_*` : Propriétés jets
- `Photon_*` : Propriétés photons
- `Tau_*` : Propriétés taus

**Quantités Globales** :
- `MET_pt`, `MET_phi` : Missing transverse energy
- `nMuon`, `nElectron`, `nJet` : Multiplicités
- `run`, `luminosityBlock`, `event` : Identifiants événements

#### Exemple Complet

```python
def load_cern_opendata(file_path):
    """
    Charge et prépare données CERN Open Data
    
    Args:
        file_path: Chemin vers fichier ROOT
    
    Returns:
        Dictionary avec données préparées
    """
    # Ouvrir fichier
    file = uproot.open(file_path)
    tree = file["Events"]
    
    # Sélectionner branches d'intérêt
    branches = [
        "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge",
        "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
        "MET_pt", "MET_phi",
        "nMuon", "nJet"
    ]
    
    # Lire données
    data = tree.arrays(branches, library="ak")
    
    # Filtrer événements (exemple: au moins 2 muons)
    mask = ak.num(data["Muon_pt"]) >= 2
    data_filtered = data[mask]
    
    print(f"Événements totaux: {len(data)}")
    print(f"Événements filtrés: {len(data_filtered)}")
    
    return data_filtered

# Utilisation
data = load_cern_opendata("data/Run2012B_DoubleMuParked.root")
```

### Jet Tagging Datasets

#### Quarks vs Gluons

**Description** : Dataset standard pour classification jets quark vs gluon.

**Caractéristiques** :
- Jets générés avec Pythia8
- Énergie: 13 TeV
- Jets anti-kT avec R=0.4
- pT > 500 GeV

**Structure** :
```python
"""
Structure typique jet tagging dataset:

- jet_features: Caractéristiques globales du jet
  - pT, eta, phi, mass, charge
  - Multiplicité particules
  - Variables de forme (thrust, sphericity, etc.)

- particle_features: Caractéristiques particules dans jet
  - pT, eta, phi, charge, PID
  - Position relative au jet axis

- label: 0 (quark) ou 1 (gluon)
"""
```

**Chargement** :
```python
import numpy as np
import h5py

def load_quark_gluon_dataset(file_path):
    """
    Charge dataset quark vs gluon
    
    Args:
        file_path: Chemin vers fichier HDF5
    
    Returns:
        Tuple (X, y) où X sont features et y labels
    """
    with h5py.File(file_path, 'r') as f:
        # Features globales jet
        jet_features = f['jet_features'][:]
        
        # Features particules (arrays irréguliers)
        particle_features = f['particle_features'][:]
        
        # Labels
        labels = f['labels'][:]
    
    return {
        'jet_features': jet_features,
        'particle_features': particle_features,
        'labels': labels
    }

# Utilisation
data = load_quark_gluon_dataset("data/quark_gluon_dataset.h5")
print(f"Nombre de jets: {len(data['labels'])}")
print(f"Distribution labels: {np.bincount(data['labels'])}")
```

#### JetClass

**Description** : Dataset complet classification jets avec 10 classes.

**Classes** :
- Quark (u, d, s, c, b)
- Gluon (g)
- Boson (W, Z)
- Top (t)
- Higgs (H)

**Utilisation** :
```python
def prepare_jetclass_data(data):
    """
    Prépare données JetClass pour ML
    
    Args:
        data: Données brutes JetClass
    
    Returns:
        Features et labels préparés
    """
    # Extraire features
    jet_features = data['jet_features']
    particle_features = data['particle_features']
    
    # Normaliser features
    jet_features = (jet_features - jet_features.mean(axis=0)) / jet_features.std(axis=0)
    
    # Convertir labels en one-hot encoding si nécessaire
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform(data['labels'])
    
    return jet_features, particle_features, labels_encoded
```

### TrackML Dataset

**Lien** : https://www.kaggle.com/c/trackml-particle-identification

**Description** : Dataset pour reconstruction traces de particules chargées dans détecteurs cylindriques multicouches.

#### Structure

```python
"""
Structure TrackML dataset:

hits.csv:
  - hit_id: Identifiant hit
  - x, y, z: Position hit (mm)
  - volume_id, layer_id, module_id: Position détecteur
  - value: Signal amplitude

particles.csv:
  - particle_id: Identifiant particule
  - vx, vy, vz: Vertex origine
  - px, py, pz: Momentum initial
  - q: Charge
  - nhits: Nombre hits associés

truth.csv:
  - hit_id, particle_id: Association hits-particules
  - weight: Poids association (pour pile-up)
"""
```

#### Chargement et Préparation

```python
import pandas as pd
import numpy as np

def load_trackml_event(event_path):
    """
    Charge événement TrackML
    
    Args:
        event_path: Chemin vers dossier événement
    
    Returns:
        Dictionary avec hits, particles, truth
    """
    hits = pd.read_csv(f"{event_path}/hits.csv")
    particles = pd.read_csv(f"{event_path}/particles.csv")
    truth = pd.read_csv(f"{event_path}/truth.csv")
    
    # Joindre truth avec hits
    hits_with_truth = hits.merge(truth, on='hit_id')
    
    return {
        'hits': hits,
        'particles': particles,
        'truth': truth,
        'hits_with_truth': hits_with_truth
    }

def prepare_trackml_for_gnn(hits, truth, particles):
    """
    Prépare données TrackML pour Graph Neural Network
    
    Args:
        hits: DataFrame hits
        truth: DataFrame truth
        particles: DataFrame particles
    
    Returns:
        Graph avec nodes (hits) et edges (candidats connexions)
    """
    # Features nodes: position, détecteur info
    node_features = hits[['x', 'y', 'z', 'volume_id', 'layer_id']].values
    
    # Créer edges: hits consécutifs dans même layer ou layers adjacents
    # (simplifié, vraie création edges plus complexe)
    
    # Labels edges: vraies connexions depuis truth
    edge_labels = create_edge_labels(hits, truth)
    
    return {
        'node_features': node_features,
        'edge_labels': edge_labels
    }
```

### Datasets Simulés Personnalisés

#### Génération avec Pythia

```python
import pythia8

def generate_jets_dataset(n_events=10000, output_file="jets_dataset.h5"):
    """
    Génère dataset jets avec Pythia8
    
    Args:
        n_events: Nombre événements générer
        output_file: Fichier sortie HDF5
    """
    pythia = pythia8.Pythia()
    pythia.readString("Beams:eCM = 13000")  # 13 TeV
    pythia.readString("HardQCD:all = on")   # QCD processus
    pythia.init()
    
    jets_data = []
    labels = []
    
    for i_event in range(n_events):
        if not pythia.next():
            continue
        
        # Extraire jets
        jets = extract_jets(pythia.event)
        
        for jet in jets:
            # Extraire features
            features = extract_jet_features(jet)
            
            # Label: quark ou gluon
            label = determine_jet_label(jet)
            
            jets_data.append(features)
            labels.append(label)
    
    # Sauvegarder
    import h5py
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('jets', data=np.array(jets_data))
        f.create_dataset('labels', data=np.array(labels))
    
    print(f"Dataset généré: {len(jets_data)} jets")
```

---

## C.2 Benchmarks de Compression

### Métriques Standard

#### Compression Ratio

$$\text{Compression Ratio} = \frac{\text{Paramètres Originaux}}{\text{Paramètres Compressés}}$$

```python
def calculate_compression_ratio(model_original, model_compressed):
    """
    Calcule ratio compression
    
    Args:
        model_original: Modèle original
        model_compressed: Modèle compressé
    
    Returns:
        Dictionary avec métriques
    """
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    params_original = count_parameters(model_original)
    params_compressed = count_parameters(model_compressed)
    
    compression_ratio = params_original / params_compressed
    
    # Taille mémoire approximative (bytes)
    size_original = params_original * 4  # float32 = 4 bytes
    size_compressed = params_compressed * 4
    memory_reduction = size_original / size_compressed
    
    return {
        'compression_ratio': compression_ratio,
        'params_original': params_original,
        'params_compressed': params_compressed,
        'memory_reduction': memory_reduction,
        'size_original_mb': size_original / (1024**2),
        'size_compressed_mb': size_compressed / (1024**2)
    }

# Exemple
metrics = calculate_compression_ratio(model_original, model_pruned)
print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")
print(f"Memory reduction: {metrics['memory_reduction']:.2f}x")
```

#### Accuracy Drop

$$\text{Accuracy Drop} = \text{Accuracy Originale} - \text{Accuracy Compressée}$$

```python
def evaluate_accuracy_drop(model_original, model_compressed, test_loader):
    """
    Évalue perte accuracy après compression
    
    Args:
        model_original: Modèle original
        model_compressed: Modèle compressé
        test_loader: DataLoader test
    
    Returns:
        Dictionary avec métriques
    """
    def evaluate_model(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total
    
    acc_original = evaluate_model(model_original, test_loader)
    acc_compressed = evaluate_model(model_compressed, test_loader)
    
    accuracy_drop = acc_original - acc_compressed
    
    return {
        'accuracy_original': acc_original,
        'accuracy_compressed': acc_compressed,
        'accuracy_drop': accuracy_drop,
        'relative_drop': 100 * accuracy_drop / acc_original
    }
```

#### Speedup

$$\text{Speedup} = \frac{\text{Latence Originale}}{\text{Latence Compressée}}$$

```python
import time
import torch

def measure_latency(model, input_shape, n_runs=100, device='cpu'):
    """
    Mesure latence modèle
    
    Args:
        model: Modèle à évaluer
        input_shape: Shape entrée (batch_size, ...)
        n_runs: Nombre runs pour moyenne
        device: Device (cpu/cuda)
    
    Returns:
        Latence moyenne (ms)
    """
    model.eval()
    model = model.to(device)
    
    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Mesure
    times = []
    for _ in range(n_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)

def calculate_speedup(model_original, model_compressed, input_shape):
    """
    Calcule speedup compression
    """
    lat_orig, _ = measure_latency(model_original, input_shape)
    lat_comp, _ = measure_latency(model_compressed, input_shape)
    
    speedup = lat_orig / lat_comp
    
    return {
        'latency_original_ms': lat_orig,
        'latency_compressed_ms': lat_comp,
        'speedup': speedup
    }
```

### Benchmarks Populaires

#### ImageNet Classification

**Dataset** : ImageNet (1.2M images, 1000 classes)

**Modèles Testés** :
- ResNet-50, ResNet-101
- EfficientNet-B0 à B7
- MobileNet-V2, V3
- Vision Transformers

**Résultats Typiques** :

| Modèle | Original Accuracy | Pruned (50% sparsity) | Quantized (INT8) |
|--------|-------------------|----------------------|------------------|
| ResNet-50 | 76.1% | 75.8% | 75.9% |
| EfficientNet-B0 | 77.1% | 76.5% | 76.8% |
| MobileNet-V2 | 72.0% | 71.2% | 71.5% |

**Code Benchmark** :
```python
def benchmark_imagenet_compression(model, pruning_ratio=0.5):
    """
    Benchmark compression sur ImageNet
    
    Args:
        model: Modèle à compresser
        pruning_ratio: Ratio pruning (0.0-1.0)
    """
    # Charger ImageNet validation set
    from torchvision.datasets import ImageNet
    val_dataset = ImageNet(root='data/imagenet', split='val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
    
    # Évaluer original
    acc_original = evaluate_accuracy(model, val_loader)
    
    # Pruning
    model_pruned = prune_model(model, pruning_ratio)
    acc_pruned = evaluate_accuracy(model_pruned, val_loader)
    
    # Quantization
    model_quantized = quantize_model(model)
    acc_quantized = evaluate_accuracy(model_quantized, val_loader)
    
    # Métriques
    compression = calculate_compression_ratio(model, model_pruned)
    speedup = calculate_speedup(model, model_pruned, (1, 3, 224, 224))
    
    return {
        'accuracy_original': acc_original,
        'accuracy_pruned': acc_pruned,
        'accuracy_quantized': acc_quantized,
        'compression_ratio': compression['compression_ratio'],
        'speedup': speedup['speedup']
    }
```

#### CIFAR-10 Classification

**Dataset** : CIFAR-10 (60k images, 10 classes)

**Modèles Testés** :
- VGG-16, VGG-19
- ResNet-18, ResNet-50
- MobileNet
- DenseNet

**Procédure Benchmark** :
```python
def benchmark_cifar10(model_name='ResNet18', compression_methods=['pruning', 'quantization']):
    """
    Benchmark compression sur CIFAR-10
    """
    from torchvision.models import resnet18
    from torchvision.datasets import CIFAR10
    
    # Charger modèle et données
    model = resnet18(num_classes=10)
    model.load_state_dict(torch.load(f'checkpoints/{model_name}_cifar10.pth'))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = CIFAR10(root='data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    
    results = {}
    
    # Original
    acc_orig = evaluate_accuracy(model, test_loader)
    results['original'] = {'accuracy': acc_orig}
    
    # Compression methods
    if 'pruning' in compression_methods:
        model_pruned = prune_model(model, ratio=0.5)
        acc_pruned = evaluate_accuracy(model_pruned, test_loader)
        results['pruning'] = {
            'accuracy': acc_pruned,
            'accuracy_drop': acc_orig - acc_pruned
        }
    
    if 'quantization' in compression_methods:
        model_quant = quantize_model(model)
        acc_quant = evaluate_accuracy(model_quant, test_loader)
        results['quantization'] = {
            'accuracy': acc_quant,
            'accuracy_drop': acc_orig - acc_quant
        }
    
    return results
```

#### GLUE Benchmark (NLP)

**Dataset** : GLUE (General Language Understanding Evaluation)

**Tâches** :
- Sentiment Analysis (SST-2)
- Natural Language Inference (MNLI, QNLI)
- Question Answering (QQP, QNLI)
- Semantic Similarity (STS-B)

**Modèles Testés** :
- BERT-base, BERT-large
- RoBERTa
- DistilBERT

**Exemple Benchmark** :
```python
def benchmark_glue_compression(model, task='sst-2'):
    """
    Benchmark compression sur GLUE
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    
    # Charger données
    dataset = load_dataset('glue', task)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Évaluer original
    acc_original = evaluate_glue_task(model, dataset['validation'])
    
    # Compression
    model_compressed = compress_transformer(model, method='quantization')
    acc_compressed = evaluate_glue_task(model_compressed, dataset['validation'])
    
    return {
        'task': task,
        'accuracy_original': acc_original,
        'accuracy_compressed': acc_compressed,
        'accuracy_drop': acc_original - acc_compressed
    }
```

### Benchmarks HEP Spécifiques

#### Jet Tagging Benchmarks

```python
def benchmark_jet_tagging_compression(model, test_loader):
    """
    Benchmark compression pour jet tagging
    """
    # Métriques HEP spécifiques
    def calculate_hep_metrics(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in loader:
                outputs = model(data)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        
        # Accuracy
        accuracy = (preds == labels).float().mean().item()
        
        # Signal efficiency et background rejection (pour classification binaire)
        if len(torch.unique(labels)) == 2:
            signal_mask = labels == 1
            background_mask = labels == 0
            
            signal_efficiency = (preds[signal_mask] == labels[signal_mask]).float().mean().item()
            background_rejection = (preds[background_mask] != labels[background_mask]).float().mean().item()
            
            return {
                'accuracy': accuracy,
                'signal_efficiency': signal_efficiency,
                'background_rejection': background_rejection
            }
        
        return {'accuracy': accuracy}
    
    # Évaluer original
    metrics_original = calculate_hep_metrics(model, test_loader)
    
    # Compression
    model_compressed = apply_compression(model)
    metrics_compressed = calculate_hep_metrics(model_compressed, test_loader)
    
    return {
        'original': metrics_original,
        'compressed': metrics_compressed
    }
```

---

## C.3 Métriques d'Évaluation

### Classification

#### Métriques de Base

**Accuracy** :

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    """
    Calcule métriques classification
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
    
    Returns:
        Dictionary avec métriques
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
```

**Precision** :

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** :

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1-Score** :

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Matrice de Confusion

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Affiche matrice confusion
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        class_names: Noms classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
```

#### ROC Curve et AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

def plot_roc_curve(y_true, y_scores):
    """
    Affiche courbe ROC
    
    Args:
        y_true: Labels vrais (binaires)
        y_scores: Scores probabilités
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

# Calcul AUC
auc_score = roc_auc_score(y_true, y_scores)
```

### Régression

#### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

#### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

#### R² Score

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    """
    Calcule métriques régression
    
    Args:
        y_true: Valeurs vraies
        y_pred: Prédictions
    
    Returns:
        Dictionary avec métriques
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }
```

### Métriques Physique des Particules

#### Signal Efficiency et Background Rejection

**Signal Efficiency** :

$$\epsilon_{\text{signal}} = \frac{\text{Signal sélectionné}}{\text{Signal total}}$$

**Background Rejection** :

$$R_{\text{background}} = 1 - \frac{\text{Background sélectionné}}{\text{Background total}}$$

```python
def calculate_hep_metrics(y_true, y_pred, y_scores=None, threshold=0.5):
    """
    Calcule métriques spécifiques HEP
    
    Args:
        y_true: Labels (0=background, 1=signal)
        y_pred: Prédictions binaires
        y_scores: Scores probabilités (optionnel)
        threshold: Seuil classification
    
    Returns:
        Dictionary avec métriques HEP
    """
    signal_mask = y_true == 1
    background_mask = y_true == 0
    
    # Si scores fournis, utiliser seuil
    if y_scores is not None:
        y_pred = (y_scores >= threshold).astype(int)
    
    # Signal efficiency
    signal_selected = np.sum((y_pred == 1) & signal_mask)
    signal_total = np.sum(signal_mask)
    signal_efficiency = signal_selected / signal_total if signal_total > 0 else 0
    
    # Background rejection
    background_selected = np.sum((y_pred == 1) & background_mask)
    background_total = np.sum(background_mask)
    background_rejection = 1 - (background_selected / background_total if background_total > 0 else 0)
    
    return {
        'signal_efficiency': signal_efficiency,
        'background_rejection': background_rejection
    }
```

#### ROC Curve pour HEP

```python
def plot_hep_roc_curve(y_true, y_scores):
    """
    Affiche courbe ROC avec métriques HEP
    
    Args:
        y_true: Labels (0=background, 1=signal)
        y_scores: Scores probabilités signal
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    # Calculer background rejection (1 - FPR)
    background_rejection = 1 - fpr
    
    plt.figure(figsize=(10, 6))
    
    # ROC standard
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate (Background Efficiency)')
    plt.ylabel('True Positive Rate (Signal Efficiency)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Signal Efficiency vs Background Rejection
    plt.subplot(1, 2, 2)
    plt.plot(background_rejection, tpr, lw=2, label=f'AUC = {auc_score:.3f}')
    plt.xlabel('Background Rejection')
    plt.ylabel('Signal Efficiency')
    plt.title('Signal Efficiency vs Background Rejection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

#### Significance

$$\text{Significance} = \frac{S}{\sqrt{S + B}}$$

où S = signal, B = background

```python
def calculate_significance(signal_count, background_count):
    """
    Calcule significance
    
    Args:
        signal_count: Nombre événements signal
        background_count: Nombre événements background
    
    Returns:
        Significance
    """
    if signal_count + background_count == 0:
        return 0.0
    return signal_count / np.sqrt(signal_count + background_count)
```

### Métriques Compression

#### Compression Metrics Summary

```python
def comprehensive_compression_report(model_original, model_compressed, 
                                     test_loader, input_shape):
    """
    Génère rapport complet compression
    
    Args:
        model_original: Modèle original
        model_compressed: Modèle compressé
        test_loader: DataLoader test
        input_shape: Shape entrée pour latence
    
    Returns:
        Dictionary avec toutes métriques
    """
    # Métriques modèle
    compression = calculate_compression_ratio(model_original, model_compressed)
    
    # Métriques performance
    accuracy = evaluate_accuracy_drop(model_original, model_compressed, test_loader)
    
    # Métriques latence
    speedup = calculate_speedup(model_original, model_compressed, input_shape)
    
    return {
        **compression,
        **accuracy,
        **speedup,
        'summary': {
            'compression_achieved': f"{compression['compression_ratio']:.2f}x",
            'memory_saved': f"{compression['memory_reduction']:.2f}x",
            'accuracy_drop': f"{accuracy['accuracy_drop']:.2f}%",
            'speedup': f"{speedup['speedup']:.2f}x"
        }
    }

# Utilisation
report = comprehensive_compression_report(
    model_original, model_compressed, test_loader, (1, 3, 224, 224)
)
print(report['summary'])
```

---

## Exercices Pratiques

### Exercice C.1
Chargez dataset CERN Open Data et analysez distribution propriétés muons.

### Exercice C.2
Créez benchmark compression pour modèle jet tagging et comparez différentes méthodes.

### Exercice C.3
Calculez métriques HEP (signal efficiency, background rejection) pour classifier binaire et tracez courbes ROC.

---

*Retour à la [Table des Matières](../../INDEX.md)*
