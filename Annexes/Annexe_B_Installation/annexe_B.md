# Annexe B : Guide d'Installation et Configuration

---

## Introduction

Cette annexe fournit des guides détaillés pour installer et configurer tous les outils nécessaires pour suivre les exemples et exercices de ce livre. Nous couvrons les environnements Python, les frameworks de deep learning, les outils FPGA, et la configuration GPU/CUDA.

---

## Plan de l'Annexe

1. [B.1 Environnement Python](#b1-environnement-python)
2. [B.2 Installation de PyTorch/TensorFlow](#b2-installation-de-pytorchtensorflow)
3. [B.3 Outils FPGA (Vivado, hls4ml)](#b3-outils-fpga-vivado-hls4ml)
4. [B.4 Configuration GPU/CUDA](#b4-configuration-gpucuda)

---

## B.1 Environnement Python

### Installation de Python

#### Vérifier Version Existante

```bash
# Vérifier version Python installée
python --version
python3 --version

# Vérifier pip
pip --version
pip3 --version
```

**Note** : Python 3.10 ou supérieur est recommandé pour ce livre.

#### Installation Python (si nécessaire)

**Linux** :
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# CentOS/RHEL
sudo yum install python3.10 python3-pip
```

**macOS** :
```bash
# Avec Homebrew
brew install python@3.10

# Ou télécharger depuis python.org
```

**Windows** :
- Télécharger depuis https://www.python.org/downloads/
- Cocher "Add Python to PATH" pendant installation

### Création d'un Environnement Virtuel

Les environnements virtuels isolent les dépendances de chaque projet.

#### Méthode 1 : venv (Recommandé)

```bash
# Créer environnement
python3 -m venv venv_hep_ml

# Activation (Linux/macOS)
source venv_hep_ml/bin/activate

# Activation (Windows)
venv_hep_ml\Scripts\activate

# Désactivation
deactivate
```

#### Méthode 2 : Conda

Conda est utile pour gérer packages compilés et dépendances complexes.

```bash
# Installation Miniconda (minimal)
# Télécharger depuis: https://docs.conda.io/en/latest/miniconda.html

# Créer environnement
conda create -n hep_ml python=3.10

# Activation
conda activate hep_ml

# Désactivation
conda deactivate
```

#### Comparaison venv vs Conda

| Critère | venv | Conda |
|---------|------|-------|
| Simplicité | Simple, intégré Python | Plus complexe |
| Packages | Via pip seulement | pip + conda packages |
| Packages compilés | Peut nécessiter compilateurs | Binaires précompilés |
| Taille | Léger | Plus volumineux |

### Gestion des Dépendances

#### Créer requirements.txt

```txt
# Core Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0

# Deep Learning Frameworks
torch>=2.0.0
tensorflow>=2.10.0

# Réseaux de Tenseurs
tensorly>=0.8.0
tntorch>=1.0.0

# FPGA Deployment
hls4ml>=0.7.0

# Physique des Particules
uproot>=5.0.0
awkward>=2.0.0
coffea>=2022.0.0

# Utilitaires
jupyter>=1.0.0
jupyterlab>=3.0.0
tqdm>=4.60.0
scikit-learn>=1.0.0

# Développement
black>=22.0.0
flake8>=4.0.0
pytest>=7.0.0
mypy>=0.950
```

#### Installation avec pip

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer depuis requirements.txt
pip install -r requirements.txt

# Installation sélective
pip install numpy scipy matplotlib
```

#### Installation avec Conda

```bash
# Créer environnement avec packages
conda create -n hep_ml python=3.10 numpy scipy matplotlib

# Activer environnement
conda activate hep_ml

# Installer packages supplémentaires
conda install -c conda-forge jupyter tensorly

# Installer depuis requirements.txt avec pip (dans environnement conda)
pip install -r requirements.txt
```

### Vérification Installation

```python
import sys
print(f"Python version: {sys.version}")

# Vérifier packages installés
packages = ['numpy', 'scipy', 'matplotlib', 'torch', 'tensorflow', 
            'tensorly', 'uproot', 'jupyter']

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {pkg}: {version}")
    except ImportError:
        print(f"✗ {pkg}: NOT INSTALLED")
```

---

## B.2 Installation de PyTorch/TensorFlow

### PyTorch

#### Installation CPU

```bash
# Version stable (CPU only)
pip install torch torchvision torchaudio

# Ou avec index spécifique
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Installation avec CUDA

```bash
# Vérifier version CUDA disponible
nvidia-smi

# PyTorch avec CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch avec CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Utiliser conda (alternative)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Vérification PyTorch

```python
import torch

# Informations de base
print(f"PyTorch version: {torch.__version__}")

# Vérifier CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Test de calcul
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)
z = torch.matmul(x, y)
print(f"Matrix multiplication successful on {device}")
```

#### Installation PyTorch depuis Source (Avancé)

```bash
# Cloner repository
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch

# Installer dépendances
pip install -r requirements.txt

# Compiler (peut prendre plusieurs heures)
python setup.py install
```

### TensorFlow

#### Installation CPU

```bash
# TensorFlow 2.x (CPU only)
pip install tensorflow

# Ou version spécifique
pip install tensorflow==2.13.0
```

#### Installation avec GPU

```bash
# TensorFlow avec GPU support
pip install tensorflow[and-cuda]

# Ou séparément (après installation CUDA/cuDNN)
pip install tensorflow-gpu

# Avec conda
conda install -c conda-forge tensorflow-gpu
```

**Note** : TensorFlow 2.x inclut généralement le support GPU. Utilisez `tensorflow[and-cuda]` pour installation complète.

#### Vérification TensorFlow

```python
import tensorflow as tf

# Informations de base
print(f"TensorFlow version: {tf.__version__}")

# Vérifier GPU
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Liste tous les devices
print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Test de calcul
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print(f"\nMatrix multiplication successful")
    print(f"Result shape: {c.shape}")
```

#### Problèmes Courants

**Problème** : CUDA not found
```bash
# Vérifier CUDA installation
nvcc --version
nvidia-smi

# Réinstaller TensorFlow avec CUDA
pip uninstall tensorflow
pip install tensorflow[and-cuda]
```

**Problème** : Version incompatibility
```bash
# Vérifier compatibilité versions
# https://www.tensorflow.org/install/source#gpu

# Installer version compatible
pip install tensorflow==2.10.0  # Exemple version spécifique
```

---

## B.3 Outils FPGA (Vivado, hls4ml)

### Installation de Vivado HLS

#### Prérequis

- Système d'exploitation supporté (Linux recommandé)
- 16+ GB RAM recommandé
- 100+ GB espace disque
- Licence Xilinx (gratuite pour étudiants/académiques)

#### Installation Vivado

1. **Créer compte Xilinx**
   - Aller sur https://www.xilinx.com/
   - Créer compte (gratuit)

2. **Télécharger Vivado**
   - Vivado Design Suite (inclut HLS)
   - Version recommandée: 2022.1 ou plus récent
   - Télécharger depuis: https://www.xilinx.com/support/download.html

3. **Installation Linux**
   ```bash
   # Rendre exécutable
   chmod +x Xilinx_Unified_2022.1_0420_0327_Lin64.bin
   
   # Lancer installation
   ./Xilinx_Unified_2022.1_0420_0327_Lin64.bin
   
   # Suivre assistant installation
   # Sélectionner: "Vivado HL Design Edition" ou "Vivado HL System Edition"
   ```

4. **Configuration Variables d'Environnement**
   
   **Bash** (`~/.bashrc`):
   ```bash
   # Vivado settings
   export XILINX_VIVADO=/tools/Xilinx/Vivado/2022.1
   source $XILINX_VIVADO/settings64.sh
   
   # Ajouter au PATH
   export PATH=$XILINX_VIVADO/bin:$PATH
   ```
   
   **Activation**:
   ```bash
   source ~/.bashrc
   ```

5. **Vérification**
   ```bash
   # Vérifier Vivado
   vivado -version
   
   # Vérifier Vivado HLS (maintenant intégré dans Vitis HLS)
   vitis_hls -version
   ```

#### Configuration Licence

```bash
# Générer license file depuis Xilinx
# Uploader sur: https://www.xilinx.com/getlicense

# Configurer licence
export XILINXD_LICENSE_FILE=/path/to/license.lic
# Ou
export LM_LICENSE_FILE=/path/to/license.lic
```

### Installation de hls4ml

#### Installation avec pip

```bash
# Installation depuis PyPI
pip install hls4ml

# Vérification
python -c "import hls4ml; print(hls4ml.__version__)"
```

#### Installation depuis Source

```bash
# Cloner repository
git clone https://github.com/fastmachinelearning/hls4ml.git
cd hls4ml

# Installation en mode développement
pip install -e .

# Ou installation normale
pip install .
```

#### Vérification hls4ml

```python
import hls4ml
import numpy as np
from tensorflow import keras

print(f"hls4ml version: {hls4ml.__version__}")

# Test simple: convertir modèle Keras
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(5,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Configuration hls4ml
config = hls4ml.utils.config_from_keras_model(model)

# Créer projet HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_test'
)

print("hls4ml configuration successful!")
```

#### Configuration hls4ml

```python
# Configuration personnalisée
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1
config['LayerName']['dense']['Strategy'] = 'Latency'
config['LayerName']['dense']['ReuseFactor'] = 1

hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_project'
)
```

### Workflow Complet

```python
"""
Exemple workflow complet hls4ml
"""
import hls4ml
from tensorflow import keras
import numpy as np

# 1. Créer/trainer modèle Keras
model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(10,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# 2. Configurer hls4ml
config = hls4ml.utils.config_from_keras_model(model)
config['Model']['Precision'] = 'ap_fixed<16,6>'

# 3. Convertir vers HLS
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='my_hls_project',
    fpga_part='xcku115-flvb2104-2-e'  # Part number FPGA
)

# 4. Vérifier configuration
hls_model.compile()

# 5. Créer projet Vivado
hls_model.build()

# 6. Vérifier ressources utilisées
print(f"Resources: {hls_model.get_used_resources()}")
```

---

## B.4 Configuration GPU/CUDA

### Installation de CUDA

#### Vérifier GPU NVIDIA

```bash
# Vérifier présence GPU NVIDIA
lspci | grep -i nvidia

# Ou sur Windows
# Device Manager → Display adapters
```

#### Installation CUDA Toolkit

**Linux** :
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Variables d'environnement
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Windows** :
1. Télécharger CUDA Toolkit depuis https://developer.nvidia.com/cuda-downloads
2. Installer exécutable (.exe)
3. Suivre assistant installation

#### Installation Drivers NVIDIA

```bash
# Linux - Ubuntu
sudo ubuntu-drivers autoinstall

# Ou manuellement
sudo apt install nvidia-driver-520

# Redémarrer système
sudo reboot
```

#### Vérification CUDA

```bash
# Vérifier drivers NVIDIA
nvidia-smi

# Vérifier CUDA compiler
nvcc --version

# Output devrait montrer:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on ...
# Cuda compilation tools, release 11.8, ...
```

### Configuration PyTorch avec CUDA

#### Installation PyTorch CUDA

```bash
# PyTorch avec CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch avec CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Test PyTorch CUDA

```python
import torch

# Vérifier disponibilité CUDA
if torch.cuda.is_available():
    print("✓ CUDA available")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    
    # Information détaillée par GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    # Test de calcul
    print("\nTesting GPU computation...")
    device = torch.device("cuda:0")
    x = torch.randn(5000, 5000).to(device)
    y = torch.randn(5000, 5000).to(device)
    z = torch.matmul(x, y)
    print(f"✓ Matrix multiplication successful")
    print(f"  Result shape: {z.shape}")
    print(f"  Result device: {z.device}")
else:
    print("✗ CUDA not available")
    print("  Using CPU instead")
```

### Configuration TensorFlow avec GPU

#### Installation TensorFlow GPU

```bash
# TensorFlow avec support CUDA complet
pip install tensorflow[and-cuda]

# Ou séparément
pip install tensorflow tensorflow-gpu
```

#### Test TensorFlow GPU

```python
import tensorflow as tf

# Vérifier GPU
print("TensorFlow version:", tf.__version__)
print("\nGPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"  {gpu}")
        print(f"    Name: {gpu.name}")
    
    # Test de calcul
    print("\nTesting GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([5000, 5000])
        b = tf.random.normal([5000, 5000])
        c = tf.matmul(a, b)
        print(f"✓ Matrix multiplication successful")
        print(f"  Result shape: {c.shape}")
else:
    print("  No GPU devices found")
```

### Optimisation GPU

#### Configuration Mémoire GPU

**PyTorch** :
```python
# Limiter mémoire allouée
torch.cuda.set_per_process_memory_fraction(0.9)

# Nettoyer cache
torch.cuda.empty_cache()

# Synchronisation explicite
torch.cuda.synchronize()
```

**TensorFlow** :
```python
# Limiter croissance mémoire
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Ou limiter mémoire totale
tf.config.experimental.set_memory_growth(gpus[0], False)
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
)
```

#### Multi-GPU Setup

**PyTorch** :
```python
import torch.nn as nn

# DataParallel (simple mais moins efficace)
model = nn.DataParallel(model)

# DistributedDataParallel (recommandé)
model = nn.parallel.DistributedDataParallel(model)
```

**TensorFlow** :
```python
# Strategy pour multi-GPU
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(...)
```

### Troubleshooting GPU

#### Problèmes Courants

**Problème** : CUDA out of memory
```python
# Solutions:
# 1. Réduire batch size
# 2. Nettoyer cache
torch.cuda.empty_cache()

# 3. Limiter mémoire
torch.cuda.set_per_process_memory_fraction(0.8)
```

**Problème** : Version mismatch
```bash
# Vérifier compatibilité
nvidia-smi  # Version driver
nvcc --version  # Version CUDA toolkit

# Réinstaller PyTorch/TensorFlow avec version compatible
```

**Problème** : GPU not detected
```bash
# Vérifier drivers
nvidia-smi

# Vérifier variables d'environnement
echo $CUDA_HOME
echo $LD_LIBRARY_PATH

# Réinstaller drivers si nécessaire
```

---

## Requirements.txt Complet

Voici un fichier `requirements.txt` complet pour ce livre:

```txt
# Core Scientific Computing
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
pandas>=1.3.0

# Deep Learning Frameworks
torch>=2.0.0
tensorflow>=2.10.0

# Réseaux de Tenseurs
tensorly>=0.8.0
tntorch>=1.0.0

# FPGA Deployment
hls4ml>=0.7.0

# Physique des Particules
uproot>=5.0.0
awkward>=2.0.0
coffea>=2022.0.0

# Utilitaires
jupyter>=1.0.0
jupyterlab>=3.0.0
tqdm>=4.60.0
scikit-learn>=1.0.0

# Visualisation
seaborn>=0.11.0
plotly>=5.0.0

# Développement
black>=22.0.0
flake8>=4.0.0
pytest>=7.0.0
mypy>=0.950

# Documentation
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
```

---

## Scripts d'Installation

### Script Linux/macOS

```bash
#!/bin/bash
# install_all.sh

set -e  # Exit on error

echo "Installing Python packages..."

# Créer environnement virtuel
python3 -m venv venv_hep_ml
source venv_hep_ml/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer packages
pip install -r requirements.txt

echo "Installation complete!"
echo "Activate environment with: source venv_hep_ml/bin/activate"
```

### Script Windows (PowerShell)

```powershell
# install_all.ps1

Write-Host "Installing Python packages..."

# Créer environnement virtuel
python -m venv venv_hep_ml
.\venv_hep_ml\Scripts\Activate.ps1

# Mettre à jour pip
python -m pip install --upgrade pip

# Installer packages
pip install -r requirements.txt

Write-Host "Installation complete!"
Write-Host "Activate environment with: .\venv_hep_ml\Scripts\Activate.ps1"
```

---

## Vérification Complète

Script Python pour vérifier toute l'installation:

```python
#!/usr/bin/env python3
"""
Script de vérification installation complète
"""

import sys

def check_package(name, import_name=None):
    """Vérifier package installé"""
    if import_name is None:
        import_name = name
    
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name:20s} {version}")
        return True
    except ImportError:
        print(f"✗ {name:20s} NOT INSTALLED")
        return False

print("="*60)
print("Vérification Installation")
print("="*60)

print("\nPython:")
print(f"  Version: {sys.version}")

packages = [
    ('numpy', None),
    ('scipy', None),
    ('matplotlib', None),
    ('pandas', None),
    ('torch', 'torch'),
    ('tensorflow', 'tensorflow'),
    ('tensorly', None),
    ('hls4ml', None),
    ('uproot', None),
    ('jupyter', None),
    ('sklearn', 'sklearn'),
]

print("\nPackages:")
all_ok = True
for name, import_name in packages:
    if not check_package(name, import_name):
        all_ok = False

print("\nGPU/CUDA:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.version.cuda}")
        print(f"  ✓ GPUs: {torch.cuda.device_count()}")
    else:
        print("  ✗ CUDA not available")
except:
    print("  ? Could not check CUDA")

print("\n" + "="*60)
if all_ok:
    print("✓ All packages installed successfully!")
else:
    print("✗ Some packages missing. Install with: pip install -r requirements.txt")
print("="*60)
```

---

*Retour à la [Table des Matières](../../INDEX.md)*
