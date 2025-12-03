# Exemples Pratiques - Livre IA HEP

---

## 📚 Vue d'Ensemble

Ce dossier contient des exemples pratiques complets et détaillés pour illustrer les concepts du livre. Chaque exemple inclut du code fonctionnel, des explications, et des résultats.

---

## 📋 Liste des Exemples

### 1. **01_Exemple_Trigger_Reel.md**
**Domain** : Physique des Particules / Trigger  
**Contenu** :
- Système de trigger IA pour LHC
- Contraintes de latence réelles (≤ 4 μs)
- Dataset CMS
- Modèle ultra-léger
- Quantification pour FPGA
- Métriques HEP (signal efficiency, background rejection)

**Objectifs** :
- Démontrer développement trigger avec contraintes temps réel
- Mesurer latence réelle
- Valider métriques performance HEP

---

### 2. **02_Compression_PyTorch_Complete.md**
**Domain** : Compression de Modèles  
**Contenu** :
- Workflow complet compression
- Pruning structuré
- Quantification INT8
- Knowledge Distillation
- Comparaison systématique

**Objectifs** :
- Montrer combinaison techniques compression
- Comparer trade-offs (accuracy, taille, latence)
- Visualiser résultats

---

### 3. **03_Tensor_Train_Probleme_Reel.md**
**Domain** : Réseaux de Tenseurs  
**Contenu** :
- Décomposition Tensor Train matrice dense
- Compression couche réseau
- Analyse compression vs erreur
- Intégration PyTorch

**Objectifs** :
- Démontrer utilisation pratique TT
- Analyser trade-offs compression
- Intégrer dans workflow ML

---

### 4. **04_Workflow_hls4ml_Complet.md**
**Domain** : Hardware / FPGA  
**Contenu** :
- Workflow complet Keras → hls4ml → FPGA
- Configuration et optimisation
- Simulation et validation
- Benchmarking ressources FPGA
- Tuning pour contraintes latence

**Objectifs** :
- Démontrer pipeline hls4ml complet
- Optimiser pour contraintes temps réel
- Valider performances

---

### 5. **05_Comparaison_FPGA_GPU_CPU.md**
**Domain** : Hardware / Performance  
**Contenu** :
- Benchmarking latence et throughput
- Comparaison consommation énergétique
- Analyse efficacité par plateforme
- Recommandations selon use case

**Objectifs** :
- Comparer performances plateformes
- Analyser trade-offs
- Guider choix plateforme

---

### 6. **06_Reconstruction_Evenement_Complet.md**
**Domain** : Physique des Particules / Reconstruction  
**Contenu** :
- Reconstruction traces depuis hits
- Classification et b-tagging jets
- Identification leptons
- Reconstruction MET avec corrections
- Analyse complète événement
- Visualisation

**Objectifs** :
- Démontrer workflow reconstruction complet
- Intégrer ML dans pipeline reconstruction
- Analyser événements type

---

## 🚀 Utilisation

### Prérequis

```bash
pip install torch torchvision numpy matplotlib tqdm
pip install tensorly tntorch
pip install hls4ml  # Pour exemples FPGA
pip install uproot awkward  # Pour données HEP
```

### Exécution

Chaque exemple est autonome. Ouvrir fichier `.md` correspondant et exécuter code sections par sections.

---

## 📊 Structure des Exemples

Chaque exemple suit cette structure :

1. **Contexte** : Description problème
2. **Objectif** : Ce qu'on va démontrer
3. **Code** : Implémentation complète
4. **Résultats** : Outputs et métriques
5. **Analyse** : Interprétation résultats
6. **Points Clés** : Takeaways

---

## 💡 Intégration dans Livre

Ces exemples peuvent être :
- Intégrés dans chapitres correspondants
- Utilisés comme exercices pratiques
- Référencés depuis sections théoriques
- Convertis en notebooks Jupyter

---

*Exemples créés pour illustrer concepts pratiques du livre*

