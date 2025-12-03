# 🔬 Intelligence Artificielle Avancée pour la Physique des Hautes Énergies

## Réseaux de Tenseurs, Compression de Modèles et Déploiement Hardware

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-yellow?style=flat-square)](https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies)

---

## 📖 À Propos de ce Livre

Ce livre est conçu comme une ressource complète pour les chercheurs et ingénieurs travaillant à l'intersection de l'intelligence artificielle et de la physique des hautes énergies. Il couvre les techniques avancées de compression de modèles deep learning, les réseaux de tenseurs, et leur déploiement sur hardware spécialisé (FPGA).

**Repository GitHub** : [https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies](https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies)

### Public Cible

Ce livre s'adresse à :

- 🔬 **Physiciens** souhaitant maîtriser les techniques d'IA modernes pour leurs recherches
- 💻 **Informaticiens** s'intéressant aux applications scientifiques et au calcul haute performance
- ⚡ **Ingénieurs hardware** travaillant sur le déploiement de modèles ML sur FPGA/ASIC
- 🧠 **Chercheurs en machine learning** explorant les réseaux de tenseurs et la compression
- 🎓 **Étudiants avancés** en physique, informatique ou ingénierie
- 🏛️ **Professionnels CERN/LHC** cherchant à optimiser les systèmes de trigger et de reconstruction

### Prérequis

- **Mathématiques** : Connaissances de base en algèbre linéaire, calcul matriciel, probabilités
- **Programmation** : Familiarité avec Python (numpy, pandas) et les concepts de programmation orientée objet
- **Machine Learning** : Notions fondamentales de deep learning (réseaux de neurones, backpropagation, optimisation)
- **Physique** (optionnel) : Intérêt pour la physique des particules facilitant la compréhension des applications

### 🎯 Objectifs d'Apprentissage

À l'issue de ce livre, vous serez capable de :

✅ Comprendre et implémenter les réseaux de tenseurs pour la compression de modèles  
✅ Appliquer les techniques de pruning, quantification et distillation  
✅ Convertir des modèles ML vers des formats optimisés pour FPGA  
✅ Déployer des modèles sur hardware spécialisé avec hls4ml  
✅ Résoudre des problèmes concrets de trigger et reconstruction en HEP  
✅ Optimiser les pipelines ML pour contraintes temps réel  
✅ Contribuer à des projets open source dans le domaine

---

## 🎯 Points Forts

- **📖 Contenu Complet** : 28 chapitres couvrant de la théorie à l'implémentation
- **💻 Code Pratique** : 6 exemples complets avec code fonctionnel et testé
- **🔬 Applications Réelles** : Cas d'usage concrets du CERN et du LHC
- **⚡ Focus Hardware** : Détails sur le déploiement FPGA avec hls4ml
- **📊 Visualisations** : Graphiques, diagrammes et exemples visuels
- **🔗 Ressources** : Glossaire complet, références, datasets et outils
- **🌍 Open Source** : Tout le contenu est librement accessible et modifiable

## 📚 Structure du Livre

```
Livre_IA_HEP/
├── Partie_I_Fondements/
│   ├── Chapitre_01_Introduction_HEP/
│   ├── Chapitre_02_Algebre_Lineaire/
│   └── Chapitre_03_Deep_Learning/
├── Partie_II_Reseaux_Tenseurs/
│   ├── Chapitre_04_Introduction_Tenseurs/
│   ├── Chapitre_05_Decompositions/
│   ├── Chapitre_06_Physique_Quantique/
│   └── Chapitre_07_Conversion_NN_TN/
├── Partie_III_Compression/
│   ├── Chapitre_08_Pruning/
│   ├── Chapitre_09_Quantification/
│   ├── Chapitre_10_Knowledge_Distillation/
│   ├── Chapitre_11_Low_Rank/
│   └── Chapitre_12_pQuant/
├── Partie_IV_Hardware/
│   ├── Chapitre_13_FPGA_Introduction/
│   ├── Chapitre_14_NN_sur_FPGA/
│   ├── Chapitre_15_hls4ml/
│   ├── Chapitre_16_Hardware_NAS/
│   └── Chapitre_17_TN_Hardware/
├── Partie_V_Applications_HEP/
│   ├── Chapitre_18_Trigger_DAQ/
│   ├── Chapitre_19_Reconstruction/
│   ├── Chapitre_20_Anomalies/
│   └── Chapitre_21_Simulation/
├── Partie_VI_Implementation/
│   ├── Chapitre_22_Python_DL/
│   ├── Chapitre_23_Decompositions_Code/
│   ├── Chapitre_24_Cpp_Performance/
│   └── Chapitre_25_Pipelines/
├── Partie_VII_Recherche/
│   ├── Chapitre_26_Methodologie/
│   ├── Chapitre_27_Open_Source/
│   └── Chapitre_28_Communication/
└── Annexes/
    ├── Annexe_A_Maths/
    ├── Annexe_B_Installation/
    ├── Annexe_C_Datasets/
    ├── Annexe_D_Glossaire/
    └── Annexe_E_Ressources/
└── Exemples_Pratiques/
    ├── 01_Exemple_Trigger_Reel.md
    ├── 02_Compression_PyTorch_Complete.md
    ├── 03_Tensor_Train_Probleme_Reel.md
    ├── 04_Workflow_hls4ml_Complet.md
    ├── 05_Comparaison_FPGA_GPU_CPU.md
    └── 06_Reconstruction_Evenement_Complet.md
```

### 📖 Détail des Parties

**Partie I : Fondements Théoriques** (Chapitres 1-3)
- Introduction à la physique des hautes énergies et au CERN
- Algèbre linéaire avancée (SVD, low-rank, produits tensoriels)
- Deep learning moderne (CNNs, Transformers, optimisation)

**Partie II : Réseaux de Tenseurs** (Chapitres 4-7)
- Fondements des réseaux de tenseurs
- Décompositions (CP, Tucker, Tensor Train, HT, Tensor Ring)
- Applications en physique quantique (MPS, PEPS, MERA)
- Conversion de réseaux de neurones en réseaux de tenseurs

**Partie III : Compression de Modèles** (Chapitres 8-12)
- Pruning (structuré, non-structuré, dynamique, Lottery Ticket)
- Quantification (PTQ, QAT, mixed-precision, binaire/ternaire)
- Knowledge Distillation (logits, features, relations)
- Approximations low-rank (SVD, LoRA)
- Bibliothèque pQuant pour compression

**Partie IV : Hardware** (Chapitres 13-17)
- Introduction aux FPGA et HLS
- Déploiement de réseaux de neurones sur FPGA
- Framework hls4ml (CERN)
- Hardware-Aware Neural Architecture Search
- Réseaux de tenseurs sur hardware

**Partie V : Applications HEP** (Chapitres 18-21)
- Systèmes de trigger et DAQ
- Reconstruction d'événements (traces, jets, leptons, MET)
- Détection d'anomalies et nouvelle physique
- Simulation Monte Carlo avec GANs et Normalizing Flows

**Partie VI : Implémentation** (Chapitres 22-25)
- Python pour deep learning (PyTorch, TensorFlow)
- Implémentation de décompositions tensorielles
- Performance avec C++ (templates, parallélisation, pybind11)
- Pipelines de compression end-to-end

**Partie VII : Recherche** (Chapitres 26-28)
- Méthodologie de recherche scientifique
- Contribution open source
- Communication scientifique (articles, présentations, posters)

---

## 🚀 Installation et Configuration

### Prérequis Système

```bash
# Python 3.8+ requis
python --version

# Git pour cloner le repository
git clone https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies.git
cd Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies
```

### Installation des Dépendances

```bash
# Environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installation des packages Python
pip install -r requirements.txt

# Packages optionnels pour exemples spécifiques
pip install torch torchvision  # PyTorch
pip install tensorflow keras  # TensorFlow
pip install tensorly tntorch  # Réseaux de tenseurs
pip install hls4ml  # Déploiement FPGA
pip install uproot awkward  # Données HEP
```

Voir [Annexe B : Guide d'Installation](./Annexes/Annexe_B_Installation/annexe_B.md) pour les détails complets.

### Configuration FPGA (Optionnel)

Pour les chapitres sur FPGA et hls4ml :
- Vivado HLS ou Vitis HLS (Xilinx)
- Voir [Annexe B](./Annexes/Annexe_B_Installation/annexe_B.md) pour l'installation

---

## 💻 Exemples Pratiques

Le livre inclut **6 exemples pratiques complets** avec code fonctionnel :

### 1. 🔥 Exemple Trigger Réel
Système de trigger IA pour le LHC avec contraintes de latence réelles (≤ 4 μs)
- Dataset CMS avec uproot
- Modèle ultra-léger avec quantification INT8
- Métriques HEP (signal efficiency, background rejection)

### 2. 🗜️ Compression PyTorch Complète
Workflow end-to-end de compression d'un modèle ResNet-18
- Pruning structuré
- Quantification INT8 post-training
- Knowledge Distillation
- Comparaison systématique avec visualisations

### 3. 🔢 Tensor Train sur Problème Réel
Décomposition Tensor Train pour compresser une couche dense 1024×1024
- Analyse trade-off compression vs erreur
- Intégration dans un modèle PyTorch
- Test sur dataset MNIST

### 4. ⚡ Workflow hls4ml Complet
Conversion d'un modèle Keras vers FPGA avec hls4ml
- Configuration et optimisation pour latence
- Simulation et validation
- Estimation des ressources FPGA
- Benchmarking et tuning

### 5. 📊 Comparaison FPGA vs GPU vs CPU
Benchmarking complet des différentes plateformes hardware
- Latence, throughput, consommation énergétique
- Visualisations comparatives
- Recommandations par use case

### 6. 🔬 Reconstruction Événement Complet
Pipeline complet de reconstruction d'événements HEP
- Reconstruction de traces avec ML
- Classification de jets et b-tagging
- Identification de leptons
- Reconstruction MET corrigée
- Visualisation d'événements

Voir le [README des Exemples](./Exemples_Pratiques/README.md) pour plus de détails.

---

## 🚀 Comment Utiliser ce Livre

### 1. **Lecture Séquentielle** 📖
Pour une compréhension complète, suivez les parties dans l'ordre :
- Commencez par la **Partie I** pour les fondements
- Poursuivez avec la **Partie II** pour les réseaux de tenseurs
- Explorez la **Partie III** pour la compression
- Appliquez avec la **Partie IV** (hardware) et **Partie V** (applications)

### 2. **Référence Rapide** 🔍
- Utilisez [INDEX.md](./INDEX.md) pour naviguer rapidement
- Consultez les **Annexes** pour des références rapides
- Utilisez le **Glossaire** (Annexe D) pour les définitions

### 3. **Apprentissage Pratique** 💻
- Chaque chapitre contient des exemples de code
- Exécutez les **6 exemples pratiques** dans `Exemples_Pratiques/`
- Adaptez le code à vos propres projets

### 4. **Recherche et Contribution** 🔬
- Consultez la **Partie VII** pour la méthodologie de recherche
- Contribuez au projet via GitHub (voir section Contribution)

---

## 📝 Conventions et Style

- `` `Code inline` `` pour les noms de fonctions, variables, et commandes
- **Gras** pour les termes importants introduits pour la première fois
- *Italique* pour l'emphase
- Les blocs de code sont annotés avec le langage utilisé (Python, C++, etc.)
- Les formules mathématiques utilisent la notation LaTeX standard
- Les références aux chapitres utilisent des liens relatifs

---

## 🔗 Ressources Complémentaires

### Datasets et Outils

- **[CERN Open Data Portal](http://opendata.cern.ch/)** - Données ouvertes du LHC
- **[TrackML Challenge](https://www.kaggle.com/c/trackml-particle-identification)** - Challenge de reconstruction de traces
- **[Jet Tagging](https://opendata.cern.ch/record/14050)** - Données pour le tagging de jets

### Bibliothèques et Frameworks

- **[hls4ml](https://fastmachinelearning.org/hls4ml/)** - Conversion ML vers FPGA (CERN)
- **[pQuant](https://github.com/cern/pquant)** - Bibliothèque de compression de modèles
- **[TensorNetwork](https://github.com/google/TensorNetwork)** - Calculs avec réseaux de tenseurs
- **[TensorLy](https://tensorly.org/)** - Décompositions tensorielles en Python
- **[uproot](https://uproot.readthedocs.io/)** - Accès aux fichiers ROOT en Python

### Documentation Technique

- **[PyTorch Documentation](https://pytorch.org/docs/)** - Framework deep learning
- **[TensorFlow Documentation](https://www.tensorflow.org/api_docs)** - Framework ML
- **[Xilinx Vivado HLS](https://www.xilinx.com/products/design-tools/vivado/integration/esl-design.html)** - High-Level Synthesis

Voir [Annexe E : Ressources et Références](./Annexes/Annexe_E_Ressources/ressources.md) pour une liste exhaustive.

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Ce livre est un projet open source et évolutif.

### Comment Contribuer

1. **Fork** le repository
2. Créez une **branche** pour votre contribution (`git checkout -b feature/AmeliorationChapitre`)
3. **Commitez** vos modifications (`git commit -m 'Ajout de contenu sur...'`)
4. **Push** vers la branche (`git push origin feature/AmeliorationChapitre`)
5. Ouvrez une **Pull Request**

### Types de Contributions Appréciées

- ✅ Correction d'erreurs (typos, formules, code)
- ✅ Amélioration d'exemples existants
- ✅ Ajout d'exemples pratiques supplémentaires
- ✅ Traduction en d'autres langues
- ✅ Amélioration de la documentation
- ✅ Ajout de visualisations et diagrammes
- ✅ Tests et validation du code

### Normes de Contribution

- Respecter le style et format Markdown utilisé
- Tester le code avant de le soumettre
- Documenter les nouvelles fonctionnalités
- Suivre les conventions de nommage existantes

Voir [Chapitre 27 : Contribution Open Source](./Partie_VII_Recherche/Chapitre_27_Open_Source/27_introduction.md) pour les bonnes pratiques.

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

Vous êtes libre de :
- ✅ Utiliser ce contenu pour vos recherches et projets
- ✅ Modifier et adapter le contenu
- ✅ Partager et redistribuer
- ✅ Utiliser commercialement (avec attribution)

---

## 👥 Auteurs et Contact

**Auteur Principal** : Michael Germini  
**Email** : michael@germini.info  
**GitHub** : [@michaelgermini](https://github.com/michaelgermini)

### Remerciements

Ce livre s'inspire des travaux de recherche menés au CERN, en particulier dans les domaines de :
- Trigger systems avec IA (CMS, ATLAS)
- Reconstruction d'événements avec machine learning
- Optimisation hardware pour applications HEP

---

## 🗺️ Roadmap et État du Projet

### ✅ Statut Actuel (100% Complet)

- [x] **Partie I** : Fondements Théoriques (3/3 chapitres)
- [x] **Partie II** : Réseaux de Tenseurs (4/4 chapitres)
- [x] **Partie III** : Compression de Modèles (5/5 chapitres)
- [x] **Partie IV** : Hardware (5/5 chapitres)
- [x] **Partie V** : Applications HEP (4/4 chapitres)
- [x] **Partie VI** : Implémentation (4/4 chapitres)
- [x] **Partie VII** : Recherche (3/3 chapitres)
- [x] **Annexes** : Toutes les annexes (5/5)
- [x] **Exemples Pratiques** : 6 exemples complets

### 🔄 Améliorations Futures

- [ ] Conversion des exemples en notebooks Jupyter interactifs
- [ ] Ajout de tests automatisés pour le code
- [ ] Création d'un site web interactif
- [ ] Génération automatique en PDF/LaTeX
- [ ] Version multilingue (anglais, français)
- [ ] Vidéos tutoriels pour les concepts clés
- [ ] Intégration avec Google Colab pour exécution en ligne

---

## 📞 Support et Questions

- **Issues GitHub** : [Ouvrir une issue](https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies/issues) pour signaler des bugs ou suggérer des améliorations
- **Discussions** : Utilisez les [Discussions GitHub](https://github.com/michaelgermini/Intelligence-Artificielle-Avancee-pour-la-Physique-des-Hautes-Energies/discussions) pour poser des questions
- **Email** : michael@germini.info (pour questions générales)

---

## ⭐ Star le Projet

Si ce livre vous est utile, n'hésitez pas à ⭐ **star** le repository ! Cela aide à faire connaître le projet.

---

## 📰 Mises à Jour

- **Décembre 2024** : Publication initiale sur GitHub
- **Décembre 2024** : Ajout de 6 exemples pratiques complets
- **Décembre 2024** : Completion de tous les chapitres et annexes

---

*Ce livre est en développement continu. Contributions et suggestions bienvenues !*  
*Dernière mise à jour : Décembre 2024*

