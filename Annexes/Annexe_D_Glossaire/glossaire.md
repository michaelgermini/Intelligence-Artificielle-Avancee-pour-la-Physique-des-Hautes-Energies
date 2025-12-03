# Annexe D : Glossaire

---

## Introduction

Ce glossaire définit les termes techniques utilisés dans ce livre, couvrant la physique des particules, le machine learning, les réseaux de tenseurs, le hardware, et la compression de modèles. Les termes sont organisés par domaine pour faciliter la recherche.

---

## Plan du Glossaire

1. [Termes de Physique des Particules](#termes-de-physique-des-particules)
2. [Termes de Machine Learning](#termes-de-machine-learning)
3. [Termes Hardware](#termes-hardware)
4. [Termes de Réseaux de Tenseurs](#termes-de-réseaux-de-tenseurs)
5. [Termes de Compression de Modèles](#termes-de-compression-de-modèles)
6. [Abréviations Communes](#abréviations-communes)

---

## Termes de Physique des Particules

### A

**ATLAS** : A Toroidal LHC ApparatuS. L'une des quatre grandes expériences du LHC, conçue pour la recherche de nouvelle physique. Utilise un aimant toroïdal pour la détection des muons.

**Axe du faisceau** : Direction principale du faisceau de particules dans l'accélérateur. Typiquement l'axe z dans le système de coordonnées du détecteur.

### B

**b-tagging** : Identification des jets provenant de quarks b, crucial pour la physique du top et du Higgs. Utilise des informations sur le vertex secondaire et le paramètre d'impact.

**Baryon** : Particule composite constituée de trois quarks. Exemples : proton (uud), neutron (udd).

**Bunch crossing** : Croisement de paquets de protons au LHC. Se produit toutes les 25 ns (40 MHz) pendant les collisions.

### C

**Calorimètre** : Détecteur mesurant l'énergie des particules en les absorbant complètement. Se divise en calorimètre électromagnétique (ECAL) et hadronique (HCAL).

**CMS** : Compact Muon Solenoid. Expérience généraliste du LHC avec un puissant solénoïde de 3.8 T. Complémentaire à ATLAS.

**CERN** : Organisation européenne pour la recherche nucléaire. Opère le LHC et d'autres accélérateurs.

**Charge électrique** : Propriété fondamentale des particules. Unité : e (charge élémentaire). Quarks : ±1/3 ou ±2/3, leptons : 0 ou ±1.

### D

**DAQ** : Data Acquisition. Système d'acquisition et de stockage des données. Filtre et stocke événements intéressants du trigger.

**Détecteur** : Instrument mesurant propriétés des particules produites dans collisions. Composants : tracker, calorimètres, détecteur muons.

**Down quark (d)** : Quark de première génération, charge -1/3e. Plus léger que le quark up.

### E

**Énergie transverse (ET)** : Composante de l'énergie perpendiculaire à l'axe du faisceau. $E_T = E \sin\theta$. Utilisée car énergie totale inconnue dans collision hadronique.

**Électron** : Lepton chargé de première génération. Charge -e, masse ~0.511 MeV/c². Particule stable.

**Événement** : Une collision proton-proton produisant particules détectées. Les événements sont filtrés par le trigger système.

### F

**Faisceau** : Flux de particules accélérées dans direction. Au LHC : faisceaux de protons tournant en sens opposés.

**Fragmentation** : Processus par lequel quark ou gluon produit jet de hadrons. Décrit par modèles (Lund, etc.).

### G

**Gluon** : Boson de jauge de l'interaction forte. Charge de couleur. Huit gluons différents (octet).

**Géométrie du détecteur** : Arrangement des composants du détecteur. Cylindrique avec endcaps aux extrémités au LHC.

### H

**Hadron** : Particule composite subissant interaction forte. Deux types : mésons (qq̄) et baryons (qqq).

**Higgs (boson de)** : Particule responsable du mécanisme de brisure de symétrie électrofaible. Découverte en 2012 au LHC. Masse ~125 GeV/c².

**HLT** : High-Level Trigger. Deuxième niveau de filtrage, basé sur software. Traite événements acceptés par L1 Trigger.

**HCAL** : Hadronic Calorimeter. Calorimètre mesurant énergie hadrons (protons, neutrons, pions).

### I

**Interaction forte** : Force fondamentale unissant quarks dans hadrons. Portée courte (~10⁻¹⁵ m). Médiateur : gluon.

**Interaction faible** : Force fondamentale responsable désintégrations radioactives. Portée très courte. Médiateurs : W⁺, W⁻, Z⁰.

**Isolation** : Critère sélectionnant particules isolées (peu d'autres particules proches). Utilisé pour identification leptons.

### J

**Jet** : Gerbe de hadrons produite par la fragmentation d'un quark ou gluon. Reconstruit avec algorithmes clustering (anti-kT, kT, etc.).

**Jet energy scale (JES)** : Calibration énergie jets. Correction systématique importante.

### L

**Lepton** : Particule élémentaire n'interagissant pas fortement. Trois générations : (e, νe), (μ, νμ), (τ, ντ).

**LHC** : Large Hadron Collider. Collisionneur de protons de 27 km de circonférence au CERN. Énergie actuelle : 13.6 TeV.

**Luminosité** : Mesure du taux de collisions. Luminosité instantanée : $\mathcal{L} = \frac{1}{\sigma_{inel}} \frac{dN}{dt}$. Unité : cm⁻² s⁻¹. Luminosité intégrée : $\int \mathcal{L} dt$ (fb⁻¹).

**L1 Trigger** : Level-1 Trigger. Premier niveau de filtrage, implémenté en hardware (FPGA). Latence ~4 μs, réduction rate 40 kHz → 100 kHz.

### M

**Méson** : Hadron constitué d'un quark et d'un antiquark. Exemples : pion (uū ou dđ̄), kaon.

**MET** : Missing Transverse Energy. Énergie transverse manquante, signature de neutrinos ou nouvelles particules. $E_T^{miss} = -\sum E_T^{visible}$.

**Modèle Standard** : Théorie décrivant particules élémentaires et interactions. Inclut électromagnétisme, force faible, force forte.

**Muon** : Lepton de deuxième génération. Masse ~105.7 MeV/c². Particule chargée traversant calorimètres.

### N

**Neutrino** : Lepton électriquement neutre. Très faible interaction. Trois saveurs : νe, νμ, ντ.

**Nouvelle physique** : Phénomènes au-delà du Modèle Standard. Recherche principale motivation LHC.

### P

**Pile-up** : Superposition de plusieurs collisions dans un même croisement de faisceaux. Au LHC Run 3 : ~50-200 collisions par croisement.

**Pion (π)** : Méson le plus léger. Types : π⁺ (uđ̄), π⁻ (ūd), π⁰ (uū ou dđ̄ mixte).

**Pseudo-rapidité (η)** : Coordonnée angulaire couramment utilisée. $\eta = -\ln[\tan(\theta/2)]$. Avantage : invariance sous boosts le long axe faisceau.

**Proton** : Baryon stable constitué de quarks uud. Charge +e. Composant principal faisceaux LHC.

**Puissance du faisceau** : Énergie totale faisceau. LHC Run 3 : ~6.5 TeV par proton × 2 faisceaux = 13 TeV centre de masse.

### Q

**Quark** : Particule élémentaire constituant hadrons. Six saveurs : up, down, charm, strange, top, bottom. Charge fractionnaire.

### R

**Run** : Période de prise de données. Ex: Run 1 (2010-2012, 7-8 TeV), Run 2 (2015-2018, 13 TeV), Run 3 (2022-2025, 13.6 TeV).

**Reconstruction** : Processus reconstruisant propriétés particules depuis signaux détecteurs. Inclut tracking, clustering, identification.

### S

**Section efficace (σ)** : Probabilité d'un processus physique. Unité : barn (1 b = 10⁻²⁴ cm²). Section efficace totale collision pp : ~110 mb.

**Saveur** : Type de quark ou lepton. Six saveurs quarks, trois saveurs leptons.

**Standard Model (SM)** : Voir Modèle Standard.

### T

**Tau (τ)** : Lepton de troisième génération. Masse ~1.78 GeV/c². Désintègre rapidement en autres particules.

**Tracker** : Détecteur mesurant trajectoires particules chargées. Utilise détecteurs silicium (pixels et strips).

**Transverse momentum (pT)** : Composante moment perpendiculaire axe faisceau. $p_T = p \sin\theta$. Crucial pour analyse HEP.

**Trigger** : Système de déclenchement sélectionnant événements intéressants. Multi-niveaux (L1 hardware, HLT software).

**Top quark (t)** : Quark le plus lourd. Masse ~173 GeV/c². Désintègre avant hadronisation.

### U

**Up quark (u)** : Quark de première génération, charge +2/3e. Plus léger que down quark.

### V

**Vertex** : Point d'interaction ou désintégration. Vertex primaire : point collision. Vertex secondaire : point désintégration particule lourde.

### W

**W boson** : Boson de jauge interaction faible. Charge ±e. Masse ~80.4 GeV/c².

**WLCG** : Worldwide LHC Computing Grid. Infrastructure de calcul distribuée pour le LHC. Traite et stocke données LHC.

### Z

**Z boson** : Boson de jauge interaction faible. Électriquement neutre. Masse ~91.2 GeV/c².

---

## Termes de Machine Learning

### A

**Activation** : Fonction non-linéaire appliquée après transformation linéaire. Exemples : ReLU ($f(x) = \max(0,x)$), sigmoid ($\sigma(x) = 1/(1+e^{-x})$), tanh.

**Adam** : Algorithme d'optimisation adaptatif combinant momentum et RMSprop. Ajuste learning rate par paramètre.

**Attention** : Mécanisme permettant au modèle de pondérer différentes parties de l'entrée. Base des Transformers.

**Autoencoder** : Réseau apprenant représentation compressée des données. Architecture : encodeur → bottleneck → décodeur.

**Accuracy** : Précision. Proportion prédictions correctes. $Acc = \frac{TP + TN}{TP + TN + FP + FN}$.

### B

**Backpropagation** : Algorithme calcul gradients par règle chaîne. Permet entraînement réseaux profonds.

**Batch** : Groupe échantillons traités ensemble pendant entraînement. Batch size : nombre échantillons par batch.

**Batch Normalization** : Normalisation activations par batch pour stabiliser entraînement. Réduit internal covariate shift.

**BDT** : Boosted Decision Tree. Ensemble arbres décision entraînés séquentiellement avec boosting.

**Bias** : Biais algorithmique ou biais statistique (terme constant dans modèle linéaire).

### C

**CNN** : Convolutional Neural Network. Réseau exploitant structure spatiale données. Utilise opérations convolution.

**Cross-entropy** : Fonction perte standard classification. $L = -\sum y_i \log(\hat{y}_i)$.

**CUDA** : Compute Unified Device Architecture. Plateforme NVIDIA pour calcul parallèle GPU.

### D

**Dataset** : Ensemble données utilisées entraînement/évaluation. Divisé en train/validation/test.

**Deep Learning** : Apprentissage avec réseaux neurones profonds (nombreuses couches). Exploite hiérarchie caractéristiques.

**Dropout** : Régularisation désactivant aléatoirement neurones pendant entraînement. Réduit overfitting.

### E

**Embedding** : Représentation vectorielle dense objets discrets (mots, particules). Espace continu facilitant apprentissage.

**Epoch** : Un passage complet sur ensemble entraînement. Multiple epochs nécessaires pour convergence.

**Early stopping** : Arrêt entraînement quand performance validation ne s'améliore plus. Évite overfitting.

### F

**Feature** : Caractéristique ou variable d'entrée. Feature engineering : création features pertinentes.

**Fine-tuning** : Ajustement modèle pré-entraîné sur nouvelle tâche. Réutilise connaissances apprises.

**FLOPs** : Floating Point Operations. Mesure complexité computationnelle. 1 GFLOP = 10⁹ opérations.

**F1-score** : Mesure performance combinant precision et recall. $F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$.

**Forward pass** : Passage données à travers réseau (entrée → sortie). Opposé backward pass (gradients).

### G

**GAN** : Generative Adversarial Network. Réseau génératif avec discriminateur adversaire. Deux réseaux en compétition.

**Gradient** : Dérivée partielle fonction par rapport paramètres. Direction plus forte augmentation.

**Gradient Descent** : Optimisation par descente direction opposée gradient. $\theta_{t+1} = \theta_t - \eta \nabla L$.

**GNN** : Graph Neural Network. Réseau opérant sur structures graphes. Important pour tracking particules.

**GPU** : Graphics Processing Unit. Processeur parallèle pour calcul intensif. Essentiel pour deep learning.

### H

**Hyperparameter** : Paramètre contrôlant apprentissage mais non appris (learning rate, batch size, architecture).

### K

**Keras** : API haut niveau pour TensorFlow. Simplifie création et entraînement modèles.

**Knowledge Distillation** : Transfert connaissances grand modèle vers petit. Utilise soft labels du teacher.

### L

**Layer** : Couche réseau neurone. Types : dense, convolution, pooling, attention, etc.

**Learning Rate** : Taux apprentissage, contrôle taille pas optimisation. Trop élevé : instabilité. Trop faible : convergence lente.

**Loss** : Fonction perte mesurant écart prédictions et cibles. Minimisée pendant entraînement.

**LSTM** : Long Short-Term Memory. Architecture récurrente avec mémoire long terme. Résout problème vanishing gradients.

### M

**MLP** : Multi-Layer Perceptron. Réseau feedforward avec couches denses. Architecture basique deep learning.

**Momentum** : Technique accélération gradient descent. Conserve vitesse direction précédente.

**Model** : Architecture réseau avec paramètres appris. Représente fonction approximation données.

### N

**Neural Network** : Réseau neurones artificiels. Composé neurones connectés calculant transformations.

**Normalization** : Normalisation données ou activations. Types : batch norm, layer norm, group norm.

**NumPy** : Bibliothèque Python calcul numérique. Arrays multidimensionnels et opérations.

### O

**Optimizer** : Algorithme optimisation. Exemples : SGD, Adam, RMSprop. Minimise fonction perte.

**Overfitting** : Sur-apprentissage. Modèle mémorise données entraînement mais généralise mal.

**ONNX** : Open Neural Network Exchange. Format échange modèles entre frameworks.

### P

**Precision** : Précision classification. $Precision = \frac{TP}{TP + FP}$.

**Pruning** : Élagage connexions ou neurones peu importants. Réduit taille et complexité modèle.

**PTQ** : Post-Training Quantization. Quantification après entraînement. Pas besoin retraining.

**PyTorch** : Framework deep learning flexible. Développé par Facebook. Popularité recherche.

### Q

**QAT** : Quantization-Aware Training. Entraînement avec quantification simulée. Meilleure performance que PTQ.

**Quantization** : Réduction précision numérique (ex: float32 → int8). Réduit mémoire et latence.

### R

**Recall** : Rappel classification. $Recall = \frac{TP}{TP + FN}$.

**ReLU** : Rectified Linear Unit. Activation $f(x) = \max(0, x)$. Standard dans CNNs.

**Regularization** : Techniques éviter sur-apprentissage. Types : L1, L2, dropout, early stopping.

**RNN** : Recurrent Neural Network. Réseau avec connexions récurrentes. Traite séquences.

**ROC curve** : Receiver Operating Characteristic. Courbe montrant performance classification binaire.

### S

**SGD** : Stochastic Gradient Descent. Descente gradient stochastique. Utilise batch aléatoire.

**Softmax** : Fonction transformant logits en probabilités. $softmax(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$.

**Sparsity** : Proportion poids nuls dans réseau. Réseaux sparse plus efficaces.

**Supervised Learning** : Apprentissage avec labels. Modèle apprend mapping entrée → sortie.

### T

**Tensor** : Tableau multidimensionnel. Généralisation matrices à dimensions arbitraires.

**TensorFlow** : Framework deep learning Google. Production-ready, large écosystème.

**Transformer** : Architecture basée attention, sans récurrence. Standard NLP et applications diverses.

**Transfer Learning** : Réutilisation modèle entraîné autre tâche. Accélère développement.

**Training** : Entraînement. Processus ajustement paramètres modèle pour minimiser perte.

### U

**Unsupervised Learning** : Apprentissage sans labels. Découvre patterns dans données.

**Underfitting** : Sous-apprentissage. Modèle trop simple pour capturer patterns données.

### V

**Validation** : Évaluation modèle sur ensemble validation. Utilisé tuning hyperparamètres.

**Vanishing Gradient** : Problème gradients devenant très petits dans réseaux profonds. Empêche apprentissage.

### W

**Weight** : Poids. Paramètre connexion neurone. Appris pendant entraînement.

**Weight Decay** : Régularisation L2 appliquée via optimiseur. Pénalise poids grands.

---

## Termes Hardware

### A

**ASIC** : Application-Specific Integrated Circuit. Circuit intégré dédié application spécifique. Performant mais coûteux développer.

**ALU** : Arithmetic Logic Unit. Unité calcul processeur effectuant opérations arithmétiques/logiques.

### B

**Bandwidth** : Bande passante. Taux transfert données. Mesure capacité communication.

**BRAM** : Block RAM. Mémoire embarquée FPGA. Accès rapide, ressources limitées.

**Bus** : Canal communication entre composants système. Transfère données/contrôle.

### C

**Clock** : Signal horloge synchronisant opérations. Fréquence clock détermine vitesse système.

**Cache** : Mémoire rapide stockant données fréquemment accédées. Réduit latence accès mémoire.

**CUDA** : Voir section Machine Learning.

### D

**DDR** : Double Data Rate. Type mémoire RAM haute performance. DDR4, DDR5 standards actuels.

**DMA** : Direct Memory Access. Accès mémoire direct sans CPU. Utilisé transferts haute performance.

**DSP** : Digital Signal Processor. Bloc calcul optimisé multiplications. Présent dans FPGAs.

**DRAM** : Dynamic RAM. Mémoire volatile nécessitant rafraîchissement périodique.

### F

**FPGA** : Field-Programmable Gate Array. Circuit logique programmable. Reprogrammable, flexible, temps réel.

**FLOPs** : Voir section Machine Learning.

**FP32/FP16/INT8** : Formats numériques. FP32 (float32), FP16 (float16), INT8 (int8). Précision vs performance.

**FIFO** : First In First Out. Structure données queue. Utilisé buffering données.

### G

**GPU** : Graphics Processing Unit. Processeur parallèle. Excellent pour calcul parallèle massif.

### H

**HDL** : Hardware Description Language. Langage description hardware. Verilog, VHDL standards.

**HLS** : High-Level Synthesis. Synthèse hardware depuis langage haut niveau (C/C++). Simplifie développement FPGA.

**HBM** : High Bandwidth Memory. Mémoire haute bande passante. Utilisée GPUs récents.

### I

**INT8** : Entier 8 bits. Format courant inférence quantifiée. Réduit mémoire et latence.

**IP Core** : Intellectual Property Core. Module hardware réutilisable. Vendus par fournisseurs FPGA.

### L

**Latency** : Latence. Temps entre entrée et sortie système. Critique applications temps réel.

**LUT** : Look-Up Table. Élément base FPGAs pour logique combinatoire. Implémente fonctions booléennes.

**Lane** : Lien communication haute vitesse. Exemple : PCIe lane.

### M

**Memory** : Mémoire. Stockage données. Types : SRAM, DRAM, Flash, HBM.

**MHz/GHz** : Fréquence. Megahertz (10⁶ Hz), Gigahertz (10⁹ Hz). Vitesse clock.

### P

**Pipeline** : Technique parallélisation temporelle opérations. Améliore throughput.

**PCIe** : Peripheral Component Interconnect Express. Interface connexion haute performance (GPU, FPGA).

**Precision** : Précision numérique. Nombre bits représentation. Trade-off précision vs performance.

### R

**Register** : Registre. Stockage données rapide dans processeur/FPGA. Plus rapide que mémoire.

**Resource Utilization** : Utilisation ressources FPGA. LUTs, BRAMs, DSPs utilisés.

### S

**SRAM** : Static RAM. Mémoire volatile rapide. Pas besoin rafraîchissement.

**Throughput** : Débit. Nombre opérations par unité temps. Opposé latency.

**SoC** : System on Chip. Système complet sur puce unique. Processeur + périphériques.

### T

**TPU** : Tensor Processing Unit. Accélérateur IA Google. Optimisé calcul tenseurs.

### V

**Verilog/VHDL** : Langages description hardware (HDL). Standards industrie.

**Vivado** : Suite outils développement Xilinx. Inclut synthèse, placement, routage, simulation.

---

## Termes de Réseaux de Tenseurs

### B

**Bond dimension** : Dimension indices internes décomposition tenseur. Contrôle complexité représentation.

### C

**Canonical form** : Forme canonique décomposition. Conditions normalisation facteurs.

**Contraction** : Sommation indices partagés entre tenseurs. $C_{ik} = \sum_j A_{ij} B_{jk}$.

**CP Decomposition** : Canonical Polyadic. Décomposition somme produits tensoriels rang 1. $\mathcal{T} = \sum_{r=1}^R \mathbf{u}_r^{(1)} \circ \cdots \circ \mathbf{u}_r^{(d)}$.

### D

**Decomposition** : Décomposition. Représentation tenseur comme somme termes plus simples.

### E

**Entanglement** : Intrication quantique. Corrélations non-classiques. Mesuré par entropie von Neumann.

**Einstein notation** : Convention notation indices répétés signifient sommation. Simplifie expressions tensorielles.

### F

**Factor matrices** : Matrices facteurs décomposition. Composants représentation factorisée.

**Frobenius norm** : Norme Frobenius tenseur. $||\mathcal{T}||_F = \sqrt{\sum T_{i_1...i_d}^2}$.

### H

**Hierarchical Tucker (HT)** : Décomposition hiérarchique. Structure arborescente. Efficace hautes dimensions.

**Hadamard product** : Produit Hadamard. Produit élément par élément. $(A \odot B)_{ij} = A_{ij} B_{ij}$.

### K

**Kronecker product** : Produit Kronecker matrices. $A \otimes B$. Utilisé produits tensoriels.

### M

**Matrix Product State (MPS)** : État quantique format Tensor Train. Utilisé physique quantique.

**Mode** : Dimension tenseur (équivalent "axe"). Tenseur d'ordre d a d modes.

**Mode-n unfolding** : Matricisation tenseur selon mode. Réarrange tenseur en matrice.

### N

**n-mode product** : Produit mode-n. Multiplication tenseur par matrice selon mode spécifique.

### O

**Order** : Ordre tenseur. Nombre dimensions (également rank, mais éviter confusion avec tensor rank).

**Outer product** : Produit externe vecteurs. $v \circ w = v w^T$. Génère matrice rang 1.

### R

**Rank (tensoriel)** : Rang tensoriel. Nombre minimal termes rang 1 représenter tenseur. NP-hard calculer.

**Rank (matriciel)** : Rang matriciel. Nombre valeurs singulières non nulles. Efficace calculer.

### S

**SVD** : Singular Value Decomposition. Décomposition valeurs singulières. $A = U \Sigma V^T$.

**SVD truncation** : Troncature SVD. Conserver seulement valeurs singulières plus grandes.

### T

**Tensor** : Tableau multidimensionnel. Généralisation scalaires (ordre 0), vecteurs (1), matrices (2).

**Tensor contraction** : Voir Contraction.

**Tensor Train (TT)** : Décomposition produit tenseurs 3D. Efficace compression haute dimension. $\mathcal{T}[i_1...i_d] = G_1[i_1] G_2[i_2] ... G_d[i_d]$.

**Tensor Ring (TR)** : Décomposition anneau tenseurs. Variation Tensor Train avec connexion circulaire.

**Tucker** : Décomposition avec noyau et matrices facteurs. $\mathcal{T} = \mathcal{G} \times_1 U_1 \times_2 U_2 \times_3 U_3$.

**TT-rank** : Rang Tensor Train. Dimensions intermédiaires décomposition TT. Contrôle compression.

### U

**Unfolding** : Matricisation tenseur selon mode. Réarrange tenseur en matrice bidimensionnelle.

---

## Termes de Compression de Modèles

### A

**Approximation error** : Erreur approximation. Différence modèle original et compressé.

### B

**Binary quantization** : Quantification binaire. Poids réduits à ±1. Compression extrême.

### C

**Channel pruning** : Pruning canaux. Suppression canaux entiers convolutions. Structured pruning.

**Compression ratio** : Ratio compression. Taille originale / taille compressée.

### D

**Distillation** : Voir Knowledge Distillation section ML.

### E

**Elasticity** : Élasticité. Capacité modèle maintenir performance sous différentes compressions.

### F

**Fine-tuning** : Ajustement après compression. Récupération performance perdue.

### K

**Knowledge Distillation** : Voir section ML.

### L

**Low-rank approximation** : Approximation rang faible. Factorisation matrices poids.

**LoRA** : Low-Rank Adaptation. Adaptation efficace par matrices rang faible. Réduit paramètres trainables.

### M

**Magnitude-based pruning** : Pruning basé magnitude. Supprime poids petits en valeur absolue.

### P

**Pruning** : Élagage. Suppression connexions/neurones/canaux peu importants.

**PTQ** : Post-Training Quantization. Voir section ML.

### Q

**QAT** : Quantization-Aware Training. Voir section ML.

**Quantization** : Voir section ML.

**Quantization error** : Erreur quantification. Perte précision due quantification.

### R

**Retraining** : Ré-entraînement après compression. Restaure performance.

### S

**Sparsity** : Sparsité. Proportion poids nuls. Réseaux sparse plus efficaces.

**Structured pruning** : Pruning structuré. Supprime structures complètes (canaux, neurones). Hardware-friendly.

**SVD decomposition** : Décomposition SVD. Factorisation poids matrices. Low-rank approximation.

### T

**Ternary quantization** : Quantification ternaire. Poids réduits à {-1, 0, +1}.

**Trade-off** : Compromis. Balance compression et performance.

### U

**Uniform quantization** : Quantification uniforme. Pas quantification constant.

**Unstructured pruning** : Pruning non-structuré. Supprime poids individuels. Meilleure compression mais moins hardware-friendly.

---

## Abréviations Communes

| Abréviation | Signification |
|-------------|---------------|
| AI/IA | Artificial Intelligence / Intelligence Artificielle |
| API | Application Programming Interface |
| AUC | Area Under Curve (ROC) |
| BN | Batch Normalization |
| CNN | Convolutional Neural Network |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture |
| DL | Deep Learning |
| ECAL | Electromagnetic Calorimeter |
| FC | Fully Connected (layer) |
| FP16/32 | Floating Point 16/32 bits |
| GAN | Generative Adversarial Network |
| GNN | Graph Neural Network |
| GPU | Graphics Processing Unit |
| HCAL | Hadronic Calorimeter |
| HEP | High Energy Physics |
| HLS | High-Level Synthesis |
| HLT | High-Level Trigger |
| L1 | Level-1 Trigger |
| LHC | Large Hadron Collider |
| LSTM | Long Short-Term Memory |
| ML | Machine Learning |
| MLP | Multi-Layer Perceptron |
| MET | Missing Transverse Energy |
| NAS | Neural Architecture Search |
| NLP | Natural Language Processing |
| NN | Neural Network |
| ONNX | Open Neural Network Exchange |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| ReLU | Rectified Linear Unit |
| ROC | Receiver Operating Characteristic |
| RNN | Recurrent Neural Network |
| SDK | Software Development Kit |
| SGD | Stochastic Gradient Descent |
| SM | Standard Model |
| SVD | Singular Value Decomposition |
| TN | Tensor Network |
| TPU | Tensor Processing Unit |
| TT | Tensor Train |
| VHDL | VHSIC Hardware Description Language |

---

## Guide d'Utilisation

### Recherche par Domaine

- **Physique des Particules** : Termes relatifs détecteurs, particules, collisions LHC
- **Machine Learning** : Termes relatifs réseaux neurones, apprentissage, optimisation
- **Hardware** : Termes relatifs FPGA, GPU, mémoire, latence
- **Réseaux de Tenseurs** : Termes relatifs décompositions, contractions, compression
- **Compression** : Termes relatifs pruning, quantification, distillation

### Recherche Alphabétique

Utilisez table des matières ou recherche dans document pour trouver terme spécifique.

### Définitions Mathématiques

Pour définitions mathématiques formelles, référez-vous sections théoriques correspondantes dans chapitres.

---

*Retour à la [Table des Matières](../../INDEX.md)*
