# Annexe E : Ressources et Références

---

## Introduction

Cette annexe regroupe les ressources essentielles pour approfondir les sujets couverts dans ce livre : articles scientifiques fondamentaux, tutoriels, bibliothèques, communautés, et outils de développement. Ces ressources permettront d'explorer plus en profondeur chaque domaine et de rester à jour avec les dernières avancées.

---

## Plan de l'Annexe

1. [E.1 Articles Fondamentaux](#e1-articles-fondamentaux)
2. [E.2 Tutoriels et Cours en Ligne](#e2-tutoriels-et-cours-en-ligne)
3. [E.3 Bibliothèques et Outils](#e3-bibliothèques-et-outils)
4. [E.4 Communautés et Forums](#e4-communautés-et-forums)
5. [E.5 Datasets](#e5-datasets)
6. [E.6 Livres Recommandés](#e6-livres-recommandés)
7. [E.7 Outils de Développement](#e7-outils-de-développement)

---

## E.1 Articles Fondamentaux

### Deep Learning Fondamentaux

1. **LeCun, Y., Bengio, Y., Hinton, G.** (2015). "Deep Learning." *Nature*, 521, 436-444.
   - Article de référence sur le deep learning moderne.
   - Lien : https://www.nature.com/articles/nature14539

2. **Rumelhart, D. E., Hinton, G. E., Williams, R. J.** (1986). "Learning representations by back-propagating errors." *Nature*, 323, 533-536.
   - Article fondateur sur la backpropagation.
   - Lien : https://www.nature.com/articles/323533a0

3. **Hochreiter, S., Schmidhuber, J.** (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780.
   - Introduction du LSTM.
   - Lien : https://direct.mit.edu/neco/article-abstract/9/8/1735/6109

4. **Krizhevsky, A., Sutskever, I., Hinton, G. E.** (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
   - Révolution du deep learning en vision.
   - Lien : https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

### Architectures Modernes

5. **He, K. et al.** (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
   - Connexions résiduelles permettant réseaux très profonds.
   - Lien : https://arxiv.org/abs/1512.03385

6. **Vaswani, A. et al.** (2017). "Attention Is All You Need." *NeurIPS*.
   - Introduction architecture Transformer.
   - Lien : https://arxiv.org/abs/1706.03762

7. **Vaswani, A. et al.** (2017). "Attention Is All You Need." *NeurIPS*.
   - Transformer architecture révolutionnaire.
   - Lien : https://arxiv.org/abs/1706.03762

### Compression de Modèles

8. **Han, S. et al.** (2015). "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." *ICLR*.
   - Compression combinée : pruning, quantification, codage.
   - Lien : https://arxiv.org/abs/1510.00149

9. **Frankle, J., Carbin, M.** (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR*.
   - Sous-réseaux sparse entraînables depuis initialisation.
   - Lien : https://arxiv.org/abs/1803.03635

10. **Hu, E. J. et al.** (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
    - Adaptation efficace par matrices rang faible.
    - Lien : https://arxiv.org/abs/2106.09685

11. **Jacob, B. et al.** (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR*.
    - Quantification entière pour inférence.
    - Lien : https://arxiv.org/abs/1712.05877

### Réseaux de Tenseurs

12. **Kolda, T. G., Bader, B. W.** (2009). "Tensor Decompositions and Applications." *SIAM Review*, 51(3), 455-500.
    - Revue complète décompositions tensorielles.
    - Lien : https://epubs.siam.org/doi/abs/10.1137/07070111X

13. **Oseledets, I. V.** (2011). "Tensor-Train Decomposition." *SIAM J. Sci. Comput.*, 33(5), 2295-2317.
    - Introduction format Tensor Train.
    - Lien : https://epubs.siam.org/doi/10.1137/090752286

14. **Novikov, A. et al.** (2015). "Tensorizing Neural Networks." *NeurIPS*.
    - Application réseaux tenseurs au deep learning.
    - Lien : https://arxiv.org/abs/1509.06569

15. **Garipov, T. et al.** (2016). "Ultimate Tensorization: Compressing Convolutional and FC Layers Alike." *arXiv*.
    - Compression couches convolutionnelles et fully-connected.
    - Lien : https://arxiv.org/abs/1611.03214

### Machine Learning pour Physique des Particules

16. **Guest, D. et al.** (2018). "Deep Learning and its Application to LHC Physics." *Ann. Rev. Nucl. Part. Sci.*, 68, 161-181.
    - Revue applications ML en HEP.
    - Lien : https://arxiv.org/abs/1806.11484

17. **Duarte, J. et al.** (2018). "Fast inference of deep neural networks in FPGAs for particle physics." *JINST*, 13(07), P07027.
    - Article fondateur hls4ml.
    - Lien : https://iopscience.iop.org/article/10.1088/1748-0221/13/07/P07027

18. **Qu, H., Gouskos, L.** (2020). "ParticleNet: Jet Tagging via Particle Clouds." *Phys. Rev. D*, 101(11), 112002.
    - GNN pour classification jets.
    - Lien : https://arxiv.org/abs/1902.08570

19. **Butter, A. et al.** (2019). "Jet Substructure as a New Higgs-Search Channel at the Large Hadron Collider." *Phys. Rev. Lett.*, 120, 241801.
    - Application ML à recherche Higgs.
    - Lien : https://arxiv.org/abs/1709.08655

20. **Komiske, P. T., Metodiev, E. M., Thaler, J.** (2019). "Energy Flow Networks: Deep Sets for Particle Jets." *JHEP*, 2019(1), 121.
    - Deep Sets pour jets.
    - Lien : https://arxiv.org/abs/1810.05165

### Hardware et Déploiement

21. **Blott, M. et al.** (2018). "FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks." *ACM TRETS*.
    - Framework quantized NNs sur FPGA.
    - Lien : https://arxiv.org/abs/1809.04570

22. **Umuroglu, Y. et al.** (2017). "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference." *FPGA*.
    - Framework binarized NNs.
    - Lien : https://arxiv.org/abs/1612.07119

### Génération et Simulation

23. **Paganini, M., de Oliveira, L., Nachman, B.** (2018). "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks." *Phys. Rev. D*, 97(1), 014021.
    - GANs pour simulation calorimètres.
    - Lien : https://arxiv.org/abs/1712.10321

24. **Bellagente, M. et al.** (2020). "Invertible Networks or Partons to Detector and Back Again." *SciPost Phys.*, 9, 074.
    - Normalizing flows pour génération événements.
    - Lien : https://arxiv.org/abs/2006.06685

---

## E.2 Tutoriels et Cours en Ligne

### Deep Learning Général

#### Cours Universitaires

- **Stanford CS231n: Convolutional Neural Networks for Visual Recognition**
  - URL : http://cs231n.stanford.edu/
  - Contenu : CNNs, vision, deep learning moderne
  - Format : Vidéos, notes, devoirs

- **MIT 6.S191: Introduction to Deep Learning**
  - URL : http://introtodeeplearning.com/
  - Contenu : Introduction moderne deep learning
  - Format : Vidéos, notebooks Jupyter

- **NYU Deep Learning (Yann LeCun)**
  - URL : https://cds.nyu.edu/deep-learning/
  - Contenu : Fondamentaux deep learning
  - Format : Vidéos, slides

#### Cours en Ligne

- **Deep Learning Specialization (Coursera, Andrew Ng)**
  - URL : https://www.coursera.org/specializations/deep-learning
  - Contenu : Deep learning complet, 5 cours
  - Niveau : Intermédiaire

- **Fast.ai Practical Deep Learning**
  - URL : https://course.fast.ai/
  - Contenu : Approche pratique, code-first
  - Format : Vidéos, notebooks

- **Full Stack Deep Learning**
  - URL : https://fullstackdeeplearning.com/
  - Contenu : Déploiement modèles production
  - Focus : Aspects pratiques

### Machine Learning pour Physique

#### Ressources Spécifiques

- **Machine Learning for Physicists (Florian Marquardt)**
  - URL : https://machine-learning-for-physicists.org/
  - Contenu : ML orienté physique
  - Format : Livre interactif, notebooks

- **CERN Machine Learning Tutorials**
  - URL : https://github.com/cernopendata/cms-opendata-ml
  - Contenu : ML appliqué données CMS
  - Format : Notebooks Jupyter

- **CERN School of Computing**
  - URL : https://indico.cern.ch/category/8729/
  - Contenu : Computing et ML pour HEP
  - Format : École d'été

#### Workshops

- **ML4Jets Workshop**
  - URL : https://indico.cern.ch/category/11324/
  - Contenu : ML pour jets et reconstruction
  - Fréquence : Annuel

- **ACAT (International Workshop on Advanced Computing and Analysis Techniques)**
  - URL : https://indico.cern.ch/category/8723/
  - Contenu : Computing techniques HEP
  - Fréquence : Biennale

- **CHEP (Computing in High Energy Physics)**
  - URL : https://indico.cern.ch/category/8721/
  - Contenu : Computing HEP
  - Fréquence : Biennale

### FPGA et Hardware

- **Xilinx Vivado HLS Tutorial**
  - URL : https://www.xilinx.com/support/documentation-navigation/design-hubs/dh0012-vivado-high-level-synthesis-hub.html
  - Contenu : HLS complet
  - Format : Documentation, exemples

- **hls4ml Tutorial**
  - URL : https://github.com/fastmachinelearning/hls4ml-tutorial
  - Contenu : ML sur FPGA avec hls4ml
  - Format : Notebooks, exemples

- **FPGA Programming for Beginners**
  - URL : https://www.fpgatutorial.com/
  - Contenu : Introduction FPGA
  - Format : Tutoriels, exemples

### Réseaux de Tenseurs

- **Tensor Networks Tutorial**
  - URL : https://tensornetwork.org/learn/
  - Contenu : Réseaux tenseurs théorie et pratique
  - Format : Tutoriels, code

- **TensorLy Documentation**
  - URL : https://tensorly.org/stable/
  - Contenu : Documentation complète TensorLy
  - Format : Documentation, exemples

---

## E.3 Bibliothèques et Outils

### Deep Learning Frameworks

| Bibliothèque | Description | Lien |
|--------------|-------------|------|
| **PyTorch** | Framework DL flexible, recherche | https://pytorch.org/ |
| **TensorFlow** | Framework DL Google, production | https://www.tensorflow.org/ |
| **JAX** | Calcul différentiable haute performance | https://github.com/google/jax |
| **ONNX** | Format échange modèles | https://onnx.ai/ |
| **TensorFlow Lite** | Déploiement mobile/embedded | https://www.tensorflow.org/lite |
| **PyTorch Mobile** | Déploiement mobile PyTorch | https://pytorch.org/mobile/ |

### Compression et Optimisation

| Bibliothèque | Description | Lien |
|--------------|-------------|------|
| **Neural Magic** | Pruning et quantification | https://neuralmagic.com/ |
| **TensorRT** | Optimisation NVIDIA | https://developer.nvidia.com/tensorrt |
| **AIMET** | Compression modèles | https://github.com/quic/aimet |
| **Brevitas** | Quantification PyTorch | https://github.com/Xilinx/brevitas |
| **QAT (PyTorch)** | Quantification PyTorch native | https://pytorch.org/docs/stable/quantization.html |
| **SparseML** | Pruning et sparse networks | https://github.com/neuralmagic/sparseml |

### Réseaux de Tenseurs

| Bibliothèque | Description | Lien |
|--------------|-------------|------|
| **TensorLy** | Décompositions tensorielles | https://tensorly.org/ |
| **tntorch** | Tensor Train PyTorch | https://github.com/rballester/tntorch |
| **TensorNetwork** | Google TN library | https://github.com/google/TensorNetwork |
| **ITensor** | TN pour physique | https://itensor.org/ |
| **Quimb** | TN pour physique quantique | https://github.com/jcmgray/quimb |

### FPGA Deployment

| Outil | Description | Lien |
|-------|-------------|------|
| **hls4ml** | ML vers FPGA | https://fastmachinelearning.org/hls4ml/ |
| **FINN** | Quantized NN sur FPGA | https://github.com/Xilinx/finn |
| **Vitis AI** | Suite Xilinx pour IA | https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html |
| **Intel OpenVINO** | Optimisation Intel | https://docs.openvino.ai/ |

### Physique des Particules

| Bibliothèque | Description | Lien |
|--------------|-------------|------|
| **ROOT** | Framework analyse HEP | https://root.cern/ |
| **Uproot** | Lecture fichiers ROOT Python | https://github.com/scikit-hep/uproot |
| **Awkward Array** | Arrays irréguliers | https://awkward-array.org/ |
| **Coffea** | Analyse HEP columnar | https://github.com/CoffeaTeam/coffea |
| **Particle** | Standard Model particules | https://github.com/scikit-hep/particle |
| **HEPML** | ML utilities HEP | https://github.com/iml-wg/hep-ml |

### Visualisation et Analyse

| Bibliothèque | Description | Lien |
|--------------|-------------|------|
| **Matplotlib** | Visualisation Python | https://matplotlib.org/ |
| **Seaborn** | Visualisation statistique | https://seaborn.pydata.org/ |
| **Plotly** | Visualisation interactive | https://plotly.com/python/ |
| **ROOT** | Visualisation HEP | https://root.cern/ |
| **mplhep** | Styling matplotlib HEP | https://github.com/scikit-hep/mplhep |

---

## E.4 Communautés et Forums

### Communautés Scientifiques

#### Organisations Internationales

- **Inter-experimental Machine Learning (IML)**
  - URL : https://iml.web.cern.ch/
  - Description : Forum CERN ML physique particules
  - Activités : Meetings, workshops, ressources

- **Fast Machine Learning Lab**
  - URL : https://fastmachinelearning.org/
  - Description : Collaboration ML temps réel HEP
  - Focus : Hardware acceleration, FPGAs

- **CERN Open Data Portal**
  - URL : http://opendata.cern.ch/
  - Description : Données LHC publiques
  - Ressources : Datasets, outils analyse

#### Workshops et Conférences

- **ML4Jets**
  - URL : https://indico.cern.ch/category/11324/
  - Focus : ML pour jets
  - Fréquence : Annuel

- **Connecting the Dots**
  - URL : https://indico.cern.ch/category/10734/
  - Focus : Tracking et reconstruction
  - Fréquence : Annuel

- **ACAT (Advanced Computing and Analysis Techniques)**
  - URL : https://indico.cern.ch/category/8723/
  - Focus : Computing techniques HEP
  - Fréquence : Biennale

### Forums Techniques

#### Forums Généraux

- **PyTorch Forums**
  - URL : https://discuss.pytorch.org/
  - Focus : Support PyTorch
  - Activité : Très active

- **TensorFlow Forums**
  - URL : https://discuss.tensorflow.org/
  - Focus : Support TensorFlow
  - Activité : Active

- **Stack Overflow**
  - Tags pertinents : `pytorch`, `tensorflow`, `fpga`, `hls`, `machine-learning`
  - URL : https://stackoverflow.com/
  - Focus : Questions techniques

#### Forums Spécialisés

- **Xilinx Forums**
  - URL : https://forums.xilinx.com/
  - Focus : FPGA, Vivado, Vitis
  - Sections : HLS, Vitis AI

- **NVIDIA Developer Forums**
  - URL : https://forums.developer.nvidia.com/
  - Focus : CUDA, TensorRT, GPUs
  - Sections : Deep Learning

- **Reddit r/MachineLearning**
  - URL : https://www.reddit.com/r/MachineLearning/
  - Focus : Discussions ML
  - Activité : Très active

### Conférences Majeures

| Conférence | Focus | Fréquence | Site |
|------------|-------|-----------|------|
| **NeurIPS** | ML général | Annuelle | https://neurips.cc/ |
| **ICML** | ML théorique | Annuelle | https://icml.cc/ |
| **ICLR** | Représentations | Annuelle | https://iclr.cc/ |
| **CVPR** | Computer Vision | Annuelle | http://cvpr.thecvf.com/ |
| **CHEP** | Computing HEP | Biennale | https://indico.cern.ch/category/8721/ |
| **ACAT** | Computing HEP | Biennale | https://indico.cern.ch/category/8723/ |

---

## E.5 Datasets

### Physique des Particules

#### Datasets Publics

| Dataset | Description | Lien |
|---------|-------------|------|
| **CERN Open Data** | Données LHC publiques | http://opendata.cern.ch/ |
| **CMS Open Data** | Données CMS publiques | http://opendata.cern.ch/docs/about-cms |
| **ATLAS Open Data** | Données ATLAS publiques | http://opendata.cern.ch/docs/about-atlas |
| **Jet Tagging** | Jets pour classification | https://zenodo.org/record/3164691 |
| **Top Quark Tagging** | Top quark jets | https://zenodo.org/record/2603256 |

#### Datasets Simulation

| Dataset | Description | Lien |
|---------|-------------|------|
| **TrackML** | Reconstruction traces | https://www.kaggle.com/c/trackml-particle-identification |
| **CaloGAN** | Images calorimètre | https://data.mendeley.com/datasets/pvn3xc3wy5 |
| **Jet Images** | Images jets | https://zenodo.org/record/3596911 |

### Benchmarks ML Généraux

| Dataset | Description | Lien |
|---------|-------------|------|
| **ImageNet** | Classification images | https://www.image-net.org/ |
| **MNIST** | Chiffres manuscrits | http://yann.lecun.com/exdb/mnist/ |
| **CIFAR-10/100** | Images naturelles | https://www.cs.toronto.edu/~kriz/cifar.html |
| **COCO** | Object detection | https://cocodataset.org/ |

### Datasets Compression

| Dataset | Description | Usage |
|---------|-------------|-------|
| **ImageNet** | Benchmark compression | Évaluation pruning/quantization |
| **CIFAR-10** | Petit dataset | Tests rapides |
| **GLUE** | NLP benchmarks | Compression modèles NLP |

---

## E.6 Livres Recommandés

### Deep Learning

1. **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press.
   - La référence complète deep learning
   - URL : https://www.deeplearningbook.org/
   - Niveau : Avancé

2. **Zhang, A. et al.** (2021). *Dive into Deep Learning*.
   - Livre interactif avec code
   - URL : https://d2l.ai/
   - Niveau : Intermédiaire à avancé
   - Format : Gratuit, interactif

3. **Chollet, F.** (2021). *Deep Learning with Python*. Manning Publications.
   - Approche pratique avec Keras
   - Niveau : Intermédiaire

### Algèbre Linéaire et Tenseurs

4. **Strang, G.** (2019). *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press.
   - Algèbre linéaire orientée ML
   - Niveau : Intermédiaire

5. **Hackbusch, W.** (2012). *Tensor Spaces and Numerical Tensor Calculus*. Springer.
   - Traitement mathématique rigoureux tenseurs
   - Niveau : Avancé

6. **Cichocki, A. et al.** (2009). *Nonnegative Matrix and Tensor Factorizations*. Wiley.
   - Factorisations matrices et tenseurs
   - Niveau : Avancé

### Physique des Particules

7. **Thomson, M.** (2013). *Modern Particle Physics*. Cambridge University Press.
   - Introduction moderne physique particules
   - Niveau : Intermédiaire

8. **Bettini, A.** (2014). *Introduction to Elementary Particle Physics*. Cambridge University Press.
   - Accessible non-spécialistes
   - Niveau : Débutant

9. **Halzen, F., Martin, A. D.** (2008). *Quarks and Leptons: An Introductory Course in Modern Particle Physics*. Wiley.
   - Introduction classique
   - Niveau : Intermédiaire

### Hardware et FPGA

10. **Pedroni, V. A.** (2019). *Circuit Design with VHDL*. MIT Press.
    - Design circuits VHDL
    - Niveau : Intermédiaire

11. **Ashenden, P. J.** (2008). *The Designer's Guide to VHDL*. Morgan Kaufmann.
    - Guide complet VHDL
    - Niveau : Avancé

---

## E.7 Outils de Développement

### Environnement Python Complet

#### Installation Conda Complète

```bash
# Créer environnement conda pour ce livre
conda create -n hep_ml python=3.10

# Activer environnement
conda activate hep_ml

# Installation packages core
conda install -c conda-forge numpy scipy matplotlib pandas

# Installation deep learning
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorflow[and-cuda]

# Installation réseaux tenseurs
pip install tensorly tntorch

# Installation FPGA
pip install hls4ml

# Installation HEP
pip install uproot awkward coffea particle mplhep

# Outils développement
pip install jupyter jupyterlab
pip install black flake8 mypy pytest
pip install sphinx sphinx-rtd-theme

# Visualisation
pip install seaborn plotly
```

### Configuration GPU Complète

#### Linux CUDA Setup

```bash
# 1. Vérifier GPU
lspci | grep -i nvidia

# 2. Installer drivers NVIDIA
sudo ubuntu-drivers autoinstall
# Ou manuellement:
sudo apt install nvidia-driver-520

# 3. Installer CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# 4. Configurer variables environnement (~/.bashrc)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 5. Vérifier installation
nvcc --version
nvidia-smi

# 6. Installer PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 7. Installer TensorFlow avec CUDA
pip install tensorflow[and-cuda]
```

### Configuration FPGA (Vivado)

#### Installation Complète

1. **Télécharger Vivado**
   - Aller sur https://www.xilinx.com/support/download.html
   - Télécharger Vivado Design Suite
   - Version recommandée: 2022.1 ou plus récent

2. **Installation**
   ```bash
   chmod +x Xilinx_Unified_2022.1_0420_0327_Lin64.bin
   ./Xilinx_Unified_2022.1_0420_0327_Lin64.bin
   ```

3. **Configuration Variables**
   ```bash
   # ~/.bashrc
   export XILINX_VIVADO=/tools/Xilinx/Vivado/2022.1
   source $XILINX_VIVADO/settings64.sh
   ```

4. **Vérification**
   ```bash
   vivado -version
   vitis_hls -version
   ```

### Outils Utilitaires

#### Version Control

```bash
# Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Git LFS pour gros fichiers
git lfs install
```

#### Development Tools

```bash
# Code formatting
pip install black isort

# Linting
pip install flake8 pylint

# Type checking
pip install mypy

# Testing
pip install pytest pytest-cov

# Documentation
pip install sphinx sphinx-rtd-theme
```

---

## E.8 Ressources Complémentaires

### Blogs et Articles

- **Distill.pub** : Visualisations interactives concepts ML
  - URL : https://distill.pub/

- **The Gradient** : Articles ML récents
  - URL : https://thegradient.pub/

- **Fast.ai Blog** : Blog pratique deep learning
  - URL : https://www.fast.ai/

### Podcasts

- **The Gradient Podcast** : Discussions ML
- **Data Skeptic** : ML et science données
- **TWIML AI Podcast** : Interviews chercheurs ML

### Newsletters

- **The Batch (DeepLearning.AI)** : Newsletter Andrew Ng
- **Import AI** : Newsletter Jack Clark
- **AI Newsletter** : Curated ML news

---

## Guide d'Utilisation des Ressources

### Par Niveau

- **Débutant** : Cours en ligne (Fast.ai, Coursera), livres introductifs
- **Intermédiaire** : Articles récents, tutoriels spécialisés, workshops
- **Avancé** : Papers scientifiques, code source bibliothèques, conférences

### Par Domaine

- **Deep Learning** : Section E.1 (articles), E.2 (tutoriels), E.3 (frameworks)
- **Réseaux Tenseurs** : Articles Kolda, Oseledets; bibliothèques TensorLy, tntorch
- **Hardware** : hls4ml tutorial, Vivado documentation, forums Xilinx
- **HEP Applications** : IML resources, ML4Jets, CERN Open Data

---

*Retour à la [Table des Matières](../../INDEX.md)*
