# 21.5 Validation et Métriques de Qualité

---

## Introduction

La **validation** est cruciale pour garantir que les modèles génératifs IA produisent des échantillons physiquement valides et statistiquement corrects. Contrairement à la simulation Monte Carlo traditionnelle qui reproduit fidèlement la physique, les modèles IA peuvent introduire des biais subtils qui doivent être détectés et corrigés.

Cette section présente les méthodes de validation, métriques de qualité, et tests statistiques pour évaluer les modèles génératifs en physique des hautes énergies.

---

## Types de Validation

### Approches de Validation

```python
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

class ValidationTypes:
    """
    Types de validation pour modèles génératifs
    """
    
    def __init__(self):
        self.validation_types = {
            'statistical': {
                'description': 'Validation distributions statistiques',
                'tests': ['Moments', 'KS test', 'Chi-squared', 'Corrélations'],
                'importance': 'Vérifie reproduction distributions'
            },
            'physical': {
                'description': 'Validation contraintes physiques',
                'tests': ['Conservation énergie', 'Masses invariantes', 'Relations cinématiques'],
                'importance': 'Vérifie validité physique'
            },
            'discrimination': {
                'description': 'Tests de discrimination',
                'tests': ['Classifier accuracy', 'ROC AUC', 'Adversarial classifier'],
                'importance': 'Vérifie qu\'on ne peut distinguer réel vs généré'
            },
            'high_level': {
                'description': 'Validation observables haute niveau',
                'tests': ['Distributions physiques', 'Régions rares', 'Observables complexes'],
                'importance': 'Vérifie qualité sur observables finales'
            }
        }
    
    def display_types(self):
        """Affiche les types"""
        print("\n" + "="*70)
        print("Types de Validation")
        print("="*70)
        
        for val_type, info in self.validation_types.items():
            print(f"\n{val_type.replace('_', ' ').title()}:")
            print(f"  Description: {info['description']}")
            print(f"  Tests: {', '.join(info['tests'])}")
            print(f"  Importance: {info['importance']}")

val_types = ValidationTypes()
val_types.display_types()
```

---

## Tests Statistiques

### Comparaison de Distributions

```python
class StatisticalValidation:
    """
    Validation statistique des distributions
    """
    
    def kolmogorov_smirnov_test(self, real_data, generated_data, feature_idx=0):
        """
        Test de Kolmogorov-Smirnov
        
        Compare distributions univariées
        """
        real_feature = real_data[:, feature_idx] if len(real_data.shape) > 1 else real_data
        gen_feature = generated_data[:, feature_idx] if len(generated_data.shape) > 1 else generated_data
        
        # Convertir en numpy
        if isinstance(real_feature, torch.Tensor):
            real_feature = real_feature.detach().numpy()
        if isinstance(gen_feature, torch.Tensor):
            gen_feature = gen_feature.detach().numpy()
        
        # KS test
        statistic, p_value = stats.ks_2samp(real_feature, gen_feature)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'reject_null': p_value < 0.05  # Rejeter si distributions différentes
        }
    
    def compare_moments(self, real_data, generated_data):
        """
        Compare moments statistiques (moyenne, variance, skewness, kurtosis)
        """
        real_np = real_data.detach().numpy() if isinstance(real_data, torch.Tensor) else real_data
        gen_np = generated_data.detach().numpy() if isinstance(generated_data, torch.Tensor) else generated_data
        
        results = {}
        
        for i in range(real_np.shape[1]):
            real_feature = real_np[:, i]
            gen_feature = gen_np[:, i]
            
            moments = {
                'mean_real': np.mean(real_feature),
                'mean_gen': np.mean(gen_feature),
                'mean_diff': abs(np.mean(real_feature) - np.mean(gen_feature)),
                'std_real': np.std(real_feature),
                'std_gen': np.std(gen_feature),
                'std_diff': abs(np.std(real_feature) - np.std(gen_feature)),
                'skew_real': stats.skew(real_feature),
                'skew_gen': stats.skew(gen_feature),
                'kurtosis_real': stats.kurtosis(real_feature),
                'kurtosis_gen': stats.kurtosis(gen_feature)
            }
            
            results[f'feature_{i}'] = moments
        
        return results
    
    def compute_correlations(self, real_data, generated_data):
        """
        Compare matrices de corrélation
        """
        real_np = real_data.detach().numpy() if isinstance(real_data, torch.Tensor) else real_data
        gen_np = generated_data.detach().numpy() if isinstance(generated_data, torch.Tensor) else generated_data
        
        corr_real = np.corrcoef(real_np.T)
        corr_gen = np.corrcoef(gen_np.T)
        
        # Différence moyenne
        corr_diff = np.abs(corr_real - corr_gen)
        
        return {
            'correlation_real': corr_real,
            'correlation_gen': corr_gen,
            'mean_difference': np.mean(corr_diff),
            'max_difference': np.max(corr_diff)
        }
    
    def chi_squared_test(self, real_data, generated_data, n_bins=20, feature_idx=0):
        """
        Test Chi-squared
        
        Compare distributions avec bins
        """
        real_feature = real_data[:, feature_idx] if len(real_data.shape) > 1 else real_data
        gen_feature = generated_data[:, feature_idx] if len(generated_data.shape) > 1 else generated_data
        
        # Convertir
        if isinstance(real_feature, torch.Tensor):
            real_feature = real_feature.detach().numpy()
        if isinstance(gen_feature, torch.Tensor):
            gen_feature = gen_feature.detach().numpy()
        
        # Créer bins
        all_values = np.concatenate([real_feature, gen_feature])
        bins = np.linspace(all_values.min(), all_values.max(), n_bins + 1)
        
        # Histogrammes
        hist_real, _ = np.histogram(real_feature, bins=bins)
        hist_gen, _ = np.histogram(gen_feature, bins=bins)
        
        # Normaliser
        hist_real = hist_real / hist_real.sum()
        hist_gen = hist_gen / hist_gen.sum()
        
        # Chi-squared
        chi2, p_value = stats.chisquare(hist_gen, hist_real)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        }

stat_validator = StatisticalValidation()

# Test avec données simulées
real_data = torch.randn(10000, 10)
generated_data = torch.randn(10000, 10) * 1.1 + 0.1  # Légèrement différent

# KS test
ks_result = stat_validator.kolmogorov_smirnov_test(real_data, generated_data, feature_idx=0)
print(f"\nKolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_result['statistic']:.4f}")
print(f"  p-value: {ks_result['p_value']:.4f}")
print(f"  Rejette H0: {ks_result['reject_null']}")

# Moments
moments = stat_validator.compare_moments(real_data, generated_data)
print(f"\nComparaison Moments (feature 0):")
print(f"  Moyenne réel: {moments['feature_0']['mean_real']:.4f}")
print(f"  Moyenne généré: {moments['feature_0']['mean_gen']:.4f}")
print(f"  Différence: {moments['feature_0']['mean_diff']:.4f}")
```

---

## Tests Physiques

### Validation des Contraintes Physiques

```python
class PhysicalValidation:
    """
    Validation des contraintes physiques
    """
    
    def check_energy_conservation(self, particles):
        """
        Vérifie conservation de l'énergie
        
        Args:
            particles: (batch, n_particles, 4) [E, px, py, pz]
        """
        # Somme énergie
        total_energy = particles[:, :, 0].sum(dim=1)
        
        # Somme momentum
        total_px = particles[:, :, 1].sum(dim=1)
        total_py = particles[:, :, 2].sum(dim=1)
        total_pz = particles[:, :, 3].sum(dim=1)
        
        # Masse invariante totale
        mass_total = torch.sqrt(
            total_energy**2 - total_px**2 - total_py**2 - total_pz**2
        )
        
        return {
            'total_energy': total_energy,
            'total_momentum': torch.stack([total_px, total_py, total_pz], dim=1),
            'invariant_mass': mass_total
        }
    
    def check_mass_constraints(self, particles, expected_mass=None):
        """
        Vérifie contraintes de masse
        
        Ex: masse invariante d'un jet devrait être positive
        """
        E = particles[:, :, 0]
        px = particles[:, :, 1]
        py = particles[:, :, 2]
        pz = particles[:, :, 3]
        
        # Masse par particule
        masses = torch.sqrt(E**2 - px**2 - py**2 - pz**2 + 1e-10)
        
        # Vérifier positivité
        negative_masses = (masses < 0).sum().item()
        
        return {
            'masses': masses,
            'negative_mass_fraction': negative_masses / masses.numel(),
            'mean_mass': masses[masses >= 0].mean().item() if (masses >= 0).any() else 0
        }
    
    def check_kinematic_relations(self, jets):
        """
        Vérifie relations cinématiques
        
        Ex: pT jet = sqrt(px^2 + py^2)
        """
        px_total = jets[:, :, 1].sum(dim=1)
        py_total = jets[:, :, 2].sum(dim=1)
        
        pT_calculated = torch.sqrt(px_total**2 + py_total**2)
        
        return {
            'pT': pT_calculated,
            'mean_pT': pT_calculated.mean().item()
        }

phys_validator = PhysicalValidation()

# Simuler particules
particles = torch.randn(100, 10, 4)
particles[:, :, 0] = torch.abs(particles[:, :, 0]) * 10  # Énergie positive

# Vérifier conservation
conservation = phys_validator.check_energy_conservation(particles)
print(f"\nValidation Physique:")
print(f"  Énergie totale moyenne: {conservation['total_energy'].mean().item():.2f} GeV")
print(f"  Masse invariante moyenne: {conservation['invariant_mass'].mean().item():.2f} GeV")
```

---

## Tests de Discrimination

### Adversarial Classifier

```python
class DiscriminationTest:
    """
    Tests de discrimination réel vs généré
    """
    
    def __init__(self):
        self.classifier = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def train_discriminator(self, real_data, generated_data, n_epochs=20, lr=0.001):
        """
        Entraîne classifier pour distinguer réel vs généré
        
        Bonne génération = classifier ne peut pas distinguer (accuracy ~50%)
        """
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Labels: 1 = réel, 0 = généré
        real_labels = torch.ones(real_data.size(0), 1)
        gen_labels = torch.zeros(generated_data.size(0), 1)
        
        all_data = torch.cat([real_data, generated_data], dim=0)
        all_labels = torch.cat([real_labels, gen_labels], dim=0)
        
        # Shuffle
        indices = torch.randperm(all_data.size(0))
        all_data = all_data[indices]
        all_labels = all_labels[indices]
        
        # Split train/val
        n_train = int(0.8 * all_data.size(0))
        train_data = all_data[:n_train]
        train_labels = all_labels[:n_train]
        val_data = all_data[n_train:]
        val_labels = all_labels[n_train:]
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward
            outputs = self.classifier(train_data)
            loss = criterion(outputs, train_labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                # Validation
                with torch.no_grad():
                    val_outputs = self.classifier(val_data)
                    val_loss = criterion(val_outputs, val_labels)
                    val_accuracy = ((val_outputs > 0.5).float() == val_labels).float().mean()
        
        # Test final
        with torch.no_grad():
            test_outputs = self.classifier(val_data)
            test_accuracy = ((test_outputs > 0.5).float() == val_labels).float().mean()
            
            # ROC AUC
            test_outputs_np = test_outputs.numpy().flatten()
            val_labels_np = val_labels.numpy().flatten()
            roc_auc = roc_auc_score(val_labels_np, test_outputs_np)
        
        return {
            'accuracy': test_accuracy.item(),
            'roc_auc': roc_auc,
            'is_undistinguishable': test_accuracy.item() < 0.55  # Proche de 50%
        }
    
    def compute_confidence_scores(self, real_data, generated_data):
        """
        Calcule scores de confiance du classifier
        """
        with torch.no_grad():
            real_scores = self.classifier(real_data)
            gen_scores = self.classifier(generated_data)
        
        return {
            'real_mean_confidence': real_scores.mean().item(),
            'gen_mean_confidence': gen_scores.mean().item(),
            'separation': abs(real_scores.mean() - gen_scores.mean()).item()
        }

discriminator_test = DiscriminationTest()

# Test
real_data = torch.randn(5000, 50)
generated_data = torch.randn(5000, 50) * 0.9  # Légèrement différent

disc_results = discriminator_test.train_discriminator(real_data, generated_data)

print(f"\nTest de Discrimination:")
print(f"  Accuracy classifier: {disc_results['accuracy']:.2%}")
print(f"  ROC AUC: {disc_results['roc_auc']:.4f}")
print(f"  Indiscernable (accuracy < 55%): {disc_results['is_undistinguishable']}")
```

---

## Métriques de Qualité

### Scores Composites

```python
class QualityMetrics:
    """
    Métriques de qualité composites
    """
    
    def compute_fid_score(self, real_data, generated_data):
        """
        Fréchet Inception Distance (simplifié)
        
        Mesure distance entre distributions dans espace de features
        """
        # En pratique: utiliser réseau pré-entraîné (Inception)
        # Ici: approximation avec statistiques
        
        real_mean = real_data.mean(dim=0)
        gen_mean = generated_data.mean(dim=0)
        
        real_cov = torch.cov(real_data.T)
        gen_cov = torch.cov(generated_data.T)
        
        # FID = ||mu_real - mu_gen||^2 + Tr(C_real + C_gen - 2*sqrt(C_real * C_gen))
        # Simplifié
        mean_diff = ((real_mean - gen_mean)**2).sum()
        
        return {
            'fid_score': mean_diff.item(),
            'mean_difference': mean_diff.item()
        }
    
    def compute_inception_score(self, generated_data, n_splits=10):
        """
        Inception Score (simplifié)
        
        Mesure qualité et diversité
        """
        # En pratique: utiliser classifier pré-entraîné
        # Ici: approximation
        
        # Score basé sur variance (diversité)
        variance = generated_data.var(dim=0).mean().item()
        
        return {
            'inception_score': variance,
            'diversity': variance
        }
    
    def compute_comprehensive_score(self, real_data, generated_data):
        """
        Score composite combinant plusieurs métriques
        """
        # Moments
        moments = stat_validator.compare_moments(real_data, generated_data)
        mean_diff = np.mean([m['mean_diff'] for m in moments.values()])
        
        # Corrélations
        corr = stat_validator.compute_correlations(real_data, generated_data)
        
        # Discrimination
        disc = discriminator_test.train_discriminator(real_data, generated_data, n_epochs=10)
        
        # Score composite (normalisé)
        score = {
            'mean_accuracy': 1.0 - abs(disc['accuracy'] - 0.5) * 2,  # Pire = 0, meilleur = 1
            'correlation_similarity': 1.0 / (1.0 + corr['mean_difference']),
            'moment_similarity': 1.0 / (1.0 + mean_diff)
        }
        
        score['overall'] = np.mean(list(score.values()))
        
        return score

quality_metrics = QualityMetrics()

# Calculer métriques
fid = quality_metrics.compute_fid_score(real_data, generated_data)
comprehensive = quality_metrics.compute_comprehensive_score(real_data, generated_data)

print(f"\nMétriques de Qualité:")
print(f"  FID score: {fid['fid_score']:.4f}")
print(f"  Score global: {comprehensive['overall']:.4f}")
print(f"  Similarité moments: {comprehensive['moment_similarity']:.4f}")
print(f"  Similarité corrélations: {comprehensive['correlation_similarity']:.4f}")
```

---

## Exercices

### Exercice 21.5.1
Implémentez une suite complète de tests statistiques (KS, Chi-squared, moments) et appliquez-les à un modèle génératif.

### Exercice 21.5.2
Créez un classifier adversarial et utilisez-le pour valider qualité de génération.

### Exercice 21.5.3
Développez une métrique composite qui combine tests statistiques, physiques, et discrimination.

### Exercice 21.5.4
Analysez comment différentes techniques de compression affectent les métriques de qualité.

---

## Points Clés à Retenir

> 📌 **La validation est cruciale pour garantir validité physique et statistique**

> 📌 **Les tests statistiques (KS, Chi-squared) comparent distributions**

> 📌 **Les tests physiques vérifient contraintes (conservation énergie, masses)**

> 📌 **Les tests de discrimination vérifient qu'on ne peut distinguer réel vs généré**

> 📌 **Les métriques composites combinent plusieurs aspects de qualité**

> 📌 **La validation doit être continue tout au long du développement**

---

*Section précédente : [21.4 Compression](./21_04_Compression.md)*

