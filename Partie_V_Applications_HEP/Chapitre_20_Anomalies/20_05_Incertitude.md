# 20.5 Quantification de l'Incertitude

---

## Introduction

La **quantification de l'incertitude** est cruciale pour la détection d'anomalies en physique des hautes énergies. Il est essentiel de distinguer les vraies anomalies des événements qui semblent anormaux simplement à cause d'incertitudes statistiques ou systématiques. De plus, l'incertitude permet d'évaluer la confiance dans les prédictions et de guider les décisions.

Cette section présente les méthodes pour quantifier l'incertitude dans les modèles de détection d'anomalies, incluant les approches bayésiennes, ensemblistes, et basées sur la calibration.

---

## Types d'Incertitude

### Épistémique vs Aléatoire

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class UncertaintyTypes:
    """
    Types d'incertitude
    """
    
    def __init__(self):
        self.uncertainty_types = {
            'epistemic': {
                'description': 'Incertitude sur le modèle (réductible avec plus de données)',
                'also_known_as': 'Incertitude de modèle',
                'sources': [
                    'Paramètres du modèle incertains',
                    'Manque de données dans certaines régions',
                    'Limites du modèle'
                ],
                'quantification': ['Bayesian neural networks', 'Dropout', 'Ensembles'],
                'reduction': 'Plus de données, meilleur modèle'
            },
            'aleatoric': {
                'description': 'Incertitude intrinsèque aux données (irréductible)',
                'also_known_as': 'Incertitude de données',
                'sources': [
                    'Bruit de mesure',
                    'Variabilité naturelle',
                    'Résolution détecteur'
                ],
                'quantification': ['Output variance', 'Heteroscedastic models'],
                'reduction': 'Ne peut pas être réduite, seulement quantifiée'
            },
            'systematic': {
                'description': 'Incertitudes systématiques expérimentales',
                'also_known_as': 'Incertitudes systématiques',
                'sources': [
                    'Calibration détecteurs',
                    'Modélisation backgrounds',
                    'Acceptance et efficacité'
                ],
                'quantification': ['Nuisance parameters', 'Systematic variations'],
                'reduction': 'Amélioration mesures, meilleure compréhension'
            }
        }
    
    def display_types(self):
        """Affiche les types"""
        print("\n" + "="*70)
        print("Types d'Incertitude")
        print("="*70)
        
        for unc_type, info in self.uncertainty_types.items():
            print(f"\n{unc_type.replace('_', ' ').title()} ({info['also_known_as']}):")
            print(f"  Description: {info['description']}")
            print(f"  Sources:")
            for source in info['sources']:
                print(f"    • {source}")
            print(f"  Quantification: {', '.join(info['quantification'])}")
            print(f"  Réduction: {info['reduction']}")

unc_types = UncertaintyTypes()
unc_types.display_types()
```

---

## Incertitude Épistémique: Bayesian Neural Networks

### Réseaux Bayésiens

```python
class BayesianLinear(nn.Module):
    """
    Couche linéaire bayésienne
    
    Poids suivent distributions au lieu de valeurs fixes
    """
    
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        
        # Moyennes des poids (paramètres à apprendre)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        
        # Log-variance des poids (paramètres à apprendre)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1)
        
        self.prior_std = prior_std
    
    def forward(self, x, sample=True):
        """
        Forward pass avec échantillonnage de poids
        
        Args:
            sample: Si True, échantillonne poids depuis distribution
        """
        if sample:
            # Échantillonner poids depuis distributions
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(weight_std)
            weight = self.weight_mu + weight_std * weight_eps
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(bias_std)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Utiliser moyennes
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        Calcule KL divergence entre posterior et prior
        
        Pour variational inference
        """
        # Prior: N(0, prior_std²)
        # Posterior: N(mu, exp(logvar))
        
        weight_kl = -0.5 * torch.sum(
            1 + self.weight_logvar - 
            (self.weight_mu / self.prior_std)**2 - 
            torch.exp(self.weight_logvar) / (self.prior_std**2)
        )
        
        bias_kl = -0.5 * torch.sum(
            1 + self.bias_logvar - 
            (self.bias_mu / self.prior_std)**2 - 
            torch.exp(self.bias_logvar) / (self.prior_std**2)
        )
        
        return weight_kl + bias_kl

class BayesianAutoencoder(nn.Module):
    """
    Autoencodeur bayésien pour quantification incertitude
    """
    
    def __init__(self, input_dim=100, latent_dim=20, hidden_dim=64):
        super().__init__()
        
        # Encodeur bayésien
        self.encoder_fc1 = BayesianLinear(input_dim, hidden_dim)
        self.encoder_fc2 = BayesianLinear(hidden_dim, latent_dim)
        
        # Décodeur bayésien
        self.decoder_fc1 = BayesianLinear(latent_dim, hidden_dim)
        self.decoder_fc2 = BayesianLinear(hidden_dim, input_dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x, sample=True):
        """Forward avec échantillonnage"""
        # Encodeur
        h = self.activation(self.encoder_fc1(x, sample=sample))
        latent = self.encoder_fc2(h, sample=sample)
        
        # Décodeur
        h = self.activation(self.decoder_fc1(latent, sample=sample))
        reconstructed = self.decoder_fc2(h, sample=sample)
        
        return reconstructed
    
    def compute_kl_loss(self):
        """Calcule KL divergence totale"""
        kl = (self.encoder_fc1.kl_divergence() + 
              self.encoder_fc2.kl_divergence() +
              self.decoder_fc1.kl_divergence() +
              self.decoder_fc2.kl_divergence())
        return kl
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Prédit avec quantification d'incertitude
        
        Échantillonne plusieurs fois pour estimer incertitude
        """
        predictions = []
        
        for _ in range(n_samples):
            pred = self.forward(x, sample=True)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Statistiques
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Incertitude épistémique = variance des prédictions
        epistemic_uncertainty = std_pred
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty
        }

bayesian_ae = BayesianAutoencoder(input_dim=50, latent_dim=10)

print(f"\nBayesian Autoencoder:")
print(f"  Paramètres: {sum(p.numel() for p in bayesian_ae.parameters()):,}")
```

---

## Incertitude via Dropout

### Monte Carlo Dropout

```python
class DropoutUncertainty:
    """
    Quantification incertitude via Monte Carlo Dropout
    """
    
    def __init__(self, model, dropout_rate=0.5):
        """
        Args:
            model: Modèle avec couches dropout
            dropout_rate: Taux de dropout
        """
        self.model = model
        self.dropout_rate = dropout_rate
        
        # S'assurer que dropout est activé même en eval
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Active dropout même en mode eval"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Force mode training pour garder dropout
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Prédit avec incertitude via MC Dropout
        
        Échantillonne plusieurs fois avec dropout activé
        """
        self.model.eval()
        self._enable_dropout()  # Important: garder dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'uncertainty': std_pred
        }

# Exemple avec autoencodeur avec dropout
dropout_ae = nn.Sequential(
    nn.Linear(50, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 10),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 50)
)

dropout_uncertainty = DropoutUncertainty(dropout_ae, dropout_rate=0.5)
```

---

## Ensembles pour Incertitude

### Ensembles de Modèles

```python
class EnsembleUncertainty:
    """
    Quantification incertitude via ensembles de modèles
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: Liste de modèles entraînés différemment
        """
        self.models = models
    
    def predict_with_uncertainty(self, x):
        """
        Prédit avec incertitude via variance d'ensemble
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'uncertainty': std_pred,
            'predictions': predictions
        }
    
    def compute_anomaly_score_with_uncertainty(self, x):
        """
        Score d'anomalie avec incertitude
        
        Score = erreur reconstruction + pénalité incertitude
        """
        results = self.predict_with_uncertainty(x)
        
        # Erreur de reconstruction moyenne
        mean_error = torch.mean((x - results['mean'])**2, dim=1)
        
        # Incertitude moyenne
        mean_uncertainty = torch.mean(results['uncertainty'], dim=1)
        
        # Score combiné: événements avec haute erreur ET haute incertitude = moins confiant
        # Score élevé si erreur haute mais incertitude basse (confiant dans anomalie)
        anomaly_score = mean_error / (mean_uncertainty + 1e-6)
        
        return {
            'anomaly_score': anomaly_score,
            'reconstruction_error': mean_error,
            'uncertainty': mean_uncertainty
        }

# Créer ensemble
ensemble_models = [
    BasicAutoencoder(input_dim=50, latent_dim=10) for _ in range(5)
]

ensemble = EnsembleUncertainty(ensemble_models)

print(f"\nEnsemble Uncertainty:")
print(f"  Nombre de modèles: {len(ensemble.models)}")
```

---

## Incertitude Aléatoire: Modèles Hétéroscédastiques

### Prédiction de Variance

```python
class HeteroscedasticAutoencoder(nn.Module):
    """
    Autoencodeur hétéroscédastique
    
    Prédit à la fois moyenne et variance (incertitude aléatoire)
    """
    
    def __init__(self, input_dim=100, latent_dim=20, hidden_dim=64):
        super().__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Décodeur pour moyenne
        self.decoder_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Décodeur pour variance (log-variance pour positivité)
        self.decoder_logvar = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        """Forward pass"""
        latent = self.encoder(x)
        
        mean = self.decoder_mean(latent)
        logvar = self.decoder_logvar(latent)
        
        return mean, logvar
    
    def compute_loss(self, x):
        """
        Loss pour modèle hétéroscédastique
        
        Log-likelihood avec variance prédite
        """
        mean, logvar = self.forward(x)
        
        # Variance prédite
        var = torch.exp(logvar)
        
        # Negative log-likelihood (gaussien)
        nll = 0.5 * torch.sum(
            torch.log(2 * np.pi * var) + (x - mean)**2 / var,
            dim=1
        ).mean()
        
        return nll
    
    def predict_with_uncertainty(self, x):
        """
        Prédit avec incertitude aléatoire
        """
        with torch.no_grad():
            mean, logvar = self.forward(x)
            std = torch.exp(0.5 * logvar)
            
            return {
                'mean': mean,
                'std': std,
                'variance': torch.exp(logvar),
                'aleatoric_uncertainty': std
            }

hetero_ae = HeteroscedasticAutoencoder(input_dim=50, latent_dim=10)

print(f"\nHeteroscedastic Autoencoder:")
print(f"  Paramètres: {sum(p.numel() for p in hetero_ae.parameters()):,}")
```

---

## Application à Détection d'Anomalies

### Utilisation de l'Incertitude

```python
class UncertaintyAwareAnomalyDetection:
    """
    Détection d'anomalies prenant en compte incertitude
    """
    
    def __init__(self, model, uncertainty_quantifier):
        """
        Args:
            model: Modèle de détection d'anomalies
            uncertainty_quantifier: Méthode pour quantifier incertitude
        """
        self.model = model
        self.uncertainty_quantifier = uncertainty_quantifier
    
    def detect_anomalies_with_confidence(self, x, 
                                        error_threshold: float,
                                        uncertainty_threshold: float):
        """
        Détecte anomalies avec seuils sur erreur ET incertitude
        
        Anomalie confiante = haute erreur + basse incertitude
        """
        # Calculer erreur de reconstruction
        with torch.no_grad():
            reconstructed = self.model(x)
            error = torch.mean((x - reconstructed)**2, dim=1)
        
        # Calculer incertitude
        unc_result = self.uncertainty_quantifier.predict_with_uncertainty(x)
        uncertainty = torch.mean(unc_result['uncertainty'], dim=1)
        
        # Décisions
        high_error = error > error_threshold
        low_uncertainty = uncertainty < uncertainty_threshold
        
        # Anomalies confiantes
        confident_anomalies = high_error & low_uncertainty
        
        # Événements avec haute incertitude (à examiner)
        high_uncertainty_events = uncertainty > uncertainty_threshold
        
        return {
            'anomaly_indices': torch.where(confident_anomalies)[0],
            'high_uncertainty_indices': torch.where(high_uncertainty_events)[0],
            'error': error,
            'uncertainty': uncertainty,
            'confidence': 1.0 / (uncertainty + 1e-6)  # Plus incertitude basse = plus confiant
        }
    
    def compute_calibrated_threshold(self, validation_data, target_fpr=0.05):
        """
        Calcule seuil calibré tenant compte incertitude
        
        Ajuste seuil selon incertitude pour maintenir FPR constant
        """
        # Calculer erreurs et incertitudes
        with torch.no_grad():
            reconstructed = self.model(validation_data)
            errors = torch.mean((validation_data - reconstructed)**2, dim=1)
        
        unc_result = self.uncertainty_quantifier.predict_with_uncertainty(validation_data)
        uncertainties = torch.mean(unc_result['uncertainty'], dim=1)
        
        # Score ajusté par incertitude
        # Plus incertitude élevée = seuil plus élevé (moins sensible)
        adjusted_scores = errors / (uncertainties + 1e-6)
        
        # Trouver seuil pour FPR cible
        threshold = np.percentile(adjusted_scores.numpy(), (1 - target_fpr) * 100)
        
        return threshold

# Application
bayesian_uncertainty = lambda model, x: model.predict_with_uncertainty(x, n_samples=50)
unc_aware_detector = UncertaintyAwareAnomalyDetection(bayesian_ae, bayesian_uncertainty)
```

---

## Exercices

### Exercice 20.5.1
Implémentez un autoencodeur bayésien complet avec variational inference et comparez incertitude épistémique vs aléatoire.

### Exercice 20.5.2
Utilisez Monte Carlo Dropout pour quantifier incertitude dans un autoencodeur et analysez l'impact du taux de dropout.

### Exercice 20.5.3
Créez un système de détection d'anomalies qui utilise incertitude pour filtrer les anomalies peu confiantes.

### Exercice 20.5.4
Comparez différentes méthodes de quantification d'incertitude (Bayesian, Dropout, Ensemble) sur même modèle.

---

## Points Clés à Retenir

> 📌 **L'incertitude épistémique est réductible avec plus de données**

> 📌 **L'incertitude aléatoire est intrinsèque et irréductible**

> 📌 **Les Bayesian Neural Networks quantifient incertitude épistémique**

> 📌 **Monte Carlo Dropout est simple et efficace pour incertitude**

> 📌 **Les ensembles de modèles donnent estimation robuste d'incertitude**

> 📌 **L'utilisation d'incertitude améliore fiabilité détection d'anomalies**

---

*Section précédente : [20.4 Réseaux de Tenseurs](./20_04_Tenseurs.md)*

