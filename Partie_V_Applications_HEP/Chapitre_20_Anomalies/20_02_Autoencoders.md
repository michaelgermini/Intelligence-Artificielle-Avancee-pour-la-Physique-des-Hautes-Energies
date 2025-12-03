# 20.2 Autoencoders pour la Détection d'Anomalies

---

## Introduction

Les **autoencoders** sont l'une des méthodes les plus populaires pour la détection d'anomalies en physique des hautes énergies. En apprenant à reconstruire les événements background du Modèle Standard, ils identifient naturellement les événements anormaux comme ceux avec une erreur de reconstruction élevée.

Cette section présente les différents types d'autoencoders utilisés pour la détection d'anomalies, leurs architectures, et leurs applications spécifiques en HEP.

---

## Principe des Autoencoders

### Architecture de Base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class BasicAutoencoder(nn.Module):
    """
    Autoencodeur basique pour détection d'anomalies
    """
    
    def __init__(self, input_dim=100, latent_dim=20, hidden_dims=[64, 32]):
        """
        Args:
            input_dim: Dimension des features d'entrée
            latent_dim: Dimension de l'espace latent
            hidden_dims: Dimensions des couches cachées
        """
        super().__init__()
        
        # Encodeur
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Décodeur (symétrique)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # Pas d'activation finale pour permettre valeurs réelles
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Encode puis décode"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode seulement"""
        return self.encoder(x)
    
    def decode(self, z):
        """Décode seulement"""
        return self.decoder(z)
    
    def compute_reconstruction_error(self, x):
        """
        Calcule erreur de reconstruction (score d'anomalie)
        
        Plus l'erreur est grande, plus l'événement est anormal
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed)**2, dim=1)
            return error

# Exemple
autoencoder = BasicAutoencoder(input_dim=50, latent_dim=10, hidden_dims=[32, 16])

print(f"\nAutoencodeur Basique:")
print(f"  Input: 50 features")
print(f"  Latent: 10 dimensions")
print(f"  Paramètres: {sum(p.numel() for p in autoencoder.parameters()):,}")
```

---

## Variational Autoencoder (VAE)

### Principe et Implémentation

```python
class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE)
    
    Modèle génératif qui apprend distribution latente
    """
    
    def __init__(self, input_dim=100, latent_dim=20, hidden_dim=64):
        super().__init__()
        
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Sorties: μ et log(σ²) pour distribution latente
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Décodeur
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode en distribution latente"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        
        z = μ + σ * ε, où ε ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Décode depuis espace latent"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass complet"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_loss(self, x, beta=1.0):
        """
        Loss VAE = Reconstruction Loss + β * KL Divergence
        
        β contrôle compromis reconstruction/regularization
        """
        reconstructed, mu, logvar = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, x, reduction='sum')
        
        # KL divergence: encourage distribution latente proche de N(0,1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def compute_anomaly_score(self, x):
        """
        Score d'anomalie pour VAE
        
        Combinaison erreur reconstruction + divergence du latent
        """
        with torch.no_grad():
            reconstructed, mu, logvar = self.forward(x)
            
            # Erreur de reconstruction
            recon_error = torch.mean((x - reconstructed)**2, dim=1)
            
            # Divergence du latent (événements anormaux ont latent différent)
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Score combiné
            anomaly_score = recon_error + 0.1 * kl_per_sample
            
            return anomaly_score

vae = VariationalAutoencoder(input_dim=50, latent_dim=10)

print(f"\nVariational Autoencoder:")
print(f"  Paramètres: {sum(p.numel() for p in vae.parameters()):,}")
```

---

## Adversarial Autoencoder (AAE)

### Principe Adversarial

```python
class AdversarialAutoencoder(nn.Module):
    """
    Adversarial Autoencoder
    
    Utilise discriminateur adversarial pour régulariser espace latent
    """
    
    def __init__(self, input_dim=100, latent_dim=20, hidden_dim=64):
        super().__init__()
        
        # Autoencodeur standard
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Discriminateur (distinguer latent réel vs échantillonné)
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode"""
        return self.encoder(x)
    
    def decode(self, z):
        """Décode"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed
    
    def discriminate(self, z):
        """Prédit si z vient de prior ou encoder"""
        return self.discriminator(z)
    
    def compute_anomaly_score(self, x):
        """Score d'anomalie"""
        with torch.no_grad():
            z = self.encode(x)
            reconstructed = self.decode(z)
            
            # Erreur reconstruction
            recon_error = torch.mean((x - reconstructed)**2, dim=1)
            
            # Score du discriminateur (devrait être bas pour anomalies)
            disc_score = self.discriminate(z).squeeze()
            
            # Combinaison
            anomaly_score = recon_error - 0.1 * disc_score  # Moins bien classé = plus anormal
            
            return anomaly_score

aae = AdversarialAutoencoder(input_dim=50, latent_dim=10)
```

---

## Autoencoders pour Événements HEP

### Features et Architecture Spécifiques

```python
class HEPEventAutoencoder(nn.Module):
    """
    Autoencodeur spécialisé pour événements HEP
    """
    
    def __init__(self, feature_groups: Dict[str, int]):
        """
        Args:
            feature_groups: Dict avec types de features et leurs dimensions
                Ex: {'jet_features': 16, 'lepton_features': 8, 'met_features': 4}
        """
        super().__init__()
        
        self.feature_groups = feature_groups
        total_dim = sum(feature_groups.values())
        
        # Encodeur par groupe de features (permet traitement spécialisé)
        self.group_encoders = nn.ModuleDict()
        for group_name, dim in feature_groups.items():
            self.group_encoders[group_name] = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            )
        
        # Encodeur global
        self.global_encoder = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Latent
        )
        
        # Décodeur global
        self.global_decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, total_dim)
        )
        
        # Décodeurs par groupe
        self.group_decoders = nn.ModuleDict()
        for group_name, dim in feature_groups.items():
            self.group_decoders[group_name] = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.ReLU(),
                nn.Linear(dim * 2, dim)
            )
    
    def forward(self, features_dict: Dict[str, torch.Tensor]):
        """
        Forward avec features groupées
        
        Args:
            features_dict: Dict {group_name: tensor}
        """
        # Encode par groupe
        encoded_groups = {}
        for group_name, feat in features_dict.items():
            encoded_groups[group_name] = self.group_encoders[group_name](feat)
        
        # Concatène
        all_encoded = torch.cat(list(encoded_groups.values()), dim=1)
        
        # Encode global
        latent = self.global_encoder(all_encoded)
        
        # Décode global
        all_decoded = self.global_decoder(latent)
        
        # Split et décode par groupe
        decoded_dict = {}
        start_idx = 0
        for group_name, dim in self.feature_groups.items():
            end_idx = start_idx + dim
            group_decoded = all_decoded[:, start_idx:end_idx]
            decoded_dict[group_name] = self.group_decoders[group_name](group_decoded)
            start_idx = end_idx
        
        return decoded_dict, latent
    
    def compute_anomaly_score(self, features_dict: Dict[str, torch.Tensor]):
        """Score d'anomalie par groupe"""
        with torch.no_grad():
            decoded_dict, latent = self.forward(features_dict)
            
            scores = {}
            total_score = 0
            
            for group_name, original in features_dict.items():
                reconstructed = decoded_dict[group_name]
                error = torch.mean((original - reconstructed)**2, dim=1)
                scores[group_name] = error
                total_score += error
            
            return scores, total_score

# Exemple pour événements HEP
hep_ae = HEPEventAutoencoder({
    'jet_features': 16,
    'lepton_features': 8,
    'met_features': 4,
    'event_features': 10
})

print(f"\nHEP Event Autoencoder:")
print(f"  Feature groups: {list(hep_ae.feature_groups.keys())}")
print(f"  Total input dim: {sum(hep_ae.feature_groups.values())}")
```

---

## Entraînement et Optimisation

### Stratégies d'Entraînement

```python
class AutoencoderTraining:
    """
    Stratégies d'entraînement pour autoencoders
    """
    
    def __init__(self):
        self.training_strategies = {
            'background_only': {
                'description': 'Entraîner seulement sur background SM',
                'advantages': ['Apprend distribution background', 'Simple'],
                'disadvantages': ['Pas de signal dans training']
            },
            'weighted_training': {
                'description': 'Pondérer événements selon importance',
                'advantages': ['Focus sur événements importants'],
                'disadvantages': ['Complexité supplémentaire']
            },
            'regularization': {
                'description': 'Régularisation pour éviter overfitting',
                'methods': ['Dropout', 'L2 regularization', 'Early stopping']
            }
        }
    
    def train_autoencoder(self, model, train_loader, n_epochs=50, lr=0.001):
        """
        Entraîne autoencodeur standard
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        model.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def train_vae(self, model, train_loader, n_epochs=50, lr=0.001, beta=1.0):
        """
        Entraîne VAE
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_recon = 0
            epoch_kl = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                loss_dict = model.compute_loss(batch, beta=beta)
                
                loss_dict['total_loss'].backward()
                optimizer.step()
                
                epoch_recon += loss_dict['recon_loss'].item()
                epoch_kl += loss_dict['kl_loss'].item()
            
            avg_recon = epoch_recon / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            losses.append({'recon': avg_recon, 'kl': avg_kl})
        
        return losses

training = AutoencoderTraining()
```

---

## Applications Pratiques

### Détection d'Anomalies dans Données Réelles

```python
class AnomalyDetectionPipeline:
    """
    Pipeline complet de détection d'anomalies
    """
    
    def __init__(self, model, threshold_percentile=95):
        """
        Args:
            model: Autoencodeur entraîné
            threshold_percentile: Percentile pour seuil d'anomalie
        """
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def fit_threshold(self, background_data):
        """
        Détermine seuil d'anomalie depuis données background
        """
        scores = self.model.compute_reconstruction_error(background_data)
        self.threshold = np.percentile(scores.numpy(), self.threshold_percentile)
        
        return self.threshold
    
    def detect_anomalies(self, data):
        """
        Détecte anomalies dans données
        
        Returns:
            anomalies: Indices des événements anormaux
            scores: Scores d'anomalie
        """
        scores = self.model.compute_reconstruction_error(data)
        
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold first.")
        
        anomaly_mask = scores > self.threshold
        anomaly_indices = torch.where(anomaly_mask)[0]
        
        return {
            'anomaly_indices': anomaly_indices.numpy(),
            'scores': scores.numpy(),
            'threshold': self.threshold,
            'n_anomalies': len(anomaly_indices)
        }
    
    def analyze_anomalies(self, data, anomaly_indices):
        """
        Analyse propriétés des anomalies trouvées
        """
        anomaly_events = data[anomaly_indices]
        
        analysis = {
            'n_anomalies': len(anomaly_indices),
            'mean_features': anomaly_events.mean(dim=0),
            'feature_distributions': {}
        }
        
        return analysis

# Exemple pipeline
pipeline = AnomalyDetectionPipeline(autoencoder, threshold_percentile=95)

# Simuler données
background = torch.randn(10000, 50)
anomalies = torch.randn(100, 50) * 3 + 5  # Distribution différente

# Fit threshold
threshold = pipeline.fit_threshold(background)
print(f"\nPipeline Anomaly Detection:")
print(f"  Seuil (95e percentile): {threshold:.4f}")

# Détecter
results = pipeline.detect_anomalies(torch.cat([background, anomalies]))
print(f"  Anomalies détectées: {results['n_anomalies']} / {len(anomalies)} (vraies anomalies)")
```

---

## Exercices

### Exercice 20.2.1
Entraînez un autoencodeur sur données background et testez sa capacité à détecter des signaux injectés de différentes intensités.

### Exercice 20.2.2
Comparez performances d'un autoencodeur standard vs VAE pour détection d'anomalies.

### Exercice 20.2.3
Implémentez un autoencodeur avec features groupées pour événements HEP (jets, leptons, MET).

### Exercice 20.2.4
Analysez l'impact de la dimension de l'espace latent sur capacité de détection et taux de faux positifs.

---

## Points Clés à Retenir

> 📌 **Les autoencoders apprennent à reconstruire background, anomalies ont erreur élevée**

> 📌 **Les VAE apprennent distribution latente, utile pour modélisation générative**

> 📌 **Les AAE utilisent adversarial training pour régulariser espace latent**

> 📌 **Les autoencoders HEP peuvent traiter features groupées spécialisées**

> 📌 **Le seuil d'anomalie doit être calibré sur données background**

> 📌 **L'analyse des anomalies trouvées est cruciale pour interprétation**

---

*Section précédente : [20.1 Nouvelle Physique](./20_01_Nouvelle_Physique.md) | Section suivante : [20.3 Méthodes Non Supervisées](./20_03_Non_Supervise.md)*

