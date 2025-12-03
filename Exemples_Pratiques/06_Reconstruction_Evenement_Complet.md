# Exemple Pratique : Reconstruction Complète d'un Événement Type

---

## Objectif

Démontrer la reconstruction complète d'un événement de collision proton-proton typique au LHC, incluant :
- Reconstruction de traces
- Identification de jets
- b-tagging
- Identification de leptons
- Reconstruction MET

---

## 1. Préparation Données

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
import uproot
import awkward as ak

# Simuler données événement (en pratique, charger depuis ROOT)
class EventGenerator:
    """
    Générateur événements simulés pour démonstration
    """
    
    def generate_event(self):
        """Génère un événement type"""
        event = {
            'hits': self._generate_hits(),
            'tracks': self._generate_tracks(),
            'jets': self._generate_jets(),
            'muons': self._generate_muons(),
            'electrons': self._generate_electrons(),
            'photons': self._generate_photons(),
            'met': self._generate_met()
        }
        return event
    
    def _generate_hits(self, n_hits=500):
        """Génère hits détecteur"""
        hits = []
        for i in range(n_hits):
            hits.append({
                'x': np.random.normal(0, 5),
                'y': np.random.normal(0, 5),
                'z': np.random.normal(0, 20),
                'layer': np.random.randint(0, 10),
                'energy': np.random.exponential(0.5)
            })
        return hits
    
    def _generate_tracks(self, n_tracks=50):
        """Génère traces reconstruites"""
        tracks = []
        for i in range(n_tracks):
            pt = np.random.exponential(5)  # GeV
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            charge = np.random.choice([-1, 1])
            
            tracks.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'charge': charge,
                'd0': np.random.normal(0, 0.01),  # Impact parameter
                'z0': np.random.normal(0, 0.1),
                'n_hits': np.random.randint(5, 15),
                'chi2': np.random.exponential(2)
            })
        return tracks
    
    def _generate_jets(self, n_jets=10):
        """Génère jets"""
        jets = []
        for i in range(n_jets):
            pt = np.random.exponential(30)  # GeV
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            mass = np.random.exponential(5)
            
            # b-tagging score (simulé)
            btag_score = np.random.beta(2, 5)  # Peu de b-jets
            
            jets.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'mass': mass,
                'btag_score': btag_score,
                'flavor': 'b' if btag_score > 0.8 else ('c' if btag_score > 0.5 else 'light'),
                'constituents': np.random.randint(10, 50)
            })
        return jets
    
    def _generate_muons(self, n_muons=2):
        """Génère muons"""
        muons = []
        for i in range(n_muons):
            pt = np.random.exponential(20)
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            muons.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'charge': np.random.choice([-1, 1]),
                'isolation': np.random.exponential(0.1),
                'is_loose': True,
                'is_medium': np.random.random() > 0.3,
                'is_tight': np.random.random() > 0.7
            })
        return muons
    
    def _generate_electrons(self, n_electrons=1):
        """Génère électrons"""
        electrons = []
        for i in range(n_electrons):
            pt = np.random.exponential(25)
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            electrons.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'charge': np.random.choice([-1, 1]),
                'isolation': np.random.exponential(0.15),
                'is_loose': True,
                'is_medium': np.random.random() > 0.4,
                'is_tight': np.random.random() > 0.8
            })
        return electrons
    
    def _generate_photons(self, n_photons=3):
        """Génère photons"""
        photons = []
        for i in range(n_photons):
            pt = np.random.exponential(15)
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            photons.append({
                'pt': pt,
                'eta': eta,
                'phi': phi,
                'is_loose': True,
                'is_tight': np.random.random() > 0.5
            })
        return photons
    
    def _generate_met(self):
        """Génère Missing Transverse Energy"""
        met_x = np.random.normal(0, 15)  # GeV
        met_y = np.random.normal(0, 15)  # GeV
        met = np.sqrt(met_x**2 + met_y**2)
        met_phi = np.arctan2(met_y, met_x)
        
        return {
            'met': met,
            'met_x': met_x,
            'met_y': met_y,
            'met_phi': met_phi
        }

# Générer événement
generator = EventGenerator()
event = generator.generate_event()

print("=== Événement Généré ===")
print(f"Hits: {len(event['hits'])}")
print(f"Tracks: {len(event['tracks'])}")
print(f"Jets: {len(event['jets'])}")
print(f"Muons: {len(event['muons'])}")
print(f"Electrons: {len(event['electrons'])}")
print(f"Photons: {len(event['photons'])}")
print(f"MET: {event['met']['met']:.2f} GeV")
```

---

## 2. Reconstruction de Traces avec ML

```python
class TrackReconstructionML:
    """
    Reconstruction traces avec modèle ML
    """
    
    def __init__(self):
        # Modèle simple pour classification hits → tracks
        self.model = self._create_model()
    
    def _create_model(self):
        """Crée modèle GNN simple pour tracking"""
        # Version simplifiée (dans pratique, utiliser vrai GNN)
        class SimpleTrackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 32)  # Features: x, y, z, layer, energy
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        
        return SimpleTrackModel()
    
    def reconstruct_tracks_from_hits(self, hits):
        """
        Reconstruit traces depuis hits
        (Version simplifiée pour démonstration)
        """
        # Extraire features hits
        hit_features = []
        for hit in hits:
            hit_features.append([
                hit['x'],
                hit['y'],
                hit['z'],
                hit['layer'],
                hit['energy']
            ])
        
        hit_features = torch.tensor(hit_features, dtype=torch.float32)
        
        # Prédire appartenance à trace (simplifié)
        with torch.no_grad():
            scores = self.model(hit_features).squeeze()
        
        # Clustering hits en traces (simplifié)
        # Dans pratique, utiliser algorithme clustering complexe
        tracks = []
        track_id = 0
        used_hits = set()
        
        for i, (hit, score) in enumerate(zip(hits, scores)):
            if score > 0.5 and i not in used_hits:
                # Créer trace depuis ce hit
                track = {
                    'track_id': track_id,
                    'hits': [hit],
                    'pt': np.random.exponential(5),
                    'eta': hit['y'] / np.sqrt(hit['x']**2 + hit['z']**2),
                    'phi': np.arctan2(hit['y'], hit['x']),
                    'score': score.item()
                }
                tracks.append(track)
                used_hits.add(i)
                track_id += 1
        
        return tracks

# Reconstruire traces
track_ml = TrackReconstructionML()
tracks_reconstructed = track_ml.reconstruct_tracks_from_hits(event['hits'])

print(f"\n=== Reconstruction Traces ===")
print(f"Traces reconstruites: {len(tracks_reconstructed)}")
```

---

## 3. Identification et Classification de Jets

```python
class JetClassifier:
    """
    Classification de jets (quark vs gluon, b-tagging)
    """
    
    def __init__(self):
        self.btag_model = self._create_btag_model()
    
    def _create_btag_model(self):
        """Modèle b-tagging"""
        class BTagModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 64)  # Features jet
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 3)  # b, c, light
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.softmax(self.fc3(x))
                return x
        
        return BTagModel()
    
    def classify_jets(self, jets):
        """Classifie jets et calcule b-tag scores"""
        for jet in jets:
            # Extraire features (simplifié)
            features = np.array([
                jet['pt'] / 100.0,  # Normaliser
                jet['eta'] / 2.5,
                jet['phi'] / np.pi,
                jet['mass'] / 50.0,
                jet['constituents'] / 100.0,
                # Ajouter autres features (width, etc.)
                0.1, 0.2, 0.3, 0.4, 0.5  # Placeholders
            ])
            
            # Prédiction (simplifié)
            features_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                scores = self.btag_model(features_tensor).squeeze()
            
            # Assigner labels
            jet['btag_score'] = scores[0].item()
            jet['ctag_score'] = scores[1].item()
            jet['light_score'] = scores[2].item()
            
            # Déterminer flavor
            if scores[0] > 0.7:
                jet['predicted_flavor'] = 'b'
            elif scores[1] > 0.6:
                jet['predicted_flavor'] = 'c'
            else:
                jet['predicted_flavor'] = 'light'
        
        return jets

# Classifier jets
jet_classifier = JetClassifier()
jets_classified = jet_classifier.classify_jets(event['jets'])

print("\n=== Classification Jets ===")
for i, jet in enumerate(jets_classified[:5]):
    print(f"Jet {i+1}: pT={jet['pt']:.1f} GeV, "
          f"b-tag={jet['btag_score']:.3f}, "
          f"Flavor={jet['predicted_flavor']}")
```

---

## 4. Identification de Leptons

```python
class LeptonIdentifier:
    """
    Identification et sélection de leptons
    """
    
    def identify_leptons(self, muons, electrons):
        """
        Identifie et sélectionne leptons avec critères
        """
        selected_muons = []
        selected_electrons = []
        
        # Sélection muons (critères LHC typiques)
        for muon in muons:
            if (muon['pt'] > 10.0 and  # pT > 10 GeV
                abs(muon['eta']) < 2.4 and  # |eta| < 2.4
                muon['isolation'] < 0.15 and  # Isolation
                muon['is_medium']):  # Medium ID
                selected_muons.append(muon)
        
        # Sélection électrons
        for electron in electrons:
            if (electron['pt'] > 10.0 and
                abs(electron['eta']) < 2.5 and
                electron['isolation'] < 0.2 and
                electron['is_medium']):
                selected_electrons.append(electron)
        
        return {
            'muons': selected_muons,
            'electrons': selected_electrons
        }
    
    def find_opposite_sign_pairs(self, muons, electrons):
        """Trouve paires leptons signe opposé"""
        pairs = []
        
        # Muon-muon pairs
        for i, m1 in enumerate(muons):
            for j, m2 in enumerate(muons[i+1:], i+1):
                if m1['charge'] * m2['charge'] < 0:  # Signe opposé
                    pairs.append({
                        'type': 'muon-muon',
                        'lepton1': m1,
                        'lepton2': m2,
                        'mass': self._calculate_dilepton_mass(m1, m2)
                    })
        
        # Electron-electron pairs
        for i, e1 in enumerate(electrons):
            for j, e2 in enumerate(electrons[i+1:], i+1):
                if e1['charge'] * e2['charge'] < 0:
                    pairs.append({
                        'type': 'electron-electron',
                        'lepton1': e1,
                        'lepton2': e2,
                        'mass': self._calculate_dilepton_mass(e1, e2)
                    })
        
        return pairs
    
    def _calculate_dilepton_mass(self, lep1, lep2):
        """Calcule masse invariante paire leptons"""
        # Approximation: masses négligées
        pt1, eta1, phi1 = lep1['pt'], lep1['eta'], lep1['phi']
        pt2, eta2, phi2 = lep2['pt'], lep2['eta'], lep2['phi']
        
        # Composantes moment
        px1 = pt1 * np.cos(phi1)
        py1 = pt1 * np.sin(phi1)
        pz1 = pt1 * np.sinh(eta1)
        
        px2 = pt2 * np.cos(phi2)
        py2 = pt2 * np.sin(phi2)
        pz2 = pt2 * np.sinh(eta2)
        
        # Masse invariante
        e1 = np.sqrt(pt1**2 + pz1**2)  # Approximation
        e2 = np.sqrt(pt2**2 + pz2**2)
        
        px_tot = px1 + px2
        py_tot = py1 + py2
        pz_tot = pz1 + pz2
        e_tot = e1 + e2
        
        mass = np.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2)
        return mass

# Identifier leptons
lepton_id = LeptonIdentifier()
selected_leptons = lepton_id.identify_leptons(event['muons'], event['electrons'])
dilepton_pairs = lepton_id.find_opposite_sign_pairs(
    selected_leptons['muons'],
    selected_leptons['electrons']
)

print("\n=== Identification Leptons ===")
print(f"Muons sélectionnés: {len(selected_leptons['muons'])}")
print(f"Électrons sélectionnés: {len(selected_leptons['electrons'])}")
print(f"Paires dileptons: {len(dilepton_pairs)}")

for pair in dilepton_pairs:
    print(f"  {pair['type']}: masse={pair['mass']:.2f} GeV")
```

---

## 5. Reconstruction MET Améliorée

```python
class METReconstruction:
    """
    Reconstruction MET avec corrections
    """
    
    def reconstruct_met(self, jets, muons, electrons, met_raw):
        """
        Reconstruit MET avec corrections
        """
        # MET raw depuis calorimètres
        met_x_corr = met_raw['met_x']
        met_y_corr = met_raw['met_y']
        
        # Correction pour muons (pas vus par calorimètres)
        for muon in muons:
            met_x_corr += muon['pt'] * np.cos(muon['phi'])
            met_y_corr += muon['pt'] * np.sin(muon['phi'])
        
        # Correction pour jets (calibration énergie)
        for jet in jets:
            if jet['pt'] > 15.0:  # Seuil
                # Facteur correction (typiquement ~1.05-1.1)
                correction_factor = 1.05
                met_x_corr -= (correction_factor - 1.0) * jet['pt'] * np.cos(jet['phi'])
                met_y_corr -= (correction_factor - 1.0) * jet['pt'] * np.sin(jet['phi'])
        
        # Calculer MET final
        met_corrected = np.sqrt(met_x_corr**2 + met_y_corr**2)
        met_phi_corrected = np.arctan2(met_y_corr, met_x_corr)
        
        # Résolution MET (approximation)
        met_resolution = 15.0 + 0.5 * np.sqrt(sum(j['pt'] for j in jets))
        
        return {
            'met': met_corrected,
            'met_x': met_x_corr,
            'met_y': met_y_corr,
            'met_phi': met_phi_corrected,
            'resolution': met_resolution,
            'significance': met_corrected / met_resolution if met_resolution > 0 else 0
        }

# Reconstruire MET
met_reco = METReconstruction()
met_corrected = met_reco.reconstruct_met(
    jets_classified,
    selected_leptons['muons'],
    selected_leptons['electrons'],
    event['met']
)

print("\n=== Reconstruction MET ===")
print(f"MET raw: {event['met']['met']:.2f} GeV")
print(f"MET corrigé: {met_corrected['met']:.2f} GeV")
print(f"Résolution: {met_corrected['resolution']:.2f} GeV")
print(f"Significativité: {met_corrected['significance']:.2f}")
```

---

## 6. Analyse Complète Événement

```python
class EventAnalyzer:
    """
    Analyse complète événement reconstruit
    """
    
    def analyze_event(self, event_reconstructed):
        """Analyse événement et calcule quantités physiques"""
        
        analysis = {
            'n_jets': len(event_reconstructed['jets']),
            'n_bjets': len([j for j in event_reconstructed['jets'] if j['predicted_flavor'] == 'b']),
            'n_leptons': (len(event_reconstructed['muons']) + 
                         len(event_reconstructed['electrons'])),
            'ht': sum(j['pt'] for j in event_reconstructed['jets'] if j['pt'] > 30),
            'met': event_reconstructed['met']['met'],
            'n_dilepton_pairs': len(event_reconstructed['dilepton_pairs'])
        }
        
        # Calculer quantités additionnelles
        if event_reconstructed['jets']:
            leading_jet = max(event_reconstructed['jets'], key=lambda x: x['pt'])
            analysis['leading_jet_pt'] = leading_jet['pt']
            analysis['leading_jet_eta'] = leading_jet['eta']
        
        if event_reconstructed['dilepton_pairs']:
            # Masse dilepton
            masses = [p['mass'] for p in event_reconstructed['dilepton_pairs']]
            analysis['dilepton_mass'] = max(masses) if masses else 0
        
        # HT + MET (quantité utile pour recherche nouvelle physique)
        analysis['ht_met'] = analysis['ht'] + analysis['met']
        
        return analysis
    
    def classify_event(self, analysis):
        """Classifie type événement"""
        event_type = "Unknown"
        
        # Détection événements type W/Z
        if analysis['n_leptons'] >= 1 and analysis['met'] > 30:
            event_type = "W-like"
        
        if analysis['n_dilepton_pairs'] > 0:
            if 80 < analysis.get('dilepton_mass', 0) < 100:
                event_type = "Z-like"
        
        # Détection événements top
        if analysis['n_bjets'] >= 1 and analysis['n_leptons'] >= 1:
            event_type = "Top-like"
        
        # Détection événements Higgs
        if (analysis['n_bjets'] >= 2 and 
            analysis['n_leptons'] >= 2 and
            analysis['met'] < 50):
            event_type = "Higgs-like"
        
        return event_type

# Analyser événement reconstruit
event_reconstructed = {
    'jets': jets_classified,
    'muons': selected_leptons['muons'],
    'electrons': selected_leptons['electrons'],
    'met': met_corrected,
    'dilepton_pairs': dilepton_pairs,
    'tracks': tracks_reconstructed
}

analyzer = EventAnalyzer()
event_analysis = analyzer.analyze_event(event_reconstructed)
event_type = analyzer.classify_event(event_analysis)

print("\n" + "="*70)
print("ANALYSE ÉVÉNEMENT COMPLÈTE")
print("="*70)
print(f"\nJets: {event_analysis['n_jets']} (dont {event_analysis['n_bjets']} b-jets)")
print(f"Leptons: {event_analysis['n_leptons']}")
print(f"HT: {event_analysis['ht']:.2f} GeV")
print(f"MET: {event_analysis['met']:.2f} GeV")
print(f"Paires dileptons: {event_analysis['n_dilepton_pairs']}")
if 'dilepton_mass' in event_analysis:
    print(f"Masse dilepton: {event_analysis['dilepton_mass']:.2f} GeV")
print(f"\nType événement: {event_type}")
```

---

## 7. Visualisation Événement

```python
def visualize_event(event_reconstructed):
    """
    Visualise événement reconstruit
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Vue transverse (eta-phi)
    ax1 = plt.subplot(2, 2, 1)
    
    # Jets
    for jet in event_reconstructed['jets']:
        color = 'red' if jet['predicted_flavor'] == 'b' else ('orange' if jet['predicted_flavor'] == 'c' else 'blue')
        size = jet['pt'] * 5  # Taille proportionnelle pT
        ax1.scatter(jet['eta'], jet['phi'], s=size, c=color, alpha=0.6, 
                   label=jet['predicted_flavor'] if jet == event_reconstructed['jets'][0] else "")
    
    # Muons
    for muon in event_reconstructed['muons']:
        ax1.scatter(muon['eta'], muon['phi'], marker='*', s=200, c='green', 
                   label='Muon' if muon == event_reconstructed['muons'][0] else "")
    
    # Électrons
    for electron in event_reconstructed['electrons']:
        ax1.scatter(electron['eta'], electron['phi'], marker='s', s=200, c='purple',
                   label='Electron' if electron == event_reconstructed['electrons'][0] else "")
    
    # MET
    met = event_reconstructed['met']
    ax1.arrow(0, 0, 
              met['met'] * np.cos(met['met_phi']) / 10,
              met['met'] * np.sin(met['met_phi']) / 10,
              head_width=0.1, head_length=0.1, fc='black', ec='black',
              label='MET', linewidth=2)
    
    ax1.set_xlabel('Pseudorapidity (η)')
    ax1.set_ylabel('Azimuth (φ)')
    ax1.set_title('Vue Transverse (η-φ)')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-np.pi, np.pi)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution pT jets
    ax2 = plt.subplot(2, 2, 2)
    jet_pts = [j['pt'] for j in event_reconstructed['jets']]
    ax2.hist(jet_pts, bins=20, alpha=0.7, color='blue')
    ax2.set_xlabel('Jet pT (GeV)')
    ax2.set_ylabel('Nombre')
    ax2.set_title('Distribution pT Jets')
    ax2.grid(True, alpha=0.3)
    
    # Distribution b-tag scores
    ax3 = plt.subplot(2, 2, 3)
    btag_scores = [j['btag_score'] for j in event_reconstructed['jets']]
    ax3.hist(btag_scores, bins=20, alpha=0.7, color='red')
    ax3.set_xlabel('b-tag Score')
    ax3.set_ylabel('Nombre')
    ax3.set_title('Distribution b-tag Scores')
    ax3.grid(True, alpha=0.3)
    
    # Masse dilepton (si disponible)
    ax4 = plt.subplot(2, 2, 4)
    if event_reconstructed['dilepton_pairs']:
        dilepton_masses = [p['mass'] for p in event_reconstructed['dilepton_pairs']]
        ax4.hist(dilepton_masses, bins=20, alpha=0.7, color='green')
        ax4.axvline(x=91.2, color='r', linestyle='--', label='Z mass (91.2 GeV)')
        ax4.set_xlabel('Masse Invariante (GeV)')
        ax4.set_ylabel('Nombre')
        ax4.set_title('Masse Invariante Dileptons')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Pas de paires dileptons', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Masse Invariante Dileptons')
    
    plt.tight_layout()
    plt.savefig('event_reconstruction.png', dpi=150)
    plt.show()

# Visualiser
visualize_event(event_reconstructed)
```

---

## 8. Résumé Reconstruction

```python
def print_reconstruction_summary(event_reconstructed, event_analysis):
    """Affiche résumé reconstruction"""
    
    print("\n" + "="*70)
    print("RÉSUMÉ RECONSTRUCTION ÉVÉNEMENT")
    print("="*70)
    
    print("\n📊 Objets Reconstruits:")
    print(f"  • Traces: {len(event_reconstructed['tracks'])}")
    print(f"  • Jets: {event_analysis['n_jets']} (dont {event_analysis['n_bjets']} b-jets)")
    print(f"  • Muons: {len(event_reconstructed['muons'])}")
    print(f"  • Électrons: {len(event_reconstructed['electrons'])}")
    print(f"  • Photons: {len(event.get('photons', []))}")
    
    print("\n⚡ Quantités Énergétiques:")
    print(f"  • HT: {event_analysis['ht']:.2f} GeV")
    print(f"  • MET: {event_analysis['met']:.2f} GeV")
    print(f"  • MET significativité: {event_reconstructed['met']['significance']:.2f}")
    print(f"  • HT + MET: {event_analysis['ht_met']:.2f} GeV")
    
    print("\n🔬 Propriétés Événement:")
    print(f"  • Type: {event_type}")
    print(f"  • Paires dileptons: {event_analysis['n_dilepton_pairs']}")
    if 'dilepton_mass' in event_analysis:
        print(f"  • Masse dilepton: {event_analysis['dilepton_mass']:.2f} GeV")
        if 80 < event_analysis['dilepton_mass'] < 100:
            print("    → Signature Z boson possible!")
    
    print("\n✅ Événement reconstruit avec succès!")
    print("   Prêt pour analyse physique")

# Afficher résumé
print_reconstruction_summary(event_reconstructed, event_analysis)
```

---

## Résultats Typiques

### Événement Type

- **Jets** : 8-12 jets, dont 1-2 b-jets
- **Leptons** : 1-3 leptons (muons + électrons)
- **MET** : 20-50 GeV (typique)
- **HT** : 200-500 GeV
- **Type** : W-like, Z-like, ou Top-like selon configuration

---

## Points Clés

✅ **Workflow complet** : Hits → Tracks → Jets → Leptons → MET  
✅ **ML intégré** : Modèles pour tracking et b-tagging  
✅ **Corrections** : MET avec corrections muons et jets  
✅ **Classification** : Identification type événement  
✅ **Visualisation** : Représentation graphique complète  
📊 **Quantités physiques** : HT, MET, masses invariantes  

---

*Cet exemple démontre reconstruction complète événement type LHC avec ML intégré.*

