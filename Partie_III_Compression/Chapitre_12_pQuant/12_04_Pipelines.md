# 12.4 Pipelines de Compression Automatisés

---

## Introduction

Les **pipelines automatisés** de pQuant permettent de compresser des modèles avec des stratégies prédéfinies ou personnalisées, en gérant automatiquement l'ordre des opérations et l'optimisation des hyperparamètres.

---

## Pipeline Standard

```python
from pquant.pipelines import StandardCompressionPipeline

class StandardCompressionPipeline:
    """
    Pipeline standard avec étapes optimisées
    """
    
    def __init__(self, target_compression=8.0, preserve_accuracy=0.98):
        """
        Args:
            target_compression: Compression cible (ex: 8x)
            preserve_accuracy: Fraction d'accuracy à préserver (ex: 0.98 = 98%)
        """
        self.target_compression = target_compression
        self.preserve_accuracy = preserve_accuracy
    
    def execute(self, model, train_loader, val_loader):
        """
        Exécute le pipeline standard
        """
        # Étape 1: Analyse du modèle
        analysis = self._analyze_model(model, val_loader)
        
        # Étape 2: Compression progressive
        compressed = self._progressive_compression(
            model, train_loader, val_loader, analysis
        )
        
        # Étape 3: Fine-tuning
        final_model = self._fine_tune(compressed, train_loader, val_loader)
        
        return final_model
    
    def _progressive_compression(self, model, train_loader, val_loader, analysis):
        """
        Compression progressive: augmente progressivement la compression
        """
        current_model = model
        current_compression = 1.0
        baseline_accuracy = analysis['baseline_accuracy']
        
        # Étape 1: Low-rank
        if current_compression < self.target_compression:
            rank = self._estimate_rank(current_model, self.target_compression / current_compression)
            
            compressor = LowRankCompression({'rank': rank})
            current_model = compressor.compress(current_model, train_loader)
            
            # Vérifie l'accuracy
            acc = self._evaluate(current_model, val_loader)
            if acc < baseline_accuracy * self.preserve_accuracy:
                # Reviens en arrière
                current_model = model
            else:
                current_compression *= (rank / min(current_model.in_features, current_model.out_features))
        
        # Étape 2: Quantification
        if current_compression < self.target_compression:
            bits = 8
            compressor = QuantizationCompression({'bits': bits})
            current_model = compressor.compress(current_model, train_loader)
            
            # Vérifie l'accuracy
            acc = self._evaluate(current_model, val_loader)
            if acc >= baseline_accuracy * self.preserve_accuracy:
                current_compression *= 4  # 32bits → 8bits
        
        return current_model
```

---

## Pipeline Auto-Tuning

```python
from pquant.pipelines import AutoTuningPipeline

class AutoTuningPipeline:
    """
    Pipeline qui trouve automatiquement les meilleurs hyperparamètres
    """
    
    def __init__(self, search_space, objective='compression_accuracy'):
        """
        Args:
            search_space: Espace de recherche des hyperparamètres
            objective: Objectif ('compression_accuracy', 'compression_only', etc.)
        """
        self.search_space = search_space
        self.objective = objective
    
    def execute(self, model, train_loader, val_loader):
        """
        Recherche les meilleurs hyperparamètres
        """
        from optuna import create_study
        
        study = create_study(direction='maximize')
        
        def objective(trial):
            # Suggère des hyperparamètres
            rank = trial.suggest_int('rank', 32, 128, step=16)
            bits = trial.suggest_int('bits', 6, 8)
            
            # Compresse
            config = {
                'low_rank': {'rank': rank},
                'quantization': {'bits': bits}
            }
            pipeline = CompressionPipeline(config)
            compressed = pipeline.compress(model, train_loader)
            
            # Évalue
            results = pipeline.evaluate(model, compressed, val_loader)
            
            # Score selon l'objectif
            if self.objective == 'compression_accuracy':
                score = results['compressed']['accuracy'] * \
                       np.log(results['total_compression']['compression_ratio'])
            else:
                score = results['total_compression']['compression_ratio']
            
            return score
        
        # Optimise
        study.optimize(objective, n_trials=50)
        
        # Meilleure configuration
        best_config = study.best_params
        print(f"Best config: {best_config}")
        
        # Compresse avec la meilleure config
        final_config = {
            'low_rank': {'rank': best_config['rank']},
            'quantization': {'bits': best_config['bits']}
        }
        pipeline = CompressionPipeline(final_config)
        final_model = pipeline.compress(model, train_loader)
        
        return final_model, best_config
```

---

## Pipeline HEP-Spécialisé

```python
from pquant.pipelines import HEPCompressionPipeline

class HEPCompressionPipeline:
    """
    Pipeline optimisé pour modèles de physique des particules
    """
    
    def compress_trigger_model(self, model, target_latency_ns=100):
        """
        Compresse pour trigger L1 avec contrainte de latence
        """
        # Configuration agressive
        config = {
            'methods': ['pruning', 'quantization'],
            'pruning': {
                'sparsity': 0.9,
                'method': 'structured'
            },
            'quantization': {
                'bits': 6,
                'method': 'ptq'
            }
        }
        
        pipeline = CompressionPipeline(config)
        compressed = pipeline.compress(model)
        
        # Vérifie la latence
        latency = self._measure_latency(compressed)
        
        if latency > target_latency_ns:
            # Compression plus agressive
            config['pruning']['sparsity'] = 0.95
            config['quantization']['bits'] = 4
            compressed = pipeline.compress(model)
        
        return compressed
    
    def compress_jet_tagger(self, model, train_loader, preserve_rare_classes=True):
        """
        Compresse un tagger de jets en préservant les classes rares
        """
        config = {
            'methods': ['low_rank', 'quantization'],
            'low_rank': {'rank': 64},
            'quantization': {'bits': 8},
            'preserve_rare_classes': preserve_rare_classes
        }
        
        pipeline = CompressionPipeline(config)
        compressed = pipeline.compress(model, train_loader)
        
        # Évaluation spéciale pour classes rares
        results = self._evaluate_rare_classes(model, compressed, train_loader)
        
        return compressed, results
```

---

## Pipeline de Validation

```python
class ValidationPipeline:
    """
    Pipeline avec validation automatique
    """
    
    def execute(self, model, train_loader, val_loader, test_loader):
        """
        Compresse avec validation à chaque étape
        """
        baseline = self._evaluate(model, test_loader)
        
        compressed = model
        compression_history = []
        
        # Compression par étapes avec validation
        steps = [
            {'method': 'low_rank', 'rank': 64},
            {'method': 'quantization', 'bits': 8},
            {'method': 'pruning', 'sparsity': 0.5}
        ]
        
        for step in steps:
            # Compresse
            compressed = self._apply_step(compressed, step, train_loader)
            
            # Valide
            acc = self._evaluate(compressed, val_loader)
            
            compression_history.append({
                'step': step,
                'accuracy': acc,
                'degradation': baseline['accuracy'] - acc
            })
            
            # Arrête si dégradation trop importante
            if acc < baseline['accuracy'] * 0.95:
                print(f"Stopping: accuracy too low after {step}")
                break
        
        return compressed, compression_history
```

---

## Exercices

### Exercice 12.4.1
Créez un pipeline personnalisé pour votre cas d'usage spécifique.

### Exercice 12.4.2
Implémentez un pipeline adaptatif qui ajuste les hyperparamètres en fonction des résultats intermédiaires.

---

## Points Clés à Retenir

> 📌 **Pipelines automatisent le processus de compression**

> 📌 **Compression progressive permet de trouver le meilleur compromis**

> 📌 **Auto-tuning trouve automatiquement les meilleurs hyperparamètres**

> 📌 **Validation à chaque étape évite la sur-compression**

---

*Section suivante : [12.5 Benchmarking et Évaluation](./12_05_Benchmarking.md)*

