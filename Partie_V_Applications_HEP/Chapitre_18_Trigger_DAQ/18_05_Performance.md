# 18.5 Latence et Requirements de Performance

---

## Introduction

Les **requirements de performance** des systèmes de trigger sont extrêmement stricts et constituent l'un des défis majeurs de la physique des hautes énergies. La latence, le throughput, l'efficacité, et la pureté doivent être optimisés simultanément sous des contraintes hardware et temporelles sévères.

Cette section détaille les métriques de performance, les techniques d'optimisation, et les méthodes de benchmark pour évaluer et améliorer les performances des systèmes de trigger.

---

## Métriques de Performance

### Définitions et Mesures

```python
import numpy as np
import time
from typing import Dict, List, Tuple

class TriggerPerformanceMetrics:
    """
    Métriques de performance pour systèmes de trigger
    """
    
    def __init__(self):
        self.metrics_definitions = {
            'latency': {
                'description': 'Temps entre input et décision',
                'l1_target_us': 4.0,
                'hlt_target_ms': 300.0,
                'measurement': 'End-to-end time'
            },
            'throughput': {
                'description': 'Taux d\'événements traités',
                'l1_target_hz': 40e6,
                'hlt_target_hz': 100e3,
                'measurement': 'Events per second'
            },
            'efficiency': {
                'description': 'Fraction de signal conservé',
                'target': 0.95,
                'measurement': 'True positive rate'
            },
            'purity': {
                'description': 'Fraction de signal dans événements acceptés',
                'target': 0.80,
                'measurement': 'Signal / (Signal + Background)'
            },
            'rate': {
                'description': 'Taux de déclenchement',
                'l1_target_khz': 100,
                'hlt_target_hz': 1000,
                'measurement': 'Output rate'
            }
        }
    
    def compute_efficiency(self, true_labels: np.ndarray, 
                          decisions: np.ndarray,
                          signal_class: int = 1) -> Dict:
        """
        Calcule l'efficacité de sélection
        
        Args:
            true_labels: Labels réels (0=background, 1=signal)
            decisions: Décisions trigger (0=reject, 1=accept)
            signal_class: Classe considérée comme signal
        """
        signal_mask = true_labels == signal_class
        
        # Efficacité signal: fraction de signal accepté
        signal_decisions = decisions[signal_mask]
        signal_efficiency = signal_decisions.mean()
        
        # Efficacité par pT (si disponible)
        return {
            'signal_efficiency': signal_efficiency,
            'n_signal_total': signal_mask.sum(),
            'n_signal_accepted': signal_decisions.sum()
        }
    
    def compute_purity(self, true_labels: np.ndarray,
                      decisions: np.ndarray,
                      signal_class: int = 1) -> Dict:
        """
        Calcule la pureté (fraction de signal dans événements acceptés)
        """
        accepted_mask = decisions == 1
        accepted_labels = true_labels[accepted_mask]
        
        if len(accepted_labels) == 0:
            return {'purity': 0.0, 'n_accepted': 0, 'n_signal_accepted': 0}
        
        n_signal_accepted = (accepted_labels == signal_class).sum()
        purity = n_signal_accepted / len(accepted_labels)
        
        return {
            'purity': purity,
            'n_accepted': len(accepted_labels),
            'n_signal_accepted': n_signal_accepted
        }
    
    def compute_rate(self, decisions: np.ndarray, 
                    input_rate_hz: float) -> Dict:
        """
        Calcule le taux de déclenchement
        """
        acceptance_rate = decisions.mean()
        output_rate_hz = input_rate_hz * acceptance_rate
        
        return {
            'acceptance_rate': acceptance_rate,
            'input_rate_hz': input_rate_hz,
            'output_rate_hz': output_rate_hz,
            'reduction_factor': 1.0 / acceptance_rate if acceptance_rate > 0 else float('inf')
        }
    
    def compute_roc_curve(self, scores: np.ndarray,
                         true_labels: np.ndarray,
                         signal_class: int = 1,
                         n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la courbe ROC (efficacité signal vs efficacité background)
        """
        signal_mask = true_labels == signal_class
        background_mask = ~signal_mask
        
        signal_scores = scores[signal_mask]
        background_scores = scores[background_mask]
        
        thresholds = np.linspace(scores.min(), scores.max(), n_points)
        
        signal_efficiencies = []
        background_efficiencies = []
        
        for threshold in thresholds:
            signal_passed = (signal_scores > threshold).mean()
            background_passed = (background_scores > threshold).mean()
            
            signal_efficiencies.append(signal_passed)
            background_efficiencies.append(background_passed)
        
        return np.array(signal_efficiencies), np.array(background_efficiencies), thresholds
    
    def display_metrics(self):
        """Affiche les définitions de métriques"""
        print("\n" + "="*70)
        print("Métriques de Performance Trigger")
        print("="*70)
        
        for metric, info in self.metrics_definitions.items():
            print(f"\n{metric.upper()}:")
            print(f"  Description: {info['description']}")
            if 'l1_target' in info:
                print(f"  L1 Target: {info['l1_target_us']} μs" if 'us' in info else 
                      f"  L1 Target: {info['l1_target_hz']/1e6:.1f} MHz" if 'hz' in info else
                      f"  L1 Target: {info['l1_target_khz']} kHz")
            if 'hlt_target' in info:
                print(f"  HLT Target: {info['hlt_target_ms']} ms" if 'ms' in info else
                      f"  HLT Target: {info['hlt_target_hz']/1e3:.1f} kHz")
            if 'target' in info:
                print(f"  Target: {info['target']}")
            print(f"  Measurement: {info['measurement']}")

metrics = TriggerPerformanceMetrics()
metrics.display_metrics()
```

---

## Mesure de Latence

### Benchmarks et Profiling

```python
class LatencyMeasurement:
    """
    Mesure et analyse de latence
    """
    
    def __init__(self):
        pass
    
    def measure_l1_latency(self, model, input_data, n_iterations=1000):
        """
        Mesure latence L1 (nanosecondes)
        """
        # Warm-up
        for _ in range(10):
            _ = model(input_data)
        
        # Mesures
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter_ns()
            _ = model(input_data)
            end = time.perf_counter_ns()
            latencies.append(end - start)
        
        latencies = np.array(latencies)
        
        return {
            'mean_ns': latencies.mean(),
            'median_ns': np.median(latencies),
            'min_ns': latencies.min(),
            'max_ns': latencies.max(),
            'std_ns': latencies.std(),
            'p99_ns': np.percentile(latencies, 99),
            'p99_9_ns': np.percentile(latencies, 99.9)
        }
    
    def measure_hlt_latency(self, processing_pipeline, event_data, n_iterations=100):
        """
        Mesure latence HLT (millisecondes)
        """
        latencies = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            result = processing_pipeline(event_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        return {
            'mean_ms': latencies.mean(),
            'median_ms': np.median(latencies),
            'min_ms': latencies.min(),
            'max_ms': latencies.max(),
            'std_ms': latencies.std(),
            'p99_ms': np.percentile(latencies, 99)
        }
    
    def profile_pipeline_stages(self, pipeline, event_data):
        """
        Profile chaque stage du pipeline
        """
        stage_times = {}
        
        # En pratique: utiliser profiling tools (cProfile, py-spy, etc.)
        # Ici: simulation
        
        stages = ['event_building', 'track_reco', 'calo_reco', 
                 'ml_inference', 'decision']
        
        for stage in stages:
            # Mesurer temps stage
            start = time.perf_counter()
            # pipeline.execute_stage(stage, event_data)
            end = time.perf_counter()
            stage_times[stage] = (end - start) * 1000  # ms
        
        return stage_times
    
    def analyze_latency_bottlenecks(self, stage_times: Dict):
        """Identifie les bottlenecks de latence"""
        total_time = sum(stage_times.values())
        
        bottlenecks = []
        for stage, time_ms in stage_times.items():
            fraction = time_ms / total_time
            if fraction > 0.2:  # Plus de 20% du temps
                bottlenecks.append({
                    'stage': stage,
                    'time_ms': time_ms,
                    'fraction': fraction
                })
        
        return sorted(bottlenecks, key=lambda x: x['time_ms'], reverse=True)

latency_meas = LatencyMeasurement()
```

---

## Optimisation de Performance

### Techniques d'Optimisation

```python
class PerformanceOptimization:
    """
    Techniques d'optimisation de performance
    """
    
    @staticmethod
    def optimize_l1_latency(model, target_latency_ns=100):
        """
        Optimise latence L1 pour respecter budget
        """
        optimizations = {
            'quantization': 'Réduire précision (8-bit → 4-bit)',
            'pruning': 'Réduire nombre de paramètres',
            'architecture_reduction': 'Réduire taille couches',
            'pipeline_optimization': 'Améliorer pipeline FPGA',
            'parallelization': 'Paralléliser calculs'
        }
        
        return optimizations
    
    @staticmethod
    def optimize_hlt_throughput(pipeline, target_throughput_hz=100e3):
        """
        Optimise throughput HLT
        """
        strategies = {
            'batch_processing': 'Traiter événements par batch',
            'parallel_execution': 'Exécuter chemins en parallèle',
            'early_stopping': 'Rejeter événements tôt',
            'model_caching': 'Cache modèles en mémoire',
            'async_processing': 'Traitement asynchrone',
            'load_balancing': 'Équilibrer charge entre nœuds'
        }
        
        return strategies
    
    @staticmethod
    def optimize_efficiency_purity_balance(model, val_data, val_labels,
                                          target_rate_hz: float):
        """
        Optimise compromis efficacité/pureté pour taux cible
        """
        # Chercher seuil optimal
        with torch.no_grad():
            scores = model(torch.tensor(val_data, dtype=torch.float32))
            if scores.dim() > 1:
                scores = scores[:, 1]  # Probabilité signal
            scores = scores.numpy()
        
        # Chercher seuil qui donne taux cible
        input_rate = len(val_data)  # Approximation
        target_acceptance = target_rate_hz / input_rate
        
        sorted_scores = np.sort(scores)[::-1]
        threshold_idx = int(target_acceptance * len(scores))
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(scores) else sorted_scores[-1]
        
        # Calculer métriques avec ce seuil
        decisions = scores > threshold
        
        metrics = TriggerPerformanceMetrics()
        efficiency = metrics.compute_efficiency(val_labels, decisions)
        purity = metrics.compute_purity(val_labels, decisions)
        rate = metrics.compute_rate(decisions, input_rate)
        
        return {
            'threshold': threshold,
            'efficiency': efficiency['signal_efficiency'],
            'purity': purity['purity'],
            'rate_hz': rate['output_rate_hz']
        }
    
    @staticmethod
    def optimize_trigger_menu(menu, target_total_rate_hz: float):
        """
        Optimise menu de trigger pour taux total cible
        """
        # Stratégie: ajuster seuils et prescales
        
        optimization = {
            'strategy': 'Adjust thresholds and prescales',
            'current_rate_hz': 0,
            'target_rate_hz': target_total_rate_hz,
            'adjustments': {}
        }
        
        return optimization

opt = PerformanceOptimization()
```

---

## Benchmarks et Validation

### Tests de Performance

```python
class TriggerBenchmarking:
    """
    Système de benchmark pour triggers
    """
    
    def __init__(self):
        self.benchmark_datasets = {
            'signal': {
                'description': 'Événements signal (ex: Higgs)',
                'size': 10000,
                'distribution': 'Simulated signal events'
            },
            'background': {
                'description': 'Événements background (QCD, etc.)',
                'size': 1000000,
                'distribution': 'Simulated background events'
            }
        }
    
    def run_full_benchmark(self, trigger_system, test_events: Dict):
        """
        Exécute benchmark complet
        """
        results = {
            'latency': {},
            'throughput': {},
            'efficiency': {},
            'purity': {},
            'rate': {}
        }
        
        # Latence
        latency_meas = LatencyMeasurement()
        results['latency'] = latency_meas.measure_hlt_latency(
            trigger_system, test_events['signal'], n_iterations=100
        )
        
        # Throughput
        start = time.perf_counter()
        n_processed = 0
        duration_s = 10  # 10 secondes de test
        
        end_time = start + duration_s
        while time.perf_counter() < end_time:
            trigger_system(test_events['signal'][n_processed % len(test_events['signal'])])
            n_processed += 1
        
        elapsed = time.perf_counter() - start
        results['throughput'] = {
            'events_per_sec': n_processed / elapsed,
            'duration_s': elapsed
        }
        
        # Efficacité et pureté
        all_events = np.concatenate([test_events['signal'], test_events['background']])
        all_labels = np.concatenate([
            np.ones(len(test_events['signal'])),
            np.zeros(len(test_events['background']))
        ])
        
        decisions = np.array([trigger_system(e) for e in all_events])
        
        metrics = TriggerPerformanceMetrics()
        results['efficiency'] = metrics.compute_efficiency(all_labels, decisions)
        results['purity'] = metrics.compute_purity(all_labels, decisions)
        results['rate'] = metrics.compute_rate(decisions, len(all_events))
        
        return results
    
    def compare_trigger_versions(self, version1, version2, test_events):
        """
        Compare deux versions de trigger
        """
        results_v1 = self.run_full_benchmark(version1, test_events)
        results_v2 = self.run_full_benchmark(version2, test_events)
        
        comparison = {
            'latency_improvement': (results_v1['latency']['mean_ms'] - 
                                  results_v2['latency']['mean_ms']) / results_v1['latency']['mean_ms'],
            'efficiency_change': results_v2['efficiency']['signal_efficiency'] - results_v1['efficiency']['signal_efficiency'],
            'purity_change': results_v2['purity']['purity'] - results_v1['purity']['purity'],
            'throughput_improvement': (results_v2['throughput']['events_per_sec'] - 
                                     results_v1['throughput']['events_per_sec']) / results_v1['throughput']['events_per_sec']
        }
        
        return comparison

benchmark = TriggerBenchmarking()
```

---

## Monitoring en Production

### Surveillance Continue

```python
class ProductionMonitoring:
    """
    Monitoring de performance en production
    """
    
    def __init__(self):
        self.metrics_history = {
            'latency': [],
            'throughput': [],
            'efficiency': [],
            'rate': []
        }
    
    def monitor_live_performance(self, trigger_system, 
                                event_stream, 
                                sampling_rate: float = 0.01):
        """
        Surveille performance en temps réel
        
        Args:
            sampling_rate: Fraction d'événements à monitorer (pour réduire overhead)
        """
        monitored_events = []
        monitored_latencies = []
        
        for i, event in enumerate(event_stream):
            if np.random.random() < sampling_rate:
                start = time.perf_counter()
                decision = trigger_system(event)
                end = time.perf_counter()
                
                monitored_events.append(decision)
                monitored_latencies.append((end - start) * 1000)
        
        return {
            'avg_latency_ms': np.mean(monitored_latencies),
            'throughput_estimate': len(monitored_events) / sampling_rate,
            'acceptance_rate': np.mean(monitored_events)
        }
    
    def detect_performance_degradation(self, current_metrics: Dict,
                                      baseline_metrics: Dict,
                                      thresholds: Dict):
        """
        Détecte dégradation de performance
        """
        alerts = []
        
        if current_metrics['latency'] > baseline_metrics['latency'] * thresholds.get('latency_multiplier', 1.5):
            alerts.append('Latency degraded')
        
        if current_metrics['efficiency'] < baseline_metrics['efficiency'] * thresholds.get('efficiency_threshold', 0.95):
            alerts.append('Efficiency degraded')
        
        if current_metrics['rate'] > baseline_metrics['rate'] * thresholds.get('rate_multiplier', 1.2):
            alerts.append('Rate too high')
        
        return alerts

monitoring = ProductionMonitoring()
```

---

## Exercices

### Exercice 18.5.1
Mesurez la latence d'un modèle L1 et optimisez-le pour respecter un budget de 80 ns.

### Exercice 18.5.2
Créez un système de benchmark qui compare l'efficacité et la pureté de deux menus de trigger différents.

### Exercice 18.5.3
Développez un système de monitoring qui détecte automatiquement les dégradations de performance en production.

### Exercice 18.5.4
Optimisez un menu de trigger pour maximiser l'efficacité signal tout en respectant un budget de taux strict.

---

## Points Clés à Retenir

> 📌 **La latence L1 doit être < 4 μs, HLT < 300 ms**

> 📌 **Le throughput L1 est de 40 MHz, HLT de 100 kHz**

> 📌 **L'efficacité signal et la pureté doivent être optimisées simultanément**

> 📌 **Le taux de déclenchement doit respecter les budgets de bande passante**

> 📌 **Le profiling identifie les bottlenecks pour optimisation ciblée**

> 📌 **Le monitoring en production est essentiel pour détecter les problèmes**

---

*Section précédente : [18.4 Intégration de l'IA](./18_04_IA_Trigger.md)*

