# 12.5 Benchmarking et Évaluation

---

## Introduction

Le **benchmarking** systématique est essentiel pour comparer différentes techniques de compression et évaluer leurs performances. Cette section présente les outils de benchmarking intégrés dans pQuant.

---

## Métriques Standard

### Métriques de Compression

```python
from pquant.benchmarks import CompressionMetrics

class CompressionMetrics:
    """
    Calcule les métriques de compression standard
    """
    
    @staticmethod
    def compute_all(original_model, compressed_model):
        """
        Calcule toutes les métriques
        """
        # Paramètres
        orig_params = sum(p.numel() for p in original_model.parameters())
        comp_params = sum(p.numel() for p in compressed_model.parameters())
        
        # Taille (bytes)
        orig_size = orig_params * 4  # FP32
        comp_size = comp_params * 4  # Approximation
        
        # Compression
        compression_ratio = orig_size / comp_size
        
        # FLOPs (approximation)
        orig_flops = CompressionMetrics._estimate_flops(original_model)
        comp_flops = CompressionMetrics._estimate_flops(compressed_model)
        flops_reduction = orig_flops / comp_flops
        
        return {
            'compression_ratio': compression_ratio,
            'parameter_reduction': 1 - (comp_params / orig_params),
            'size_reduction_mb': (orig_size - comp_size) / (1024**2),
            'flops_reduction': flops_reduction,
            'original_params': orig_params,
            'compressed_params': comp_params
        }
    
    @staticmethod
    def _estimate_flops(model):
        """Estime les FLOPs (simplifié)"""
        # Compte approximatif des opérations
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                # Approximation simplifiée
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                total_flops += 2 * module.in_channels * module.out_channels * kernel_flops
        
        return total_flops
```

---

## Benchmarks Automatisés

```python
from pquant.benchmarks import benchmark_compression

def benchmark_all_methods(model, train_loader, test_loader):
    """
    Benchmark toutes les méthodes de compression
    """
    methods = {
        'baseline': None,
        'low_rank_64': {'method': 'low_rank', 'rank': 64},
        'low_rank_32': {'method': 'low_rank', 'rank': 32},
        'quantization_8': {'method': 'quantization', 'bits': 8},
        'quantization_6': {'method': 'quantization', 'bits': 6},
        'tensor_train': {'method': 'tensor_train', 'rank': 32},
        'combined': {
            'methods': ['low_rank', 'quantization'],
            'low_rank_rank': 64,
            'quantization_bits': 8
        }
    }
    
    results = {}
    
    for name, config in methods.items():
        print(f"Benchmarking: {name}")
        
        if config is None:
            # Baseline
            compressed = model
        else:
            pipeline = CompressionPipeline(config)
            compressed = pipeline.compress(model, train_loader)
        
        # Évalue
        metrics = CompressionMetrics.compute_all(model, compressed)
        accuracy = evaluate_model(compressed, test_loader)
        
        results[name] = {
            **metrics,
            'accuracy': accuracy,
            'degradation': evaluate_model(model, test_loader) - accuracy
        }
    
    return results

# Affichage des résultats
def print_benchmark_results(results):
    """
    Affiche les résultats de benchmark de manière structurée
    """
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Method':<20} | {'Compression':<12} | {'Accuracy':<10} | {'Degradation':<12}")
    print("-"*80)
    
    for name, res in results.items():
        print(f"{name:<20} | {res['compression_ratio']:>10.2f}x | "
              f"{res['accuracy']:>8.2f}% | {res['degradation']:>10.2f}%")
    
    print("="*80)

# Exécution
results = benchmark_all_methods(model, train_loader, test_loader)
print_benchmark_results(results)
```

---

## Benchmark de Latence

```python
from pquant.benchmarks import LatencyBenchmark

class LatencyBenchmark:
    """
    Benchmark de latence d'inférence
    """
    
    @staticmethod
    def measure_latency(model, input_shape, num_warmup=10, num_runs=100):
        """
        Mesure la latence d'inférence
        
        Args:
            model: Modèle à benchmarker
            input_shape: Shape de l'input (sans batch)
            num_warmup: Nombre de runs de warmup
            num_runs: Nombre de runs pour mesure
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronisation GPU si nécessaire
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Mesure
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.time()
                times.append((end - start) * 1000)  # ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p99_ms': np.percentile(times, 99)
        }

# Benchmark latence
latency_original = LatencyBenchmark.measure_latency(model, (784,))
latency_compressed = LatencyBenchmark.measure_latency(compressed, (784,))

print(f"Latence:")
print(f"  Original: {latency_original['mean_ms']:.2f}ms")
print(f"  Compressé: {latency_compressed['mean_ms']:.2f}ms")
print(f"  Accélération: {latency_original['mean_ms'] / latency_compressed['mean_ms']:.2f}x")
```

---

## Benchmark de Mémoire

```python
from pquant.benchmarks import MemoryBenchmark

class MemoryBenchmark:
    """
    Benchmark de consommation mémoire
    """
    
    @staticmethod
    def measure_memory(model, input_shape):
        """
        Mesure la consommation mémoire
        """
        import torch
        
        # Reset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        device = next(model.parameters()).device
        
        # Forward
        dummy_input = torch.randn(1, *input_shape).to(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Mémoire GPU
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            current_memory_mb = torch.cuda.memory_allocated() / (1024**2)
        else:
            peak_memory_mb = 0
            current_memory_mb = 0
        
        # Taille du modèle (paramètres)
        model_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': peak_memory_mb,
            'current_memory_mb': current_memory_mb
        }
```

---

## Rapport de Benchmark Complet

```python
from pquant.benchmarks import generate_benchmark_report

def generate_complete_report(model, compressed, train_loader, test_loader):
    """
    Génère un rapport complet de benchmark
    """
    report = {
        'compression_metrics': CompressionMetrics.compute_all(model, compressed),
        'accuracy': {
            'original': evaluate_model(model, test_loader),
            'compressed': evaluate_model(compressed, test_loader)
        },
        'latency': {
            'original': LatencyBenchmark.measure_latency(model, (784,)),
            'compressed': LatencyBenchmark.measure_latency(compressed, (784,))
        },
        'memory': {
            'original': MemoryBenchmark.measure_memory(model, (784,)),
            'compressed': MemoryBenchmark.measure_memory(compressed, (784,))
        }
    }
    
    # Génère rapport markdown
    report_md = generate_markdown_report(report)
    
    return report, report_md

def generate_markdown_report(report):
    """Génère un rapport en Markdown"""
    md = "# Compression Benchmark Report\n\n"
    
    md += "## Compression Metrics\n\n"
    cm = report['compression_metrics']
    md += f"- Compression Ratio: {cm['compression_ratio']:.2f}x\n"
    md += f"- Parameter Reduction: {cm['parameter_reduction']*100:.1f}%\n"
    md += f"- Size Reduction: {cm['size_reduction_mb']:.2f} MB\n\n"
    
    md += "## Accuracy\n\n"
    acc = report['accuracy']
    md += f"- Original: {acc['original']:.2f}%\n"
    md += f"- Compressed: {acc['compressed']:.2f}%\n"
    md += f"- Degradation: {acc['original'] - acc['compressed']:.2f}%\n\n"
    
    md += "## Latency\n\n"
    lat = report['latency']
    md += f"- Original: {lat['original']['mean_ms']:.2f}ms\n"
    md += f"- Compressed: {lat['compressed']['mean_ms']:.2f}ms\n"
    md += f"- Speedup: {lat['original']['mean_ms'] / lat['compressed']['mean_ms']:.2f}x\n"
    
    return md
```

---

## Exercices

### Exercice 12.5.1
Créez un benchmark personnalisé pour votre application spécifique avec des métriques adaptées.

### Exercice 12.5.2
Implémentez un système de comparaison automatique entre différentes configurations.

---

## Points Clés à Retenir

> 📌 **Métriques standard facilitent la comparaison**

> 📌 **Latence et mémoire sont aussi importants que compression**

> 📌 **Rapports automatisés facilitent l'analyse**

> 📌 **Benchmarks complets incluent accuracy, compression, latence, mémoire**

---

*Section suivante : [12.6 Contribution Open-Source et Bonnes Pratiques](./12_06_Contribution.md)*

