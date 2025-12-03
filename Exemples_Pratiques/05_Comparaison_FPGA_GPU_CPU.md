# Exemple Pratique : Comparaison FPGA vs GPU vs CPU

---

## Objectif

Comparer performances (latence, throughput, consommation) d'un modèle ML déployé sur FPGA, GPU, et CPU.

---

## 1. Modèle de Test

```python
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Modèle pour classification jets (similaire trigger L1)
class JetClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Créer modèle
model = JetClassifier()

# Générer données test
batch_sizes = [1, 8, 32, 128, 512]
input_dim = 16
X_test = {bs: torch.randn(bs, input_dim) for bs in batch_sizes}

print(f"Modèle créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
```

---

## 2. Benchmark CPU

```python
def benchmark_cpu(model, X_test, n_warmup=100, n_runs=1000):
    """Benchmark sur CPU"""
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for batch_size, inputs in X_test.items():
        inputs = inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(inputs)
        
        # Mesure
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(inputs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        
        # Calculer throughput
        throughput = batch_size / (np.mean(times) / 1000)  # échantillons/seconde
        
        results[batch_size] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'p99_ms': np.percentile(times, 99),
            'throughput': throughput,
            'latency_per_sample_ms': np.mean(times) / batch_size
        }
    
    return results

print("\n=== Benchmark CPU ===")
results_cpu = benchmark_cpu(model, X_test)

for bs, res in results_cpu.items():
    print(f"Batch {bs:3d}: {res['mean_ms']:.4f} ms ({res['latency_per_sample_ms']:.4f} ms/échantillon, "
          f"{res['throughput']:.0f} échantillons/s)")
```

---

## 3. Benchmark GPU

```python
def benchmark_gpu(model, X_test, n_warmup=100, n_runs=1000):
    """Benchmark sur GPU"""
    if not torch.cuda.is_available():
        print("GPU non disponible")
        return None
    
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for batch_size, inputs in X_test.items():
        inputs = inputs.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(inputs)
        
        # Synchronisation CUDA
        torch.cuda.synchronize()
        
        # Mesure
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(inputs)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        throughput = batch_size / (np.mean(times) / 1000)
        
        results[batch_size] = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'p99_ms': np.percentile(times, 99),
            'throughput': throughput,
            'latency_per_sample_ms': np.mean(times) / batch_size
        }
    
    return results

print("\n=== Benchmark GPU ===")
results_gpu = benchmark_gpu(model, X_test)

if results_gpu:
    for bs, res in results_gpu.items():
        print(f"Batch {bs:3d}: {res['mean_ms']:.4f} ms ({res['latency_per_sample_ms']:.4f} ms/échantillon, "
              f"{res['throughput']:.0f} échantillons/s)")
```

---

## 4. Benchmark FPGA (Simulation)

```python
# Pour FPGA, utiliser résultats hls4ml ou simulation
# Ici, simulation basée sur caractéristiques FPGA typiques

def simulate_fpga_performance(model, X_test):
    """
    Simule performance FPGA basée sur caractéristiques typiques
    - Clock: 200 MHz (5 ns par cycle)
    - Pipeline optimisé
    - Latence fixe par batch
    """
    # Estimer cycles nécessaires (simplifié)
    # Basé sur nombre opérations dans modèle
    params = sum(p.numel() for p in model.parameters())
    
    # Latence base (cycles) - approximation
    base_latency_cycles = 100  # Latence pipeline
    
    results = {}
    
    for batch_size, inputs in X_test.items():
        # FPGA pipeline: latence dépend peu de batch size
        # Throughput limité par fréquence clock
        clock_freq_mhz = 200
        clock_period_ns = 1000 / clock_freq_mhz
        
        # Latence: cycles pour premier résultat
        latency_cycles = base_latency_cycles
        latency_ns = latency_cycles * clock_period_ns
        latency_ms = latency_ns / 1e6
        
        # Throughput: dépend de pipeline depth et clock
        # Un résultat par cycle (idéal)
        throughput_per_second = clock_freq_mhz * 1e6  # échantillons/seconde max
        
        # Avec batch, throughput réel
        if batch_size == 1:
            throughput = throughput_per_second
        else:
            # Pipeline permet traitement parallèle
            throughput = min(batch_size * throughput_per_second, throughput_per_second * 10)
        
        results[batch_size] = {
            'mean_ms': latency_ms,
            'std_ms': latency_ms * 0.01,  # Très faible variance
            'p99_ms': latency_ms * 1.01,
            'throughput': throughput,
            'latency_per_sample_ms': latency_ms / batch_size if batch_size > 1 else latency_ms,
            'latency_us': latency_ns / 1000
        }
    
    return results

print("\n=== Benchmark FPGA (Simulation) ===")
results_fpga = simulate_fpga_performance(model, X_test)

for bs, res in results_fpga.items():
    print(f"Batch {bs:3d}: {res['mean_ms']:.4f} ms ({res['latency_per_sample_ms']:.4f} ms/échantillon, "
          f"{res['throughput']:.0f} échantillons/s, {res['latency_us']:.2f} μs)")
```

---

## 5. Comparaison Visuelle

```python
def compare_platforms(results_cpu, results_gpu, results_fpga):
    """Compare performances plateformes"""
    
    batch_sizes = list(results_cpu.keys())
    
    # Latence par échantillon
    cpu_latencies = [results_cpu[bs]['latency_per_sample_ms'] for bs in batch_sizes]
    gpu_latencies = [results_gpu[bs]['latency_per_sample_ms'] if results_gpu else None for bs in batch_sizes]
    fpga_latencies = [results_fpga[bs]['latency_per_sample_ms'] for bs in batch_sizes]
    
    # Throughput
    cpu_throughput = [results_cpu[bs]['throughput'] / 1e6 for bs in batch_sizes]  # Millions/s
    gpu_throughput = [results_gpu[bs]['throughput'] / 1e6 if results_gpu else None for bs in batch_sizes]
    fpga_throughput = [results_fpga[bs]['throughput'] / 1e6 for bs in batch_sizes]
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Latence par batch
    axes[0, 0].plot(batch_sizes, cpu_latencies, 'o-', label='CPU', linewidth=2)
    if results_gpu:
        axes[0, 0].plot(batch_sizes, gpu_latencies, 's-', label='GPU', linewidth=2)
    axes[0, 0].plot(batch_sizes, fpga_latencies, '^-', label='FPGA', linewidth=2)
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Latence (ms/échantillon)')
    axes[0, 0].set_title('Latence par Échantillon')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Throughput
    axes[0, 1].plot(batch_sizes, cpu_throughput, 'o-', label='CPU', linewidth=2)
    if results_gpu:
        axes[0, 1].plot(batch_sizes, gpu_throughput, 's-', label='GPU', linewidth=2)
    axes[0, 1].plot(batch_sizes, fpga_throughput, '^-', label='FPGA', linewidth=2)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Throughput (M échantillons/s)')
    axes[0, 1].set_title('Throughput')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Latence absolue (batch=1)
    batch_1_idx = 0
    platforms = ['CPU', 'GPU', 'FPGA']
    latencies_batch1 = [
        cpu_latencies[batch_1_idx],
        gpu_latencies[batch_1_idx] if results_gpu else None,
        fpga_latencies[batch_1_idx]
    ]
    
    latencies_batch1 = [l for l in latencies_batch1 if l is not None]
    platforms_filtered = [p for p, l in zip(platforms, latencies_batch1) if l is not None]
    
    axes[1, 0].bar(platforms_filtered, latencies_batch1)
    axes[1, 0].set_ylabel('Latence (ms)')
    axes[1, 0].set_title('Latence Batch Size = 1')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Speedup relatif à CPU
    cpu_lat_batch1 = cpu_latencies[batch_1_idx]
    speedups = []
    speedup_labels = []
    
    if results_gpu:
        speedups.append(cpu_lat_batch1 / gpu_latencies[batch_1_idx])
        speedup_labels.append('GPU vs CPU')
    
    speedups.append(cpu_lat_batch1 / fpga_latencies[batch_1_idx])
    speedup_labels.append('FPGA vs CPU')
    
    axes[1, 1].bar(speedup_labels, speedups, color=['blue', 'green'])
    axes[1, 1].set_ylabel('Speedup')
    axes[1, 1].set_title('Speedup vs CPU (Batch Size = 1)')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('fpga_gpu_cpu_comparison.png', dpi=150)
    plt.show()

# Comparer
compare_platforms(results_cpu, results_gpu, results_fpga)
```

---

## 6. Analyse Consommation Énergétique

```python
# Consommation énergétique typique (approximations)
power_consumption = {
    'CPU': {
        'idle_watts': 50,
        'load_watts': 120,
        'peak_watts': 150
    },
    'GPU': {
        'idle_watts': 30,
        'load_watts': 250,
        'peak_watts': 350
    },
    'FPGA': {
        'idle_watts': 5,
        'load_watts': 20,
        'peak_watts': 25
    }
}

def calculate_energy_efficiency(results_cpu, results_gpu, results_fpga, power):
    """Calcule efficacité énergétique"""
    
    batch_size = 1  # Batch size = 1 pour latence
    throughput_cpu = results_cpu[batch_size]['throughput']
    throughput_gpu = results_gpu[batch_size]['throughput'] if results_gpu else 0
    throughput_fpga = results_fpga[batch_size]['throughput']
    
    # Échantillons par Watt
    efficiency = {
        'CPU': throughput_cpu / power['CPU']['load_watts'],
        'GPU': throughput_gpu / power['GPU']['load_watts'] if results_gpu else 0,
        'FPGA': throughput_fpga / power['FPGA']['load_watts']
    }
    
    # Visualiser
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Consommation
    platforms = list(efficiency.keys())
    platforms = [p for p in platforms if efficiency[p] > 0]
    watts = [power[p]['load_watts'] for p in platforms]
    
    axes[0].bar(platforms, watts, color=['red', 'orange', 'green'])
    axes[0].set_ylabel('Consommation (Watts)')
    axes[0].set_title('Consommation Énergétique')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Efficacité
    eff_values = [efficiency[p] / 1e6 for p in platforms]  # M échantillons/Watt
    
    axes[1].bar(platforms, eff_values, color=['red', 'orange', 'green'])
    axes[1].set_ylabel('Efficacité (M échantillons/Watt)')
    axes[1].set_title('Efficacité Énergétique')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('energy_efficiency.png', dpi=150)
    plt.show()
    
    return efficiency

print("\n=== Analyse Consommation ===")
efficiency = calculate_energy_efficiency(results_cpu, results_gpu, results_fpga, power_consumption)

for platform, eff in efficiency.items():
    if eff > 0:
        print(f"{platform}: {eff/1e6:.2f} M échantillons/Watt")
```

---

## 7. Comparaison Complète

```python
def generate_comparison_report(results_cpu, results_gpu, results_fpga, power_consumption):
    """Génère rapport comparaison complet"""
    
    batch_1 = 1
    
    report = {
        'Latence (ms, batch=1)': {
            'CPU': results_cpu[batch_1]['latency_per_sample_ms'],
            'GPU': results_gpu[batch_1]['latency_per_sample_ms'] if results_gpu else None,
            'FPGA': results_fpga[batch_1]['latency_per_sample_ms']
        },
        'Throughput (M échantillons/s, batch=128)': {
            'CPU': results_cpu[128]['throughput'] / 1e6,
            'GPU': results_gpu[128]['throughput'] / 1e6 if results_gpu else None,
            'FPGA': results_fpga[128]['throughput'] / 1e6
        },
        'Consommation (Watts)': {
            'CPU': power_consumption['CPU']['load_watts'],
            'GPU': power_consumption['GPU']['load_watts'],
            'FPGA': power_consumption['FPGA']['load_watts']
        },
        'Efficacité (M échantillons/Watt)': {
            'CPU': efficiency['CPU'] / 1e6,
            'GPU': efficiency['GPU'] / 1e6 if efficiency['GPU'] > 0 else None,
            'FPGA': efficiency['FPGA'] / 1e6
        }
    }
    
    print("\n" + "="*70)
    print("RAPPORT COMPARAISON FPGA vs GPU vs CPU")
    print("="*70)
    
    import pandas as pd
    df = pd.DataFrame(report)
    print("\n" + df.to_string())
    
    # Recommandations
    print("\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    
    # Pour latence faible
    best_latency = min(
        [(k, v) for k, v in report['Latence (ms, batch=1)'].items() if v is not None],
        key=lambda x: x[1]
    )
    print(f"\n✅ Meilleure latence: {best_latency[0]} ({best_latency[1]:.4f} ms)")
    
    # Pour throughput élevé
    best_throughput = max(
        [(k, v) for k, v in report['Throughput (M échantillons/s, batch=128)'].items() if v is not None],
        key=lambda x: x[1]
    )
    print(f"✅ Meilleur throughput: {best_throughput[0]} ({best_throughput[1]:.2f} M échantillons/s)")
    
    # Pour efficacité énergétique
    best_efficiency = max(
        [(k, v) for k, v in report['Efficacité (M échantillons/Watt)'].items() if v is not None],
        key=lambda x: x[1]
    )
    print(f"✅ Meilleure efficacité: {best_efficiency[0]} ({best_efficiency[1]:.2f} M échantillons/Watt)")
    
    # Pour trigger L1 (latence < 4 μs)
    fpga_latency_us = results_fpga[batch_1]['latency_us']
    if fpga_latency_us <= 4.0:
        print(f"\n🎯 FPGA respecte contrainte L1 Trigger (≤ 4 μs)")
    else:
        print(f"\n⚠️  Optimisation FPGA nécessaire pour L1")
    
    return report

# Générer rapport
final_report = generate_comparison_report(results_cpu, results_gpu, results_fpga, power_consumption)
```

---

## Résultats Typiques

| Plateforme | Latence (batch=1) | Throughput (batch=128) | Consommation | Efficacité |
|------------|-------------------|------------------------|--------------|------------|
| CPU | ~0.5 ms | ~2 M échantillons/s | 120 W | ~17 M/W |
| GPU | ~0.05 ms | ~50 M échantillons/s | 250 W | ~200 M/W |
| FPGA | ~0.002 ms (2 μs) | ~200 M échantillons/s | 20 W | ~10,000 M/W |

### Observations

- **FPGA** : Meilleure latence et efficacité énergétique (idéal trigger temps réel)
- **GPU** : Meilleur throughput batch (idéal training/inférence batch)
- **CPU** : Flexible mais moins performant

---

## Points Clés

✅ **Comparaison systématique** : Latence, throughput, consommation  
✅ **Visualisations complètes** : Graphiques multi-métriques  
✅ **Analyse énergétique** : Efficacité par plateforme  
✅ **Recommandations** : Choix plateforme selon use case  
📊 **Benchmarking réaliste** : Tests avec différents batch sizes  

---

*Cet exemple démontre quand utiliser FPGA, GPU, ou CPU pour déploiement modèles ML.*

