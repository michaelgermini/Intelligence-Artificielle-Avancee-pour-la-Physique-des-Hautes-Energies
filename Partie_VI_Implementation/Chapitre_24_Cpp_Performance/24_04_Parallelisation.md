# 24.4 Parallélisation (OpenMP, TBB)

---

## Introduction

La **parallélisation** est essentielle pour exploiter les architectures multi-cores modernes. **OpenMP** offre une parallélisation simple avec directives, tandis que **Threading Building Blocks (TBB)** de Intel fournit une bibliothèque C++ pour parallélisation plus avancée.

---

## OpenMP

### Installation et Configuration

```cpp
/*
Installation:
- GCC/Clang: -fopenmp flag
- CMake: find_package(OpenMP REQUIRED)
- Compilation: g++ -fopenmp program.cpp

Directives:
- #pragma omp parallel: créer thread pool
- #pragma omp for: paralléliser boucle
- #pragma omp sections: sections parallèles
*/

#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

void openmp_basics() {
    // Nombre de threads
    int num_threads = omp_get_max_threads();
    std::cout << "Max threads: " << num_threads << std::endl;
    
    // Paralléliser boucle
    const int N = 1000000;
    std::vector<double> vec(N);
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        vec[i] = i * 2.0;
    }
    
    // Réduction
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += vec[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
}
```

---

## OpenMP: Sections et Tasks

### Parallélisme Structuré

```cpp
#include <omp.h>
#include <iostream>

void openmp_sections() {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            std::cout << "Section 1, thread " << omp_get_thread_num() << std::endl;
            // Tâche 1
        }
        
        #pragma omp section
        {
            std::cout << "Section 2, thread " << omp_get_thread_num() << std::endl;
            // Tâche 2
        }
        
        #pragma omp section
        {
            std::cout << "Section 3, thread " << omp_get_thread_num() << std::endl;
            // Tâche 3
        }
    }
}

void openmp_tasks() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < 10; ++i) {
                #pragma omp task
                {
                    // Tâche asynchrone
                    int tid = omp_get_thread_num();
                    std::cout << "Task " << i << " on thread " << tid << std::endl;
                }
            }
        }
    }
}
```

---

## OpenMP: Synchronisation

### Critical, Barrier, Atomic

```cpp
#include <omp.h>
#include <iostream>
#include <vector>

void openmp_synchronization() {
    const int N = 1000;
    std::vector<int> shared_data(N, 0);
    int shared_counter = 0;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Critical section
        #pragma omp critical
        {
            shared_counter++;
            std::cout << "Thread " << tid << " incremented counter" << std::endl;
        }
        
        // Barrier: attendre tous threads
        #pragma omp barrier
        
        // Atomic operations
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            #pragma omp atomic
            shared_data[i]++;
        }
    }
    
    std::cout << "Final counter: " << shared_counter << std::endl;
}

// Reduction personnalisée
struct ReductionOp {
    double value;
    
    ReductionOp() : value(0.0) {}
    ReductionOp(double v) : value(v) {}
    
    ReductionOp& operator+=(const ReductionOp& other) {
        value += other.value;
        return *this;
    }
};

void custom_reduction() {
    ReductionOp sum;
    
    #pragma omp declare reduction(plus: ReductionOp: \
        omp_out += omp_in) initializer(omp_priv = ReductionOp())
    
    #pragma omp parallel for reduction(plus:sum)
    for (int i = 0; i < 1000; ++i) {
        sum += ReductionOp(i * 0.1);
    }
    
    std::cout << "Custom reduction sum: " << sum.value << std::endl;
}
```

---

## OpenMP: Performance

### Optimisations

```cpp
#include <omp.h>
#include <vector>

void matrix_multiply_omp(const std::vector<double>& A, int rows_A, int cols_A,
                         const std::vector<double>& B, int rows_B, int cols_B,
                         std::vector<double>& C) {
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cols_A; ++k) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

// Scheduling strategies
void scheduling_examples() {
    const int N = 10000;
    std::vector<double> vec(N);
    
    // Static: découpage fixe
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        vec[i] = i * 2.0;
    }
    
    // Dynamic: découpage dynamique (bon pour charges inégales)
    #pragma omp parallel for schedule(dynamic, 100)
    for (int i = 0; i < N; ++i) {
        vec[i] = vec[i] * 2.0;
    }
    
    // Guided: taille chunks diminue
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; ++i) {
        vec[i] = vec[i] + 1.0;
    }
}
```

---

## Threading Building Blocks (TBB)

### Installation et Utilisation

```cpp
/*
Installation:
- Package: sudo apt-get install libtbb-dev
- CMake: find_package(TBB REQUIRED)
- Include: #include <tbb/tbb.h>
*/

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include <iostream>

void tbb_basics() {
    const int N = 1000000;
    std::vector<double> vec(N);
    
    // Paralléliser boucle
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, N),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                vec[i] = i * 2.0;
            }
        }
    );
    
    // Réduction
    double sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, N),
        0.0,
        [&](const tbb::blocked_range<size_t>& r, double init) {
            double partial_sum = init;
            for (size_t i = r.begin(); i != r.end(); ++i) {
                partial_sum += vec[i];
            }
            return partial_sum;
        },
        std::plus<double>()
    );
    
    std::cout << "TBB sum: " << sum << std::endl;
}
```

---

## TBB: Parallel Algorithms

### Algorithmes Parallèles

```cpp
#include <tbb/parallel_sort.h>
#include <tbb/parallel_for_each.h>
#include <algorithm>
#include <vector>

void tbb_algorithms() {
    std::vector<int> data(1000000);
    
    // Générer données
    std::iota(data.begin(), data.end(), 0);
    std::random_shuffle(data.begin(), data.end());
    
    // Tri parallèle
    tbb::parallel_sort(data.begin(), data.end());
    
    // For each parallèle
    std::vector<int> processed(data.size());
    tbb::parallel_for_each(
        data.begin(), data.end(),
        [&](int& value) {
            value *= 2;
        }
    );
    
    // Transform parallèle
    std::vector<int> result(data.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, data.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            std::transform(
                data.begin() + r.begin(),
                data.begin() + r.end(),
                result.begin() + r.begin(),
                [](int x) { return x * x; }
            );
        }
    );
}
```

---

## TBB: Task-Based Parallelism

### Système de Tasks

```cpp
#include <tbb/task_group.h>
#include <tbb/parallel_invoke.h>

void tbb_tasks() {
    // Task group
    tbb::task_group tg;
    
    tg.run([]() {
        std::cout << "Task 1" << std::endl;
    });
    
    tg.run([]() {
        std::cout << "Task 2" << std::endl;
    });
    
    tg.wait();  // Attendre toutes tâches
    
    // Parallel invoke
    tbb::parallel_invoke(
        []() { std::cout << "Task A" << std::endl; },
        []() { std::cout << "Task B" << std::endl; },
        []() { std::cout << "Task C" << std::endl; }
    );
}
```

---

## Comparaison OpenMP vs TBB

### Trade-offs

```cpp
/*
OpenMP:
+ Simple (directives)
+ Standardisé
+ Supporté largement
+ Bon pour parallélisation régulière
- Moins flexible
- Moins adapté patterns complexes

TBB:
+ Très flexible
+ Task-based (meilleur pour charges irrégulières)
+ Gestion mémoire améliorée
+ Algorithmes parallèles intégrés
- Plus complexe
- Dépendance externe
*/
```

---

## Exercices

### Exercice 24.4.1
Parallélisez une multiplication matricielle avec OpenMP et mesurez speedup.

### Exercice 24.4.2
Implémentez une réduction parallèle personnalisée avec OpenMP.

### Exercice 24.4.3
Utilisez TBB pour paralléliser un tri de vecteur et comparez avec version séquentielle.

### Exercice 24.4.4
Comparez performance OpenMP vs TBB pour un problème avec charges inégales.

---

## Points Clés à Retenir

> 📌 **OpenMP offre parallélisation simple avec directives**

> 📌 **TBB fournit système de tasks flexible et puissant**

> 📌 **Scheduling (static/dynamic) impact performance selon charge**

> 📌 **Synchronisation (critical/atomic) nécessaire pour données partagées**

> 📌 **Réduction évite race conditions automatiquement**

> 📌 **Choisir OpenMP ou TBB selon complexité problème**

---

*Section précédente : [24.3 Algèbre Linéaire](./24_03_Algebre_Lineaire.md) | Section suivante : [24.5 Pybind11](./24_05_Pybind11.md)*

