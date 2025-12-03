# 24.1 C++ Moderne (C++17/20)

---

## Introduction

Les standards **C++17** et **C++20** introduisent de nombreuses fonctionnalités qui simplifient l'écriture de code tout en maintenant la performance. Cette section présente les fonctionnalités modernes les plus utiles pour le calcul scientifique et le deep learning.

---

## Auto et Déduction de Type

### Simplification du Code

```cpp
#include <vector>
#include <type_traits>
#include <iostream>

// C++11+: auto pour déduction automatique
auto x = 42;                    // int
auto y = 3.14;                  // double
auto vec = std::vector<int>{1, 2, 3};  // std::vector<int>

// C++14+: auto pour fonctions lambda
auto add = [](auto a, auto b) { return a + b; };
auto result = add(5, 3.14);     // double

// C++17: if constexpr (compile-time condition)
template<typename T>
auto process_value(T value) {
    if constexpr (std::is_integral_v<T>) {
        return value * 2;
    } else {
        return value * 2.0;
    }
}

// Utilisation
int int_val = process_value(5);      // 10
double double_val = process_value(3.14);  // 6.28
```

---

## Structured Bindings (C++17)

### Décomposition de Structures

```cpp
#include <tuple>
#include <map>
#include <iostream>

// Structured bindings pour tuples
auto get_values() {
    return std::make_tuple(1, 2.5, "hello");
}

void example_structured_bindings() {
    // C++17: Décomposition automatique
    auto [x, y, z] = get_values();
    std::cout << x << ", " << y << ", " << z << std::endl;
    
    // Avec références
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}};
    for (const auto& [key, value] : m) {
        std::cout << key << ": " << value << std::endl;
    }
    
    // Pour paires
    std::pair<int, double> p{10, 20.5};
    auto [first, second] = p;
}
```

---

## Smart Pointers

### Gestion Automatique de Mémoire

```cpp
#include <memory>
#include <vector>

// unique_ptr: propriété unique
std::unique_ptr<double[]> allocate_array(size_t size) {
    return std::make_unique<double[]>(size);
}

// shared_ptr: propriété partagée
std::shared_ptr<std::vector<int>> create_shared_vector() {
    return std::make_shared<std::vector<int>>();
}

// Exemple pour matrices
class Matrix {
private:
    size_t rows_, cols_;
    std::unique_ptr<double[]> data_;

public:
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), 
          data_(std::make_unique<double[]>(rows * cols)) {}
    
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }
    
    const double& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }
    
    // Move semantics (C++11)
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    
    // Delete copy (évite copies coûteuses)
    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
};
```

---

## Lambda Expressions Avancées

### Captures et Generic Lambdas

```cpp
#include <algorithm>
#include <vector>
#include <numeric>

void lambda_examples() {
    std::vector<int> vec{1, 2, 3, 4, 5};
    
    // Lambda avec capture
    int multiplier = 2;
    std::transform(vec.begin(), vec.end(), vec.begin(),
        [multiplier](int x) { return x * multiplier; });
    
    // Lambda générique (C++14)
    auto generic_add = [](auto a, auto b) { return a + b; };
    auto sum_int = generic_add(5, 3);
    auto sum_double = generic_add(5.5, 3.3);
    
    // Capture par référence
    int sum = 0;
    std::for_each(vec.begin(), vec.end(),
        [&sum](int x) { sum += x; });
    
    // Lambda récursive (C++14)
    auto factorial = [](auto f, int n) {
        return n <= 1 ? 1 : n * f(f, n - 1);
    };
    int fact_5 = factorial(factorial, 5);
}
```

---

## Variadic Templates

### Fonctions à Nombre Variable d'Arguments

```cpp
#include <iostream>

// Template variadic (C++11)
template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...);  // Fold expression (C++17)
    std::cout << std::endl;
}

// Exemple: fonction max avec arguments variables
template<typename T>
T max_value(T value) {
    return value;
}

template<typename T, typename... Args>
T max_value(T first, Args... args) {
    T rest_max = max_value(args...);
    return first > rest_max ? first : rest_max;
}

// Ou avec fold expression (C++17)
template<typename... Args>
auto max_fold(Args... args) {
    return std::max({args...});
}

void example_variadic() {
    print_all(1, 2.5, "hello", 'c');
    int max = max_value(5, 3, 9, 2, 7);
    std::cout << "Max: " << max << std::endl;
}
```

---

## Concepts (C++20)

### Contraintes sur Templates

```cpp
// C++20: Concepts pour contraindre templates
#include <concepts>
#include <type_traits>

// Concept simple
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Fonction avec concept
template<Numeric T>
T multiply(T a, T b) {
    return a * b;
}

// Concept plus complexe
template<typename T>
concept MatrixLike = requires(T m) {
    m.rows();
    m.cols();
    m(0, 0);
};

// Utilisation
template<MatrixLike Matrix>
auto compute_trace(const Matrix& m) {
    auto sum = m(0, 0);
    for (size_t i = 1; i < m.rows(); ++i) {
        sum += m(i, i);
    }
    return sum;
}
```

---

## Ranges (C++20)

### Algorithmes Modernes

```cpp
#include <ranges>
#include <vector>
#include <algorithm>
#include <iostream>

void ranges_example() {
    std::vector<int> vec{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // C++20: Ranges avec pipe operators
    auto result = vec 
        | std::views::filter([](int x) { return x % 2 == 0; })
        | std::views::transform([](int x) { return x * x; })
        | std::views::take(3);
    
    // Convertir en vector si nécessaire
    std::vector<int> squares;
    std::ranges::copy(result, std::back_inserter(squares));
    
    for (auto val : squares) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
```

---

## Exemple: Classe Matrix Moderne

### Implémentation Complète

```cpp
#include <memory>
#include <iostream>
#include <algorithm>
#include <stdexcept>

template<typename T>
class ModernMatrix {
private:
    size_t rows_, cols_;
    std::unique_ptr<T[]> data_;

public:
    // Constructeur
    ModernMatrix(size_t rows, size_t cols, T init_val = T{})
        : rows_(rows), cols_(cols),
          data_(std::make_unique<T[]>(rows * cols)) {
        std::fill(data_.get(), data_.get() + rows * cols, init_val);
    }
    
    // Accès éléments
    T& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }
    
    const T& operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }
    
    // Dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    // Multiplication matricielle
    template<typename U>
    auto operator*(const ModernMatrix<U>& other) const {
        if (cols_ != other.rows_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        ModernMatrix<decltype(T{} * U{})> result(rows_, other.cols_);
        
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                T sum{};
                for (size_t k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        
        return result;
    }
    
    // Addition
    ModernMatrix operator+(const ModernMatrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw std::invalid_argument("Matrix dimensions mismatch");
        }
        
        ModernMatrix result(rows_, cols_);
        for (size_t i = 0; i < rows_ * cols_; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
};

// Exemple utilisation
void matrix_example() {
    ModernMatrix<double> A(3, 3, 1.0);
    ModernMatrix<double> B(3, 2, 2.0);
    
    auto C = A * B;
    
    std::cout << "Result matrix: " << C.rows() << "x" << C.cols() << std::endl;
}
```

---

## Exercices

### Exercice 24.1.1
Créez une classe `Vector` moderne avec smart pointers et opérateurs surchargés.

### Exercice 24.1.2
Implémentez une fonction template variadic qui calcule la moyenne de ses arguments.

### Exercice 24.1.3
Utilisez structured bindings pour décomposer le retour d'une fonction qui renvoie un tuple.

### Exercice 24.1.4
Créez un concept (C++20) pour types numériques et utilisez-le dans une fonction template.

---

## Points Clés à Retenir

> 📌 **auto simplifie déclarations de types**

> 📌 **Structured bindings simplifient décomposition structures**

> 📌 **Smart pointers gèrent mémoire automatiquement**

> 📌 **Lambdas permettent code fonctionnel moderne**

> 📌 **Variadic templates supportent fonctions flexibles**

> 📌 **Concepts (C++20) contraignent templates de manière expressive**

> 📌 **Ranges (C++20) simplifient algorithmes sur conteneurs**

---

*Section précédente : [24.0 Introduction](./24_introduction.md) | Section suivante : [24.2 Templates](./24_02_Templates.md)*

