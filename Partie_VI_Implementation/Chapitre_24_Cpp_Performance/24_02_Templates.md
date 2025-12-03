# 24.2 Templates et Métaprogrammation

---

## Introduction

Les **templates** permettent d'écrire du code générique sans perte de performance. La **métaprogrammation** template (TMP) permet d'exécuter des calculs au moment de la compilation, optimisant ainsi le runtime. Cette section présente les techniques avancées de templates pour le calcul scientifique.

---

## Templates de Base

### Fonctions et Classes Templates

```cpp
#include <iostream>

// Fonction template
template<typename T>
T maximum(T a, T b) {
    return a > b ? a : b;
}

// Spécialisation
template<>
const char* maximum(const char* a, const char* b) {
    return strcmp(a, b) > 0 ? a : b;
}

// Classe template
template<typename T, size_t N>
class FixedArray {
private:
    T data_[N];

public:
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    size_t size() const { return N; }
};

// Exemple
void basic_templates() {
    auto max_int = maximum(5, 3);
    auto max_double = maximum(5.5, 3.3);
    
    FixedArray<int, 10> arr;
    arr[0] = 42;
    std::cout << "Size: " << arr.size() << std::endl;
}
```

---

## Template Specialization

### Spécialisations Totales et Partielles

```cpp
// Template primaire
template<typename T>
class TypeInfo {
public:
    static const char* name() { return "unknown"; }
};

// Spécialisation totale
template<>
class TypeInfo<int> {
public:
    static const char* name() { return "int"; }
};

template<>
class TypeInfo<double> {
public:
    static const char* name() { return "double"; }
};

// Spécialisation partielle
template<typename T>
class TypeInfo<T*> {
public:
    static const char* name() { return "pointer"; }
};

template<typename T, size_t N>
class TypeInfo<T[N]> {
public:
    static const char* name() { return "array"; }
};
```

---

## SFINAE et Type Traits

### Substitution Failure Is Not An Error

```cpp
#include <type_traits>
#include <iostream>

// SFINAE: vérifier si type a méthode size()
template<typename T>
auto has_size_impl(int) -> decltype(std::declval<T>().size(), std::true_type{});

template<typename T>
std::false_type has_size_impl(...);

template<typename T>
struct has_size : decltype(has_size_impl<T>(0)) {};

// Utilisation
template<typename Container>
void process_container(const Container& c) {
    if constexpr (has_size<Container>::value) {
        std::cout << "Size: " << c.size() << std::endl;
    } else {
        std::cout << "No size method" << std::endl;
    }
}

// Type traits utilitaires
template<typename T>
void print_type_info() {
    std::cout << "Is integral: " << std::is_integral_v<T> << std::endl;
    std::cout << "Is floating point: " << std::is_floating_point_v<T> << std::endl;
    std::cout << "Size: " << sizeof(T) << " bytes" << std::endl;
}
```

---

## Constexpr et Compile-Time Computation

### Calculs à la Compilation

```cpp
#include <array>

// Fonction constexpr (C++11)
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

// Variable constexpr
constexpr int fact_10 = factorial(10);  // Calculé à la compilation

// Classe avec méthodes constexpr
template<int N>
class Factorial {
public:
    static constexpr int value = N * Factorial<N - 1>::value;
};

template<>
class Factorial<0> {
public:
    static constexpr int value = 1;
};

// Utilisation avec array size
std::array<int, Factorial<5>::value> arr;  // Array de taille 120

// C++17: if constexpr
template<typename T>
constexpr auto get_size() {
    if constexpr (std::is_integral_v<T>) {
        return sizeof(T);
    } else {
        return 0;
    }
}

// constexpr pour boucles (C++17)
constexpr int sum_squares(int n) {
    int sum = 0;
    for (int i = 1; i <= n; ++i) {
        sum += i * i;
    }
    return sum;
}

constexpr int squares_10 = sum_squares(10);  // Calculé compile-time
```

---

## Template Métaprogrammation

### Calculs au Compile-Time

```cpp
// TMP: Calcul Fibonacci à la compilation
template<int N>
struct Fibonacci {
    static constexpr int value = 
        Fibonacci<N - 1>::value + Fibonacci<N - 2>::value;
};

template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

// Utilisation
constexpr int fib_10 = Fibonacci<10>::value;  // 55

// TMP: Power function
template<int Base, int Exp>
struct Power {
    static constexpr long long value = Base * Power<Base, Exp - 1>::value;
};

template<int Base>
struct Power<Base, 0> {
    static constexpr long long value = 1;
};

constexpr long long pow_2_10 = Power<2, 10>::value;  // 1024
```

---

## CRTP (Curiously Recurring Template Pattern)

### Polymorphisme Statique

```cpp
// CRTP: Base template
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
    
    void common_functionality() {
        // Code commun
    }
};

// Derived classes
class Derived1 : public Base<Derived1> {
public:
    void implementation() {
        std::cout << "Derived1 implementation" << std::endl;
    }
};

class Derived2 : public Base<Derived2> {
public:
    void implementation() {
        std::cout << "Derived2 implementation" << std::endl;
    }
};

// Utilisation (pas de virtual functions, performance meilleure)
template<typename Derived>
void use_base(Base<Derived>& obj) {
    obj.interface();  // Appel statique, pas virtuel
}
```

---

## Expression Templates

### Optimisation d'Expressions

```cpp
// Expression template pour éviter temporaires
template<typename Left, typename Right>
class AddExpr {
private:
    const Left& left_;
    const Right& right_;

public:
    AddExpr(const Left& l, const Right& r) : left_(l), right_(r) {}
    
    double operator[](size_t i) const {
        return left_[i] + right_[i];
    }
};

template<typename T>
class Vector {
private:
    std::vector<T> data_;

public:
    Vector(size_t size) : data_(size) {}
    
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    
    // Template pour expressions
    template<typename Expr>
    Vector& operator=(const Expr& expr) {
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = expr[i];
        }
        return *this;
    }
    
    // Opérateur +
    template<typename Other>
    auto operator+(const Other& other) const {
        return AddExpr<Vector, Other>(*this, other);
    }
};

// Utilisation (évite temporaires)
void expression_template_example() {
    Vector<double> v1(1000), v2(1000), v3(1000);
    
    // Pas de temporaires créés
    v3 = v1 + v2 + v1;  // Évalué en une passe
}
```

---

## Concepts Avancés (C++20)

### Contraintes Expressives

```cpp
#include <concepts>

// Concept pour types numériques
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

// Concept pour matrices
template<typename T>
concept Matrix = requires(T m, size_t i, size_t j) {
    m.rows();
    m.cols();
    m(i, j);
    { m(i, j) } -> std::convertible_to<double>;
};

// Fonction template avec concept
template<Numeric T>
T square(T x) {
    return x * x;
}

// Fonction avec concept Matrix
template<Matrix M>
auto compute_trace(const M& m) {
    auto sum = m(0, 0);
    for (size_t i = 1; i < m.rows(); ++i) {
        sum += m(i, i);
    }
    return sum;
}

// Concept composé
template<typename T>
concept NumericMatrix = Matrix<T> && Numeric<decltype(std::declval<T>()(0, 0))>;
```

---

## Exercices

### Exercice 24.2.1
Implémentez une fonction template qui calcule la norme L2 d'un vecteur pour différents types numériques.

### Exercice 24.2.2
Créez une classe template `Stack` avec spécialisation pour types entiers avec optimisations.

### Exercice 24.2.3
Implémentez un système de type traits pour détecter si un type a un opérateur `operator+`.

### Exercice 24.2.4
Créez un concept (C++20) pour types "Addable" et utilisez-le dans une fonction template.

---

## Points Clés à Retenir

> 📌 **Templates permettent généricité sans overhead runtime**

> 📌 **Spécialisations permettent optimisations spécifiques par type**

> 📌 **SFINAE permet sélection conditionnelle de fonctions**

> 📌 **constexpr permet calculs à la compilation**

> 📌 **TMP permet métaprogrammation au compile-time**

> 📌 **Expression templates optimisent évaluation d'expressions**

> 📌 **Concepts (C++20) rendent contraintes templates plus expressives**

---

*Section précédente : [24.1 C++ Moderne](./24_01_Cpp_Moderne.md) | Section suivante : [24.3 Algèbre Linéaire](./24_03_Algebre_Lineaire.md)*

