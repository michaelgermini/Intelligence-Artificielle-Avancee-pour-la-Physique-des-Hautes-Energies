# 24.3 Bibliothèques d'Algèbre Linéaire (Eigen, BLAS)

---

## Introduction

Les bibliothèques **Eigen** et **BLAS/LAPACK** sont essentielles pour le calcul scientifique haute performance. **Eigen** fournit une interface C++ moderne et expressive, tandis que **BLAS** offre des routines optimisées au niveau assembleur.

---

## Eigen

### Installation et Configuration

```cpp
/*
Installation:
- Via package manager: sudo apt-get install libeigen3-dev
- Via CMake: find_package(Eigen3 REQUIRED)
- Inclusion: #include <Eigen/Dense>
*/

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

void eigen_basics() {
    // Vecteurs
    VectorXd v(5);  // Vector double, taille dynamique
    v << 1, 2, 3, 4, 5;
    
    Vector3d v3(1, 2, 3);  // Vector double, taille fixe 3
    
    // Matrices
    MatrixXd m(3, 3);  // Matrix double, taille dynamique
    m << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    
    Matrix3d m3;  // Matrix double, taille fixe 3x3
    m3 << 1, 2, 3,
          4, 5, 6,
          7, 8, 9;
    
    // Accès éléments
    double element = m(0, 1);  // 2
    m(0, 1) = 10;
    
    // Dimensions
    std::cout << "Rows: " << m.rows() << ", Cols: " << m.cols() << std::endl;
}
```

---

## Opérations Eigen

### Algèbre Linéaire

```cpp
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

void eigen_operations() {
    Matrix3d A, B;
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
    B << 9, 8, 7,
         6, 5, 4,
         3, 2, 1;
    
    // Opérations élément par élément
    Matrix3d C = A + B;
    Matrix3d D = A * B;  // Multiplication matricielle
    
    // Multiplication scalaire
    Matrix3d E = 2.0 * A;
    
    // Transposition
    Matrix3d A_T = A.transpose();
    
    // Inverse
    Matrix3d A_inv = A.inverse();
    
    // Déterminant
    double det = A.determinant();
    
    // Trace
    double trace = A.trace();
    
    // Norme
    double norm = A.norm();  // Frobenius norm
    
    // Résolution système linéaire: Ax = b
    Vector3d b(1, 2, 3);
    Vector3d x = A.colPivHouseholderQr().solve(b);
    
    // Décomposition LU
    PartialPivLU<Matrix3d> lu(A);
    Matrix3d L = lu.matrixLU().triangularView<Lower>();
    Matrix3d U = lu.matrixLU().triangularView<Upper>();
    
    // SVD
    JacobiSVD<Matrix3d> svd(A, ComputeFullU | ComputeFullV);
    Vector3d singular_values = svd.singularValues();
    Matrix3d U_svd = svd.matrixU();
    Matrix3d V_svd = svd.matrixV();
    
    // Valeurs propres
    EigenSolver<Matrix3d> es(A);
    Vector3cd eigenvalues = es.eigenvalues();
    Matrix3cd eigenvectors = es.eigenvectors();
}
```

---

## Eigen: Types et Optimisations

### Fixed vs Dynamic

```cpp
#include <Eigen/Dense>

void eigen_types() {
    // Fixed size: connu à la compilation, plus rapide
    Matrix3d fixed_matrix;
    Vector4d fixed_vector;
    
    // Dynamic size: taille connue runtime, plus flexible
    MatrixXd dynamic_matrix(10, 10);
    VectorXd dynamic_vector(100);
    
    // Expression templates: pas d'évaluation immédiate
    MatrixXd A(100, 100), B(100, 100), C(100, 100);
    
    // Pas de temporaires créés
    C = A + B + A;  // Évalué en une passe
    
    // Lazy evaluation
    MatrixXd D = (A + B).transpose();  // Pas d'évaluation intermédiaire
    
    // Bloc operations
    MatrixXd large(100, 100);
    
    // Extraire bloc
    MatrixXd block = large.block(10, 10, 20, 20);
    
    // Assigner bloc
    large.block(0, 0, 10, 10) = MatrixXd::Identity(10, 10);
    
    // Colonnes/lignes
    VectorXd col = large.col(0);
    VectorXd row = large.row(0);
    
    // Diagonale
    VectorXd diag = large.diagonal();
}
```

---

## BLAS (Basic Linear Algebra Subprograms)

### Interface C

```cpp
/*
BLAS levels:
- Level 1: Vector operations (saxpy, dot product)
- Level 2: Matrix-vector operations (gemv)
- Level 3: Matrix-matrix operations (gemm)

Installation:
- OpenBLAS: sudo apt-get install libopenblas-dev
- Intel MKL: Intel's optimized implementation
*/

extern "C" {
    // Level 1: saxpy (single precision)
    void saxpy_(int* n, float* alpha, float* x, int* incx, 
                float* y, int* incy);
    
    // Level 1: dot product
    float sdot_(int* n, float* x, int* incx, 
                float* y, int* incy);
    
    // Level 3: matrix multiplication (general)
    void sgemm_(char* transa, char* transb, int* m, int* n, int* k,
                float* alpha, float* A, int* lda, float* B, int* ldb,
                float* beta, float* C, int* ldc);
}

void blas_example() {
    int n = 1000;
    float alpha = 2.0;
    float* x = new float[n];
    float* y = new float[n];
    
    // Initialiser
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0;
        y[i] = 2.0;
    }
    
    // y = alpha * x + y
    int incx = 1, incy = 1;
    saxpy_(&n, &alpha, x, &incx, y, &incy);
    
    // Dot product
    float dot = sdot_(&n, x, &incx, y, &incy);
    
    delete[] x;
    delete[] y;
}
```

---

## Wrapper C++ pour BLAS

### Interface Modernisée

```cpp
#include <vector>
#include <cblas.h>  // CBLAS interface

class BLASWrapper {
public:
    // Matrix multiplication: C = alpha * A * B + beta * C
    static void gemm(const std::vector<double>& A, int rows_A, int cols_A,
                     const std::vector<double>& B, int rows_B, int cols_B,
                     std::vector<double>& C,
                     double alpha = 1.0, double beta = 0.0,
                     bool transpose_A = false, bool transpose_B = false) {
        
        CBLAS_TRANSPOSE transA = transpose_A ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE transB = transpose_B ? CblasTrans : CblasNoTrans;
        
        int M = rows_A;
        int N = cols_B;
        int K = transpose_A ? rows_A : cols_A;
        
        cblas_dgemm(CblasRowMajor, transA, transB,
                    M, N, K, alpha,
                    A.data(), cols_A,
                    B.data(), cols_B,
                    beta, C.data(), N);
    }
    
    // Vector operations
    static void axpy(int n, double alpha, const std::vector<double>& x,
                     std::vector<double>& y) {
        cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
    }
    
    static double dot(int n, const std::vector<double>& x,
                     const std::vector<double>& y) {
        return cblas_ddot(n, x.data(), 1, y.data(), 1);
    }
};

// Exemple
void blas_wrapper_example() {
    std::vector<double> A(100 * 100, 1.0);
    std::vector<double> B(100 * 100, 2.0);
    std::vector<double> C(100 * 100, 0.0);
    
    BLASWrapper::gemm(A, 100, 100, B, 100, 100, C);
    
    std::cout << "Matrix multiplication complete" << std::endl;
}
```

---

## Comparaison Eigen vs BLAS

### Trade-offs

```cpp
/*
Eigen:
+ Interface C++ moderne et expressive
+ Expression templates (optimisations automatiques)
+ Header-only (facile intégration)
+ Support excellent pour petits/moyens problèmes
- Peut être plus lent que BLAS optimisé pour grands problèmes

BLAS:
+ Performance maximale (optimisé assembleur)
+ Standard industriel
+ Supporté par Intel MKL, OpenBLAS, etc.
- Interface C/Fortran (moins moderne)
- Nécessite compilation séparée
*/
```

---

## Exercices

### Exercice 24.3.1
Utilisez Eigen pour résoudre un système linéaire Ax=b et comparer avec solution analytique.

### Exercice 24.3.2
Implémentez une fonction qui calcule la SVD d'une matrice avec Eigen.

### Exercice 24.3.3
Créez un wrapper C++ moderne pour BLAS gemm et testez performance vs Eigen.

### Exercice 24.3.4
Comparez performance Eigen vs BLAS pour multiplication matricielle de grandes tailles.

---

## Points Clés à Retenir

> 📌 **Eigen offre interface C++ moderne avec optimisations automatiques**

> 📌 **BLAS fournit performance maximale pour opérations critiques**

> 📌 **Expression templates d'Eigen évitent temporaires inutiles**

> 📌 **Fixed-size matrices Eigen sont optimisées compile-time**

> 📌 **BLAS Level 3 (gemm) est optimisé pour cache hierarchy**

> 📌 **Combiner Eigen et BLAS selon besoins spécifiques**

---

*Section précédente : [24.2 Templates](./24_02_Templates.md) | Section suivante : [24.4 Parallélisation](./24_04_Parallelisation.md)*

