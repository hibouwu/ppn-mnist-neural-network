#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>  // size_t

using Scalar = float;

class Matrix {
public:
    // Ordre : data, rows, cols (dans le même ordre que dans les constructeurs)
    std::vector<Scalar> data;  // 1er
    size_t rows;               // 2ème
    size_t cols;               // 3ème

    // Constructeurs
    Matrix(size_t r, size_t c);
    Matrix(size_t r, size_t c, double init_value);

    // Constructeur de copie, opérateur d'assignation
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // Destructeur
    ~Matrix() = default;

    // Accès à un élément (ligne, colonne)
    Scalar& operator()(size_t r, size_t c);
    const Scalar& operator()(size_t r, size_t c) const;

    // Opérations de base
    Matrix add(const Matrix& other) const;
    void addInPlace(const Matrix& other);
    Matrix mul(const Matrix& other) const;        // multiplication élément par élément
    Matrix matmul(const Matrix& other) const;     // produit matriciel
    void matmul_into(const Matrix& other, Matrix& out, bool transA = false, bool transB = false) const;


    // Initialisation aléatoire
    // use_normal=false -> Uniform(param1, param2)
    // use_normal=true  -> Normal(mean=param1, stddev=param2)
    // seed=0 -> random_device, seed!=0 -> fixed
    void randomInit(double param1 = -1.0, double param2 = 1.0, bool use_normal = false, unsigned int seed = 0);
  
    // Affichage
    void print() const;
};

#endif
