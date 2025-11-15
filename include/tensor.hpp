#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>  // size_t

class Matrix {
public:
    // ✅ Ordre : data, rows, cols (dans le même ordre que dans les constructeurs)
    std::vector<double> data;  // 1er
    size_t rows;               // 2ème
    size_t cols;               // 3ème

    // Constructeurs
    Matrix(size_t r, size_t c);
    Matrix(size_t r, size_t c, double init_value);

    // Constructeur de copie, opérateur d'assignation
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    // Destructeur
    ~Matrix() = default;

    // Accès à un élément (ligne, colonne)
    double& operator()(size_t r, size_t c);
    const double& operator()(size_t r, size_t c) const;

    // Opérations de base
    Matrix add(const Matrix& other) const;
    Matrix mul(const Matrix& other) const;        // multiplication élément par élément
    Matrix matmul(const Matrix& other) const;     // produit matriciel
    Matrix transpose() const;

    // Initialisation aléatoire
    void randomInit(double min = -1.0, double max = 1.0);

    // Affichage
    void print() const;
};

#endif
