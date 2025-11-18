#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>  // size_t

class Tensor {
public:
    // Stockage : data flattenée + tailles
    std::vector<double> data;
    size_t rows;
    size_t cols;

    // Constructeurs
    Tensor(size_t r, size_t c);
    Tensor(size_t r, size_t c, double init_value);

    // Constructeur de copie, opérateur d'assignation
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // Destructeur
    ~Tensor() = default;

    // Accès aux éléments (r, c)
    double& operator()(size_t r, size_t c);
    const double& operator()(size_t r, size_t c) const;

    // Opérations de base
    Tensor add(const Tensor& other) const;     // élément par élément : +
    Tensor mul(const Tensor& other) const;     // élément par élément : *
    Tensor matmul(const Tensor& other) const;  // produit matriciel
    Tensor transpose() const;

    // Initialisation aléatoire
    void randomInit(double min = -1.0, double max = 1.0);

    // Affichage
    void print() const;
};

#endif

