#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class LinearLayer {
public:
    Matrix weights;  // Shape: (in_dim, out_dim)
    Matrix bias;     // Shape: (1, out_dim)

    size_t in_dim;
    size_t out_dim;

    // Constructeur
    LinearLayer(size_t in, size_t out);

    // Propagation avant
    Matrix forward(const Matrix& input) const;

    // Initialisation aléatoire
    void randomInit(double min = -1.0, double max = 1.0);
};

#endif
