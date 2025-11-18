#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

class LinearLayer {
public:
    Tensor weights;  // Shape: (in_dim, out_dim)
    Tensor bias;     // Shape: (1, out_dim)

    size_t in_dim;
    size_t out_dim;

    // Constructeur
    LinearLayer(size_t in, size_t out);

    // Propagation avant : input shape (batch, in_dim)
    Tensor forward(const Tensor& input) const;

    // Initialisation aléatoire
    void randomInit(double min = -1.0, double max = 1.0);
};

#endif

