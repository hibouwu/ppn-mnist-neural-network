#include "layer.hpp"
#include <stdexcept>

LinearLayer::LinearLayer(size_t in, size_t out) : in_dim(in), out_dim(out), weights(in, out), bias(1, out) {
    randomInit();
}

Matrix LinearLayer::forward(const Matrix& input) const {
    // Input shape: (batch_size, in_dim)
    if (input.cols != in_dim) {
        throw std::invalid_argument("Input dimension mismatch");
    }

    // Calcul: input @ weights + bias
    Matrix output = input.matmul(weights);
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output(i, j) += bias(0, j);
        }
    }
    return output;
}

void LinearLayer::randomInit(double min, double max) {
    weights.randomInit(min, max);
    bias.randomInit(min, max);
}
