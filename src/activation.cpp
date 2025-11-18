#include "activation.hpp"
#include <cmath>

Tensor ReLU::forward(const Tensor& input) const {
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::max(0.0, input.data[i]);
    }
    return output;
}

Tensor ReLU::backward(const Tensor& input, const Tensor& grad_output) const {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = (input.data[i] > 0) ? grad_output.data[i] : 0.0;
    }
    return grad;
}

Tensor Sigmoid::forward(const Tensor& input) const {
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = 1.0 / (1.0 + std::exp(-input.data[i]));
    }
    return output;
}

Tensor Sigmoid::backward(const Tensor& input, const Tensor& grad_output) const {
    Tensor sig = forward(input);  // sigma(x)
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = grad_output.data[i] * sig.data[i] * (1.0 - sig.data[i]);
    }
    return grad;
}

Tensor Tanh::forward(const Tensor& input) const {
    Tensor output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::tanh(input.data[i]);
    }
    return output;
}

Tensor Tanh::backward(const Tensor& input, const Tensor& grad_output) const {
    Tensor tanh_x = forward(input);  // tanh(x)
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = grad_output.data[i] * (1.0 - tanh_x.data[i] * tanh_x.data[i]);
    }
    return grad;
}
