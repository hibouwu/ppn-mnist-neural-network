#include "activation.hpp"
#include <cmath>

Matrix ReLU::forward(const Matrix& input) const {
    Matrix output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::max(0.0, input.data[i]);
    }
    return output;
}

Matrix ReLU::backward(const Matrix& input, const Matrix& grad_output) const {
    Matrix grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = (input.data[i] > 0) ? grad_output.data[i] : 0.0;
    }
    return grad;
}

Matrix Sigmoid::forward(const Matrix& input) const {
    Matrix output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = 1.0 / (1.0 + std::exp(-input.data[i]));
    }
    return output;
}

Matrix Sigmoid::backward(const Matrix& input, const Matrix& grad_output) const {
    Matrix sig = forward(input);  // sigma(x)
    Matrix grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = grad_output.data[i] * sig.data[i] * (1.0 - sig.data[i]);  // grad * sigma(x)(1 - sigma(x))
    }
    return grad;
}

Matrix Tanh::forward(const Matrix& input) const {
    Matrix output(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        output.data[i] = std::tanh(input.data[i]);
    }
    return output;
}

Matrix Tanh::backward(const Matrix& input, const Matrix& grad_output) const {
    Matrix tanh_x = forward(input);  // tanh(x)
    Matrix grad(input.rows, input.cols);
    for (size_t i = 0; i < input.data.size(); ++i) {
        grad.data[i] = grad_output.data[i] * (1.0 - tanh_x.data[i] * tanh_x.data[i]);  // grad * (1 - tanh^2(x))
    }
    return grad;
}
