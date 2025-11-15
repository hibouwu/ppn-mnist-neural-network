#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "tensor.hpp"

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    // Propagation avant
    virtual Matrix forward(const Matrix& input) const = 0;

    // Rétropropagation (gradient de la fonction d'activation)
    virtual Matrix backward(const Matrix& input, const Matrix& grad_output) const = 0;
};

// Implémentations concrètes
class ReLU : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& grad_output) const override;
};

class Sigmoid : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& grad_output) const override;
};

class Tanh : public ActivationFunction {
public:
    Matrix forward(const Matrix& input) const override;
    Matrix backward(const Matrix& input, const Matrix& grad_output) const override;
};

#endif
