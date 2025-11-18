#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "tensor.hpp"

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    // Avant : propagation (Tensor in → Tensor out)
    virtual Tensor forward(const Tensor& input) const = 0;

    // Arrière : rétropropagation (Tensor in, Tensor grad_out → Tensor grad_in)
    virtual Tensor backward(const Tensor& input, const Tensor& grad_output) const = 0;
};

// Implémentation ReLU
class ReLU : public ActivationFunction {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
};

// Implémentation Sigmoid
class Sigmoid : public ActivationFunction {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
};

// Implémentation Tanh
class Tanh : public ActivationFunction {
public:
    Tensor forward(const Tensor& input) const override;
    Tensor backward(const Tensor& input, const Tensor& grad_output) const override;
};

#endif

