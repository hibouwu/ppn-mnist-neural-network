/**
 * @file activation.hpp
 * @brief Activation function interfaces and implementations.
 */
#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "node.hpp"

class ActivationFunction {
public:
    /**
     * @brief Base class for activation functions.
     */
    virtual ~ActivationFunction() = default;

    /**
     * @brief Forward pass that builds the computation graph.
     * @param input Input node.
     * @return Output node after applying the activation.
     */
    virtual Node::Ptr forward(const Node::Ptr& input) const = 0;
};

// Implémentations concrètes
class ReLU : public ActivationFunction {
public:
    /** @brief Apply ReLU activation. */
    Node::Ptr forward(const Node::Ptr& input) const override;
};

class Sigmoid : public ActivationFunction {
public:
    /** @brief Apply Sigmoid activation. */
    Node::Ptr forward(const Node::Ptr& input) const override;
};

class Tanh : public ActivationFunction {
public:
    /** @brief Apply Tanh activation. */
    Node::Ptr forward(const Node::Ptr& input) const override;
};

#endif
