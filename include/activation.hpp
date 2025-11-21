#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "node.hpp"

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    // Propagation avant (construit le graphe)
    virtual Node::Ptr forward(const Node::Ptr& input) const = 0;
};

// Implémentations concrètes
class ReLU : public ActivationFunction {
public:
    Node::Ptr forward(const Node::Ptr& input) const override;
};

class Sigmoid : public ActivationFunction {
public:
    Node::Ptr forward(const Node::Ptr& input) const override;
};

class Tanh : public ActivationFunction {
public:
    Node::Ptr forward(const Node::Ptr& input) const override;
};

#endif
