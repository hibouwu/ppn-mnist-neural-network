/**
 * @file neural_network.hpp
 * @brief Minimal interface for neural network models (MLP, CNN, etc.).
 */
#pragma once

#include "node.hpp"
#include <vector>

class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;

    /**
     * @brief Forward pass through the network.
     * @param input Input node.
     * @return Output node.
     */
    virtual Node::Ptr forward(const Node::Ptr& input) const = 0;

    /**
     * @brief Get all trainable parameters.
     */
    virtual std::vector<Node::Ptr> getParameters() const = 0;
};
