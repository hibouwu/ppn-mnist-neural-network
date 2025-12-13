/**
 * @file network.hpp
 * @brief Simple MLP network composition helpers.
 */
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "node.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include <vector>
#include <memory>

class MLPNetwork {
public:
    struct LayerNode {
        std::unique_ptr<LinearLayer> linear;
        std::unique_ptr<ActivationFunction> activation;

        LayerNode(std::unique_ptr<LinearLayer> l, std::unique_ptr<ActivationFunction> a)
            : linear(std::move(l)), activation(std::move(a)) {}
    };

    std::vector<LayerNode> layers;

    /**
     * @brief Append a linear layer followed by an activation.
     */
    void addLayer(std::unique_ptr<LinearLayer> linear, std::unique_ptr<ActivationFunction> activation);

    /**
     * @brief Forward pass through all layers.
     * @param input Input node (batch, in_dim of first layer).
     * @return Output node after the last activation.
     */
    Node::Ptr forward(const Node::Ptr& input) const;

    /**
     * @brief Get all trainable parameters (weights and biases) from all layers.
     */
    std::vector<Node::Ptr> getParameters() const;
};

#endif
