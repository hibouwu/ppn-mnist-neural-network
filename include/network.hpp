/**
 * @file network.hpp
 * @brief Simple MLP network composition helpers.
 */
#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "neural_network.hpp"
#include "node.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include <vector>
#include <memory>
#include <string>

class MLPNetwork : public NeuralNetwork {
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
    Node::Ptr forward(const Node::Ptr& input) const override;

    /**
     * @brief Get all trainable parameters (weights and biases) from all layers.
     */
    std::vector<Node::Ptr> getParameters() const override;

    /**
     * @brief Build a MLP with ONE hidden layer:
     *        input -> hidden -> output
     *
     * @param input_dim   e.g. 784
     * @param hidden_dim  e.g. 128
     * @param output_dim  e.g. 10
     * @param activation_name  e.g. "relu", "tanh"
     * @param init_name        e.g. "he", "xavier"
     */
    static MLPNetwork createSingleHidden(
        int input_dim,
        int hidden_dim,
        int output_dim,
        const std::string& activation_name,
        const std::string& init_name,
        unsigned int seed = 0
    );

    static MLPNetwork createMultiHidden(
        int input_dim,
        const std::vector<int>& hidden_dims,
        int output_dim,
        const std::string& activation_name,
        const std::string& init_name,
        unsigned int seed = 0
    );

};

#endif
