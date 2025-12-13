#include "network.hpp"

void MLPNetwork::addLayer(std::unique_ptr<LinearLayer> linear, std::unique_ptr<ActivationFunction> activation) {
    layers.emplace_back(std::move(linear), std::move(activation));
}

Node::Ptr MLPNetwork::forward(const Node::Ptr& input) const {
    Node::Ptr current = input;
    for (const auto& layer : layers) {
        // Linear
        current = layer.linear->forward(current);
        // Activation
        if (layer.activation) {
            current = layer.activation->forward(current);
        }
    }
    return current;
}

std::vector<Node::Ptr> MLPNetwork::getParameters() const {
    std::vector<Node::Ptr> params;
    for (const auto& layer : layers) {
        auto layer_params = layer.linear->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}