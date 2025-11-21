#include "network.hpp"

void MLPNetwork::addLayer(std::unique_ptr<LinearLayer> linear, std::unique_ptr<ActivationFunction> activation) {
    layers.emplace_back(std::move(linear), std::move(activation));
}

Node::Ptr MLPNetwork::forward(const Node::Ptr& input) const {
    Node::Ptr current = input;
    for (const auto& layer : layers) {
        current = layer.linear->forward(current);
        current = layer.activation->forward(current);
    }
    return current;
}