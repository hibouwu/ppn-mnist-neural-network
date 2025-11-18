#include "network.hpp"

void MLPNetwork::addLayer(std::unique_ptr<LinearLayer> linear,
                          std::unique_ptr<ActivationFunction> activation)
{
    layers.emplace_back(std::move(linear), std::move(activation));
}

Tensor MLPNetwork::forward(const Tensor& input) const {
    Tensor current = input;
    for (const auto& layer : layers) {
        current = layer.linear->forward(current);
        current = layer.activation->forward(current);
    }
    return current;
}
