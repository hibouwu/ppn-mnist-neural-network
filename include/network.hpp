#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"
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

    // Ajouter une couche linéaire + activation
    void addLayer(std::unique_ptr<LinearLayer> linear, std::unique_ptr<ActivationFunction> activation);

    // Propagation avant
    Tensor forward(const Tensor&) const;

};

#endif
