#include "network.hpp"
#include <stdexcept>

// ---------- helpers ----------
static LinearLayer::InitType parseInit(const std::string& init_name) {
    if (init_name == "he") return LinearLayer::InitType::He;
    if (init_name == "xavier") return LinearLayer::InitType::Xavier;
    return LinearLayer::InitType::Manual;
}

static std::unique_ptr<ActivationFunction> makeActivation(const std::string& name) {
    if (name == "relu")       return std::make_unique<ReLU>();
    if (name == "leaky_relu") return std::make_unique<LeakyReLU>();
    if (name == "gelu")       return std::make_unique<GELU>();
    if (name == "sigmoid")    return std::make_unique<Sigmoid>();
    if (name == "tanh")       return std::make_unique<Tanh>();
    throw std::invalid_argument(
        "unsupported activation '" + name +
        "'. Expected one of: relu, leaky_relu, gelu, sigmoid, tanh.");
}

static std::unique_ptr<LinearLayer> makeLinear(
    int in_dim, int out_dim,
    const std::string& init_name,
    unsigned int seed
) {
    auto layer = std::make_unique<LinearLayer>(
        static_cast<size_t>(in_dim),
        static_cast<size_t>(out_dim),
        false);
    LinearLayer::InitType initType = parseInit(init_name);

    // Keep same convention as your main.cpp: [-0.1, 0.1] for Manual
    layer->randomInit(-0.1, 0.1, initType, seed);
    return layer;
}
// ----------------------------

void MLPNetwork::addLayer(std::unique_ptr<LinearLayer> linear,
                          std::unique_ptr<ActivationFunction> activation) {
    layers.emplace_back(std::move(linear), std::move(activation));
}

Node::Ptr MLPNetwork::forward(const Node::Ptr& input) const {
    Node::Ptr current = input;
    for (const auto& layer : layers) {
        const bool fuse_relu = layer.activation &&
            dynamic_cast<const ReLU*>(layer.activation.get()) != nullptr;
        current = fuse_relu ? layer.linear->forwardReLU(current)
                            : layer.linear->forward(current);
        if (layer.activation && !fuse_relu) {
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

MLPNetwork MLPNetwork::createSingleHidden(
    int input_dim,
    int hidden_dim,
    int output_dim,
    const std::string& activation_name,
    const std::string& init_name,
    unsigned int seed
) {
    MLPNetwork net;

    // Layer 1: input -> hidden + activation
    net.addLayer(
        makeLinear(input_dim, hidden_dim, init_name, seed),
        makeActivation(activation_name)
    );

    // Layer 2: hidden -> output (logits) + no activation (nullptr)
    // For reproducibility: use a different seed for next layer if seed != 0.
    unsigned int seed2 = (seed == 0) ? 0u : (seed + 1u);
    net.addLayer(
        makeLinear(hidden_dim, output_dim, init_name, seed2),
        nullptr
    );

    return net;
}

MLPNetwork MLPNetwork::createMultiHidden(
    int input_dim,
    const std::vector<int>& hidden_dims,
    int output_dim,
    const std::string& activation_name,
    const std::string& init_name,
    unsigned int seed
) {
    if (hidden_dims.empty()) {
        // If user gives empty list, fallback to no hidden layer makes little sense here.
        // You can also choose to throw:
        throw std::runtime_error("createMultiHidden: hidden_dims is empty.");
    }

    MLPNetwork net;

    int prev = input_dim;
    unsigned int local_seed = seed;

    // Hidden layers: prev -> h + activation
    for (int h : hidden_dims) {
        net.addLayer(
            makeLinear(prev, h, init_name, local_seed),
            makeActivation(activation_name)
        );
        prev = h;
        if (seed != 0) local_seed++; // different seed per layer if user fixed seed
    }

    // Output layer: last_hidden -> output (logits), no activation
    net.addLayer(
        makeLinear(prev, output_dim, init_name, local_seed),
        nullptr
    );

    return net;
}
