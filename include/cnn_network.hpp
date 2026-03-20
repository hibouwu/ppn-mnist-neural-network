/**
 * @file cnn_network.hpp
 * @brief Configurable CNN network with dataset-dependent input shape.
 *        Default configuration reproduces LeNet-5 on MNIST.
 */
#pragma once

#include "neural_network.hpp"
#include "conv2d_layer.hpp"
#include "maxpool2d_layer.hpp"
#include "layer.hpp"
#include "activation.hpp"
#include <vector>
#include <string>
#include <stdexcept>

/**
 * @brief Configuration for a CNN with variable conv stages and FC layers.
 */
struct CNNConfig {
    size_t input_channels = 1;
    size_t input_height = 28;
    size_t input_width = 28;
    std::vector<size_t> conv_channels;    // Output channels per stage  [required]
    std::vector<size_t> conv_kernels;     // Square kernel size         [default 3]
    std::vector<size_t> conv_strides;     // Conv stride                [default 1]
    std::vector<size_t> conv_paddings;    // Padding                    [default 1]
    std::vector<bool>   pool_after;       // Pool after stage?          [default false]
    std::vector<size_t> pool_kernels;     // Pool kernel size           [default 2]
    std::vector<size_t> pool_strides;     // Pool stride                [default 2]
    std::vector<size_t> fc_hidden_sizes;  // FC hidden dimensions       [default {120,84}]
    size_t num_classes = 10;

    /** Number of conv stages. */
    size_t stages() const { return conv_channels.size(); }

    /** Expand single-value or empty lists to stages() length with defaults. */
    void expandDefaults();

    /** Validate lengths match and values are legal. Throws on error. */
    void validate() const;

    /** Return the standard LeNet-5 configuration. */
    static CNNConfig lenet5();
};

/**
 * @brief Configurable CNN network.
 *        Topology: [Conv → ReLU → (Pool?)] × S  →  Flatten  →  [FC → ReLU] × H  →  FC(logits)
 */
class CNNNetwork : public NeuralNetwork {
public:
    /**
     * @brief Construct from config.
     * @param cfg  Network configuration (will be expanded and validated).
     * @param seed Random seed (0 = random).
     */
    explicit CNNNetwork(const CNNConfig& cfg, unsigned int seed = 0);

    Node::Ptr forward(const Node::Ptr& input) const override;
    std::vector<Node::Ptr> getParameters() const override;

private:
    std::vector<Conv2DLayer>     convs_;
    std::vector<MaxPool2DLayer>  pools_;
    std::vector<bool>            pool_after_;
    std::vector<LinearLayer>     fcs_;     // hidden + output
    ReLU relu_;
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;
    size_t input_dim_;
    size_t flatten_dim_;                   // C*H*W after all conv/pool stages
};
