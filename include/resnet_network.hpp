/**
 * @file resnet_network.hpp
 * @brief Small serial ResNet network for image classification.
 */
#pragma once

#include "activation.hpp"
#include "conv2d_layer.hpp"
#include "layer.hpp"
#include "neural_network.hpp"

#include <memory>
#include <string>
#include <vector>

struct ResNetConfig {
    size_t input_channels = 1;
    size_t input_height = 28;
    size_t input_width = 28;
    size_t stem_channels = 16;
    std::vector<size_t> stage_channels = {16, 32, 64};
    std::vector<size_t> blocks_per_stage = {2, 2, 2};
    size_t num_classes = 10;
    ConvBackend conv_backend = ConvBackend::Reference;

    void validate() const;
};

class ResNetNetwork : public NeuralNetwork {
public:
    explicit ResNetNetwork(const ResNetConfig& cfg, unsigned int seed = 0);
    explicit ResNetNetwork(
        const ResNetConfig& cfg,
        const std::string& activation_name,
        const std::string& init_name,
        unsigned int seed = 0
    );

    Node::Ptr forward(const Node::Ptr& input) const override;
    std::vector<Node::Ptr> getParameters() const override;

private:
    class BasicBlock {
    public:
        BasicBlock(size_t in_channels,
                   size_t out_channels,
                   size_t stride,
                   ConvBackend backend);

        Node::Ptr forward(const Node::Ptr& input,
                          size_t N,
                          size_t& C,
                          size_t& H,
                          size_t& W,
                          const ActivationFunction& activation) const;
        void randomInit(unsigned int& seed);
        std::vector<Node::Ptr> parameters() const;

        std::pair<size_t, size_t> outputShape(size_t H, size_t W) const;
        size_t outChannels() const { return out_channels_; }

    private:
        Conv2DLayer conv1_;
        Conv2DLayer conv2_;
        std::unique_ptr<Conv2DLayer> projection_;
        size_t in_channels_;
        size_t out_channels_;
        size_t stride_;
    };

    Conv2DLayer stem_;
    std::vector<BasicBlock> blocks_;
    LinearLayer classifier_;
    std::unique_ptr<ActivationFunction> activation_;
    std::string init_name_;
    size_t input_channels_;
    size_t input_height_;
    size_t input_width_;
    size_t input_dim_;
    size_t flatten_dim_;
};
