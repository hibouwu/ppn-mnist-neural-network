#pragma once

#include "node.hpp"

#include <cstdint>
#include <string>
#include <vector>

class Optimizer;

namespace checkpoint {

struct Metadata {
    std::uint64_t epoch = 0;
    int batch_size = 64;
    double learning_rate = 0.01;
    unsigned int seed = 0;

    std::string dataset;
    std::string data_dir;
    std::string model;
    std::string activation;
    std::string init;
    std::string optimizer;
    std::string hidden_sizes;

    double momentum = 0.9;
    bool nesterov = false;
    double weight_decay = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;

    std::string cnn_conv_channels;
    std::string cnn_conv_kernels;
    std::string cnn_conv_strides;
    std::string cnn_conv_paddings;
    std::string cnn_pool_after;
    std::string cnn_pool_kernels;
    std::string cnn_pool_strides;
    std::string cnn_fc_hidden_sizes;
    std::string conv_backend;
};

Metadata loadMetadata(const std::string& path);

void saveCheckpoint(const std::string& path,
                    const Metadata& metadata,
                    const std::vector<Node::Ptr>& parameters,
                    const Optimizer& optimizer);

Metadata loadCheckpoint(const std::string& path,
                        const std::vector<Node::Ptr>& parameters,
                        Optimizer& optimizer);

}  // namespace checkpoint
