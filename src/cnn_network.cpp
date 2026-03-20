/**
 * @file cnn_network.cpp
 * @brief Configurable CNN network implementation.
 */
#include "cnn_network.hpp"
#include "math_ops.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>

// =====================================================================
// CNNConfig
// =====================================================================

CNNConfig CNNConfig::lenet5() {
    CNNConfig cfg;
    cfg.conv_channels   = {6, 16};
    cfg.conv_kernels    = {5, 5};
    cfg.conv_strides    = {1, 1};
    cfg.conv_paddings   = {2, 0};
    cfg.pool_after      = {true, true};
    cfg.pool_kernels    = {2, 2};
    cfg.pool_strides    = {2, 2};
    cfg.fc_hidden_sizes = {120, 84};
    cfg.num_classes     = 10;
    return cfg;
}

static void expandVec(std::vector<size_t>& v, size_t n,
                      size_t default_val, const std::string& name) {
    if (v.empty()) {
        v.assign(n, default_val);
    } else if (v.size() == 1) {
        v.assign(n, v[0]);
    } else if (v.size() != n) {
        throw std::invalid_argument(
            name + ": expected 0, 1, or " + std::to_string(n) +
            " values, got " + std::to_string(v.size()));
    }
}

static void expandBoolVec(std::vector<bool>& v, size_t n,
                          bool default_val, const std::string& name) {
    if (v.empty()) {
        v.assign(n, default_val);
    } else if (v.size() == 1) {
        v.assign(n, v[0]);
    } else if (v.size() != n) {
        throw std::invalid_argument(
            name + ": expected 0, 1, or " + std::to_string(n) +
            " values, got " + std::to_string(v.size()));
    }
}

void CNNConfig::expandDefaults() {
    if (conv_channels.empty()) {
        throw std::invalid_argument(
            "CNNConfig: conv_channels must not be empty.");
    }
    size_t n = stages();
    expandVec(conv_kernels,  n, 3, "conv_kernels");
    expandVec(conv_strides,  n, 1, "conv_strides");
    expandVec(conv_paddings, n, 1, "conv_paddings");
    expandBoolVec(pool_after, n, false, "pool_after");
    expandVec(pool_kernels,  n, 2, "pool_kernels");
    expandVec(pool_strides,  n, 2, "pool_strides");
    if (fc_hidden_sizes.empty()) {
        fc_hidden_sizes = {120, 84};
    }
}

void CNNConfig::validate() const {
    size_t n = stages();
    if (n == 0) {
        throw std::invalid_argument("CNNConfig: need at least 1 conv stage.");
    }
    if (input_channels == 0 || input_height == 0 || input_width == 0) {
        throw std::invalid_argument("input shape must be strictly positive.");
    }

    auto check_len = [&](const auto& v, const std::string& name) {
        if (v.size() != n) {
            throw std::invalid_argument(
                name + ": length " + std::to_string(v.size()) +
                " != stages " + std::to_string(n));
        }
    };
    check_len(conv_kernels,  "conv_kernels");
    check_len(conv_strides,  "conv_strides");
    check_len(conv_paddings, "conv_paddings");
    check_len(pool_after,    "pool_after");
    check_len(pool_kernels,  "pool_kernels");
    check_len(pool_strides,  "pool_strides");

    for (size_t i = 0; i < n; ++i) {
        if (conv_channels[i] == 0)
            throw std::invalid_argument("conv_channels[" + std::to_string(i) + "] must be > 0.");
        if (conv_kernels[i] == 0)
            throw std::invalid_argument("conv_kernels[" + std::to_string(i) + "] must be > 0.");
        if (conv_strides[i] == 0)
            throw std::invalid_argument("conv_strides[" + std::to_string(i) + "] must be > 0.");
        if (pool_kernels[i] == 0)
            throw std::invalid_argument("pool_kernels[" + std::to_string(i) + "] must be > 0.");
        if (pool_strides[i] == 0)
            throw std::invalid_argument("pool_strides[" + std::to_string(i) + "] must be > 0.");
    }

    for (size_t i = 0; i < fc_hidden_sizes.size(); ++i) {
        if (fc_hidden_sizes[i] == 0)
            throw std::invalid_argument("fc_hidden_sizes[" + std::to_string(i) + "] must be > 0.");
    }

    if (num_classes == 0) {
        throw std::invalid_argument("num_classes must be > 0.");
    }
}

// =====================================================================
// CNNNetwork
// =====================================================================

CNNNetwork::CNNNetwork(const CNNConfig& cfg, unsigned int seed)
    : input_channels_(0),
      input_height_(0),
      input_width_(0),
      input_dim_(0),
      flatten_dim_(0)
{
    // Local copy for expand + validate
    CNNConfig c = cfg;
    c.expandDefaults();
    c.validate();
    input_channels_ = c.input_channels;
    input_height_ = c.input_height;
    input_width_ = c.input_width;
    input_dim_ = input_channels_ * input_height_ * input_width_;

    size_t n = c.stages();
    pool_after_ = std::vector<bool>(c.pool_after.begin(), c.pool_after.end());

    // --- Build conv/pool layers and compute flatten_dim ---
    size_t C = c.input_channels;
    size_t H = c.input_height;
    size_t W = c.input_width;

    for (size_t i = 0; i < n; ++i) {
        size_t in_ch = (i == 0) ? c.input_channels : c.conv_channels[i - 1];
        convs_.emplace_back(in_ch, c.conv_channels[i],
                            c.conv_kernels[i], c.conv_kernels[i],
                            c.conv_strides[i], c.conv_paddings[i]);

        // Create pool layer (even if not used, cost is negligible)
        pools_.emplace_back(c.pool_kernels[i], c.pool_kernels[i],
                            c.pool_strides[i]);

        // Shape after conv
        auto [cH, cW] = convs_.back().outputShape(H, W);
        if (cH == 0 || cW == 0) {
            throw std::invalid_argument(
                "Stage " + std::to_string(i) +
                ": conv output shape is 0 (H=" + std::to_string(cH) +
                ", W=" + std::to_string(cW) + ").");
        }
        H = cH; W = cW;
        C = c.conv_channels[i];

        // Shape after pool (if enabled)
        if (pool_after_[i]) {
            auto [pH, pW] = pools_.back().outputShape(H, W);
            if (pH == 0 || pW == 0) {
                throw std::invalid_argument(
                    "Stage " + std::to_string(i) +
                    ": pool output shape is 0 (H=" + std::to_string(pH) +
                    ", W=" + std::to_string(pW) + ").");
            }
            H = pH; W = pW;
        }
    }

    flatten_dim_ = C * H * W;

    // --- Build FC layers ---
    size_t prev = flatten_dim_;
    for (size_t h : c.fc_hidden_sizes) {
        fcs_.emplace_back(prev, h);
        prev = h;
    }
    fcs_.emplace_back(prev, c.num_classes);  // Output layer

    // --- Initialize weights ---
    unsigned int s = seed;
    for (size_t i = 0; i < convs_.size(); ++i) {
        convs_[i].randomInit(s);
        if (s != 0) ++s;
    }
    for (size_t i = 0; i < fcs_.size(); ++i) {
        fcs_[i].randomInit(0.0, 0.0, LinearLayer::InitType::He, s);
        if (s != 0) ++s;
    }
}

Node::Ptr CNNNetwork::forward(const Node::Ptr& input) const {
    if (!input) {
        throw std::invalid_argument("CNNNetwork::forward: input node is null.");
    }
    const Matrix& in = input->value();
    if (in.cols != input_dim_) {
        throw std::invalid_argument(
            "CNNNetwork::forward: expected input (N, " + std::to_string(input_dim_) + ").");
    }

    size_t N = in.rows;
    size_t C = input_channels_;
    size_t H = input_height_;
    size_t W = input_width_;

    auto x = input;

    // Conv stages
    for (size_t i = 0; i < convs_.size(); ++i) {
        x = convs_[i].forward(x, N, C, H, W);
        x = relu_.forward(x);

        auto [cH, cW] = convs_[i].outputShape(H, W);
        C = convs_[i].outChannels();
        H = cH; W = cW;

        if (pool_after_[i]) {
            x = pools_[i].forward(x, N, C, H, W);
            auto [pH, pW] = pools_[i].outputShape(H, W);
            H = pH; W = pW;
        }
    }

    // Flatten is implicit: Matrix(N, C*H*W) is already the layout

    // FC layers: hidden + ReLU, last without activation
    for (size_t i = 0; i < fcs_.size(); ++i) {
        x = fcs_[i].forward(x);
        if (i + 1 < fcs_.size()) {
            x = relu_.forward(x);
        }
    }

    return x;
}

std::vector<Node::Ptr> CNNNetwork::getParameters() const {
    std::vector<Node::Ptr> params;

    for (const auto& conv : convs_) {
        auto p = conv.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    for (const auto& fc : fcs_) {
        auto p = fc.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }

    return params;
}
