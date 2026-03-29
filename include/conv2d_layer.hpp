/**
 * @file conv2d_layer.hpp
 * @brief 2D convolution layer using im2col + matmul.
 */
#pragma once

#include "node.hpp"
#include <string>
#include <vector>
#include <cstddef>
#include <memory>
#include <utility>

enum class ConvBackend {
    Reference,
    OneDnn,
};

class Conv2DLayer {
public:
    class Backend;

    /**
     * @brief Create a Conv2D layer.
     * @param in_ch   Input channels.
     * @param out_ch  Output channels (number of filters).
     * @param kH      Kernel height.
     * @param kW      Kernel width.
     * @param stride  Stride (default 1).
     * @param padding Padding (default 0).
     */
    Conv2DLayer(size_t in_ch, size_t out_ch,
                size_t kH, size_t kW,
                size_t stride = 1, size_t padding = 0,
                ConvBackend backend = ConvBackend::Reference);

    /**
     * @brief Forward pass.
     *
     * Input is stored as Matrix(N, C_in*H*W). Shape info is passed explicitly.
     * Output is Matrix(N, C_out*H_out*W_out).
     *
     * @param input  Input node, shape (N, C_in*H*W).
     * @param N      Batch size.
     * @param C      Input channels.
     * @param H      Input height.
     * @param W      Input width.
     * @return Output node, shape (N, C_out*H_out*W_out).
     */
    Node::Ptr forward(const Node::Ptr& input,
                      size_t N, size_t C, size_t H, size_t W) const;

    /**
     * @brief Compute output spatial dimensions.
     */
    std::pair<size_t, size_t> outputShape(size_t H, size_t W) const;

    /**
     * @brief Initialize weights (He initialization).
     */
    void randomInit(unsigned int seed = 0);

    /**
     * @brief Get trainable parameters {kernels, bias}.
     */
    std::vector<Node::Ptr> parameters() const;

    size_t outChannels() const { return out_channels_; }
    ConvBackend backendKind() const { return backend_kind_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_h_;
    size_t kernel_w_;
    size_t stride_;
    size_t padding_;

    Node::Ptr kernels_;  // Shape: (out_ch, in_ch * kH * kW)
    Node::Ptr bias_;     // Shape: (1, out_ch)
    ConvBackend backend_kind_;
    std::shared_ptr<const Backend> backend_;
};

std::string toString(ConvBackend backend);
ConvBackend parseConvBackend(const std::string& backend_name);
