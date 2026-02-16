/**
 * @file conv2d_layer.hpp
 * @brief 2D convolution layer using im2col + matmul.
 */
#pragma once

#include "node.hpp"
#include <vector>
#include <cstddef>
#include <utility>

class Conv2DLayer {
public:
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
                size_t stride = 1, size_t padding = 0);

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

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_h_;
    size_t kernel_w_;
    size_t stride_;
    size_t padding_;

    Node::Ptr kernels_;  // Shape: (out_ch, in_ch * kH * kW)
    Node::Ptr bias_;     // Shape: (1, out_ch)

    /**
     * @brief Transform input into column matrix for matmul-based convolution.
     *
     * Input:  Matrix(N, C*H*W), interpreted as (N, C, H, W).
     * Output: Matrix(N*H_out*W_out, C*kH*kW).
     */
    Matrix im2col(const Matrix& input,
                  size_t N, size_t C, size_t H, size_t W,
                  size_t H_out, size_t W_out) const;

    /**
     * @brief Inverse of im2col: scatter column gradients back to input shape.
     *
     * Input:  Matrix(N*H_out*W_out, C*kH*kW).
     * Output: Matrix(N, C*H*W).
     */
    Matrix col2im(const Matrix& cols,
                  size_t N, size_t C, size_t H, size_t W,
                  size_t H_out, size_t W_out) const;
};
