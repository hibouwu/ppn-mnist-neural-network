/**
 * @file maxpool2d_layer.hpp
 * @brief 2D max pooling layer.
 */
#pragma once

#include "node.hpp"
#include <vector>
#include <cstddef>
#include <utility>

class MaxPool2DLayer {
public:
    /**
     * @brief Create a MaxPool2D layer.
     * @param pH     Pool height.
     * @param pW     Pool width.
     * @param stride Stride (default = pH).
     */
    MaxPool2DLayer(size_t pH, size_t pW, size_t stride);

    /**
     * @brief Forward pass.
     *
     * Input: Matrix(N, C*H*W), shape passed explicitly.
     * Output: Matrix(N, C*H_out*W_out).
     *
     * @param input Input node.
     * @param N     Batch size.
     * @param C     Channels.
     * @param H     Input height.
     * @param W     Input width.
     * @return Output node.
     */
    Node::Ptr forward(const Node::Ptr& input,
                      size_t N, size_t C, size_t H, size_t W) const;

    /**
     * @brief Compute output spatial dimensions.
     */
    std::pair<size_t, size_t> outputShape(size_t H, size_t W) const;

    /**
     * @brief No trainable parameters.
     */
    std::vector<Node::Ptr> parameters() const { return {}; }

private:
    size_t pool_h_;
    size_t pool_w_;
    size_t stride_;
};
