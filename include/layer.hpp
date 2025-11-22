/**
 * @file layer.hpp
 * @brief Linear layer definition.
 */
#ifndef LAYER_HPP
#define LAYER_HPP

#include "node.hpp"

class LinearLayer {
public:
    /**
     * @brief Create a linear layer y = x @ W + b.
     * @param in  Input dimension.
     * @param out Output dimension.
     */
    LinearLayer(size_t in, size_t out);

    /**
     * @brief Forward pass.
     * @param input Input node shaped (batch, in_dim).
     * @return Output node shaped (batch, out_dim).
     */
    Node::Ptr forward(const Node::Ptr& input) const;

    /**
     * @brief Randomly initialize weights and bias.
     * @param min Minimum value.
     * @param max Maximum value.
     */
    void randomInit(double min = -1.0, double max = 1.0);

    /**
     * @brief Return trainable parameters.
     * @return {weights, bias}.
     */
    std::vector<Node::Ptr> parameters() const;

private:
    size_t in_dim;
    size_t out_dim;
    Node::Ptr weights_;  // Shape: (in_dim, out_dim)
    Node::Ptr bias_;     // Shape: (1, out_dim)
};

#endif
