/**
 * @file math_ops.hpp
 * @brief Basic math operations on computation graph nodes.
 */
#pragma once

#include "operation_node.hpp"

namespace MathOps {

    /** @brief Element-wise addition (supports bias broadcasting). */
    Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b);

    /** @brief Element-wise multiplication. */
    Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b);

    /** @brief Matrix multiplication. */
    Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b);
    
    /** @brief Sum all elements to a scalar node. */
    Node::Ptr sum(const Node::Ptr& x); 
    
    /** @brief Mean of all elements to a scalar node. */
    Node::Ptr mean(const Node::Ptr& x);

    /** @brief ReLU activation. */
    Node::Ptr relu(const Node::Ptr& x);
    /** @brief LeakyReLU activation. */
    Node::Ptr leaky_relu(const Node::Ptr& x, double alpha = 0.01);
    /** @brief GELU activation (tanh approximation). */
    Node::Ptr gelu(const Node::Ptr& x);
    /** @brief Sigmoid activation. */
    Node::Ptr sigmoid(const Node::Ptr& x);
    /** @brief Tanh activation. */
    Node::Ptr tanh(const Node::Ptr& x);

}
