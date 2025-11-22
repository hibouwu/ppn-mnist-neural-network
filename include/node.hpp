/**
 * @file node.hpp
 * @brief Core computation graph node and utilities.
 */
#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>

/**
 * @brief Computation graph node holding value, gradient, parents, and backward function.
 */
class Node : public std::enable_shared_from_this<Node> {
public:
    using Ptr = std::shared_ptr<Node>;

    /**
     * @brief Create a leaf node with given value (grad initialized to zero).
     */
    explicit Node(const Matrix& v);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    const Matrix& value() const { return value_; }
    const Matrix& grad()  const { return grad_;  }
    Matrix&       grad()        { return grad_;  }

    void addParent(const Ptr& p) { parents_.push_back(p); }

    /**
     * @brief Set backward function. Input is the gradient of this node.
     */
    void setBackwardFn(std::function<void(const Matrix&)> fn) {
        backwardFn_ = std::move(fn);
    }

    /**
     * @brief Accumulate gradient into grad_.
     */
    void addGrad(const Matrix& g);
    void zeroGrad();

    /**
     * @brief Run backward pass from this node through the graph.
     */
    void backward();

    /**
     * @brief Topological sort of the graph (parents first).
     */
    static std::vector<Ptr> topoSort(const Ptr& root);

private:
    Matrix value_;
    Matrix grad_;
    std::vector<std::shared_ptr<Node>> parents_;
    std::function<void(const Matrix&)> backwardFn_;
};

inline std::shared_ptr<Node> constant(const Matrix& t){
    return std::make_shared<Node>(t);
}
