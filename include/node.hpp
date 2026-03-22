/**
 * @file node.hpp
 * @brief Core computation graph node and autograd metadata.
 */
#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>

struct BackwardContext;
class GradFn;

/**
 * @brief Computation graph node holding value, gradient, inputs, and autograd state.
 */
class Node : public std::enable_shared_from_this<Node> {
public:
    using Ptr = std::shared_ptr<Node>;

    /**
     * @brief Create a leaf node with given value (grad initialized to zero).
     */
    explicit Node(const Matrix& v, bool requiresGrad = true);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    const Matrix& value() const { return value_; }
    const Matrix& grad()  const;
    Matrix&       grad();

    bool requiresGrad() const { return requires_grad_; }
    void setRequiresGrad(bool value) { requires_grad_ = value; }

    bool isLeaf() const { return is_leaf_; }
    void setIsLeaf(bool value) { is_leaf_ = value; }

    bool isParameter() const { return is_parameter_; }
    void setIsParameter(bool value) { is_parameter_ = value; }

    bool hasAllocatedGrad() const { return static_cast<bool>(grad_); }
    bool hasExplicitGradSeed() const { return has_explicit_seed_grad_; }

    void setInputs(std::vector<Ptr> inputs);
    const std::vector<Ptr>& inputs() const { return inputs_; }

    void setGradFn(std::shared_ptr<GradFn> gradFn) { grad_fn_ = std::move(gradFn); }
    const std::shared_ptr<GradFn>& gradFn() const { return grad_fn_; }

    void setBackwardContext(std::shared_ptr<BackwardContext> context) { context_ = std::move(context); }
    const std::shared_ptr<BackwardContext>& backwardContext() const { return context_; }

    /**
     * @brief Accumulate gradient into grad_.
     */
    void addGrad(const Matrix& g);
    void zeroGrad();

    /**
     * @brief Run backward pass from this node through the graph.
     */
    void backward();

private:
    Matrix value_;
    mutable std::unique_ptr<Matrix> grad_;
    bool requires_grad_ = true;
    bool is_leaf_ = true;
    bool is_parameter_ = false;
    bool has_explicit_seed_grad_ = false;
    std::vector<std::shared_ptr<Node>> inputs_;
    std::shared_ptr<GradFn> grad_fn_;
    std::shared_ptr<BackwardContext> context_;
};

inline std::shared_ptr<Node> constant(const Matrix& t){
    return std::make_shared<Node>(t, false);
}
