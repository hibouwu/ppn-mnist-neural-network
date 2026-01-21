#include "optimizer.hpp"

Optimizer::Optimizer(std::vector<Node::Ptr> params, double lr)
    : parameters_(std::move(params)), lr_(lr) {}

void Optimizer::zeroGrad() {
    for (auto& p : parameters_) {
        p->zeroGrad();
    }
}

SGDOptimizer::SGDOptimizer(std::vector<Node::Ptr> params, double lr)
    : Optimizer(std::move(params), lr) {}

void SGDOptimizer::step() {
    for (auto& p : parameters_) {
        // W = W - lr * grad; const_cast used because Node lacks a mutable accessor.
        Matrix& val = const_cast<Matrix&>(p->value());
        const Matrix& grad = p->grad();

        size_t n = val.data.size();
        for(size_t i=0; i<n; ++i) {
            val.data[i] -= lr_ * grad.data[i];
        }
    }
}
