#include "node.hpp"
#include "autograd/engine.hpp"
#include <algorithm>
#include <stdexcept>

namespace {
Matrix makeZeroGradLike(const Matrix& v) {
    return Matrix(v.rows, v.cols, 0.0);
}
}

Node::Node(const Matrix& v, bool requiresGrad)
    : value_(v),
      requires_grad_(requiresGrad) {}

const Matrix& Node::grad() const {
    if (!grad_) {
        grad_ = std::make_unique<Matrix>(makeZeroGradLike(value_));
    }
    return *grad_;
}

Matrix& Node::grad() {
    if (!grad_) {
        grad_ = std::make_unique<Matrix>(makeZeroGradLike(value_));
    }
    return *grad_;
}

void Node::setInputs(std::vector<Ptr> inputs) {
    inputs_ = std::move(inputs);
    is_leaf_ = inputs_.empty();
}

// Ajoute un gradient au gradient actuel : grad_ += g
void Node::addGrad(const Matrix& g) {
    Matrix& grad_ref = grad();
    if (g.rows != grad_ref.rows || g.cols != grad_ref.cols) {
        throw std::invalid_argument("Dimensions incompatibles dans addGrad.");
    }
    const std::size_t n = grad_ref.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_ref.data[i] += g.data[i];
    }
    has_explicit_seed_grad_ = true;
}

// Met le gradient à zéro
void Node::zeroGrad() {
    Matrix& grad_ref = grad();
    std::fill(grad_ref.data.begin(), grad_ref.data.end(), 0.0);
    has_explicit_seed_grad_ = false;
}

// Lance la rétropropagation du gradient
void Node::backward() {
    auto self = shared_from_this();
    AutogradEngine engine;
    engine.backward(self);
}
