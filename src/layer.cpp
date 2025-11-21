#include "layer.hpp"
#include "math_ops.hpp"
#include <stdexcept>

LinearLayer::LinearLayer(size_t in, size_t out) : in_dim(in), out_dim(out) {
    randomInit();
}

Node::Ptr LinearLayer::forward(const Node::Ptr& input) const {
    // Input shape: (batch_size, in_dim)
    // We assume input is (batch_size, in_dim)
    // weights is (in_dim, out_dim)
    // bias is (1, out_dim)
    
    // y = x @ W + b
    Node::Ptr z = MathOps::matmul(input, weights_);
    Node::Ptr output = MathOps::add(z, bias_);
    
    return output;
}

void LinearLayer::randomInit(double min, double max) {
    Matrix w(in_dim, out_dim);
    w.randomInit(min, max);
    weights_ = std::make_shared<Node>(w);

    Matrix b(1, out_dim);
    b.randomInit(min, max);
    bias_ = std::make_shared<Node>(b);
}

std::vector<Node::Ptr> LinearLayer::parameters() const {
    return {weights_, bias_};
}