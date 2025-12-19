#include "layer.hpp"
#include "math_ops.hpp"
#include <stdexcept>
#include <cmath>

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

void LinearLayer::randomInit(double min, double max, InitType initType, unsigned int seed) {
    if (initType == InitType::Manual) {
        Matrix w(in_dim, out_dim);
        w.randomInit(min, max, false, seed);
        weights_ = std::make_shared<Node>(w);

        Matrix b(1, out_dim);
        b.randomInit(min, max, false, seed);
        bias_ = std::make_shared<Node>(b);
    } 
    else if (initType == InitType::He) {
        // He Initialization (Kaiming): Normal(0, sqrt(2/fan_in))
        // Suitable for ReLU
        double stddev = std::sqrt(2.0 / static_cast<double>(in_dim));
        Matrix w(in_dim, out_dim);
        w.randomInit(0.0, stddev, true, seed);
        weights_ = std::make_shared<Node>(w);
        
        // Bias usually 0
        Matrix b(1, out_dim, 0.0);
        bias_ = std::make_shared<Node>(b);
    } 
    else if (initType == InitType::Xavier) {
        // Xavier (Glorot): Normal(0, sqrt(2/(fan_in + fan_out))) 
        // Or Uniform [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
        // Let's use Normal variant for consistency with typical defaults: Normal(0, sqrt(2/(fan_in+fan_out)))
        double stddev = std::sqrt(2.0 / static_cast<double>(in_dim + out_dim));
        Matrix w(in_dim, out_dim);
        w.randomInit(0.0, stddev, true, seed);
        weights_ = std::make_shared<Node>(w);
        
        // Bias usually 0
        Matrix b(1, out_dim, 0.0);
        bias_ = std::make_shared<Node>(b);
    }
}

std::vector<Node::Ptr> LinearLayer::parameters() const {
    return {weights_, bias_};
}