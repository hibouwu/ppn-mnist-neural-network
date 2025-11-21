#include "activation.hpp"
#include "math_ops.hpp"

Node::Ptr ReLU::forward(const Node::Ptr& input) const {
    return MathOps::relu(input);
}

Node::Ptr Sigmoid::forward(const Node::Ptr& input) const {
    return MathOps::sigmoid(input);
}

Node::Ptr Tanh::forward(const Node::Ptr& input) const {
    return MathOps::tanh(input);
}
