#include "math_ops.hpp"

namespace MathOps {

Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().add(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::ADD, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a, b](const Matrix& grad){
        a->addGrad(grad);
        b->addGrad(grad);
    });

    return node;
}

Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().mul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a, b](const Matrix& grad){
        a->addGrad(grad.mul(b->value()));
        b->addGrad(grad.mul(a->value()));
    });

    return node;
}

Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().matmul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MATMUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a, b](const Matrix& grad){
        // dA = grad @ B^T
        Matrix dA = grad.matmul(b->value().transpose());
        a->addGrad(dA);

        // dB = A^T @ grad
        Matrix dB = a->value().transpose().matmul(grad);
        b->addGrad(dB);
    });

    return node;
}

} // namespace MathOps

