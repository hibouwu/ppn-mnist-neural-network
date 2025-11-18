#include "math_ops.hpp"

namespace MathOps {

Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().add(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::ADD, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b](const Matrix& grad){
        a_ptr->addGrad(grad);
        b_ptr->addGrad(grad);
    });

    return node;
}

Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().mul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b](const Matrix& grad){
        a_ptr->addGrad(grad.mul(b_ptr->value()));
        b_ptr->addGrad(grad.mul(a_ptr->value()));
    });

    return node;
}

Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().matmul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MATMUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b](const Matrix& grad){
        // dA = grad @ B^T
        Matrix dA = grad.matmul(b_ptr->value().transpose());
        a_ptr->addGrad(dA);

        // dB = A^T @ grad
        Matrix dB = a_ptr->value().transpose().matmul(grad);
        b_ptr->addGrad(dB);
    });

    return node;
}

} // namespace MathOps
