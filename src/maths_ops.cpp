#include "math_ops.hpp"

namespace MathOps {

using NodePtr = std::shared_ptr<Node>;

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


Node::Ptr sum(const Node::Ptr& x) {
    const Matrix& xv = x->value();

    double s = 0.0;
    for (double v : xv.data) s += v;
    Matrix out(1, 1);
    out.data[0] = s;

    auto node = std::make_shared<Node>(out);
    node->addParent(x);

    // backward
    node->setBackwardFn([x](const Matrix& grad_out) {
        // grad_out  1x1
        double g = grad_out.data[0];
        const Matrix& xv = x->value();
        Matrix gx(xv.rows, xv.cols);
        std::fill(gx.data.begin(), gx.data.end(), g);
        x->addGrad(gx);
    });

    return node;
}


Node::Ptr mean(const Node::Ptr& x) {
    const Matrix& xv = x->value();
    std::size_t N = xv.data.size();

    // forward
    double s = 0.0;
    for (double v : xv.data) s += v;
    double m = (N > 0 ? s / static_cast<double>(N) : 0.0);
    Matrix out(1, 1);
    out.data[0] = m;

    auto node = std::make_shared<Node>(out);
    node->addParent(x);

    // backward
    node->setBackwardFn([x, N](const Matrix& grad_out) {
        double g = grad_out.data[0];
        const Matrix& xv = x->value();
        Matrix gx(xv.rows, xv.cols);
        double coeff = (N > 0 ? g / static_cast<double>(N) : 0.0);
        std::fill(gx.data.begin(), gx.data.end(), coeff);
        x->addGrad(gx);
    });

    return node;
}


} // namespace MathOps
