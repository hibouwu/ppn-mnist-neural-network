#include "math_ops.hpp"
#include <cmath>
#include <algorithm>

namespace MathOps {

using NodePtr = std::shared_ptr<Node>;

Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b) {
    const Matrix& val_a = a->value();
    const Matrix& val_b = b->value();
    
    // Check for broadcasting: (N, M) + (1, M)
    bool broadcast_b = (val_b.rows == 1 && val_b.cols == val_a.cols && val_a.rows > 1);
    
    Matrix out(val_a.rows, val_a.cols);
    
    if (broadcast_b) {
        for (size_t i = 0; i < val_a.rows; ++i) {
            for (size_t j = 0; j < val_a.cols; ++j) {
                out(i, j) = val_a(i, j) + val_b(0, j);
            }
        }
    } else {
        // Standard addition
        out = val_a.add(val_b);
    }

    auto node = std::make_shared<OperationNode>(OpKind::ADD, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b, broadcast_b](const Matrix& grad){
        a_ptr->addGrad(grad);
        
        if (broadcast_b) {
            // If b was broadcasted, we sum gradients along rows
            Matrix grad_b(1, grad.cols);
            for (size_t j = 0; j < grad.cols; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < grad.rows; ++i) {
                    sum += grad(i, j);
                }
                grad_b(0, j) = sum;
            }
            b_ptr->addGrad(grad_b);
        } else {
            b_ptr->addGrad(grad);
        }
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

Node::Ptr relu(const Node::Ptr& x) {
    Matrix out = x->value();
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::max(0.0, out.data[i]);

    auto node = std::make_shared<OperationNode>(OpKind::RELU, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x,out](const Matrix& grad){
        Matrix gx(grad.rows, grad.cols);
        for (size_t i = 0; i < out.data.size(); ++i)
            gx.data[i] = (out.data[i] > 0) ? grad.data[i] : 0.0;
        x->addGrad(gx);
    });

    return node;
}

Node::Ptr sigmoid(const Node::Ptr& x) {
    Matrix out = x->value();
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = 1.0 / (1.0 + std::exp(-out.data[i]));

    auto node = std::make_shared<OperationNode>(OpKind::SIGMOID, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x,out](const Matrix& grad){
        Matrix gx(grad.rows, grad.cols);
        for (size_t i = 0; i < out.data.size(); ++i)
            gx.data[i] = grad.data[i] * out.data[i] * (1 - out.data[i]);
        x->addGrad(gx);
    });

    return node;
}

Node::Ptr tanh(const Node::Ptr& x) {
    Matrix out = x->value();
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::tanh(out.data[i]);

    auto node = std::make_shared<OperationNode>(OpKind::TANH, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x,out](const Matrix& grad){
        Matrix gx(grad.rows, grad.cols);
        for (size_t i = 0; i < out.data.size(); ++i)
            gx.data[i] =
                grad.data[i] * (1 - out.data[i] * out.data[i]);
        x->addGrad(gx);
    });

    return node;
}

} // namespace MathOps
