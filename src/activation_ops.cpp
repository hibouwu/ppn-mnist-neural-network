#include "activation_ops.hpp"
#include <cmath>

namespace ActivationOps {

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

} // namespace ActivationOps
