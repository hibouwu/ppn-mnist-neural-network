#include "math_ops.hpp"
#include <cmath>
#include <algorithm>

namespace MathOps {

using NodePtr = std::shared_ptr<Node>;
constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kGeluCubicCoeff = 0.044715;

/**
 * @brief Element-wise addition between two nodes, with basic broadcasting support.
 *
 * The forward pass computes:
 *   - Standard element-wise sum if both matrices have identical shapes.
 *   - Broadcasting on the second operand if its shape is (1, M) and the first
 *     operand is (N, M), i.e. the same row is added to each row of the first matrix.
 *
 * The backward pass:
 *   - Propagates the gradient directly to both parents in the standard case.
 *   - If broadcasting was used, the gradient for the broadcasted parent is
 *     obtained by summing gradients over all rows.
 *
 * @param a First input node.
 * @param b Second input node.
 * @return A new OperationNode representing the addition.
 */
Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b) {
    const Matrix& val_a = a->value();
    const Matrix& val_b = b->value();
    
    // Check for broadcasting pattern: (N, M) + (1, M)
    bool broadcast_b = (val_b.rows == 1 && val_b.cols == val_a.cols && val_a.rows > 1);
    
    Matrix out(val_a.rows, val_a.cols);
    
    if (broadcast_b) {
        // Broadcasting: each row of 'a' adds the single row of 'b'
        for (size_t i = 0; i < val_a.rows; ++i) {
            for (size_t j = 0; j < val_a.cols; ++j) {
                out(i, j) = val_a(i, j) + val_b(0, j);
            }
        }
    } else {
        // Standard element-wise addition
        out = val_a.add(val_b);
    }

    auto node = std::make_shared<OperationNode>(OpKind::ADD, out, std::vector<Node::Ptr>{a, b});

    // Backward: dL/dA = grad, dL/dB = grad (or reduced along rows if B was broadcasted)
    node->setBackwardFn([a_ptr=a, b_ptr=b, broadcast_b](const Matrix& grad){
        // Gradient w.r.t. 'a' has the same shape as grad
        a_ptr->addGrad(grad);
        
        if (broadcast_b) {
            // If 'b' was broadcasted, we sum gradients along rows to get a (1, M) gradient.
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
            // No broadcasting: gradient has the same shape as grad.
            b_ptr->addGrad(grad);
        }
    });

    return node;
}

/**
 * @brief Element-wise multiplication (Hadamard product) between two nodes.
 *
 * Forward:
 *   out = a ⊙ b
 *
 * Backward:
 *   dL/da = grad ⊙ b
 *   dL/db = grad ⊙ a
 *
 * @param a First input node.
 * @param b Second input node.
 * @return A new OperationNode representing element-wise multiplication.
 */
Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().mul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b](const Matrix& grad){
        // dL/dA = grad ⊙ B
        a_ptr->addGrad(grad.mul(b_ptr->value()));
        // dL/dB = grad ⊙ A
        b_ptr->addGrad(grad.mul(a_ptr->value()));
    });

    return node;
}

/**
 * @brief Matrix multiplication between two nodes.
 *
 * Forward:
 *   out = A @ B
 *
 * Backward:
 *   dL/dA = grad @ B^T
 *   dL/dB = A^T @ grad
 *
 * @param a Left-hand side node (A).
 * @param b Right-hand side node (B).
 * @return A new OperationNode representing the matrix multiplication.
 */
Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b) {
    Matrix out = a->value().matmul(b->value());
    auto node = std::make_shared<OperationNode>(OpKind::MATMUL, out, std::vector<Node::Ptr>{a, b});

    node->setBackwardFn([a_ptr=a, b_ptr=b](const Matrix& grad){
        // dA = grad @ B^T (sans allouer B^T)
        Matrix dA(grad.rows, b_ptr->value().rows);
        grad.matmul_into(b_ptr->value(), dA, false, true);
        a_ptr->addGrad(dA);

        // dB = A^T @ grad (sans allouer A^T)
        Matrix dB(a_ptr->value().cols, grad.cols);
        a_ptr->value().matmul_into(grad, dB, true, false);
        b_ptr->addGrad(dB);
    });

    return node;
}


/**
 * @brief Sums all elements of the input node into a scalar node (1x1 matrix).
 *
 * Forward:
 *   out = ∑_i x_i
 *
 * Backward:
 *   dL/dx_i = grad_out (same scalar for all elements).
 *
 * @param x Input node.
 * @return A new Node storing the scalar sum.
 */
Node::Ptr sum(const Node::Ptr& x) {
    const Matrix& xv = x->value();

    double s = 0.0;
    for (double v : xv.data) s += v;
    Matrix out(1, 1);
    out.data[0] = s;

    auto node = std::make_shared<Node>(out);
    node->addParent(x);

    // Backward: propagate the same scalar gradient to all elements.
    node->setBackwardFn([x](const Matrix& grad_out) {
        // grad_out is 1x1
        double g = grad_out.data[0];
        const Matrix& xv = x->value();
        Matrix gx(xv.rows, xv.cols);
        std::fill(gx.data.begin(), gx.data.end(), g);
        x->addGrad(gx);
    });

    return node;
}


/**
 * @brief Computes the mean of all elements of the input node.
 *
 * Forward:
 *   out = (1/N) * ∑_i x_i
 *
 * Backward:
 *   dL/dx_i = grad_out / N (same value for all elements).
 *
 * @param x Input node.
 * @return A new Node storing the scalar mean (1x1 matrix).
 */
Node::Ptr mean(const Node::Ptr& x) {
    const Matrix& xv = x->value();
    std::size_t N = xv.data.size();

    // Forward: compute arithmetic mean of all elements
    double s = 0.0;
    for (double v : xv.data) s += v;
    double m = (N > 0 ? s / static_cast<double>(N) : 0.0);
    Matrix out(1, 1);
    out.data[0] = m;

    auto node = std::make_shared<Node>(out);
    node->addParent(x);

    // Backward: each element receives grad_out / N
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

/**
 * @brief Applies the ReLU activation function element-wise.
 *
 * Forward:
 *   out_i = max(0, x_i)
 *
 * Backward:
 *   dL/dx_i = grad_i if x_i > 0, else 0.
 *
 * @param x Input node.
 * @return A new OperationNode representing ReLU(x).
 */
Node::Ptr relu(const Node::Ptr& x) {
    Matrix out = x->value();
    for (size_t i = 0; i < out.data.size(); ++i)
        out.data[i] = std::max(0.0, out.data[i]);

    auto node = std::make_shared<OperationNode>(OpKind::RELU, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x,out](const Matrix& grad){
        Matrix gx(grad.rows, grad.cols);
        // Gradient flows only where the ReLU output was strictly positive
        for (size_t i = 0; i < out.data.size(); ++i)
            gx.data[i] = (out.data[i] > 0) ? grad.data[i] : 0.0;
        x->addGrad(gx);
    });

    return node;
}

/**
 * @brief Applies the LeakyReLU activation function element-wise.
 *
 * Forward:
 *   out_i = x_i if x_i > 0 else alpha * x_i
 *
 * Backward:
 *   dL/dx_i = grad_i * (1 if x_i > 0 else alpha)
 *
 * @param x Input node.
 * @param alpha Negative slope coefficient.
 * @return A new OperationNode representing LeakyReLU(x).
 */
Node::Ptr leaky_relu(const Node::Ptr& x, double alpha) {
    Matrix in = x->value();
    Matrix out = in;
    for (size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = (out.data[i] > 0.0) ? out.data[i] : (alpha * out.data[i]);
    }

    auto node = std::make_shared<OperationNode>(OpKind::LEAKY_RELU, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x, in, alpha](const Matrix& grad) {
        Matrix gx(grad.rows, grad.cols);
        for (size_t i = 0; i < in.data.size(); ++i) {
            const double slope = (in.data[i] > 0.0) ? 1.0 : alpha;
            gx.data[i] = grad.data[i] * slope;
        }
        x->addGrad(gx);
    });

    return node;
}

/**
 * @brief Applies the GELU activation element-wise using tanh approximation.
 *
 * Forward:
 *   out_i = 0.5 * x_i * (1 + tanh(k * (x_i + c * x_i^3)))
 *   where k = sqrt(2/pi), c = 0.044715
 *
 * Backward:
 *   dL/dx_i = grad_i * d(gelu_tanh_approx)/dx_i
 *
 * @param x Input node.
 * @return A new OperationNode representing GELU(x).
 */
Node::Ptr gelu(const Node::Ptr& x) {
    Matrix in = x->value();
    Matrix out = in;

    for (size_t i = 0; i < out.data.size(); ++i) {
        const double xi = out.data[i];
        const double xi3 = xi * xi * xi;
        const double u = kSqrt2OverPi * (xi + kGeluCubicCoeff * xi3);
        const double t = std::tanh(u);
        out.data[i] = 0.5 * xi * (1.0 + t);
    }

    auto node = std::make_shared<OperationNode>(OpKind::GELU, out, std::vector<Node::Ptr>{x});

    node->setBackwardFn([x, in](const Matrix& grad) {
        Matrix gx(grad.rows, grad.cols);
        for (size_t i = 0; i < in.data.size(); ++i) {
            const double xi = in.data[i];
            const double xi2 = xi * xi;
            const double xi3 = xi2 * xi;
            const double u = kSqrt2OverPi * (xi + kGeluCubicCoeff * xi3);
            const double t = std::tanh(u);
            const double sech2 = 1.0 - t * t;
            const double du_dx = kSqrt2OverPi * (1.0 + 3.0 * kGeluCubicCoeff * xi2);
            const double dy_dx = 0.5 * (1.0 + t) + 0.5 * xi * sech2 * du_dx;
            gx.data[i] = grad.data[i] * dy_dx;
        }
        x->addGrad(gx);
    });

    return node;
}

/**
 * @brief Applies the sigmoid activation function element-wise.
 *
 * Forward:
 *   out_i = 1 / (1 + exp(-x_i))
 *
 * Backward:
 *   dL/dx_i = grad_i * out_i * (1 - out_i)
 *   (using the identity σ'(x) = σ(x) * (1 - σ(x)) ).
 *
 * @param x Input node.
 * @return A new OperationNode representing sigmoid(x).
 */
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

/**
 * @brief Applies the hyperbolic tangent activation function element-wise.
 *
 * Forward:
 *   out_i = tanh(x_i)
 *
 * Backward:
 *   dL/dx_i = grad_i * (1 - out_i^2)
 *   (using the identity (tanh x)' = 1 - tanh^2 x ).
 *
 * @param x Input node.
 * @return A new OperationNode representing tanh(x).
 */
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
