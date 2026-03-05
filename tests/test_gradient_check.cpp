#include "node.hpp"
#include "math_ops.hpp"
#include "activation.hpp"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <vector>

// Numerical gradient check
// f: function that takes a Matrix (input) and returns a Node::Ptr (output scalar usually, but here we sum it)
// x: input node
// epsilon: small perturbation
// returns: numerical gradient matrix
Matrix compute_numerical_gradient(std::function<Node::Ptr(Node::Ptr)> f, Node::Ptr x, double epsilon = 1e-4) {
    Matrix grad_num(x->value().rows, x->value().cols);
    Matrix original_val = x->value();

    for (size_t i = 0; i < original_val.data.size(); ++i) {
        // f(x + eps)
        Matrix x_plus = original_val;
        x_plus.data[i] += epsilon;
        x = std::make_shared<Node>(x_plus); // Reset node with new value
        Node::Ptr y_plus = f(x);
        double loss_plus = y_plus->value().data[0]; // Assume scalar output for simplicity or sum

        // f(x - eps)
        Matrix x_minus = original_val;
        x_minus.data[i] -= epsilon;
        x = std::make_shared<Node>(x_minus);
        Node::Ptr y_minus = f(x);
        double loss_minus = y_minus->value().data[0];

        // Central difference
        grad_num.data[i] = (loss_plus - loss_minus) / (2 * epsilon);
    }
    
    return grad_num;
}

bool check_gradient(const Matrix& grad_auto, const Matrix& grad_num, double tol = 1e-3) {
    for (size_t i = 0; i < grad_auto.data.size(); ++i) {
        double diff = std::abs(grad_auto.data[i] - grad_num.data[i]);
        double max_val = std::max(std::abs(grad_auto.data[i]), std::abs(grad_num.data[i]));
        double rel_err = (max_val > 0) ? diff / max_val : diff;
        
        if (rel_err > tol && diff > tol) {
            std::cerr << "Gradient mismatch at index " << i << ": auto=" << grad_auto.data[i] << ", num=" << grad_num.data[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

void test_relu_gradient() {
    std::cout << "Testing ReLU Gradient..." << std::endl;
    Matrix val(2, 2);
    val.data = {-1.0, 0.5, 2.0, -0.5};
    Node::Ptr x = std::make_shared<Node>(val);

    // Function: y = sum(relu(x))
    auto func = [](Node::Ptr input) -> Node::Ptr {
        return MathOps::sum(MathOps::relu(input));
    };

    // Autodiff
    Node::Ptr y = func(x);
    y->backward();
    Matrix grad_auto = x->grad();

    // Numerical
    Matrix grad_num = compute_numerical_gradient(func, x);

    if (check_gradient(grad_auto, grad_num)) {
        std::cout << "ReLU Gradient Passed!" << std::endl;
    } else {
        std::cout << "ReLU Gradient Failed!" << std::endl;
        exit(1);
    }
}

void test_matmul_gradient() {
    std::cout << "Testing Matmul Gradient..." << std::endl;
    Matrix val_a(2, 3);
    val_a.randomInit();
    Node::Ptr a = std::make_shared<Node>(val_a);

    Matrix val_b(3, 2);
    val_b.randomInit();
    Node::Ptr b = std::make_shared<Node>(val_b);

    // Function: y = sum(a @ b)
    // We only check gradient w.r.t 'a' here for simplicity, but 'b' works similarly
    auto func_a = [&](Node::Ptr input_a) -> Node::Ptr {
        // Re-create b to ensure graph is fresh if needed, but here b is constant w.r.t a
        return MathOps::sum(MathOps::matmul(input_a, b));
    };

    // Autodiff
    Node::Ptr y = MathOps::sum(MathOps::matmul(a, b));
    y->backward();
    Matrix grad_auto_a = a->grad();

    // Numerical
    Matrix grad_num_a = compute_numerical_gradient(func_a, a);

    if (check_gradient(grad_auto_a, grad_num_a)) {
        std::cout << "Matmul Gradient (w.r.t A) Passed!" << std::endl;
    } else {
        std::cout << "Matmul Gradient (w.r.t A) Failed!" << std::endl;
        exit(1);
    }
}

void test_leaky_relu_gradient() {
    std::cout << "Testing LeakyReLU Gradient..." << std::endl;
    Matrix val(2, 2);
    val.data = {-1.2, 0.7, 2.1, -0.3};
    Node::Ptr x = std::make_shared<Node>(val);

    auto func = [](Node::Ptr input) -> Node::Ptr {
        return MathOps::sum(MathOps::leaky_relu(input, 0.01));
    };

    Node::Ptr y = func(x);
    y->backward();
    Matrix grad_auto = x->grad();

    Matrix grad_num = compute_numerical_gradient(func, x);

    if (check_gradient(grad_auto, grad_num)) {
        std::cout << "LeakyReLU Gradient Passed!" << std::endl;
    } else {
        std::cout << "LeakyReLU Gradient Failed!" << std::endl;
        exit(1);
    }
}

void test_gelu_gradient() {
    std::cout << "Testing GELU Gradient..." << std::endl;
    Matrix val(2, 2);
    val.data = {-1.0, 0.2, 1.5, -0.4};
    Node::Ptr x = std::make_shared<Node>(val);

    auto func = [](Node::Ptr input) -> Node::Ptr {
        return MathOps::sum(MathOps::gelu(input));
    };

    Node::Ptr y = func(x);
    y->backward();
    Matrix grad_auto = x->grad();

    Matrix grad_num = compute_numerical_gradient(func, x);

    if (check_gradient(grad_auto, grad_num)) {
        std::cout << "GELU Gradient Passed!" << std::endl;
    } else {
        std::cout << "GELU Gradient Failed!" << std::endl;
        exit(1);
    }
}

int main() {
    test_relu_gradient();
    test_leaky_relu_gradient();
    test_gelu_gradient();
    test_matmul_gradient();
    std::cout << "All gradient checks passed!" << std::endl;
    return 0;
}
