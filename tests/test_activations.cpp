#include "activation.hpp"
#include "math_ops.hpp"
#include "node.hpp"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kGeluCubicCoeff = 0.044715;

void assertMatrixClose(const Matrix& got,
                       const std::vector<double>& expected,
                       double tol,
                       const std::string& name) {
    assert(got.data.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double diff = std::abs(got.data[i] - expected[i]);
        if (diff > tol) {
            std::cerr << name << " mismatch at index " << i
                      << ": got=" << got.data[i]
                      << ", expected=" << expected[i]
                      << ", diff=" << diff << std::endl;
            std::exit(1);
        }
    }
}

double geluApprox(double x) {
    const double x3 = x * x * x;
    const double u = kSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    return 0.5 * x * (1.0 + std::tanh(u));
}

double geluApproxDerivative(double x) {
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double u = kSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    const double t = std::tanh(u);
    const double sech2 = 1.0 - t * t;
    const double du_dx = kSqrt2OverPi * (1.0 + 3.0 * kGeluCubicCoeff * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx;
}

} // namespace

int main() {
    Matrix input_val(2, 3);
    input_val(0, 0) = -1.0; input_val(0, 1) = 0.0; input_val(0, 2) = 1.0;
    input_val(1, 0) = -2.0; input_val(1, 1) = 0.5; input_val(1, 2) = 2.0;

    // ReLU
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        ReLU relu;
        Node::Ptr out_relu = relu.forward(input);
        assertMatrixClose(out_relu->value(), {0.0, 0.0, 1.0, 0.0, 0.5, 2.0}, 1e-6, "ReLU forward");

        MathOps::sum(out_relu)->backward();
        assertMatrixClose(input->grad(), {0.0, 0.0, 1.0, 0.0, 1.0, 1.0}, 1e-6, "ReLU backward");
    }

    // LeakyReLU (alpha=0.01)
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        LeakyReLU leaky_relu;
        Node::Ptr out = leaky_relu.forward(input);
        assertMatrixClose(out->value(), {-0.01, 0.0, 1.0, -0.02, 0.5, 2.0}, 1e-6, "LeakyReLU forward");

        MathOps::sum(out)->backward();
        assertMatrixClose(input->grad(), {0.01, 0.01, 1.0, 0.01, 1.0, 1.0}, 1e-6, "LeakyReLU backward");
    }

    // Sigmoid
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        Sigmoid sigmoid;
        Node::Ptr out_sigmoid = sigmoid.forward(input);
        std::vector<double> expected_forward;
        std::vector<double> expected_backward;
        expected_forward.reserve(input_val.data.size());
        expected_backward.reserve(input_val.data.size());
        for (double x : input_val.data) {
            const double s = 1.0 / (1.0 + std::exp(-x));
            expected_forward.push_back(s);
            expected_backward.push_back(s * (1.0 - s));
        }
        assertMatrixClose(out_sigmoid->value(), expected_forward, 1e-6, "Sigmoid forward");

        MathOps::sum(out_sigmoid)->backward();
        assertMatrixClose(input->grad(), expected_backward, 1e-6, "Sigmoid backward");
    }

    // Tanh
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        Tanh tanh;
        Node::Ptr out_tanh = tanh.forward(input);
        std::vector<double> expected_forward;
        std::vector<double> expected_backward;
        expected_forward.reserve(input_val.data.size());
        expected_backward.reserve(input_val.data.size());
        for (double x : input_val.data) {
            const double t = std::tanh(x);
            expected_forward.push_back(t);
            expected_backward.push_back(1.0 - t * t);
        }
        assertMatrixClose(out_tanh->value(), expected_forward, 1e-6, "Tanh forward");

        MathOps::sum(out_tanh)->backward();
        assertMatrixClose(input->grad(), expected_backward, 1e-6, "Tanh backward");
    }

    // GELU (tanh approximation)
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        GELU gelu;
        Node::Ptr out = gelu.forward(input);
        std::vector<double> expected_forward;
        std::vector<double> expected_backward;
        expected_forward.reserve(input_val.data.size());
        expected_backward.reserve(input_val.data.size());
        for (double x : input_val.data) {
            expected_forward.push_back(geluApprox(x));
            expected_backward.push_back(geluApproxDerivative(x));
        }
        assertMatrixClose(out->value(), expected_forward, 1e-5, "GELU forward");

        MathOps::sum(out)->backward();
        assertMatrixClose(input->grad(), expected_backward, 1e-5, "GELU backward");
    }

    std::cout << "All activation tests passed!\n";
    return 0;
}
