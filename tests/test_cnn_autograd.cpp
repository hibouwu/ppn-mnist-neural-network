#include "autograd/engine.hpp"
#include "cnn_network.hpp"
#include "conv2d_layer.hpp"
#include "loss.hpp"
#include "math_ops.hpp"
#include "maxpool2d_layer.hpp"
#include "network.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

bool almostEqual(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) <= eps;
}

double scalarValue(const Node::Ptr& node) {
    return node->value().data[0];
}

void test_maxpool_scatter_correctness() {
    MaxPool2DLayer pool(2, 2, 2);
    Matrix input(1, 4);
    input.data = {1.0, 2.0, 3.0, 4.0};
    auto x = std::make_shared<Node>(input);

    auto out = pool.forward(x, 1, 1, 2, 2);
    out->backward();

    assert(almostEqual(x->grad().data[0], 0.0));
    assert(almostEqual(x->grad().data[1], 0.0));
    assert(almostEqual(x->grad().data[2], 0.0));
    assert(almostEqual(x->grad().data[3], 1.0));
}

void test_maxpool_metadata() {
    MaxPool2DLayer pool(2, 2, 2);
    Matrix input(1, 4);
    input.data = {1.0, 2.0, 3.0, 4.0};
    auto x = constant(input);

    auto out = pool.forward(x, 1, 1, 2, 2);
    assert(!out->requiresGrad());
    assert(!out->gradFn());
    assert(!out->backwardContext());
    assert(out->inputs().empty());
    out->backward();
    assert(!x->hasAllocatedGrad());
}

Node::Ptr build_maxpool_sum(const Matrix& in) {
    MaxPool2DLayer pool(2, 2, 2);
    auto x = std::make_shared<Node>(in);
    return MathOps::sum(pool.forward(x, 1, 1, 2, 2));
}

void test_maxpool_numerical_gradient() {
    Matrix input(1, 4);
    input.data = {1.0, 2.0, 3.0, 4.0};
    auto x = std::make_shared<Node>(input);
    auto y = MathOps::sum(MaxPool2DLayer(2, 2, 2).forward(x, 1, 1, 2, 2));
    y->backward();

    const double eps = 1e-5;
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        Matrix xp = input;
        Matrix xn = input;
        xp.data[i] += eps;
        xn.data[i] -= eps;
        const double fp = scalarValue(build_maxpool_sum(xp));
        const double fn = scalarValue(build_maxpool_sum(xn));
        const double num = (fp - fn) / (2.0 * eps);
        assert(almostEqual(x->grad().data[i], num, 1e-4));
    }
}

Node::Ptr build_conv_sum(const Matrix& input,
                         const Matrix& kernel,
                         const Matrix& bias) {
    Conv2DLayer conv(1, 1, 2, 2, 1, 0);
    conv.parameters()[0]->grad();
    Matrix& k = const_cast<Matrix&>(conv.parameters()[0]->value());
    Matrix& b = const_cast<Matrix&>(conv.parameters()[1]->value());
    k = kernel;
    b = bias;

    auto x = std::make_shared<Node>(input);
    return MathOps::sum(conv.forward(x, 1, 1, 4, 4));
}

void test_conv_metadata() {
    Conv2DLayer conv(1, 1, 2, 2, 1, 0);
    Matrix input(1, 16, 0.5);
    for (const auto& param : conv.parameters()) {
        param->setRequiresGrad(false);
    }
    auto x = constant(input);
    auto out = conv.forward(x, 1, 1, 4, 4);
    assert(!out->requiresGrad());
    assert(!out->gradFn());
    assert(!out->backwardContext());
    assert(out->inputs().empty());
    out->backward();
    assert(!x->hasAllocatedGrad());
}

void test_conv_parameter_contract() {
    Conv2DLayer conv(2, 3, 2, 2, 1, 0);
    auto params = conv.parameters();

    assert(params.size() == 2);

    const auto& kernels = params[0];
    const auto& bias = params[1];

    assert(kernels->value().rows == 3);
    assert(kernels->value().cols == 2 * 2 * 2);
    assert(bias->value().rows == 1);
    assert(bias->value().cols == 3);

    assert(kernels->isParameter());
    assert(bias->isParameter());
    assert(kernels->isLeaf());
    assert(bias->isLeaf());
    assert(kernels->inputs().empty());
    assert(bias->inputs().empty());
    assert(kernels->requiresGrad());
    assert(bias->requiresGrad());
}

void test_conv_backward_shapes_and_finite_grads() {
    Conv2DLayer conv(1, 1, 2, 2, 1, 0);
    Matrix input(1, 16);
    input.data = {
        0.2, -0.1, 0.3, 0.0,
        0.4,  0.5, -0.2, 0.1,
       -0.3,  0.6, 0.7, -0.4,
        0.2, -0.5, 0.8, 0.9
    };
    auto x = std::make_shared<Node>(input);
    auto out = conv.forward(x, 1, 1, 4, 4);
    auto loss = MathOps::sum(out);
    loss->backward();

    assert(x->grad().rows == 1 && x->grad().cols == 16);
    auto params = conv.parameters();
    assert(params[0]->grad().rows == 1 && params[0]->grad().cols == 4);
    assert(params[1]->grad().rows == 1 && params[1]->grad().cols == 1);

    bool any_nonzero = false;
    for (double v : params[0]->grad().data) {
        assert(std::isfinite(v));
        any_nonzero = any_nonzero || std::abs(v) > 1e-12;
    }
    for (double v : params[1]->grad().data) {
        assert(std::isfinite(v));
        any_nonzero = any_nonzero || std::abs(v) > 1e-12;
    }
    for (double v : x->grad().data) {
        assert(std::isfinite(v));
    }
    assert(any_nonzero);
}

void test_conv_numerical_gradient() {
    Matrix input(1, 16);
    input.data = {
        0.2, -0.1, 0.3, 0.0,
        0.4,  0.5, -0.2, 0.1,
       -0.3,  0.6, 0.7, -0.4,
        0.2, -0.5, 0.8, 0.9
    };
    Matrix kernel(1, 4);
    kernel.data = {0.1, -0.2, 0.3, 0.4};
    Matrix bias(1, 1);
    bias.data = {0.05};

    Conv2DLayer conv(1, 1, 2, 2, 1, 0);
    Matrix& k = const_cast<Matrix&>(conv.parameters()[0]->value());
    Matrix& b = const_cast<Matrix&>(conv.parameters()[1]->value());
    k = kernel;
    b = bias;

    auto x = std::make_shared<Node>(input);
    auto y = MathOps::sum(conv.forward(x, 1, 1, 4, 4));
    y->backward();

    const double eps = 1e-5;
    {
        Matrix xp = input;
        Matrix xn = input;
        xp.data[0] += eps;
        xn.data[0] -= eps;
        const double fp = scalarValue(build_conv_sum(xp, kernel, bias));
        const double fn = scalarValue(build_conv_sum(xn, kernel, bias));
        const double num = (fp - fn) / (2.0 * eps);
        assert(almostEqual(x->grad().data[0], num, 1e-4));
    }
    {
        Matrix kp = kernel;
        Matrix kn = kernel;
        kp.data[0] += eps;
        kn.data[0] -= eps;
        const double fp = scalarValue(build_conv_sum(input, kp, bias));
        const double fn = scalarValue(build_conv_sum(input, kn, bias));
        const double num = (fp - fn) / (2.0 * eps);
        assert(almostEqual(conv.parameters()[0]->grad().data[0], num, 1e-4));
    }
}

void test_cnn_main_path_and_parameter_ready() {
    CNNConfig cfg;
    cfg.input_channels = 1;
    cfg.input_height = 4;
    cfg.input_width = 4;
    cfg.conv_channels = {1};
    cfg.conv_kernels = {2};
    cfg.conv_strides = {1};
    cfg.conv_paddings = {0};
    cfg.pool_after = {true};
    cfg.pool_kernels = {2};
    cfg.pool_strides = {2};
    cfg.fc_hidden_sizes = {2};
    cfg.num_classes = 2;

    CNNNetwork net(cfg, 42);
    Matrix input(1, 16);
    input.data = {
        0.2, -0.1, 0.3, 0.0,
        0.4,  0.5, -0.2, 0.1,
       -0.3,  0.6, 0.7, -0.4,
        0.2, -0.5, 0.8, 0.9
    };
    Matrix target(1, 2, 0.0);
    target(0, 1) = 1.0;

    auto x = constant(input);
    auto y = constant(target);
    auto logits = net.forward(x);
    auto loss = CrossEntropyLoss().forward(logits, y);

    int ready_count = 0;
    AutogradEngine engine;
    engine.setParameterReadyHook([&ready_count](Node& node) {
        (void)node;
        ready_count += 1;
    });
    engine.backward(loss);

    auto params = net.getParameters();
    assert(ready_count == static_cast<int>(params.size()));
    for (const auto& p : params) {
        assert(p->hasAllocatedGrad());
        bool finite = true;
        for (double v : p->grad().data) {
            finite = finite && std::isfinite(v);
        }
        assert(finite);
    }
}

}

int main() {
    test_maxpool_scatter_correctness();
    test_maxpool_metadata();
    test_maxpool_numerical_gradient();
    test_conv_metadata();
    test_conv_parameter_contract();
    test_conv_backward_shapes_and_finite_grads();
    test_conv_numerical_gradient();
    test_cnn_main_path_and_parameter_ready();
    std::cout << "CNN autograd tests passed!" << std::endl;
    return 0;
}
