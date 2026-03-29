#include "conv2d_layer.hpp"
#include "autograd/engine.hpp"
#include "cnn_network.hpp"
#include "loss.hpp"
#include "math_ops.hpp"
#include "optimizer.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

bool almostEqual(double a, double b, double eps = 1e-4) {
    return std::abs(a - b) <= eps;
}

void assertMatrixNear(const Matrix& actual, const Matrix& expected, double eps = 1e-4) {
    assert(actual.rows == expected.rows);
    assert(actual.cols == expected.cols);
    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        assert(almostEqual(actual.data[i], expected.data[i], eps));
    }
}

Matrix makeConvInput() {
    Matrix input(2, 2 * 4 * 5);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.05 * static_cast<double>(i + 1);
    }
    return input;
}

Matrix makeConvWeights() {
    Matrix weights(3, 2 * 3 * 2);
    for (std::size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = -0.1 + 0.01 * static_cast<double>(i + 1);
    }
    return weights;
}

Matrix makeConvBias() {
    Matrix bias(1, 3);
    bias.data = {-0.2, 0.05, 0.3};
    return bias;
}

Matrix makeConvGradOutput() {
    Matrix grad_output(2, 3 * 4 * 6);
    for (std::size_t i = 0; i < grad_output.data.size(); ++i) {
        grad_output.data[i] = 0.02 * static_cast<double>((i % 17) + 1);
    }
    return grad_output;
}

void loadConvParameters(Conv2DLayer& conv, const Matrix& weights, const Matrix& bias) {
    auto params = conv.parameters();
    Matrix& kernels_value = const_cast<Matrix&>(params[0]->value());
    Matrix& bias_value = const_cast<Matrix&>(params[1]->value());
    kernels_value = weights;
    bias_value = bias;
}

void test_onednn_parameter_contract() {
    Conv2DLayer conv(2, 3, 2, 2, 1, 0, ConvBackend::OneDnn);
    auto params = conv.parameters();

    assert(params.size() == 2);
    assert(conv.backendKind() == ConvBackend::OneDnn);

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

void test_onednn_forward_backward_parity() {
    const Matrix input = makeConvInput();
    const Matrix weights = makeConvWeights();
    const Matrix bias = makeConvBias();
    const Matrix grad_output = makeConvGradOutput();

    Conv2DLayer reference(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    Conv2DLayer onednn(2, 3, 3, 2, 1, 1, ConvBackend::OneDnn);
    loadConvParameters(reference, weights, bias);
    loadConvParameters(onednn, weights, bias);

    auto x_ref = std::make_shared<Node>(input);
    auto out_ref = reference.forward(x_ref, 2, 2, 4, 5);
    out_ref->addGrad(grad_output);
    AutogradEngine ref_engine;
    ref_engine.backward(out_ref);

    auto x_dnn = std::make_shared<Node>(input);
    auto out_dnn = onednn.forward(x_dnn, 2, 2, 4, 5);
    out_dnn->addGrad(grad_output);
    AutogradEngine dnn_engine;
    dnn_engine.backward(out_dnn);

    assertMatrixNear(out_dnn->value(), out_ref->value());
    assertMatrixNear(x_dnn->grad(), x_ref->grad());
    assertMatrixNear(onednn.parameters()[0]->grad(), reference.parameters()[0]->grad());
    assertMatrixNear(onednn.parameters()[1]->grad(), reference.parameters()[1]->grad());
}

void test_onednn_optimizer_step_parity() {
    const Matrix input = makeConvInput();
    const Matrix weights = makeConvWeights();
    const Matrix bias = makeConvBias();

    Conv2DLayer reference(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    Conv2DLayer onednn(2, 3, 3, 2, 1, 1, ConvBackend::OneDnn);
    loadConvParameters(reference, weights, bias);
    loadConvParameters(onednn, weights, bias);

    auto x_ref = std::make_shared<Node>(input);
    auto loss_ref = MathOps::sum(reference.forward(x_ref, 2, 2, 4, 5));
    AutogradEngine ref_engine;
    ref_engine.backward(loss_ref);

    auto x_dnn = std::make_shared<Node>(input);
    auto loss_dnn = MathOps::sum(onednn.forward(x_dnn, 2, 2, 4, 5));
    AutogradEngine dnn_engine;
    dnn_engine.backward(loss_dnn);

    SGDOptimizer opt_ref(reference.parameters(), 0.01);
    SGDOptimizer opt_dnn(onednn.parameters(), 0.01);
    opt_ref.step();
    opt_dnn.step();

    assertMatrixNear(onednn.parameters()[0]->value(), reference.parameters()[0]->value());
    assertMatrixNear(onednn.parameters()[1]->value(), reference.parameters()[1]->value());
}

void test_onednn_cnn_parameter_ready_hook() {
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
    cfg.conv_backend = ConvBackend::OneDnn;

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
        assert(node.isParameter());
        ready_count += 1;
    });
    engine.backward(loss);

    const auto params = net.getParameters();
    assert(ready_count == static_cast<int>(params.size()));
    for (const auto& p : params) {
        assert(p->hasAllocatedGrad());
    }
}

} // namespace

int main() {
#if !PPN_HAVE_ONEDNN_CONV_BACKEND
    std::cout << "Conv2D oneDNN backend tests skipped: binary built without oneDNN backend support." << std::endl;
    return 0;
#else
    test_onednn_parameter_contract();
    test_onednn_forward_backward_parity();
    test_onednn_optimizer_step_parity();
    test_onednn_cnn_parameter_ready_hook();
    std::cout << "Conv2D oneDNN backend tests passed!" << std::endl;
    return 0;
#endif
}
