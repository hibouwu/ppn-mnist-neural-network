/**
 * @file test_cnn_configurable.cpp
 * @brief Tests for CNNConfig and CNNNetwork.
 */
#include "cnn_network.hpp"
#include "node.hpp"
#include "loss.hpp"
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdexcept>

static void testLeNet5Default() {
    std::cout << "  testLeNet5Default... ";
    CNNConfig cfg = CNNConfig::lenet5();
    CNNNetwork net(cfg, 42);

    // Input: batch=2, 784 features
    Matrix input_data(2, 784);
    input_data.randomInit(-0.5, 0.5, false, 42);
    auto x = constant(input_data);

    auto out = net.forward(x);
    assert(out->value().rows == 2);
    assert(out->value().cols == 10);
    std::cout << "PASSED" << std::endl;
}

static void testSingleStage() {
    std::cout << "  testSingleStage... ";
    CNNConfig cfg;
    cfg.input_channels = 3;
    cfg.input_height = 64;
    cfg.input_width = 64;
    cfg.conv_channels = {8};
    cfg.conv_kernels = {3};
    cfg.conv_strides = {1};
    cfg.conv_paddings = {1};
    cfg.pool_after = {true};
    cfg.pool_kernels = {2};
    cfg.pool_strides = {2};
    cfg.fc_hidden_sizes = {64};

    CNNNetwork net(cfg, 42);

    Matrix input_data(4, 64 * 64 * 3);
    input_data.randomInit(-0.5, 0.5, false, 42);
    auto x = constant(input_data);

    auto out = net.forward(x);
    assert(out->value().rows == 4);
    assert(out->value().cols == 10);
    std::cout << "PASSED" << std::endl;
}

static void testMultiStage() {
    std::cout << "  testMultiStage... ";
    CNNConfig cfg;
    cfg.input_channels = 3;
    cfg.input_height = 64;
    cfg.input_width = 64;
    cfg.conv_channels = {6, 16, 32};
    cfg.conv_kernels = {5, 3, 3};
    cfg.conv_paddings = {2, 1, 1};
    cfg.pool_after = {true, true, false};
    cfg.fc_hidden_sizes = {256, 128};

    CNNNetwork net(cfg, 42);

    Matrix input_data(2, 64 * 64 * 3);
    input_data.randomInit(-0.5, 0.5, false, 42);
    auto x = constant(input_data);

    auto out = net.forward(x);
    assert(out->value().rows == 2);
    assert(out->value().cols == 10);
    std::cout << "PASSED" << std::endl;
}

static void testBroadcastDefaults() {
    std::cout << "  testBroadcastDefaults... ";
    CNNConfig cfg;
    cfg.conv_channels = {8, 16, 32};
    // Leave others empty → should expand to defaults (kernel=3, stride=1, padding=1, etc.)
    cfg.expandDefaults();

    assert(cfg.conv_kernels.size() == 3);
    assert(cfg.conv_kernels[0] == 3);
    assert(cfg.conv_strides.size() == 3);
    assert(cfg.conv_strides[0] == 1);
    assert(cfg.conv_paddings.size() == 3);
    assert(cfg.conv_paddings[0] == 1);
    assert(cfg.pool_after.size() == 3);
    assert(cfg.pool_after[0] == false);

    cfg.validate();
    std::cout << "PASSED" << std::endl;
}

static void testSingleValueBroadcast() {
    std::cout << "  testSingleValueBroadcast... ";
    CNNConfig cfg;
    cfg.conv_channels = {8, 16};
    cfg.conv_kernels = {5};   // single value → broadcast to 2
    cfg.conv_paddings = {2};  // single value → broadcast to 2
    cfg.expandDefaults();

    assert(cfg.conv_kernels.size() == 2);
    assert(cfg.conv_kernels[0] == 5);
    assert(cfg.conv_kernels[1] == 5);
    assert(cfg.conv_paddings[0] == 2);
    assert(cfg.conv_paddings[1] == 2);
    std::cout << "PASSED" << std::endl;
}

static void testLengthMismatch() {
    std::cout << "  testLengthMismatch... ";
    CNNConfig cfg;
    cfg.conv_channels = {8, 16, 32};
    cfg.conv_kernels = {5, 3};  // 2 != 3 → should throw
    bool threw = false;
    try {
        cfg.expandDefaults();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "PASSED" << std::endl;
}

static void testInvalidShape() {
    std::cout << "  testInvalidShape... ";
    CNNConfig cfg;
    cfg.input_channels = 3;
    cfg.input_height = 64;
    cfg.input_width = 64;
    cfg.conv_channels = {8};
    cfg.conv_kernels = {99};   // kernel too large for 64x64
    cfg.conv_paddings = {0};
    cfg.conv_strides = {1};

    bool threw = false;
    try {
        CNNNetwork net(cfg, 42);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "PASSED" << std::endl;
}

static void testBackwardGradients() {
    std::cout << "  testBackwardGradients... ";
    CNNConfig cfg;
    cfg.input_channels = 3;
    cfg.input_height = 64;
    cfg.input_width = 64;
    cfg.conv_channels = {4};
    cfg.conv_kernels = {3};
    cfg.conv_paddings = {1};
    cfg.pool_after = {true};
    cfg.fc_hidden_sizes = {32};

    CNNNetwork net(cfg, 42);

    Matrix input_data(2, 64 * 64 * 3);
    input_data.randomInit(-0.5, 0.5, false, 42);
    auto x = constant(input_data);

    // One-hot targets
    Matrix targets(2, 10, 0.0);
    targets(0, 3) = 1.0;
    targets(1, 7) = 1.0;
    auto y = constant(targets);

    auto pred = net.forward(x);
    CrossEntropyLoss loss;
    auto loss_node = loss.forward(pred, y);

    loss_node->backward();

    // Check that parameters have non-zero gradients
    auto params = net.getParameters();
    bool any_nonzero = false;
    for (const auto& p : params) {
        const Matrix& g = p->grad();
        for (size_t i = 0; i < g.rows; ++i) {
            for (size_t j = 0; j < g.cols; ++j) {
                if (std::abs(g(i, j)) > 1e-12) {
                    any_nonzero = true;
                    break;
                }
            }
            if (any_nonzero) break;
        }
        if (any_nonzero) break;
    }
    assert(any_nonzero);

    // Check all gradients are finite
    for (const auto& p : params) {
        const Matrix& g = p->grad();
        for (size_t i = 0; i < g.rows; ++i) {
            for (size_t j = 0; j < g.cols; ++j) {
                assert(std::isfinite(g(i, j)));
            }
        }
    }

    std::cout << "PASSED" << std::endl;
}

static void testPoolKernelZeroRejected() {
    std::cout << "  testPoolKernelZeroRejected... ";
    CNNConfig cfg;
    cfg.conv_channels = {8};
    cfg.conv_kernels = {3};
    cfg.conv_paddings = {1};
    cfg.pool_after = {false};
    cfg.pool_kernels = {0};   // invalid even when pool_after=false
    cfg.pool_strides = {2};

    bool threw = false;
    try {
        cfg.expandDefaults();
        cfg.validate();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running CNN configurable tests..." << std::endl;

    testLeNet5Default();
    testSingleStage();
    testMultiStage();
    testBroadcastDefaults();
    testSingleValueBroadcast();
    testLengthMismatch();
    testInvalidShape();
    testBackwardGradients();
    testPoolKernelZeroRejected();

    std::cout << "\nAll CNN configurable tests PASSED!" << std::endl;
    return 0;
}
