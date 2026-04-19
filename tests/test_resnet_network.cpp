/**
 * @file test_resnet_network.cpp
 * @brief Tests for ResNetNetwork.
 */
#include "resnet_network.hpp"
#include "loss.hpp"
#include "node.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void testForwardMnistShape() {
    std::cout << "  testForwardMnistShape... ";
    ResNetConfig cfg;
    cfg.input_channels = 1;
    cfg.input_height = 28;
    cfg.input_width = 28;
    cfg.stem_channels = 4;
    cfg.stage_channels = {4, 8};
    cfg.blocks_per_stage = {1, 1};
    cfg.num_classes = 10;

    ResNetNetwork net(cfg, 42);

    Matrix input_data(2, 28 * 28);
    input_data.randomInit(-0.5, 0.5, false, 42);
    auto x = constant(input_data);

    auto out = net.forward(x);
    assert(out->value().rows == 2);
    assert(out->value().cols == 10);
    assert(!net.getParameters().empty());
    std::cout << "PASSED" << std::endl;
}

static void testBackwardGradients() {
    std::cout << "  testBackwardGradients... ";
    ResNetConfig cfg;
    cfg.input_channels = 1;
    cfg.input_height = 28;
    cfg.input_width = 28;
    cfg.stem_channels = 4;
    cfg.stage_channels = {4, 8};
    cfg.blocks_per_stage = {1, 1};
    cfg.num_classes = 10;

    ResNetNetwork net(cfg, 7);

    Matrix input_data(2, 28 * 28);
    input_data.randomInit(-0.5, 0.5, false, 7);
    auto x = constant(input_data);

    Matrix targets(2, 10, 0.0);
    targets(0, 1) = 1.0;
    targets(1, 9) = 1.0;
    auto y = constant(targets);

    CrossEntropyLoss loss;
    auto pred = net.forward(x);
    auto loss_node = loss.forward(pred, y);
    loss_node->backward();

    bool any_nonzero = false;
    for (const auto& p : net.getParameters()) {
        const Matrix& g = p->grad();
        for (size_t i = 0; i < g.rows; ++i) {
            for (size_t j = 0; j < g.cols; ++j) {
                assert(std::isfinite(g(i, j)));
                if (std::abs(g(i, j)) > 1e-12) {
                    any_nonzero = true;
                }
            }
        }
    }
    if (!any_nonzero) {
        throw std::runtime_error("Expected at least one non-zero ResNet gradient.");
    }
    std::cout << "PASSED" << std::endl;
}

int main() {
    std::cout << "Running ResNet tests..." << std::endl;
    testForwardMnistShape();
    testBackwardGradients();
    std::cout << "\nAll ResNet tests PASSED!" << std::endl;
    return 0;
}
