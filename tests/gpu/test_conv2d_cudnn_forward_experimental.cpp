#include "conv2d_layer.hpp"
#include "autograd/backward_context.hpp"

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
#include <cuda_runtime.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool almostEqual(double a, double b, double eps = 1e-5) {
    return std::abs(a - b) <= eps;
}

void assertMatrixNear(const Matrix& actual, const Matrix& expected, double eps = 1e-5) {
    assert(actual.rows == expected.rows);
    assert(actual.cols == expected.cols);
    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        assert(almostEqual(actual.data[i], expected.data[i], eps));
    }
}

struct ScopedExperimentalGpuForward {
    explicit ScopedExperimentalGpuForward(bool enabled) {
        const char* current = std::getenv("PPN_EXPERIMENTAL_CUDNN_CONV_FORWARD");
        had_original_ = current != nullptr;
        if (had_original_) {
            original_value_ = current;
        }
        setenv("PPN_EXPERIMENTAL_CUDNN_CONV_FORWARD", enabled ? "1" : "0", 1);
    }

    ~ScopedExperimentalGpuForward() {
        if (had_original_) {
            setenv("PPN_EXPERIMENTAL_CUDNN_CONV_FORWARD", original_value_.c_str(), 1);
        } else {
            unsetenv("PPN_EXPERIMENTAL_CUDNN_CONV_FORWARD");
        }
    }

private:
    bool had_original_ = false;
    std::string original_value_;
};

Matrix makeInput() {
    Matrix input(2, 2 * 4 * 5);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.05 * static_cast<double>(i + 1);
    }
    return input;
}

Matrix makeWeights() {
    Matrix weights(3, 2 * 3 * 2);
    for (std::size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = -0.1 + 0.01 * static_cast<double>(i + 1);
    }
    return weights;
}

Matrix makeBias() {
    Matrix bias(1, 3);
    bias.data = {-0.2, 0.05, 0.3};
    return bias;
}

void loadConvParameters(Conv2DLayer& conv, const Matrix& weights, const Matrix& bias) {
    auto params = conv.parameters();
    Matrix& kernels_value = const_cast<Matrix&>(params[0]->value());
    Matrix& bias_value = const_cast<Matrix&>(params[1]->value());
    kernels_value = weights;
    bias_value = bias;
}

void disableConvParameterGrad(Conv2DLayer& conv) {
    for (const auto& param : conv.parameters()) {
        param->setRequiresGrad(false);
    }
}

bool hasExperimentalForwardMarker(const Node::Ptr& node) {
    return node &&
           node->backwardContext() &&
           !node->backwardContext()->flags.empty();
}

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
bool gpuRuntimeAvailable() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

void test_gpu_path_selection_and_parity() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    Conv2DLayer cpu_conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    Conv2DLayer gpu_conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    loadConvParameters(cpu_conv, weights, bias);
    loadConvParameters(gpu_conv, weights, bias);
    disableConvParameterGrad(cpu_conv);
    disableConvParameterGrad(gpu_conv);

    {
        ScopedExperimentalGpuForward guard(false);
        auto out = cpu_conv.forward(constant(input), 2, 2, 4, 5);
        assert(!out->requiresGrad());
        assert(!out->gradFn());
        assert(!out->backwardContext());
        assert(out->inputs().empty());
    }

    Matrix cpu_value = [&]() {
        ScopedExperimentalGpuForward guard(false);
        return cpu_conv.forward(constant(input), 2, 2, 4, 5)->value();
    }();

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    {
        ScopedExperimentalGpuForward guard(true);
        auto out = gpu_conv.forward(constant(input), 2, 2, 4, 5);
        assert(!out->requiresGrad());
        assert(!out->gradFn());
        assert(out->backwardContext());
        assert(out->inputs().empty());
        assert(hasExperimentalForwardMarker(out));
        assertMatrixNear(out->value(), cpu_value);
    }
}

void test_gpu_forward_output_shape_consistency() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    Conv2DLayer conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    loadConvParameters(conv, weights, bias);
    disableConvParameterGrad(conv);

    const auto [H_out, W_out] = conv.outputShape(4, 5);
    ScopedExperimentalGpuForward guard(true);
    auto out = conv.forward(constant(input), 2, 2, 4, 5);

    assert(out->value().rows == 2);
    assert(out->value().cols == 3 * H_out * W_out);
}

void test_gpu_forward_requires_grad_contract() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    Conv2DLayer cpu_conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    Conv2DLayer gpu_conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference);
    loadConvParameters(cpu_conv, weights, bias);
    loadConvParameters(gpu_conv, weights, bias);

    auto x = std::make_shared<Node>(input);
    auto x_gpu = std::make_shared<Node>(input);

    auto cpu_out = [&]() {
        ScopedExperimentalGpuForward guard(false);
        return cpu_conv.forward(x, 2, 2, 4, 5);
    }();

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    auto gpu_out = [&]() {
        ScopedExperimentalGpuForward guard(true);
        return gpu_conv.forward(x_gpu, 2, 2, 4, 5);
    }();

    assert(cpu_out->requiresGrad());
    assert(cpu_out->gradFn());
    assert(cpu_out->backwardContext());
    assert(!cpu_out->inputs().empty());

    assert(gpu_out->requiresGrad());
    assert(gpu_out->gradFn());
    assert(gpu_out->backwardContext());
    assert(!gpu_out->inputs().empty());
    assert(hasExperimentalForwardMarker(gpu_out));
    assertMatrixNear(gpu_out->value(), cpu_out->value());
}

}  // namespace

int main() {
#if !PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    std::cout << "Experimental cuDNN Conv2D forward tests skipped: binary built without cuDNN experimental forward support." << std::endl;
    return 0;
#else
    test_gpu_path_selection_and_parity();
    test_gpu_forward_output_shape_consistency();
    test_gpu_forward_requires_grad_contract();
    std::cout << "Experimental cuDNN Conv2D forward tests passed!" << std::endl;
    return 0;
#endif
}
