#include "conv2d_layer.hpp"
#include "autograd/engine.hpp"
#include "autograd/backward_context.hpp"
#include "math_ops.hpp"

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

struct ScopedExperimentalGpuBackward {
    explicit ScopedExperimentalGpuBackward(bool enabled) {
        const char* current = std::getenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD");
        had_original_ = current != nullptr;
        if (had_original_) {
            original_value_ = current;
        }
        setenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD", enabled ? "1" : "0", 1);
    }

    ~ScopedExperimentalGpuBackward() {
        if (had_original_) {
            setenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD", original_value_.c_str(), 1);
        } else {
            unsetenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD");
        }
    }

private:
    bool had_original_ = false;
    std::string original_value_;
};

struct ScopedExperimentalGpuBackwardForceFail {
    explicit ScopedExperimentalGpuBackwardForceFail(bool enabled) {
        const char* current = std::getenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD_FORCE_FAIL");
        had_original_ = current != nullptr;
        if (had_original_) {
            original_value_ = current;
        }
        setenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD_FORCE_FAIL", enabled ? "1" : "0", 1);
    }

    ~ScopedExperimentalGpuBackwardForceFail() {
        if (had_original_) {
            setenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD_FORCE_FAIL", original_value_.c_str(), 1);
        } else {
            unsetenv("PPN_EXPERIMENTAL_CUDNN_CONV_BACKWARD_FORCE_FAIL");
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

bool hasExperimentalForwardMarker(const Node::Ptr& node) {
    return node &&
           node->backwardContext() &&
           node->backwardContext()->flags.size() >= 2;
}

bool hasExperimentalBackwardMarker(const Node::Ptr& node) {
    return node &&
           node->backwardContext() &&
           node->backwardContext()->flags.size() >= 3;
}

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
bool gpuRuntimeAvailable() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
#endif

struct RunResult {
    Conv2DLayer conv;
    Node::Ptr x;
    Node::Ptr loss;
    Node::Ptr out;

    RunResult()
        : conv(2, 3, 3, 2, 1, 1, ConvBackend::Reference) {}
};

RunResult runConvSum(const Matrix& input,
                     const Matrix& weights,
                     const Matrix& bias,
                     bool input_requires_grad,
                     bool weight_requires_grad,
                     bool bias_requires_grad) {
    RunResult result;
    loadConvParameters(result.conv, weights, bias);
    result.x = std::make_shared<Node>(input, input_requires_grad);
    result.conv.parameters()[0]->setRequiresGrad(weight_requires_grad);
    result.conv.parameters()[1]->setRequiresGrad(bias_requires_grad);

    result.out = result.conv.forward(result.x, 2, 2, 4, 5);
    result.loss = MathOps::sum(result.out);
    return result;
}

void assertGradState(const RunResult& actual,
                     const RunResult& expected,
                     bool expect_backward_marker,
                     bool expect_input_grad,
                     bool expect_weight_grad,
                     bool expect_bias_grad) {
    assert(hasExperimentalForwardMarker(actual.out) || !expect_backward_marker);
    assert(hasExperimentalBackwardMarker(actual.out) == expect_backward_marker);

    if (expect_input_grad) {
        assert(actual.x->hasAllocatedGrad());
        assert(expected.x->hasAllocatedGrad());
        assertMatrixNear(actual.x->grad(), expected.x->grad());
    } else {
        assert(!actual.x->hasAllocatedGrad());
    }

    const auto actual_params = actual.conv.parameters();
    const auto expected_params = expected.conv.parameters();

    if (expect_weight_grad) {
        assert(actual_params[0]->hasAllocatedGrad());
        assert(expected_params[0]->hasAllocatedGrad());
        assertMatrixNear(actual_params[0]->grad(), expected_params[0]->grad());
    } else {
        assert(!actual_params[0]->hasAllocatedGrad());
    }

    if (expect_bias_grad) {
        assert(actual_params[1]->hasAllocatedGrad());
        assert(expected_params[1]->hasAllocatedGrad());
        assertMatrixNear(actual_params[1]->grad(), expected_params[1]->grad());
    } else {
        assert(!actual_params[1]->hasAllocatedGrad());
    }
}

void test_experimental_backward_parity() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, true, true, true);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward gpu_forward_guard(true);
    ScopedExperimentalGpuBackward gpu_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail gpu_fail_guard(false);
    auto gpu_run = runConvSum(input, weights, bias, true, true, true);
    cpu_run.loss->backward();
    gpu_run.loss->backward();

    assertGradState(gpu_run, cpu_run, true, true, true, true);
}

void test_backward_opt_in_disabled_falls_back_to_cpu() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, true, true, true);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward guarded_forward_guard(true);
    ScopedExperimentalGpuBackward guarded_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail guarded_fail_guard(false);
    auto guarded_run = runConvSum(input, weights, bias, true, true, true);
    cpu_run.loss->backward();
    guarded_run.loss->backward();

    assert(hasExperimentalForwardMarker(guarded_run.out));
    assertGradState(guarded_run, cpu_run, false, true, true, true);
}

void test_non_experimental_forward_keeps_cpu_backward() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, true, true, true);

    ScopedExperimentalGpuForward guarded_forward_guard(false);
    ScopedExperimentalGpuBackward guarded_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail guarded_fail_guard(false);
    auto guarded_run = runConvSum(input, weights, bias, true, true, true);
    cpu_run.loss->backward();
    guarded_run.loss->backward();

    assert(!hasExperimentalForwardMarker(guarded_run.out));
    assertGradState(guarded_run, cpu_run, false, true, true, true);
}

void test_helper_failure_falls_back_to_cpu() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, true, true, true);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward guarded_forward_guard(true);
    ScopedExperimentalGpuBackward guarded_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail guarded_fail_guard(true);
    auto guarded_run = runConvSum(input, weights, bias, true, true, true);
    cpu_run.loss->backward();
    guarded_run.loss->backward();

    assert(hasExperimentalForwardMarker(guarded_run.out));
    assertGradState(guarded_run, cpu_run, false, true, true, true);
}

void test_partial_grad_input_only() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, true, false, false);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward gpu_forward_guard(true);
    ScopedExperimentalGpuBackward gpu_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail gpu_fail_guard(false);
    auto gpu_run = runConvSum(input, weights, bias, true, false, false);
    cpu_run.loss->backward();
    gpu_run.loss->backward();

    assertGradState(gpu_run, cpu_run, true, true, false, false);
}

void test_partial_grad_weight_only() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, false, true, false);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward gpu_forward_guard(true);
    ScopedExperimentalGpuBackward gpu_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail gpu_fail_guard(false);
    auto gpu_run = runConvSum(input, weights, bias, false, true, false);
    cpu_run.loss->backward();
    gpu_run.loss->backward();

    assertGradState(gpu_run, cpu_run, true, false, true, false);
}

void test_partial_grad_bias_only_keeps_cpu_backward() {
    const Matrix input = makeInput();
    const Matrix weights = makeWeights();
    const Matrix bias = makeBias();

    ScopedExperimentalGpuForward cpu_forward_guard(false);
    ScopedExperimentalGpuBackward cpu_backward_guard(false);
    ScopedExperimentalGpuBackwardForceFail cpu_fail_guard(false);
    auto cpu_run = runConvSum(input, weights, bias, false, false, true);

#if PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD
    if (!gpuRuntimeAvailable()) {
        return;
    }
#endif

    ScopedExperimentalGpuForward gpu_forward_guard(true);
    ScopedExperimentalGpuBackward gpu_backward_guard(true);
    ScopedExperimentalGpuBackwardForceFail gpu_fail_guard(false);
    auto gpu_run = runConvSum(input, weights, bias, false, false, true);
    cpu_run.loss->backward();
    gpu_run.loss->backward();

    assert(hasExperimentalForwardMarker(gpu_run.out));
    assertGradState(gpu_run, cpu_run, false, false, false, true);
}

}  // namespace

int main() {
#if !(PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_FORWARD && PPN_HAVE_EXPERIMENTAL_CUDNN_CONV_BACKWARD)
    std::cout << "Experimental cuDNN Conv2D backward tests skipped: binary built without cuDNN experimental forward/backward support." << std::endl;
    return 0;
#else
    test_experimental_backward_parity();
    test_backward_opt_in_disabled_falls_back_to_cpu();
    test_non_experimental_forward_keeps_cpu_backward();
    test_helper_failure_falls_back_to_cpu();
    test_partial_grad_input_only();
    test_partial_grad_weight_only();
    test_partial_grad_bias_only_keeps_cpu_backward();
    std::cout << "Experimental cuDNN Conv2D backward tests passed!" << std::endl;
    return 0;
#endif
}
