#include "autograd/engine.hpp"
#include "conv2d_layer.hpp"
#include "gpu/cudnn_conv_backward.hpp"
#include "node.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kParityTol = 1e-4;
constexpr double kBiasTol = 1e-9;

struct ProbeCase {
    const char* name = "";
    GpuConv2dProblem problem;
};

struct ReferenceResult {
    Matrix diff_src;
    Matrix diff_weights;
    Matrix diff_bias;
    int out_h = 0;
    int out_w = 0;
};

Matrix makeInput(const GpuConv2dProblem& problem) {
    Matrix input(problem.n, problem.c * problem.h * problem.w);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.03 * static_cast<double>(i + 1);
    }
    return input;
}

Matrix makeWeights(const GpuConv2dProblem& problem) {
    Matrix weights(problem.k, problem.c * problem.r * problem.s);
    for (std::size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] = -0.07 + 0.01 * static_cast<double>(i + 1);
    }
    return weights;
}

Matrix makeBias(const GpuConv2dProblem& problem) {
    Matrix bias(1, problem.k);
    for (int k = 0; k < problem.k; ++k) {
        bias(0, static_cast<std::size_t>(k)) = -0.1 + 0.05 * static_cast<double>(k);
    }
    return bias;
}

std::pair<int, int> inferOutputShape(const GpuConv2dProblem& problem) {
    const int effective_kernel_h = (problem.r - 1) * problem.dilation_h + 1;
    const int effective_kernel_w = (problem.s - 1) * problem.dilation_w + 1;
    const int numerator_h = problem.h + 2 * problem.pad_h - effective_kernel_h;
    const int numerator_w = problem.w + 2 * problem.pad_w - effective_kernel_w;
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::runtime_error("Backward probe shape inference failed: invalid convolution geometry.");
    }
    return {
        numerator_h / problem.stride_h + 1,
        numerator_w / problem.stride_w + 1,
    };
}

Matrix makeGradOutput(const GpuConv2dProblem& problem, int out_h, int out_w) {
    Matrix grad_output(problem.n, problem.k * out_h * out_w);
    for (std::size_t i = 0; i < grad_output.data.size(); ++i) {
        grad_output.data[i] = 0.02 * static_cast<double>(i + 1);
    }
    return grad_output;
}

std::vector<float> toFloatBuffer(const Matrix& matrix) {
    std::vector<float> buffer(matrix.data.size());
    for (std::size_t i = 0; i < matrix.data.size(); ++i) {
        buffer[i] = static_cast<float>(matrix.data[i]);
    }
    return buffer;
}

Matrix cpuReduceBiasGrad(const Matrix& grad_output, int out_channels, int out_h, int out_w) {
    Matrix diff_bias(1, static_cast<std::size_t>(out_channels), 0.0);
    const std::size_t spatial = static_cast<std::size_t>(out_h) * out_w;
    for (std::size_t n = 0; n < grad_output.rows; ++n) {
        const std::size_t row_base = n * grad_output.cols;
        for (int oc = 0; oc < out_channels; ++oc) {
            double acc = 0.0;
            const std::size_t channel_base = static_cast<std::size_t>(oc) * spatial;
            for (std::size_t hw = 0; hw < spatial; ++hw) {
                acc += grad_output.data[row_base + channel_base + hw];
            }
            diff_bias(0, static_cast<std::size_t>(oc)) += acc;
        }
    }
    return diff_bias;
}

ReferenceResult runCpuReference(const ProbeCase& probe_case,
                                const Matrix& input,
                                const Matrix& weights,
                                const Matrix& bias,
                                const Matrix& grad_output) {
    const GpuConv2dProblem& problem = probe_case.problem;
    if (problem.stride_h != problem.stride_w ||
        problem.pad_h != problem.pad_w ||
        problem.dilation_h != 1 ||
        problem.dilation_w != 1) {
        throw std::runtime_error(
            "CPU reference for backward probe only supports symmetric stride/padding and dilation=1.");
    }

    Conv2DLayer conv(
        static_cast<std::size_t>(problem.c),
        static_cast<std::size_t>(problem.k),
        static_cast<std::size_t>(problem.r),
        static_cast<std::size_t>(problem.s),
        static_cast<std::size_t>(problem.stride_h),
        static_cast<std::size_t>(problem.pad_h),
        ConvBackend::Reference);
    auto params = conv.parameters();
    Matrix& kernels_value = const_cast<Matrix&>(params[0]->value());
    Matrix& bias_value = const_cast<Matrix&>(params[1]->value());
    kernels_value = weights;
    bias_value = bias;

    auto x = std::make_shared<Node>(input);
    auto out = conv.forward(
        x,
        static_cast<std::size_t>(problem.n),
        static_cast<std::size_t>(problem.c),
        static_cast<std::size_t>(problem.h),
        static_cast<std::size_t>(problem.w));
    out->addGrad(grad_output);

    AutogradEngine engine;
    engine.backward(out);

    const auto [out_h, out_w] = inferOutputShape(problem);
    return ReferenceResult{
        x->grad(),
        params[0]->grad(),
        params[1]->grad(),
        out_h,
        out_w,
    };
}

double maxAbsError(const std::vector<float>& actual, const Matrix& expected) {
    if (actual.size() != expected.data.size()) {
        throw std::runtime_error("comparison shape mismatch.");
    }
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(static_cast<double>(actual[i]) - expected.data[i]));
    }
    return max_abs_error;
}

double maxAbsError(const Matrix& actual, const Matrix& expected) {
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        throw std::runtime_error("comparison shape mismatch.");
    }
    double max_abs_error = 0.0;
    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(actual.data[i] - expected.data[i]));
    }
    return max_abs_error;
}

void printShape(const char* label, int n, int c, int h, int w) {
    std::cout << label << ": [" << n << ", " << c << ", " << h << ", " << w << "]\n";
}

bool runCase(const ProbeCase& probe_case) {
    std::cout << "Case: " << probe_case.name << '\n';
    printShape("Input shape",
               probe_case.problem.n,
               probe_case.problem.c,
               probe_case.problem.h,
               probe_case.problem.w);
    printShape("Filter shape",
               probe_case.problem.k,
               probe_case.problem.c,
               probe_case.problem.r,
               probe_case.problem.s);

    try {
        const Matrix input = makeInput(probe_case.problem);
        const Matrix weights = makeWeights(probe_case.problem);
        const Matrix bias = makeBias(probe_case.problem);
        const auto [out_h, out_w] = inferOutputShape(probe_case.problem);
        const Matrix grad_output = makeGradOutput(probe_case.problem, out_h, out_w);

        const ReferenceResult cpu_result =
            runCpuReference(probe_case, input, weights, bias, grad_output);
        const GpuConv2dBackwardDataResult gpu_diff_src =
            gpuConv2dBackwardDataNchw(
                toFloatBuffer(weights), toFloatBuffer(grad_output), probe_case.problem);
        const GpuConv2dBackwardFilterResult gpu_diff_weights =
            gpuConv2dBackwardFilterNchw(
                toFloatBuffer(input), toFloatBuffer(grad_output), probe_case.problem);
        const Matrix cpu_diff_bias = cpuReduceBiasGrad(
            grad_output, probe_case.problem.k, cpu_result.out_h, cpu_result.out_w);

        printShape("Grad output shape",
                   probe_case.problem.n,
                   probe_case.problem.k,
                   cpu_result.out_h,
                   cpu_result.out_w);
        std::cout << "Backward-data algo: " << gpu_diff_src.algorithm_name << '\n';
        std::cout << "Backward-data workspace bytes: " << gpu_diff_src.workspace_bytes << '\n';
        std::cout << "Backward-filter algo: " << gpu_diff_weights.algorithm_name << '\n';
        std::cout << "Backward-filter workspace bytes: " << gpu_diff_weights.workspace_bytes << '\n';

        const bool diff_src_shape_ok =
            gpu_diff_src.diff_n == probe_case.problem.n &&
            gpu_diff_src.diff_c == probe_case.problem.c &&
            gpu_diff_src.diff_h == probe_case.problem.h &&
            gpu_diff_src.diff_w == probe_case.problem.w &&
            gpu_diff_src.diff_input.size() == input.data.size();
        const bool diff_weights_shape_ok =
            gpu_diff_weights.diff_k == probe_case.problem.k &&
            gpu_diff_weights.diff_c == probe_case.problem.c &&
            gpu_diff_weights.diff_r == probe_case.problem.r &&
            gpu_diff_weights.diff_s == probe_case.problem.s &&
            gpu_diff_weights.diff_filter.size() == weights.data.size();
        const bool diff_bias_shape_ok =
            cpu_diff_bias.rows == cpu_result.diff_bias.rows &&
            cpu_diff_bias.cols == cpu_result.diff_bias.cols;

        if (!diff_src_shape_ok || !diff_weights_shape_ok || !diff_bias_shape_ok) {
            std::cout << "Parity: FAIL\n";
            std::cout << "Max abs error diff_src: shape-mismatch\n";
            std::cout << "Max abs error diff_weights: shape-mismatch\n";
            std::cout << "Max abs error diff_bias: shape-mismatch\n\n";
            return false;
        }

        const double diff_src_error = maxAbsError(gpu_diff_src.diff_input, cpu_result.diff_src);
        const double diff_weights_error = maxAbsError(gpu_diff_weights.diff_filter, cpu_result.diff_weights);
        const double diff_bias_error = maxAbsError(cpu_diff_bias, cpu_result.diff_bias);

        const bool parity_ok =
            diff_src_error <= kParityTol &&
            diff_weights_error <= kParityTol &&
            diff_bias_error <= kBiasTol;

        std::cout << "Max abs error diff_src: " << diff_src_error << '\n';
        std::cout << "Max abs error diff_weights: " << diff_weights_error << '\n';
        std::cout << "Max abs error diff_bias: " << diff_bias_error << '\n';
        std::cout << "Parity: " << (parity_ok ? "PASS" : "FAIL") << "\n\n";
        return parity_ok;
    } catch (const std::exception& err) {
        std::cout << "Parity: FAIL\n";
        std::cout << "Error: " << err.what() << "\n\n";
        return false;
    }
}

bool runInvalidGradOutputCase() {
    std::cout << "Case: invalid_grad_output_size\n";
    const GpuConv2dProblem problem{1, 1, 4, 4, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    const Matrix weights = makeWeights(problem);
    try {
        (void)gpuConv2dBackwardDataNchw(toFloatBuffer(weights), {1.0f, 2.0f}, problem);
        std::cout << "Contract: FAIL\n";
        std::cout << "Error: expected invalid_argument was not thrown\n\n";
        return false;
    } catch (const std::invalid_argument& err) {
        std::cout << "Contract: PASS\n";
        std::cout << "Error: " << err.what() << "\n\n";
        return true;
    } catch (const std::exception& err) {
        std::cout << "Contract: FAIL\n";
        std::cout << "Error: " << err.what() << "\n\n";
        return false;
    }
}

}  // namespace

int main() {
    const std::vector<ProbeCase> probe_cases{
        {
            "single_channel_valid_no_padding",
            {1, 1, 4, 4, 1, 2, 2, 0, 0, 1, 1, 1, 1},
        },
        {
            "multi_batch_multi_channel_padding",
            {2, 2, 4, 5, 3, 3, 2, 1, 1, 1, 1, 1, 1},
        },
    };

    bool all_passed = true;
    for (const ProbeCase& probe_case : probe_cases) {
        all_passed = runCase(probe_case) && all_passed;
    }
    all_passed = runInvalidGradOutputCase() && all_passed;

    std::cout << "Summary: " << (all_passed ? "PASS" : "FAIL")
              << " (" << probe_cases.size() + 1 << " checks)\n";
    return all_passed ? 0 : 1;
}
