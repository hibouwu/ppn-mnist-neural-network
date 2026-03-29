#include "gpu/cudnn_conv_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct OutputShape {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
};

struct ProbeCase {
    const char* name = "";
    GpuConv2dProblem problem;
    std::vector<float> input;
    std::vector<float> filter;
};

OutputShape inferOutputShape(const GpuConv2dProblem& problem) {
    const int effective_kernel_h = (problem.r - 1) * problem.dilation_h + 1;
    const int effective_kernel_w = (problem.s - 1) * problem.dilation_w + 1;
    const int numerator_h = problem.h + 2 * problem.pad_h - effective_kernel_h;
    const int numerator_w = problem.w + 2 * problem.pad_w - effective_kernel_w;
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::runtime_error("CPU reference shape inference failed: invalid convolution geometry.");
    }

    return OutputShape{
        problem.n,
        problem.k,
        numerator_h / problem.stride_h + 1,
        numerator_w / problem.stride_w + 1,
    };
}

std::vector<float> cpuReferenceForward(const std::vector<float>& input,
                                       const std::vector<float>& filter,
                                       const GpuConv2dProblem& problem,
                                       const OutputShape& out_shape) {
    std::vector<float> output(
        static_cast<std::size_t>(out_shape.n) * out_shape.c * out_shape.h * out_shape.w,
        0.0f);

    for (int n = 0; n < problem.n; ++n) {
        for (int k = 0; k < problem.k; ++k) {
            for (int oh = 0; oh < out_shape.h; ++oh) {
                for (int ow = 0; ow < out_shape.w; ++ow) {
                    float acc = 0.0f;
                    for (int c = 0; c < problem.c; ++c) {
                        for (int kh = 0; kh < problem.r; ++kh) {
                            for (int kw = 0; kw < problem.s; ++kw) {
                                const int ih = oh * problem.stride_h - problem.pad_h + kh * problem.dilation_h;
                                const int iw = ow * problem.stride_w - problem.pad_w + kw * problem.dilation_w;
                                if (ih < 0 || ih >= problem.h || iw < 0 || iw >= problem.w) {
                                    continue;
                                }

                                const std::size_t input_idx =
                                    static_cast<std::size_t>(((n * problem.c + c) * problem.h + ih) * problem.w + iw);
                                const std::size_t filter_idx =
                                    static_cast<std::size_t>(((k * problem.c + c) * problem.r + kh) * problem.s + kw);
                                acc += input[input_idx] * filter[filter_idx];
                            }
                        }
                    }

                    const std::size_t output_idx =
                        static_cast<std::size_t>(((n * out_shape.c + k) * out_shape.h + oh) * out_shape.w + ow);
                    output[output_idx] = acc;
                }
            }
        }
    }
    return output;
}

void printShape(const char* label, int n, int c, int h, int w) {
    std::cout << label << ": [" << n << ", " << c << ", " << h << ", " << w << "]\n";
}

void printVector(const char* label, const std::vector<float>& values) {
    std::cout << label << ':';
    for (float value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

bool runCase(const ProbeCase& probe_case, float tolerance) {
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
        const OutputShape expected_shape = inferOutputShape(probe_case.problem);
        const GpuConv2dForwardResult gpu_result =
            gpuConv2dForwardNchw(probe_case.input, probe_case.filter, probe_case.problem);
        const std::vector<float> cpu_output =
            cpuReferenceForward(probe_case.input, probe_case.filter, probe_case.problem, expected_shape);

        printShape("Output shape", gpu_result.out_n, gpu_result.out_c, gpu_result.out_h, gpu_result.out_w);
        std::cout << "Selected algo: " << gpu_result.algorithm_name << '\n';
        std::cout << "Workspace bytes: " << gpu_result.workspace_bytes << '\n';

        bool shape_ok = gpu_result.out_n == expected_shape.n &&
                        gpu_result.out_c == expected_shape.c &&
                        gpu_result.out_h == expected_shape.h &&
                        gpu_result.out_w == expected_shape.w;
        if (!shape_ok) {
            std::cout << "Parity: FAIL\n";
            std::cout << "Max abs error: shape-mismatch\n\n";
            return false;
        }

        float max_abs_error = 0.0f;
        for (std::size_t i = 0; i < gpu_result.output.size(); ++i) {
            max_abs_error = std::max(max_abs_error, std::fabs(gpu_result.output[i] - cpu_output[i]));
        }
        const bool parity_ok = max_abs_error <= tolerance;

        printVector("GPU output", gpu_result.output);
        printVector("CPU output", cpu_output);
        std::cout << "Parity: " << (parity_ok ? "PASS" : "FAIL") << '\n';
        std::cout << "Max abs error: " << max_abs_error << "\n\n";
        return parity_ok;
    } catch (const std::exception& err) {
        std::cout << "Parity: FAIL\n";
        std::cout << "Max abs error: exception\n";
        std::cout << "Error: " << err.what() << "\n\n";
        return false;
    }
}

}  // namespace

int main() {
    constexpr float tolerance = 1e-5f;
    const std::vector<ProbeCase> probe_cases{
        {
            "single_channel_valid_no_padding",
            {1, 1, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
            },
            {
                1.0f, 0.0f,
                0.0f, 1.0f,
            },
        },
        {
            "padding_one_same_spatial",
            {1, 1, 3, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
            },
            {
                0.0f, 1.0f, 0.0f,
                1.0f, 1.0f, 1.0f,
                0.0f, 1.0f, 0.0f,
            },
        },
        {
            "stride_two_downsample",
            {1, 1, 4, 4, 1, 2, 2, 0, 0, 2, 2, 1, 1},
            {
                1.0f, 2.0f, 3.0f, 4.0f,
                5.0f, 6.0f, 7.0f, 8.0f,
                9.0f, 10.0f, 11.0f, 12.0f,
                13.0f, 14.0f, 15.0f, 16.0f,
            },
            {
                1.0f, -1.0f,
                0.5f, 2.0f,
            },
        },
        {
            "multi_input_channel",
            {1, 2, 3, 3, 1, 2, 2, 0, 0, 1, 1, 1, 1},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                9.0f, 8.0f, 7.0f,
                6.0f, 5.0f, 4.0f,
                3.0f, 2.0f, 1.0f,
            },
            {
                1.0f, 0.0f,
                0.0f, 1.0f,
                0.5f, 0.5f,
                0.5f, 0.5f,
            },
        },
        {
            "batch_two_multi_output_channel",
            {2, 2, 3, 3, 2, 2, 2, 0, 0, 1, 1, 1, 1},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                0.0f, 1.0f, 0.0f,
                1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f,
                2.0f, 1.0f, 0.0f,
                1.0f, 2.0f, 1.0f,
                0.0f, 1.0f, 2.0f,
                1.0f, 1.0f, 1.0f,
                2.0f, 2.0f, 2.0f,
                3.0f, 3.0f, 3.0f,
            },
            {
                1.0f, 0.0f,
                0.0f, 1.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f,
                1.0f, 1.0f,
                -1.0f, 0.0f,
                0.0f, -1.0f,
                0.5f, 0.0f,
                0.0f, 0.5f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, -1.0f,
                -1.0f, 1.0f,
                0.25f, 0.75f,
                0.75f, 0.25f,
            },
        },
    };

    bool all_passed = true;
    for (const ProbeCase& probe_case : probe_cases) {
        all_passed = runCase(probe_case, tolerance) && all_passed;
    }

    std::cout << "Summary: " << (all_passed ? "PASS" : "FAIL") << " (" << probe_cases.size()
              << " cases)\n";
    return all_passed ? 0 : 1;
}
