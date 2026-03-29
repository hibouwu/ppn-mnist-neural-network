#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Host-side problem description for NCHW input and OIHW/KCRS filters.
struct GpuConv2dProblem {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    int k = 0;
    int r = 0;
    int s = 0;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
};

struct GpuConv2dForwardResult {
    int out_n = 0;
    int out_c = 0;
    int out_h = 0;
    int out_w = 0;
    std::size_t workspace_bytes = 0;
    std::string algorithm_name;
    std::vector<float> output;
};

GpuConv2dForwardResult gpuConv2dForwardNchw(const std::vector<float>& input,
                                            const std::vector<float>& filter,
                                            const GpuConv2dProblem& problem);
