#pragma once

#include "gpu/cudnn_conv_forward.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct GpuConv2dBackwardDataResult {
    int diff_n = 0;
    int diff_c = 0;
    int diff_h = 0;
    int diff_w = 0;
    std::size_t workspace_bytes = 0;
    std::string algorithm_name;
    std::vector<float> diff_input;
};

struct GpuConv2dBackwardFilterResult {
    int diff_k = 0;
    int diff_c = 0;
    int diff_r = 0;
    int diff_s = 0;
    std::size_t workspace_bytes = 0;
    std::string algorithm_name;
    std::vector<float> diff_filter;
};

GpuConv2dBackwardDataResult gpuConv2dBackwardDataNchw(
    const std::vector<float>& filter,
    const std::vector<float>& grad_output,
    const GpuConv2dProblem& problem);

GpuConv2dBackwardFilterResult gpuConv2dBackwardFilterNchw(
    const std::vector<float>& input,
    const std::vector<float>& grad_output,
    const GpuConv2dProblem& problem);
