#include "gpu/cudnn_conv_backward.hpp"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct OutputShape {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
};

std::string backwardDataAlgoName(cudnnConvolutionBwdDataAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            return "ALGO_0";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            return "ALGO_1";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            return "FFT";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
            return "FFT_TILING";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            return "WINOGRAD";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD_NONFUSED";
        case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
            return "COUNT";
    }
    return "UNKNOWN";
}

std::string backwardFilterAlgoName(cudnnConvolutionBwdFilterAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            return "ALGO_0";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            return "ALGO_1";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
            return "FFT";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
            return "ALGO_3";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD_NONFUSED";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
            return "FFT_TILING";
        case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
            return "COUNT";
    }
    return "UNKNOWN";
}

void throwCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
    }
}

void throwCudnn(cudnnStatus_t status, const char* what) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed: " + cudnnGetErrorString(status));
    }
}

OutputShape inferOutputShape(const GpuConv2dProblem& problem) {
    const int effective_kernel_h = (problem.r - 1) * problem.dilation_h + 1;
    const int effective_kernel_w = (problem.s - 1) * problem.dilation_w + 1;
    const int numerator_h = problem.h + 2 * problem.pad_h - effective_kernel_h;
    const int numerator_w = problem.w + 2 * problem.pad_w - effective_kernel_w;
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::invalid_argument(
            "cuDNN backward helper has invalid geometry: effective kernel exceeds padded input.");
    }

    return OutputShape{
        problem.n,
        problem.k,
        numerator_h / problem.stride_h + 1,
        numerator_w / problem.stride_w + 1,
    };
}

void validateProblemGeometry(const GpuConv2dProblem& problem) {
    if (problem.n <= 0 || problem.c <= 0 || problem.h <= 0 || problem.w <= 0 ||
        problem.k <= 0 || problem.r <= 0 || problem.s <= 0) {
        throw std::invalid_argument("cuDNN backward helper requires positive N/C/H/W/K/R/S.");
    }
    if (problem.stride_h <= 0 || problem.stride_w <= 0 ||
        problem.dilation_h <= 0 || problem.dilation_w <= 0) {
        throw std::invalid_argument("cuDNN backward helper requires positive stride/dilation.");
    }
    if (problem.pad_h < 0 || problem.pad_w < 0) {
        throw std::invalid_argument("cuDNN backward helper requires non-negative padding.");
    }
    (void)inferOutputShape(problem);
}

void validateBackwardDataInputs(const GpuConv2dProblem& problem,
                                const std::vector<float>& filter,
                                const std::vector<float>& grad_output) {
    validateProblemGeometry(problem);
    const OutputShape out_shape = inferOutputShape(problem);
    const std::size_t filter_size =
        static_cast<std::size_t>(problem.k) * problem.c * problem.r * problem.s;
    const std::size_t grad_output_size =
        static_cast<std::size_t>(out_shape.n) * out_shape.c * out_shape.h * out_shape.w;
    if (filter.size() != filter_size) {
        throw std::invalid_argument(
            "gpuConv2dBackwardDataNchw filter size does not match KCRS shape.");
    }
    if (grad_output.size() != grad_output_size) {
        throw std::invalid_argument(
            "gpuConv2dBackwardDataNchw grad_output size does not match NKHW output shape.");
    }
}

void validateBackwardFilterInputs(const GpuConv2dProblem& problem,
                                  const std::vector<float>& input,
                                  const std::vector<float>& grad_output) {
    validateProblemGeometry(problem);
    const OutputShape out_shape = inferOutputShape(problem);
    const std::size_t input_size =
        static_cast<std::size_t>(problem.n) * problem.c * problem.h * problem.w;
    const std::size_t grad_output_size =
        static_cast<std::size_t>(out_shape.n) * out_shape.c * out_shape.h * out_shape.w;
    if (input.size() != input_size) {
        throw std::invalid_argument(
            "gpuConv2dBackwardFilterNchw input size does not match NCHW shape.");
    }
    if (grad_output.size() != grad_output_size) {
        throw std::invalid_argument(
            "gpuConv2dBackwardFilterNchw grad_output size does not match NKHW output shape.");
    }
}

struct CudnnHandleDeleter {
    void operator()(cudnnHandle_t handle) const {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }
};

struct TensorDescDeleter {
    void operator()(cudnnTensorDescriptor_t desc) const {
        if (desc != nullptr) {
            cudnnDestroyTensorDescriptor(desc);
        }
    }
};

struct FilterDescDeleter {
    void operator()(cudnnFilterDescriptor_t desc) const {
        if (desc != nullptr) {
            cudnnDestroyFilterDescriptor(desc);
        }
    }
};

struct ConvDescDeleter {
    void operator()(cudnnConvolutionDescriptor_t desc) const {
        if (desc != nullptr) {
            cudnnDestroyConvolutionDescriptor(desc);
        }
    }
};

struct DeviceMemoryDeleter {
    void operator()(float* ptr) const {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct RawDeviceMemoryDeleter {
    void operator()(void* ptr) const {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

using UniqueCudnnHandle =
    std::unique_ptr<std::remove_pointer<cudnnHandle_t>::type, CudnnHandleDeleter>;
using UniqueTensorDesc =
    std::unique_ptr<std::remove_pointer<cudnnTensorDescriptor_t>::type, TensorDescDeleter>;
using UniqueFilterDesc =
    std::unique_ptr<std::remove_pointer<cudnnFilterDescriptor_t>::type, FilterDescDeleter>;
using UniqueConvDesc =
    std::unique_ptr<std::remove_pointer<cudnnConvolutionDescriptor_t>::type, ConvDescDeleter>;
using UniqueDeviceFloat = std::unique_ptr<float, DeviceMemoryDeleter>;
using UniqueDeviceBuffer = std::unique_ptr<void, RawDeviceMemoryDeleter>;

UniqueCudnnHandle makeHandle() {
    cudnnHandle_t handle = nullptr;
    throwCudnn(cudnnCreate(&handle), "cudnnCreate");
    return UniqueCudnnHandle(handle);
}

UniqueTensorDesc makeTensorDesc() {
    cudnnTensorDescriptor_t desc = nullptr;
    throwCudnn(cudnnCreateTensorDescriptor(&desc), "cudnnCreateTensorDescriptor");
    return UniqueTensorDesc(desc);
}

UniqueFilterDesc makeFilterDesc() {
    cudnnFilterDescriptor_t desc = nullptr;
    throwCudnn(cudnnCreateFilterDescriptor(&desc), "cudnnCreateFilterDescriptor");
    return UniqueFilterDesc(desc);
}

UniqueConvDesc makeConvDesc() {
    cudnnConvolutionDescriptor_t desc = nullptr;
    throwCudnn(cudnnCreateConvolutionDescriptor(&desc), "cudnnCreateConvolutionDescriptor");
    return UniqueConvDesc(desc);
}

UniqueDeviceFloat makeDeviceFloatBuffer(std::size_t elements, const char* what) {
    float* ptr = nullptr;
    throwCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), elements * sizeof(float)), what);
    return UniqueDeviceFloat(ptr);
}

UniqueDeviceBuffer makeDeviceBuffer(std::size_t bytes, const char* what) {
    void* ptr = nullptr;
    throwCuda(cudaMalloc(&ptr, bytes), what);
    return UniqueDeviceBuffer(ptr);
}

void initializeCudaDevice(const char* what) {
    int device_count = 0;
    throwCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error(std::string(what) + " found no CUDA-capable device.");
    }
    throwCuda(cudaSetDevice(0), "cudaSetDevice");
}

void setInputTensorDesc(cudnnTensorDescriptor_t desc, const GpuConv2dProblem& problem) {
    throwCudnn(
        cudnnSetTensor4dDescriptor(
            desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, problem.n, problem.c, problem.h, problem.w),
        "cudnnSetTensor4dDescriptor(input)");
}

void setFilterDesc(cudnnFilterDescriptor_t desc, const GpuConv2dProblem& problem) {
    throwCudnn(
        cudnnSetFilter4dDescriptor(
            desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, problem.k, problem.c, problem.r, problem.s),
        "cudnnSetFilter4dDescriptor");
}

void setConvDesc(cudnnConvolutionDescriptor_t desc, const GpuConv2dProblem& problem) {
    throwCudnn(
        cudnnSetConvolution2dDescriptor(
            desc,
            problem.pad_h,
            problem.pad_w,
            problem.stride_h,
            problem.stride_w,
            problem.dilation_h,
            problem.dilation_w,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor");
}

void setGradOutputTensorDesc(cudnnTensorDescriptor_t desc, const OutputShape& out_shape) {
    throwCudnn(
        cudnnSetTensor4dDescriptor(
            desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            out_shape.n,
            out_shape.c,
            out_shape.h,
            out_shape.w),
        "cudnnSetTensor4dDescriptor(grad_output)");
}

}  // namespace

GpuConv2dBackwardDataResult gpuConv2dBackwardDataNchw(
    const std::vector<float>& filter,
    const std::vector<float>& grad_output,
    const GpuConv2dProblem& problem) {
    validateBackwardDataInputs(problem, filter, grad_output);
    initializeCudaDevice("gpuConv2dBackwardDataNchw");

    const OutputShape out_shape = inferOutputShape(problem);
    const auto handle = makeHandle();
    const auto diff_input_desc = makeTensorDesc();
    const auto diff_output_desc = makeTensorDesc();
    const auto filter_desc = makeFilterDesc();
    const auto conv_desc = makeConvDesc();

    setInputTensorDesc(diff_input_desc.get(), problem);
    setGradOutputTensorDesc(diff_output_desc.get(), out_shape);
    setFilterDesc(filter_desc.get(), problem);
    setConvDesc(conv_desc.get(), problem);

    int algo_count = 0;
    throwCudnn(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.get(), &algo_count),
               "cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
    if (algo_count <= 0) {
        throw std::runtime_error("cuDNN reported no backward-data algorithms.");
    }

    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(static_cast<std::size_t>(algo_count));
    int returned_algo_count = 0;
    throwCudnn(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle.get(),
            filter_desc.get(),
            diff_output_desc.get(),
            conv_desc.get(),
            diff_input_desc.get(),
            algo_count,
            &returned_algo_count,
            perf_results.data()),
        "cudnnGetConvolutionBackwardDataAlgorithm_v7");

    cudnnConvolutionBwdDataAlgo_t selected_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    bool found_algo = false;
    for (int i = 0; i < returned_algo_count; ++i) {
        if (perf_results[static_cast<std::size_t>(i)].status == CUDNN_STATUS_SUCCESS) {
            selected_algo = perf_results[static_cast<std::size_t>(i)].algo;
            found_algo = true;
            break;
        }
    }
    if (!found_algo) {
        throw std::runtime_error("No cuDNN backward-data algorithm reported success.");
    }

    std::size_t workspace_bytes = 0;
    throwCudnn(
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle.get(),
            filter_desc.get(),
            diff_output_desc.get(),
            conv_desc.get(),
            diff_input_desc.get(),
            selected_algo,
            &workspace_bytes),
        "cudnnGetConvolutionBackwardDataWorkspaceSize");

    GpuConv2dBackwardDataResult result;
    result.diff_n = problem.n;
    result.diff_c = problem.c;
    result.diff_h = problem.h;
    result.diff_w = problem.w;
    result.workspace_bytes = workspace_bytes;
    result.algorithm_name = backwardDataAlgoName(selected_algo);
    result.diff_input.resize(static_cast<std::size_t>(problem.n) * problem.c * problem.h * problem.w);

    const auto device_filter = makeDeviceFloatBuffer(filter.size(), "cudaMalloc(filter)");
    const auto device_grad_output = makeDeviceFloatBuffer(grad_output.size(), "cudaMalloc(grad_output)");
    const auto device_diff_input = makeDeviceFloatBuffer(result.diff_input.size(), "cudaMalloc(diff_input)");
    const auto workspace = workspace_bytes > 0
        ? makeDeviceBuffer(workspace_bytes, "cudaMalloc(workspace)")
        : UniqueDeviceBuffer(nullptr);

    throwCuda(
        cudaMemcpy(device_filter.get(), filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(filter)");
    throwCuda(
        cudaMemcpy(device_grad_output.get(),
                   grad_output.data(),
                   grad_output.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(grad_output)");

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    throwCudnn(
        cudnnConvolutionBackwardData(
            handle.get(),
            &alpha,
            filter_desc.get(),
            device_filter.get(),
            diff_output_desc.get(),
            device_grad_output.get(),
            conv_desc.get(),
            selected_algo,
            workspace.get(),
            workspace_bytes,
            &beta,
            diff_input_desc.get(),
            device_diff_input.get()),
        "cudnnConvolutionBackwardData");
    throwCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    throwCuda(
        cudaMemcpy(result.diff_input.data(),
                   device_diff_input.get(),
                   result.diff_input.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(diff_input)");

    return result;
}

GpuConv2dBackwardFilterResult gpuConv2dBackwardFilterNchw(
    const std::vector<float>& input,
    const std::vector<float>& grad_output,
    const GpuConv2dProblem& problem) {
    validateBackwardFilterInputs(problem, input, grad_output);
    initializeCudaDevice("gpuConv2dBackwardFilterNchw");

    const OutputShape out_shape = inferOutputShape(problem);
    const auto handle = makeHandle();
    const auto input_desc = makeTensorDesc();
    const auto diff_output_desc = makeTensorDesc();
    const auto diff_filter_desc = makeFilterDesc();
    const auto conv_desc = makeConvDesc();

    setInputTensorDesc(input_desc.get(), problem);
    setGradOutputTensorDesc(diff_output_desc.get(), out_shape);
    setFilterDesc(diff_filter_desc.get(), problem);
    setConvDesc(conv_desc.get(), problem);

    int algo_count = 0;
    throwCudnn(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.get(), &algo_count),
               "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    if (algo_count <= 0) {
        throw std::runtime_error("cuDNN reported no backward-filter algorithms.");
    }

    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(static_cast<std::size_t>(algo_count));
    int returned_algo_count = 0;
    throwCudnn(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle.get(),
            input_desc.get(),
            diff_output_desc.get(),
            conv_desc.get(),
            diff_filter_desc.get(),
            algo_count,
            &returned_algo_count,
            perf_results.data()),
        "cudnnGetConvolutionBackwardFilterAlgorithm_v7");

    cudnnConvolutionBwdFilterAlgo_t selected_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    bool found_algo = false;
    for (int i = 0; i < returned_algo_count; ++i) {
        if (perf_results[static_cast<std::size_t>(i)].status == CUDNN_STATUS_SUCCESS) {
            selected_algo = perf_results[static_cast<std::size_t>(i)].algo;
            found_algo = true;
            break;
        }
    }
    if (!found_algo) {
        throw std::runtime_error("No cuDNN backward-filter algorithm reported success.");
    }

    std::size_t workspace_bytes = 0;
    throwCudnn(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle.get(),
            input_desc.get(),
            diff_output_desc.get(),
            conv_desc.get(),
            diff_filter_desc.get(),
            selected_algo,
            &workspace_bytes),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize");

    GpuConv2dBackwardFilterResult result;
    result.diff_k = problem.k;
    result.diff_c = problem.c;
    result.diff_r = problem.r;
    result.diff_s = problem.s;
    result.workspace_bytes = workspace_bytes;
    result.algorithm_name = backwardFilterAlgoName(selected_algo);
    result.diff_filter.resize(static_cast<std::size_t>(problem.k) * problem.c * problem.r * problem.s);

    const auto device_input = makeDeviceFloatBuffer(input.size(), "cudaMalloc(input)");
    const auto device_grad_output = makeDeviceFloatBuffer(grad_output.size(), "cudaMalloc(grad_output)");
    const auto device_diff_filter = makeDeviceFloatBuffer(result.diff_filter.size(), "cudaMalloc(diff_filter)");
    const auto workspace = workspace_bytes > 0
        ? makeDeviceBuffer(workspace_bytes, "cudaMalloc(workspace)")
        : UniqueDeviceBuffer(nullptr);

    throwCuda(
        cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");
    throwCuda(
        cudaMemcpy(device_grad_output.get(),
                   grad_output.data(),
                   grad_output.size() * sizeof(float),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(grad_output)");

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    throwCudnn(
        cudnnConvolutionBackwardFilter(
            handle.get(),
            &alpha,
            input_desc.get(),
            device_input.get(),
            diff_output_desc.get(),
            device_grad_output.get(),
            conv_desc.get(),
            selected_algo,
            workspace.get(),
            workspace_bytes,
            &beta,
            diff_filter_desc.get(),
            device_diff_filter.get()),
        "cudnnConvolutionBackwardFilter");
    throwCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    throwCuda(
        cudaMemcpy(result.diff_filter.data(),
                   device_diff_filter.get(),
                   result.diff_filter.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(diff_filter)");

    return result;
}
