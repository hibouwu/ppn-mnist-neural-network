#include "gpu/cudnn_conv_forward.hpp"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string algoName(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "IMPLICIT_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "IMPLICIT_PRECOMP_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "DIRECT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "FFT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "FFT_TILING";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "WINOGRAD";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "WINOGRAD_NONFUSED";
        case CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
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

void validateProblem(const GpuConv2dProblem& problem,
                     const std::vector<float>& input,
                     const std::vector<float>& filter) {
    if (problem.n <= 0 || problem.c <= 0 || problem.h <= 0 || problem.w <= 0 ||
        problem.k <= 0 || problem.r <= 0 || problem.s <= 0) {
        throw std::invalid_argument("gpuConv2dForwardNchw requires positive N/C/H/W/K/R/S.");
    }
    if (problem.stride_h <= 0 || problem.stride_w <= 0 ||
        problem.dilation_h <= 0 || problem.dilation_w <= 0) {
        throw std::invalid_argument("gpuConv2dForwardNchw requires positive stride/dilation.");
    }
    if (problem.pad_h < 0 || problem.pad_w < 0) {
        throw std::invalid_argument("gpuConv2dForwardNchw requires non-negative padding.");
    }

    const int effective_kernel_h = (problem.r - 1) * problem.dilation_h + 1;
    const int effective_kernel_w = (problem.s - 1) * problem.dilation_w + 1;
    const int numerator_h = problem.h + 2 * problem.pad_h - effective_kernel_h;
    const int numerator_w = problem.w + 2 * problem.pad_w - effective_kernel_w;
    if (numerator_h < 0 || numerator_w < 0) {
        throw std::invalid_argument(
            "gpuConv2dForwardNchw has invalid geometry: effective kernel exceeds padded input.");
    }

    const std::size_t input_size =
        static_cast<std::size_t>(problem.n) * problem.c * problem.h * problem.w;
    const std::size_t filter_size =
        static_cast<std::size_t>(problem.k) * problem.c * problem.r * problem.s;
    if (input.size() != input_size) {
        throw std::invalid_argument("gpuConv2dForwardNchw input size does not match NCHW shape.");
    }
    if (filter.size() != filter_size) {
        throw std::invalid_argument("gpuConv2dForwardNchw filter size does not match KCRS shape.");
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

using UniqueCudnnHandle = std::unique_ptr<std::remove_pointer<cudnnHandle_t>::type, CudnnHandleDeleter>;
using UniqueTensorDesc = std::unique_ptr<std::remove_pointer<cudnnTensorDescriptor_t>::type, TensorDescDeleter>;
using UniqueFilterDesc = std::unique_ptr<std::remove_pointer<cudnnFilterDescriptor_t>::type, FilterDescDeleter>;
using UniqueConvDesc = std::unique_ptr<std::remove_pointer<cudnnConvolutionDescriptor_t>::type, ConvDescDeleter>;
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

}  // namespace

GpuConv2dForwardResult gpuConv2dForwardNchw(const std::vector<float>& input,
                                            const std::vector<float>& filter,
                                            const GpuConv2dProblem& problem) {
    validateProblem(problem, input, filter);

    int device_count = 0;
    throwCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("gpuConv2dForwardNchw found no CUDA-capable device.");
    }
    throwCuda(cudaSetDevice(0), "cudaSetDevice");

    const auto handle = makeHandle();
    const auto input_desc = makeTensorDesc();
    const auto output_desc = makeTensorDesc();
    const auto filter_desc = makeFilterDesc();
    const auto conv_desc = makeConvDesc();

    throwCudnn(
        cudnnSetTensor4dDescriptor(
            input_desc.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, problem.n, problem.c, problem.h, problem.w),
        "cudnnSetTensor4dDescriptor(input)");
    throwCudnn(
        cudnnSetFilter4dDescriptor(
            filter_desc.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, problem.k, problem.c, problem.r, problem.s),
        "cudnnSetFilter4dDescriptor");
    throwCudnn(
        cudnnSetConvolution2dDescriptor(
            conv_desc.get(),
            problem.pad_h,
            problem.pad_w,
            problem.stride_h,
            problem.stride_w,
            problem.dilation_h,
            problem.dilation_w,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor");

    GpuConv2dForwardResult result;
    throwCudnn(
        cudnnGetConvolution2dForwardOutputDim(
            conv_desc.get(),
            input_desc.get(),
            filter_desc.get(),
            &result.out_n,
            &result.out_c,
            &result.out_h,
            &result.out_w),
        "cudnnGetConvolution2dForwardOutputDim");
    if (result.out_n != problem.n || result.out_c != problem.k ||
        result.out_h <= 0 || result.out_w <= 0) {
        throw std::runtime_error(
            "gpuConv2dForwardNchw produced an invalid output shape from cuDNN.");
    }

    throwCudnn(
        cudnnSetTensor4dDescriptor(
            output_desc.get(),
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            result.out_n,
            result.out_c,
            result.out_h,
            result.out_w),
        "cudnnSetTensor4dDescriptor(output)");

    int algo_count = 0;
    throwCudnn(cudnnGetConvolutionForwardAlgorithmMaxCount(handle.get(), &algo_count),
               "cudnnGetConvolutionForwardAlgorithmMaxCount");
    if (algo_count <= 0) {
        throw std::runtime_error("cuDNN reported no forward algorithms.");
    }

    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(static_cast<std::size_t>(algo_count));
    int returned_algo_count = 0;
    throwCudnn(
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle.get(),
            input_desc.get(),
            filter_desc.get(),
            conv_desc.get(),
            output_desc.get(),
            algo_count,
            &returned_algo_count,
            perf_results.data()),
        "cudnnGetConvolutionForwardAlgorithm_v7");

    cudnnConvolutionFwdAlgo_t selected_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    bool found_algo = false;
    for (int i = 0; i < returned_algo_count; ++i) {
        if (perf_results[static_cast<std::size_t>(i)].status == CUDNN_STATUS_SUCCESS) {
            selected_algo = perf_results[static_cast<std::size_t>(i)].algo;
            found_algo = true;
            break;
        }
    }
    if (!found_algo) {
        throw std::runtime_error("No cuDNN forward algorithm reported success.");
    }

    std::size_t workspace_bytes = 0;
    throwCudnn(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle.get(),
            input_desc.get(),
            filter_desc.get(),
            conv_desc.get(),
            output_desc.get(),
            selected_algo,
            &workspace_bytes),
        "cudnnGetConvolutionForwardWorkspaceSize");
    result.workspace_bytes = workspace_bytes;
    result.algorithm_name = algoName(selected_algo);

    result.output.resize(
        static_cast<std::size_t>(result.out_n) * result.out_c * result.out_h * result.out_w);

    const auto device_input = makeDeviceFloatBuffer(input.size(), "cudaMalloc(input)");
    const auto device_filter = makeDeviceFloatBuffer(filter.size(), "cudaMalloc(filter)");
    const auto device_output = makeDeviceFloatBuffer(result.output.size(), "cudaMalloc(output)");
    const auto workspace = workspace_bytes > 0
        ? makeDeviceBuffer(workspace_bytes, "cudaMalloc(workspace)")
        : UniqueDeviceBuffer(nullptr);

    throwCuda(
        cudaMemcpy(device_input.get(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");
    throwCuda(
        cudaMemcpy(device_filter.get(), filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(filter)");

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    throwCudnn(
        cudnnConvolutionForward(
            handle.get(),
            &alpha,
            input_desc.get(),
            device_input.get(),
            filter_desc.get(),
            device_filter.get(),
            conv_desc.get(),
            selected_algo,
            workspace.get(),
            workspace_bytes,
            &beta,
            output_desc.get(),
            device_output.get()),
        "cudnnConvolutionForward");
    throwCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    throwCuda(
        cudaMemcpy(result.output.data(),
                   device_output.get(),
                   result.output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(output)");

    return result;
}
