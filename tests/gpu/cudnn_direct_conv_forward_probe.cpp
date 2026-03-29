#include <cudnn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::cerr << what << " failed: " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

void checkCudnn(cudnnStatus_t status, const char* what) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << what << " failed: " << cudnnGetErrorString(status) << '\n';
        std::exit(1);
    }
}

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

std::vector<float> cpuReferenceForward(const std::vector<float>& input,
                                       const std::vector<float>& filter,
                                       int out_h,
                                       int out_w) {
    std::vector<float> output(static_cast<std::size_t>(out_h * out_w), 0.0f);
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            float acc = 0.0f;
            for (int kh = 0; kh < 2; ++kh) {
                for (int kw = 0; kw < 2; ++kw) {
                    const int ih = oh + kh;
                    const int iw = ow + kw;
                    acc += input[static_cast<std::size_t>(ih * 3 + iw)] *
                           filter[static_cast<std::size_t>(kh * 2 + kw)];
                }
            }
            output[static_cast<std::size_t>(oh * out_w + ow)] = acc;
        }
    }
    return output;
}

void printVector(const char* label, const std::vector<float>& values) {
    std::cout << label << ':';
    for (float value : values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}

}  // namespace

int main() {
    constexpr int n = 1;
    constexpr int c = 1;
    constexpr int h = 3;
    constexpr int w = 3;
    constexpr int k = 1;
    constexpr int r = 2;
    constexpr int s = 2;
    constexpr int pad_h = 0;
    constexpr int pad_w = 0;
    constexpr int stride_h = 1;
    constexpr int stride_w = 1;
    constexpr int dilation_h = 1;
    constexpr int dilation_w = 1;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    constexpr float tolerance = 1e-5f;

    const std::vector<float> host_input{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const std::vector<float> host_filter{
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    float* device_input = nullptr;
    float* device_filter = nullptr;
    float* device_output = nullptr;
    void* workspace = nullptr;

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::cerr << "No CUDA-capable device is available.\n";
        return 1;
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice");

    checkCudnn(cudnnCreate(&handle), "cudnnCreate");
    checkCudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor(input)");
    checkCudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor(output)");
    checkCudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor");
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    checkCudnn(
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w),
        "cudnnSetTensor4dDescriptor(input)");
    checkCudnn(
        cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s),
        "cudnnSetFilter4dDescriptor");
    checkCudnn(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor");

    int out_n = 0;
    int out_c = 0;
    int out_h = 0;
    int out_w = 0;
    checkCudnn(
        cudnnGetConvolution2dForwardOutputDim(
            conv_desc, input_desc, filter_desc, &out_n, &out_c, &out_h, &out_w),
        "cudnnGetConvolution2dForwardOutputDim");
    checkCudnn(
        cudnnSetTensor4dDescriptor(
            output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w),
        "cudnnSetTensor4dDescriptor(output)");

    int algo_count = 0;
    checkCudnn(
        cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &algo_count),
        "cudnnGetConvolutionForwardAlgorithmMaxCount");
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(static_cast<std::size_t>(algo_count));
    int returned_algo_count = 0;
    checkCudnn(
        cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
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
        std::cerr << "No cuDNN forward algorithm reported success.\n";
        return 1;
    }

    std::size_t workspace_bytes = 0;
    checkCudnn(
        cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
            selected_algo,
            &workspace_bytes),
        "cudnnGetConvolutionForwardWorkspaceSize");

    std::vector<float> host_output(static_cast<std::size_t>(out_n * out_c * out_h * out_w), 0.0f);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), host_input.size() * sizeof(float)), "cudaMalloc(input)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_filter), host_filter.size() * sizeof(float)), "cudaMalloc(filter)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), host_output.size() * sizeof(float)), "cudaMalloc(output)");
    if (workspace_bytes > 0) {
        checkCuda(cudaMalloc(&workspace, workspace_bytes), "cudaMalloc(workspace)");
    }

    checkCuda(
        cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");
    checkCuda(
        cudaMemcpy(device_filter, host_filter.data(), host_filter.size() * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy(filter)");

    checkCudnn(
        cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc,
            device_input,
            filter_desc,
            device_filter,
            conv_desc,
            selected_algo,
            workspace,
            workspace_bytes,
            &beta,
            output_desc,
            device_output),
        "cudnnConvolutionForward");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    checkCuda(
        cudaMemcpy(host_output.data(), device_output, host_output.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy(output)");

    const std::vector<float> cpu_output = cpuReferenceForward(host_input, host_filter, out_h, out_w);
    float max_abs_error = 0.0f;
    for (std::size_t i = 0; i < host_output.size(); ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(host_output[i] - cpu_output[i]));
    }
    const bool parity_ok = max_abs_error <= tolerance;

    std::cout << "Input shape: [" << n << ", " << c << ", " << h << ", " << w << "]\n";
    std::cout << "Filter shape: [" << k << ", " << c << ", " << r << ", " << s << "]\n";
    std::cout << "Output shape: [" << out_n << ", " << out_c << ", " << out_h << ", " << out_w << "]\n";
    std::cout << "Selected algo: " << algoName(selected_algo) << '\n';
    std::cout << "Workspace bytes: " << workspace_bytes << '\n';
    printVector("cuDNN output", host_output);
    printVector("CPU reference", cpu_output);
    std::cout << "Parity: " << (parity_ok ? "PASS" : "FAIL") << '\n';
    std::cout << "Max abs error: " << max_abs_error << '\n';

    checkCuda(cudaFree(workspace), "cudaFree(workspace)");
    checkCuda(cudaFree(device_output), "cudaFree(output)");
    checkCuda(cudaFree(device_filter), "cudaFree(filter)");
    checkCuda(cudaFree(device_input), "cudaFree(input)");
    checkCudnn(cudnnDestroyConvolutionDescriptor(conv_desc), "cudnnDestroyConvolutionDescriptor");
    checkCudnn(cudnnDestroyFilterDescriptor(filter_desc), "cudnnDestroyFilterDescriptor");
    checkCudnn(cudnnDestroyTensorDescriptor(output_desc), "cudnnDestroyTensorDescriptor(output)");
    checkCudnn(cudnnDestroyTensorDescriptor(input_desc), "cudnnDestroyTensorDescriptor(input)");
    checkCudnn(cudnnDestroy(handle), "cudnnDestroy");

    return parity_ok ? 0 : 1;
}
