#include <cuda_runtime.h>

#include <iostream>

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: "
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }

    std::cout << "CUDA device count: " << device_count << '\n';
    for (int device_index = 0; device_index < device_count; ++device_index) {
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, device_index);
        if (status != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties failed for device "
                      << device_index << ": "
                      << cudaGetErrorString(status) << '\n';
            return 1;
        }

        std::cout << "Device " << device_index << ": "
                  << properties.name << '\n';
    }

    return 0;
}
