#include <cudnn.h>

#include <iostream>

int main() {
    std::cout << "cuDNN version: " << cudnnGetVersion() << '\n';

    cudnnHandle_t handle = nullptr;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnCreate failed: "
                  << cudnnGetErrorString(status) << '\n';
        return 1;
    }

    status = cudnnDestroy(handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cudnnDestroy failed: "
                  << cudnnGetErrorString(status) << '\n';
        return 1;
    }

    return 0;
}
