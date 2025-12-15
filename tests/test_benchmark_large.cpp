#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

// Simple random matrix generator
Matrix random_matrix(size_t rows, size_t cols) {
    Matrix m(rows, cols);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < rows * cols; ++i) {
        m.data[i] = dis(gen);
    }
    return m;
}

int main(int argc, char** argv) {
    size_t size = 1024; // Default to 1024x1024
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }

    std::cout << "Benchmarking Matrix Multiplication (" << size << "x" << size << ")..." << std::endl;

    // Create two large random matrices
    Matrix A = random_matrix(size, size);
    Matrix B = random_matrix(size, size);

    // Warmup
    std::cout << "Warming up..." << std::endl;
    A.matmul(B); 

    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform multiplication
    Matrix C = A.matmul(B);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Done in " << std::fixed << std::setprecision(9) << diff.count() << " seconds." << std::endl;
    // Prevent optimization
    std::cout << "Result check: " << C.data[0] << std::endl;

    return 0;
}
