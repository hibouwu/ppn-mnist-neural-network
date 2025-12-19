#include "tensor.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

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
    size_t M = 1024, K = 1024, N = 1024; // Defaults
    int reps = 1;

    // Cases:
    // 1. ./bench size reps
    // 2. ./bench M K N reps
    if (argc == 3) {
        M = K = N = std::stoi(argv[1]);
        reps = std::stoi(argv[2]);
    } else if (argc == 5) {
        M = std::stoi(argv[1]);
        K = std::stoi(argv[2]);
        N = std::stoi(argv[3]);
        reps = std::stoi(argv[4]);
    } else if (argc > 1) {
        M = K = N = std::stoi(argv[1]);
        if (argc > 2) reps = std::stoi(argv[2]);
    }

    // std::cout << "Benchmarking Matrix Multiplication (" << M << "x" << K << " * " << K << "x" << N << "), Reps: " << reps << "..." << std::endl;

    // Create matrices A (MxK) and B (KxN)
    Matrix A = random_matrix(M, K);
    Matrix B = random_matrix(K, N);
    Matrix C(M, N); 

    // Warmup (Always run a few times to heat up cache)
    // For very small sizes, we might want more warmups, but 5 is a safe minimum.
    int warmups = 5;
    for (int i = 0; i < warmups; ++i) {
        C = A.matmul(B);
    }

    std::vector<double> timings;
    timings.reserve(reps);

    // Perform multiplication Reps times
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        C = A.matmul(B);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> run_diff = t1 - t0;
        timings.push_back(run_diff.count());
    }

    // Drop the lowest/highest 10% to reduce outlier influence
    std::vector<double> filtered = timings;
    std::sort(filtered.begin(), filtered.end());
    size_t trim = static_cast<size_t>(filtered.size() * 0.10);
    if (trim * 2 >= filtered.size()) {
        trim = 0; // Not enough samples to trim
    }
    auto beginIt = filtered.begin() + trim;
    auto endIt   = filtered.end() - trim;

    // Compute Mean on trimmed sample
    double sum = 0.0;
    size_t used = static_cast<size_t>(std::distance(beginIt, endIt));
    for (auto it = beginIt; it != endIt; ++it) sum += *it;
    double mean = sum / static_cast<double>(used);

    // Compute StdDev on trimmed sample
    double sq_sum = 0.0;
    for (auto it = beginIt; it != endIt; ++it) sq_sum += (*it - mean) * (*it - mean);
    double stddev = (used > 1) ? std::sqrt(sq_sum / static_cast<double>(used - 1)) : 0.0;

    // Output output to be parsed: "Mean: <MEAN> s, StdDev: <STDDEV> s"
    std::cout << "Done. Mean: " << std::fixed << std::setprecision(12) << mean 
              << " s, StdDev: " << stddev << " s" << std::endl;
    
    // Prevent optimization
    std::cout << "Result check: " << C.data[0] << std::endl;

    return 0;
}
