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
    timings.reserve(2000); // Reserve rough max

    // Fixed Repetitions Configuration
    // User requested: Always run exactly max_reps times (no adaptive stopping)
    int min_reps = 100;
    int max_reps = 2000; 
    
    // If user provides reps > 0 via command line, use it as FIXED value
    if (reps > 0) {
        max_reps = reps;
        min_reps = reps;  // Force exact repetitions
    }

    int current_reps = 0;
    double current_mean = 0.0;
    double current_std = 0.0;
    
    // Run EXACTLY max_reps iterations (no early stopping)
    while (current_reps < max_reps) {
        auto t0 = std::chrono::high_resolution_clock::now();
        C = A.matmul(B);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> run_diff = t1 - t0;
        timings.push_back(run_diff.count());
        current_reps++;
        
        // Calculate running stats (for monitoring, but no early exit)
        double sum = 0.0;
        for (double t : timings) sum += t;
        current_mean = sum / current_reps;
        
        double sq_sum = 0.0;
        for (double t : timings) sq_sum += (t - current_mean) * (t - current_mean);
        current_std = (current_reps > 1) ? std::sqrt(sq_sum / (current_reps - 1)) : 0.0;
        
        // NO convergence check - always run to max_reps
    }

    // Final Post-Processing (Trimming Outliers as before for robust result)
    std::vector<double> filtered = timings;
    std::sort(filtered.begin(), filtered.end());
    size_t trim = static_cast<size_t>(filtered.size() * 0.10);
    if (trim * 2 >= filtered.size()) trim = 0;

    auto beginIt = filtered.begin() + trim;
    auto endIt   = filtered.end() - trim;
    
    // Final Mean
    double f_sum = 0.0;
    size_t f_count = 0;
    for (auto it = beginIt; it != endIt; ++it) { f_sum += *it; f_count++; }
    double final_mean = (f_count > 0) ? f_sum / f_count : 0.0;

    // Final StdDev
    double f_sq_sum = 0.0;
    for (auto it = beginIt; it != endIt; ++it) f_sq_sum += (*it - final_mean) * (*it - final_mean);
    double final_std = (f_count > 1) ? std::sqrt(f_sq_sum / (f_count - 1)) : 0.0;

    // Output output to be parsed: "Mean: <MEAN> s, StdDev: <STDDEV> s, Reps: <REPS>"
    // Note: Added Reps to output so script can parse it
    std::cout << "Done. Mean: " << std::fixed << std::setprecision(12) << final_mean 
              << " s, StdDev: " << final_std << " s, Reps: " << current_reps << std::endl;
    
    // Prevent optimization
    std::cout << "Result check: " << C.data[0] << std::endl;

    return 0;
}
