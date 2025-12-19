#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "layer.hpp"
#include "tensor.hpp"

// Utility to calculate mean and stddev of a matrix
std::pair<double, double> calc_stats(const Matrix& m) {
    double sum = 0.0;
    double sq_sum = 0.0;
    size_t n = m.rows * m.cols;
    for (double val : m.data) {
        sum += val;
        sq_sum += val * val;
    }
    double mean = sum / n;
    double var = (sq_sum / n) - (mean * mean);
    return {mean, std::sqrt(var)};
}

int main() {
    std::cout << "Running Initialization Tests..." << std::endl;

    // 1. Test Seed Reproducibility
    unsigned int seed = 12345;
    Matrix m1(10, 10);
    m1.randomInit(-1.0, 1.0, false, seed);
    
    Matrix m2(10, 10);
    m2.randomInit(-1.0, 1.0, false, seed);

    for(size_t i=0; i<m1.data.size(); ++i) {
        assert(std::abs(m1.data[i] - m2.data[i]) < 1e-9);
    }
    std::cout << "[PASS] Seed Reproducibility (Matrix Level)" << std::endl;

    // 2. Test He Initialization (Large layer for stats)
    size_t in = 1000;
    size_t out = 1000;
    LinearLayer heLayer(in, out);
    heLayer.randomInit(0, 0, LinearLayer::InitType::He, seed); 
    // He StdDev = sqrt(2/in) = sqrt(2/1000) = sqrt(0.002) ≈ 0.0447
    
    // Hack verify: LinearLayer doesn't expose weights easily unless we inspect node
    // But parameters() returns {weights, bias}
    auto params = heLayer.parameters();
    // Assuming params[0] is weights (layer.cpp: returns {weights_, bias_})
    const Matrix& w_he = params[0]->value();
    
    auto stats = calc_stats(w_he);
    std::cout << "He Init Stats: Mean=" << stats.first << " (Exp ~0), Std=" << stats.second << " (Exp " << std::sqrt(2.0/in) << ")" << std::endl;
    assert(std::abs(stats.first) < 0.01);
    assert(std::abs(stats.second - std::sqrt(2.0/in)) < 0.01); 
    std::cout << "[PASS] He Initialization Statistics" << std::endl;

    // 3. Test Xavier Initialization
    LinearLayer xavierLayer(in, out);
    xavierLayer.randomInit(0, 0, LinearLayer::InitType::Xavier, seed);
    // Xavier StdDev = sqrt(2/(in+out)) = sqrt(1/1000) ≈ 0.0316
    
    auto paramsX = xavierLayer.parameters();
    const Matrix& w_x = paramsX[0]->value();
    auto statsX = calc_stats(w_x);
    std::cout << "Xavier Init Stats: Mean=" << statsX.first << " (Exp ~0), Std=" << statsX.second << " (Exp " << std::sqrt(2.0/(in+out)) << ")" << std::endl;
    assert(std::abs(statsX.first) < 0.01);
    assert(std::abs(statsX.second - std::sqrt(2.0/(in+out))) < 0.01);
    std::cout << "[PASS] Xavier Initialization Statistics" << std::endl;

    std::cout << "All initialization tests passed!" << std::endl;
    return 0;
}
