#include "math_ops.hpp"
#include "node.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double run_chain_benchmark(std::size_t iterations) {
    const Matrix x_val(1, 32, 0.5);
    auto start = Clock::now();
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        auto x = std::make_shared<Node>(x_val);
        auto y = x;
        for (int depth = 0; depth < 64; ++depth) {
            y = MathOps::relu(MathOps::add(y, y));
        }
        auto loss = MathOps::sum(y);
        loss->backward();
    }
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double run_fanin_benchmark(std::size_t iterations) {
    const Matrix x_val(1, 32, 0.5);
    auto start = Clock::now();
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        auto x = std::make_shared<Node>(x_val);
        std::vector<Node::Ptr> branches;
        branches.reserve(16);
        for (int i = 0; i < 16; ++i) {
            branches.push_back(MathOps::relu(MathOps::mul(x, x)));
        }
        auto acc = branches[0];
        for (std::size_t i = 1; i < branches.size(); ++i) {
            acc = MathOps::add(acc, branches[i]);
        }
        auto loss = MathOps::sum(acc);
        loss->backward();
    }
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double run_large_tensor_benchmark(std::size_t iterations) {
    Matrix a_val(256, 256, 0.01);
    Matrix b_val(256, 256, 0.02);
    auto start = Clock::now();
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        auto a = std::make_shared<Node>(a_val);
        auto b = std::make_shared<Node>(b_val);
        auto y = MathOps::matmul(a, b);
        auto loss = MathOps::sum(y);
        loss->backward();
    }
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_result(const std::string& name, double seconds, std::size_t iterations) {
    const double avg_ms = (seconds * 1000.0) / static_cast<double>(iterations);
    std::cout << std::fixed << std::setprecision(3)
              << name << ": total_s=" << seconds
              << ", avg_ms=" << avg_ms
              << ", iterations=" << iterations
              << '\n';
}

} // namespace

int main() {
    const std::size_t chain_iters = 50;
    const std::size_t fanin_iters = 50;
    const std::size_t large_iters = 5;

    print_result("chain_small_tensor", run_chain_benchmark(chain_iters), chain_iters);
    print_result("fanin_small_tensor", run_fanin_benchmark(fanin_iters), fanin_iters);
    print_result("large_tensor", run_large_tensor_benchmark(large_iters), large_iters);
    return 0;
}
