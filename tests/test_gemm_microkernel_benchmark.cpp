#include "gemm/matmul_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool kernel_available(const gemm::KernelSpec& spec) {
    if (spec.isa == gemm::KernelIsa::Avx2) {
        return gemm::cpu_supports_avx2_fma();
    }
    if (spec.isa == gemm::KernelIsa::Avx512) {
        return gemm::cpu_supports_avx512f();
    }
    return false;
}

enum class BenchmarkMode {
    Timing,
    Perf,
};

BenchmarkMode parse_mode(const std::string& mode) {
    if (mode == "timing") return BenchmarkMode::Timing;
    if (mode == "perf") return BenchmarkMode::Perf;
    throw std::invalid_argument("Unknown benchmark mode: " + mode);
}

void run_hot_loop(const gemm::KernelSpec& spec,
                  const Scalar* packed_a,
                  const Scalar* packed_b,
                  Scalar* c,
                  size_t ldc,
                  size_t kc,
                  uint64_t calls) {
    for (uint64_t i = 0; i < calls; ++i) {
        spec.fn(packed_a, packed_b, c, ldc, kc);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string kernel_name = "avx2_6x8";
    size_t kc = 256;
    int reps = 200;
    int inner_iters = 10000;
    BenchmarkMode mode = BenchmarkMode::Timing;
    uint64_t perf_call_scale = 1;

    if (argc > 1) kernel_name = argv[1];
    if (argc > 2) kc = static_cast<size_t>(std::stoul(argv[2]));
    if (argc > 3) reps = std::stoi(argv[3]);
    if (argc > 4) inner_iters = std::stoi(argv[4]);
    if (argc > 5) mode = parse_mode(argv[5]);
    if (argc > 6) perf_call_scale = static_cast<uint64_t>(std::stoull(argv[6]));

    const gemm::KernelSpec* spec = gemm::find_kernel_by_name(kernel_name.c_str());
    if (!spec) {
        std::cerr << "Unknown kernel: " << kernel_name << "\n";
        return 1;
    }
    if (!kernel_available(*spec)) {
        std::cerr << "Kernel unavailable on this CPU: " << kernel_name << "\n";
        return 2;
    }

    const size_t ldc = spec->nr + 8;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<Scalar> packed_a(kc * spec->mr);
    std::vector<Scalar> packed_b(kc * spec->nr);
    std::vector<Scalar> c(spec->mr * ldc, 0.0f);

    for (Scalar& v : packed_a) v = dist(rng);
    for (Scalar& v : packed_b) v = dist(rng);
    for (Scalar& v : c) v = dist(rng) * 0.01f;

    run_hot_loop(*spec, packed_a.data(), packed_b.data(), c.data(), ldc, kc, 100);

    const uint64_t timed_calls = static_cast<uint64_t>(reps) * static_cast<uint64_t>(inner_iters);
    const uint64_t perf_calls = timed_calls * perf_call_scale;
    const uint64_t flops_per_call = 2ull * spec->mr * spec->nr * kc;
    const uint64_t perf_flops = flops_per_call * perf_calls;

    if (mode == BenchmarkMode::Perf) {
        run_hot_loop(*spec, packed_a.data(), packed_b.data(), c.data(), ldc, kc, perf_calls);

        double checksum = 0.0;
        for (size_t i = 0; i < std::min<size_t>(c.size(), 32); ++i) checksum += c[i];

        std::cout << "Mode: perf\n";
        std::cout << "Kernel: " << spec->name
                  << ", MR: " << spec->mr
                  << ", NR: " << spec->nr
                  << ", KC: " << kc << "\n";
        std::cout << "TimedCalls: " << timed_calls
                  << ", PerfCallScale: " << perf_call_scale << "\n";
        std::cout << "PerfCalls: " << perf_calls
                  << ", FLOPsPerCall: " << flops_per_call
                  << ", PerfFLOPs: " << perf_flops << "\n";
        std::cout << "Result check: " << checksum << "\n";
        return 0;
    }

    std::vector<double> timings;
    timings.reserve(reps);

    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_hot_loop(*spec, packed_a.data(), packed_b.data(), c.data(), ldc, kc,
                     static_cast<uint64_t>(inner_iters));
        auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> diff = t1 - t0;
        timings.push_back(diff.count() / static_cast<double>(inner_iters));
    }

    std::sort(timings.begin(), timings.end());
    size_t trim = static_cast<size_t>(timings.size() * 0.10);
    if (trim * 2 >= timings.size()) trim = 0;

    auto begin = timings.begin() + static_cast<std::ptrdiff_t>(trim);
    auto end = timings.end() - static_cast<std::ptrdiff_t>(trim);

    double mean = 0.0;
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        mean += *it;
        ++count;
    }
    mean = count ? mean / static_cast<double>(count) : 0.0;

    double sq_sum = 0.0;
    for (auto it = begin; it != end; ++it) {
        const double d = *it - mean;
        sq_sum += d * d;
    }
    const double stddev = count > 1 ? std::sqrt(sq_sum / static_cast<double>(count - 1)) : 0.0;

    double checksum = 0.0;
    for (size_t i = 0; i < std::min<size_t>(c.size(), 32); ++i) checksum += c[i];

    std::cout << "Mode: timing\n";
    std::cout << "Done. Mean: " << std::fixed << std::setprecision(12) << mean
              << " s, StdDev: " << stddev
              << " s, Reps: " << reps
              << ", InnerIters: " << inner_iters << "\n";
    std::cout << "Kernel: " << spec->name
              << ", MR: " << spec->mr
              << ", NR: " << spec->nr
              << ", KC: " << kc << "\n";
    std::cout << "TimedCalls: " << timed_calls
              << ", PerfCallScale: " << perf_call_scale << "\n";
    std::cout << "PerfCalls: " << perf_calls
              << ", FLOPsPerCall: " << flops_per_call
              << ", PerfFLOPs: " << perf_flops << "\n";
    std::cout << "Result check: " << checksum << "\n";

    return 0;
}
