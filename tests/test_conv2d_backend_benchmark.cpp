#include "autograd/engine.hpp"
#include "cnn_network.hpp"
#include "conv2d_layer.hpp"
#include "loss.hpp"
#include "math_ops.hpp"
#include "optimizer.hpp"
#include "profiling.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkResult {
    double mean_ms = 0.0;
    std::vector<OpTimingStat> op_stats;
};

Matrix makeConvInput(std::size_t batch, std::size_t channels, std::size_t height, std::size_t width) {
    Matrix input(batch, channels * height * width);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = 0.01 * static_cast<double>((i % 97) + 1);
    }
    return input;
}

Matrix makeTargets(std::size_t batch, std::size_t num_classes) {
    Matrix target(batch, num_classes, 0.0);
    for (std::size_t n = 0; n < batch; ++n) {
        target(n, n % num_classes) = 1.0;
    }
    return target;
}

void loadConvParameters(Conv2DLayer& conv) {
    auto params = conv.parameters();
    Matrix& kernels = const_cast<Matrix&>(params[0]->value());
    Matrix& bias = const_cast<Matrix&>(params[1]->value());
    for (std::size_t i = 0; i < kernels.data.size(); ++i) {
        kernels.data[i] = -0.1 + 0.002 * static_cast<double>((i % 71) + 1);
    }
    for (std::size_t i = 0; i < bias.data.size(); ++i) {
        bias.data[i] = -0.05 + 0.01 * static_cast<double>(i);
    }
}

void zeroParameterGrads(const std::vector<Node::Ptr>& params) {
    for (const auto& p : params) {
        p->zeroGrad();
    }
}

std::unordered_map<std::string, OpTimingStat> toMap(const std::vector<OpTimingStat>& stats) {
    std::unordered_map<std::string, OpTimingStat> out;
    for (const auto& stat : stats) {
        out.emplace(stat.name, stat);
    }
    return out;
}

void printSelectedStats(const std::vector<OpTimingStat>& stats,
                        const std::vector<std::string>& names,
                        std::size_t iterations) {
    const auto by_name = toMap(stats);
    for (const auto& name : names) {
        const auto it = by_name.find(name);
        if (it == by_name.end()) {
            continue;
        }
        const double avg_us = it->second.calls > 0
            ? static_cast<double>(it->second.total_us) / static_cast<double>(it->second.calls)
            : 0.0;
        const double per_iter_us = iterations > 0
            ? static_cast<double>(it->second.total_us) / static_cast<double>(iterations)
            : 0.0;
        std::cout << "  - " << name
                  << ": calls=" << it->second.calls
                  << ", total_us=" << it->second.total_us
                  << ", avg_call_us=" << std::fixed << std::setprecision(2) << avg_us
                  << ", per_iter_us=" << per_iter_us
                  << '\n';
    }
}

template <typename Fn>
BenchmarkResult runTimedLoop(std::size_t warmup_iters,
                             std::size_t measure_iters,
                             Fn&& fn) {
    for (std::size_t i = 0; i < warmup_iters; ++i) {
        fn();
    }

    opProfileEpochReset();
    const auto start = Clock::now();
    for (std::size_t i = 0; i < measure_iters; ++i) {
        fn();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
    return BenchmarkResult{
        static_cast<double>(elapsed) / 1000.0 / static_cast<double>(measure_iters),
        opProfileEpochSnapshot()
    };
}

BenchmarkResult benchmarkSingleConvForward(ConvBackend backend,
                                           std::size_t warmup_iters,
                                           std::size_t measure_iters) {
    const Matrix input = makeConvInput(32, 8, 28, 28);
    Conv2DLayer conv(8, 16, 3, 3, 1, 1, backend);
    loadConvParameters(conv);
    return runTimedLoop(warmup_iters, measure_iters, [&] {
        auto x = constant(input);
        volatile std::size_t sink = conv.forward(x, 32, 8, 28, 28)->value().cols;
        (void)sink;
    });
}

BenchmarkResult benchmarkSingleConvForwardBackward(ConvBackend backend,
                                                   std::size_t warmup_iters,
                                                   std::size_t measure_iters) {
    const Matrix input = makeConvInput(32, 8, 28, 28);
    Conv2DLayer conv(8, 16, 3, 3, 1, 1, backend);
    loadConvParameters(conv);
    return runTimedLoop(warmup_iters, measure_iters, [&] {
        zeroParameterGrads(conv.parameters());
        auto x = std::make_shared<Node>(input);
        auto loss = MathOps::sum(conv.forward(x, 32, 8, 28, 28));
        AutogradEngine engine;
        engine.backward(loss);
    });
}

BenchmarkResult benchmarkMiniCnnTrainStep(ConvBackend backend,
                                          std::size_t warmup_iters,
                                          std::size_t measure_iters) {
    CNNConfig cfg;
    cfg.input_channels = 1;
    cfg.input_height = 28;
    cfg.input_width = 28;
    cfg.conv_channels = {8};
    cfg.conv_kernels = {3};
    cfg.conv_strides = {1};
    cfg.conv_paddings = {1};
    cfg.pool_after = {true};
    cfg.pool_kernels = {2};
    cfg.pool_strides = {2};
    cfg.fc_hidden_sizes = {32};
    cfg.num_classes = 10;
    cfg.conv_backend = backend;

    CNNNetwork net(cfg, 42);
    SGDOptimizer optimizer(net.getParameters(), 0.01);
    CrossEntropyLoss loss_fn;
    const Matrix input = makeConvInput(32, 1, 28, 28);
    const Matrix target = makeTargets(32, 10);

    return runTimedLoop(warmup_iters, measure_iters, [&] {
        optimizer.zeroGrad();
        auto x = constant(input);
        auto y = constant(target);
        auto logits = net.forward(x);
        auto loss = loss_fn.forward(logits, y);
        AutogradEngine engine;
        engine.backward(loss);
        optimizer.step();
    });
}

void printBenchmarkSection(const std::string& title,
                           const std::string& backend_name,
                           const BenchmarkResult& result,
                           const std::vector<std::string>& op_names,
                           std::size_t iterations) {
    std::cout << title << " backend=" << backend_name
              << " mean_ms=" << std::fixed << std::setprecision(3) << result.mean_ms << '\n';
    printSelectedStats(result.op_stats, op_names, iterations);
}

} // namespace

int main() {
#if !PPN_HAVE_ONEDNN_CONV_BACKEND
    std::cout << "Conv2D backend benchmark skipped: binary built without oneDNN backend support." << std::endl;
    return 0;
#else
    const char* omp_threads = std::getenv("OMP_NUM_THREADS");
    std::cout << "Benchmark config: OMP_NUM_THREADS=" << (omp_threads ? omp_threads : "unset")
              << ", batch=32, seed=42, warmup=3, measure=10" << '\n';

    const std::size_t warmup_iters = 3;
    const std::size_t measure_iters = 10;

    const BenchmarkResult ref_forward =
        benchmarkSingleConvForward(ConvBackend::Reference, warmup_iters, measure_iters);
    const BenchmarkResult dnn_forward =
        benchmarkSingleConvForward(ConvBackend::OneDnn, warmup_iters, measure_iters);

    const BenchmarkResult ref_forward_backward =
        benchmarkSingleConvForwardBackward(ConvBackend::Reference, warmup_iters, measure_iters);
    const BenchmarkResult dnn_forward_backward =
        benchmarkSingleConvForwardBackward(ConvBackend::OneDnn, warmup_iters, measure_iters);

    const BenchmarkResult ref_train_step =
        benchmarkMiniCnnTrainStep(ConvBackend::Reference, warmup_iters, measure_iters);
    const BenchmarkResult dnn_train_step =
        benchmarkMiniCnnTrainStep(ConvBackend::OneDnn, warmup_iters, measure_iters);

    std::cout << "[single_conv_forward]" << '\n';
    printBenchmarkSection(
        "single_conv_forward",
        "reference",
        ref_forward,
        {"conv2d_forward_im2col", "conv2d_forward_gemm", "conv2d_forward_bias_reshape"},
        measure_iters);
    printBenchmarkSection(
        "single_conv_forward",
        "onednn",
        dnn_forward,
        {"conv2d_onednn_forward_setup", "conv2d_onednn_forward_bridge",
         "conv2d_onednn_forward_reorder", "conv2d_onednn_forward_execute"},
        measure_iters);

    std::cout << "[single_conv_forward_backward]" << '\n';
    printBenchmarkSection(
        "single_conv_forward_backward",
        "reference",
        ref_forward_backward,
        {"conv2d_forward_im2col", "conv2d_forward_gemm", "conv2d_forward_bias_reshape",
         "conv2d_backward_grad_reshape", "conv2d_backward_dW_gemm",
         "conv2d_backward_db_reduce", "conv2d_backward_dX_gemm", "conv2d_backward_col2im"},
        measure_iters);
    printBenchmarkSection(
        "single_conv_forward_backward",
        "onednn",
        dnn_forward_backward,
        {"conv2d_onednn_forward_setup", "conv2d_onednn_forward_bridge",
         "conv2d_onednn_forward_reorder", "conv2d_onednn_forward_execute",
         "conv2d_onednn_backward_setup", "conv2d_onednn_backward_bridge",
         "conv2d_onednn_backward_reorder", "conv2d_onednn_backward_data_execute",
         "conv2d_onednn_backward_weights_execute"},
        measure_iters);

    std::cout << "[mini_cnn_train_step]" << '\n';
    printBenchmarkSection(
        "mini_cnn_train_step",
        "reference",
        ref_train_step,
        {"conv2d_forward_im2col", "conv2d_forward_gemm", "conv2d_forward_bias_reshape",
         "conv2d_backward_grad_reshape", "conv2d_backward_dW_gemm",
         "conv2d_backward_db_reduce", "conv2d_backward_dX_gemm", "conv2d_backward_col2im"},
        measure_iters);
    printBenchmarkSection(
        "mini_cnn_train_step",
        "onednn",
        dnn_train_step,
        {"conv2d_onednn_forward_setup", "conv2d_onednn_forward_bridge",
         "conv2d_onednn_forward_reorder", "conv2d_onednn_forward_execute",
         "conv2d_onednn_backward_setup", "conv2d_onednn_backward_bridge",
         "conv2d_onednn_backward_reorder", "conv2d_onednn_backward_data_execute",
         "conv2d_onednn_backward_weights_execute"},
        measure_iters);

    return 0;
#endif
}
