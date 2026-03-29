#include "autograd/engine.hpp"
#include "conv2d_layer.hpp"
#include "node.hpp"
#include "tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if ONEDNN_PARITY_PROBE_HAVE_DNNL
#include <dnnl.hpp>
#endif

namespace {

constexpr double kContractTol = 1e-9;
constexpr double kParityTol = 1e-4;

enum class ProbeStatus {
    Pass,
    SkipCompileNoOneDnn,
    SkipNoEngine,
    FailRuntime,
    FailParity,
    FailContract,
};

const char* toString(ProbeStatus status) {
    switch (status) {
    case ProbeStatus::Pass: return "PASS";
    case ProbeStatus::SkipCompileNoOneDnn: return "SKIP-COMPILE-NO-ONEDNN";
    case ProbeStatus::SkipNoEngine: return "SKIP-NO-ENGINE";
    case ProbeStatus::FailRuntime: return "FAIL-RUNTIME";
    case ProbeStatus::FailParity: return "FAIL-PARITY";
    case ProbeStatus::FailContract: return "FAIL-CONTRACT";
    }
    return "UNKNOWN";
}

struct ContractFailure : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct NoEngineFailure : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct PrimitiveRuntimeFailure : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ParityFailure : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct ConvCase {
    std::size_t N;
    std::size_t C;
    std::size_t H;
    std::size_t W;
    std::size_t O;
    std::size_t kH;
    std::size_t kW;
    std::size_t stride;
    std::size_t padding;
};

struct ReorderSummary {
    bool src = false;
    bool weights = false;
    bool bias = false;
    bool dst = false;
};

struct ReferenceResult {
    Matrix output;
    Matrix diff_src;
    Matrix diff_weights;
    Matrix diff_bias;
};

struct ProbeResult {
    Matrix output;
    Matrix diff_src;
    Matrix diff_weights;
    Matrix diff_bias;
    ReorderSummary forward;
    ReorderSummary backward_data;
    ReorderSummary backward_weights;
};

struct ComparisonSummary {
    std::string name;
    bool pass = true;
    double tolerance = kContractTol;
    double max_abs_error = 0.0;
    std::size_t max_index = 0;
    double actual_at_max = 0.0;
    double expected_at_max = 0.0;
    std::size_t rows = 0;
    std::size_t cols = 0;
};

struct ProbeReport {
    ProbeStatus status = ProbeStatus::Pass;
    std::string message;
    std::vector<ComparisonSummary> comparisons;
    bool contract_verified = false;
    bool compiled_with_onednn = false;
    bool engine_created = false;
    bool attempted_execution = false;
    bool execution_completed = false;
    std::size_t H_out = 0;
    std::size_t W_out = 0;
    ReorderSummary forward;
    ReorderSummary backward_data;
    ReorderSummary backward_weights;
};

std::int64_t i64(std::size_t value) {
    return static_cast<std::int64_t>(value);
}

std::size_t inputFlatIndex(const ConvCase& cfg,
                           std::size_t c,
                           std::size_t h,
                           std::size_t w) {
    return c * cfg.H * cfg.W + h * cfg.W + w;
}

std::size_t outputFlatIndex(std::size_t H_out,
                            std::size_t W_out,
                            std::size_t oc,
                            std::size_t oh,
                            std::size_t ow) {
    return oc * H_out * W_out + oh * W_out + ow;
}

bool almostEqual(double a, double b, double eps = kContractTol) {
    return std::abs(a - b) <= eps;
}

Matrix makeInput(const ConvCase& cfg) {
    Matrix input(cfg.N, cfg.C * cfg.H * cfg.W);
    for (std::size_t n = 0; n < cfg.N; ++n) {
        for (std::size_t c = 0; c < cfg.C; ++c) {
            for (std::size_t h = 0; h < cfg.H; ++h) {
                for (std::size_t w = 0; w < cfg.W; ++w) {
                    const std::size_t flat = inputFlatIndex(cfg, c, h, w);
                    input(n, flat) = 0.05 * static_cast<double>(1 + n * 100 + c * 10 + h * 3 + w);
                }
            }
        }
    }
    return input;
}

Matrix makeWeights(const ConvCase& cfg) {
    Matrix weights(cfg.O, cfg.C * cfg.kH * cfg.kW);
    for (std::size_t oc = 0; oc < cfg.O; ++oc) {
        for (std::size_t ic = 0; ic < cfg.C; ++ic) {
            for (std::size_t kh = 0; kh < cfg.kH; ++kh) {
                for (std::size_t kw = 0; kw < cfg.kW; ++kw) {
                    const std::size_t col = ic * cfg.kH * cfg.kW + kh * cfg.kW + kw;
                    weights(oc, col) = 0.01 * static_cast<double>(1 + oc * 50 + ic * 7 + kh * 3 + kw);
                }
            }
        }
    }
    return weights;
}

Matrix makeBias(const ConvCase& cfg) {
    Matrix bias(1, cfg.O);
    for (std::size_t oc = 0; oc < cfg.O; ++oc) {
        bias(0, oc) = -0.2 + 0.15 * static_cast<double>(oc);
    }
    return bias;
}

Matrix makeGradOutput(std::size_t N,
                      std::size_t O,
                      std::size_t H_out,
                      std::size_t W_out) {
    Matrix grad_output(N, O * H_out * W_out);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t oc = 0; oc < O; ++oc) {
            for (std::size_t oh = 0; oh < H_out; ++oh) {
                for (std::size_t ow = 0; ow < W_out; ++ow) {
                    const std::size_t flat = outputFlatIndex(H_out, W_out, oc, oh, ow);
                    grad_output(n, flat) = 0.02 * static_cast<double>(1 + n * 80 + oc * 9 + oh * 2 + ow);
                }
            }
        }
    }
    return grad_output;
}

void verifyMatrixContractAgainstNchwOihwView(const ConvCase& cfg,
                                             const Matrix& input,
                                             const Matrix& weights) {
    if (input.rows != cfg.N || input.cols != cfg.C * cfg.H * cfg.W) {
        throw ContractFailure("Input Matrix shape is incompatible with external NCHW view.");
    }
    if (weights.rows != cfg.O || weights.cols != cfg.C * cfg.kH * cfg.kW) {
        throw ContractFailure("Weight Matrix shape is incompatible with external OIHW view.");
    }

    for (std::size_t n = 0; n < cfg.N; ++n) {
        for (std::size_t c = 0; c < cfg.C; ++c) {
            for (std::size_t h = 0; h < cfg.H; ++h) {
                for (std::size_t w = 0; w < cfg.W; ++w) {
                    const std::size_t flat = inputFlatIndex(cfg, c, h, w);
                    const double via_matrix = input(n, flat);
                    const double via_nchw_view = input.data[n * input.cols + flat];
                    if (!almostEqual(via_matrix, via_nchw_view)) {
                        throw ContractFailure("Input Matrix storage does not match external NCHW indexing.");
                    }
                }
            }
        }
    }

    for (std::size_t oc = 0; oc < cfg.O; ++oc) {
        for (std::size_t ic = 0; ic < cfg.C; ++ic) {
            for (std::size_t kh = 0; kh < cfg.kH; ++kh) {
                for (std::size_t kw = 0; kw < cfg.kW; ++kw) {
                    const std::size_t col = ic * cfg.kH * cfg.kW + kh * cfg.kW + kw;
                    const double via_matrix = weights(oc, col);
                    const double via_oihw_view = weights.data[oc * weights.cols + col];
                    if (!almostEqual(via_matrix, via_oihw_view)) {
                        throw ContractFailure("Weight Matrix storage does not match external OIHW indexing.");
                    }
                }
            }
        }
    }
}

Matrix manualForwardNchwOihw(const Matrix& input,
                             const Matrix& weights,
                             const Matrix& bias,
                             const ConvCase& cfg,
                             std::size_t H_out,
                             std::size_t W_out) {
    Matrix output(cfg.N, cfg.O * H_out * W_out, 0.0);
    for (std::size_t n = 0; n < cfg.N; ++n) {
        for (std::size_t oc = 0; oc < cfg.O; ++oc) {
            for (std::size_t oh = 0; oh < H_out; ++oh) {
                for (std::size_t ow = 0; ow < W_out; ++ow) {
                    double acc = bias(0, oc);
                    for (std::size_t ic = 0; ic < cfg.C; ++ic) {
                        for (std::size_t kh = 0; kh < cfg.kH; ++kh) {
                            for (std::size_t kw = 0; kw < cfg.kW; ++kw) {
                                const int ih = static_cast<int>(oh * cfg.stride + kh) - static_cast<int>(cfg.padding);
                                const int iw = static_cast<int>(ow * cfg.stride + kw) - static_cast<int>(cfg.padding);
                                if (ih < 0 || iw < 0 ||
                                    ih >= static_cast<int>(cfg.H) ||
                                    iw >= static_cast<int>(cfg.W)) {
                                    continue;
                                }
                                const std::size_t input_flat = inputFlatIndex(
                                    cfg, ic, static_cast<std::size_t>(ih), static_cast<std::size_t>(iw));
                                const std::size_t weight_col = ic * cfg.kH * cfg.kW + kh * cfg.kW + kw;
                                acc += input(n, input_flat) * weights(oc, weight_col);
                            }
                        }
                    }
                    output(n, outputFlatIndex(H_out, W_out, oc, oh, ow)) = acc;
                }
            }
        }
    }
    return output;
}

ReferenceResult runReferenceAutograd(const Matrix& input,
                                     const Matrix& weights,
                                     const Matrix& bias,
                                     const Matrix& grad_output,
                                     const ConvCase& cfg) {
    Conv2DLayer conv(cfg.C, cfg.O, cfg.kH, cfg.kW, cfg.stride, cfg.padding);
    auto params = conv.parameters();
    Matrix& kernels_value = const_cast<Matrix&>(params[0]->value());
    Matrix& bias_value = const_cast<Matrix&>(params[1]->value());
    kernels_value = weights;
    bias_value = bias;

    auto x = std::make_shared<Node>(input);
    auto out = conv.forward(x, cfg.N, cfg.C, cfg.H, cfg.W);
    out->addGrad(grad_output);

    AutogradEngine engine;
    engine.backward(out);

    return ReferenceResult{
        out->value(),
        x->grad(),
        params[0]->grad(),
        params[1]->grad()
    };
}

ComparisonSummary summarizeComparison(const Matrix& actual,
                                     const Matrix& expected,
                                     const std::string& name,
                                     double tolerance) {
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        throw ParityFailure(name + ": shape mismatch.");
    }

    ComparisonSummary summary;
    summary.name = name;
    summary.tolerance = tolerance;
    summary.rows = actual.rows;
    summary.cols = actual.cols;

    for (std::size_t i = 0; i < actual.data.size(); ++i) {
        const double err = std::abs(actual.data[i] - expected.data[i]);
        if (err > summary.max_abs_error) {
            summary.max_abs_error = err;
            summary.max_index = i;
            summary.actual_at_max = actual.data[i];
            summary.expected_at_max = expected.data[i];
        }
    }
    summary.pass = summary.max_abs_error <= tolerance;
    return summary;
}

void printComparisonSummary(const ComparisonSummary& summary) {
    std::cout
        << "CHECK " << summary.name
        << " status=" << (summary.pass ? "PASS" : "FAIL")
        << " tol=" << summary.tolerance
        << " max_abs_error=" << summary.max_abs_error
        << " max_index=" << summary.max_index
        << " actual=" << summary.actual_at_max
        << " expected=" << summary.expected_at_max
        << " shape=(" << summary.rows << "," << summary.cols << ")"
        << '\n';
}

void printBoundarySummary(const ReorderSummary& summary,
                          const std::string& phase) {
    std::cout << "BOUNDARY " << phase
              << " src=" << (summary.src ? "reorder" : "zero-copy")
              << " weights=" << (summary.weights ? "reorder" : "zero-copy")
              << " bias=" << (summary.bias ? "reorder" : "zero-copy")
              << " dst=" << (summary.dst ? "reorder" : "zero-copy")
              << '\n';
}

void printExternalContractPolicy() {
    std::cout
        << "CONTRACT external Matrix->oneDNN user-view policy: "
        << "src NCHW zero-copy, weights OIHW zero-copy, bias X zero-copy, "
        << "dst/diff tensors zero-copy when exposed as explicit-stride user memory; "
        << "reorder is required only when a primitive selects a different internal descriptor."
        << '\n';
}

void printEnvironmentSummary(const ProbeReport& report) {
    std::cout
        << "ENV compiled_with_onednn=" << (report.compiled_with_onednn ? "yes" : "no")
        << " contract_verified=" << (report.contract_verified ? "yes" : "no")
        << " engine_created=" << (report.engine_created ? "yes" : "no")
        << " attempted_execution=" << (report.attempted_execution ? "yes" : "no")
        << " execution_completed=" << (report.execution_completed ? "yes" : "no")
        << " output_spatial=(" << report.H_out << "," << report.W_out << ")"
        << '\n';
}

#if ONEDNN_PARITY_PROBE_HAVE_DNNL

dnnl::memory::desc makeUserNchwMd(const ConvCase& cfg,
                                  dnnl::memory::data_type dt) {
    return dnnl::memory::desc(
        {i64(cfg.N), i64(cfg.C), i64(cfg.H), i64(cfg.W)},
        dt,
        {i64(cfg.C * cfg.H * cfg.W), i64(cfg.H * cfg.W), i64(cfg.W), 1});
}

dnnl::memory::desc makeUserOihwMd(const ConvCase& cfg,
                                  dnnl::memory::data_type dt) {
    return dnnl::memory::desc(
        {i64(cfg.O), i64(cfg.C), i64(cfg.kH), i64(cfg.kW)},
        dt,
        {i64(cfg.C * cfg.kH * cfg.kW), i64(cfg.kH * cfg.kW), i64(cfg.kW), 1});
}

dnnl::memory::desc makeUserDstMd(const ConvCase& cfg,
                                 std::size_t H_out,
                                 std::size_t W_out,
                                 dnnl::memory::data_type dt) {
    return dnnl::memory::desc(
        {i64(cfg.N), i64(cfg.O), i64(H_out), i64(W_out)},
        dt,
        {i64(cfg.O * H_out * W_out), i64(H_out * W_out), i64(W_out), 1});
}

dnnl::memory::desc makeBiasMd(const ConvCase& cfg,
                              dnnl::memory::data_type dt) {
    return dnnl::memory::desc(
        {i64(cfg.O)},
        dt,
        dnnl::memory::format_tag::x);
}

std::vector<float> toFloatBuffer(const Matrix& matrix) {
    std::vector<float> buffer(matrix.data.size());
    for (std::size_t i = 0; i < matrix.data.size(); ++i) {
        buffer[i] = static_cast<float>(matrix.data[i]);
    }
    return buffer;
}

void copyFloatBufferToMatrix(const std::vector<float>& buffer, Matrix& matrix) {
    if (buffer.size() != matrix.data.size()) {
        throw PrimitiveRuntimeFailure("Float bridge buffer size does not match Matrix size.");
    }
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        matrix.data[i] = static_cast<double>(buffer[i]);
    }
}

template <typename Fn>
void runPrimitiveStep(const std::string& label, Fn&& fn) {
    try {
        fn();
    } catch (const dnnl::error& err) {
        throw PrimitiveRuntimeFailure(label + ": " + err.what());
    }
}

ProbeResult runOneDnnProbe(const Matrix& input,
                           const Matrix& weights,
                           const Matrix& bias,
                           const Matrix& grad_output,
                           const ConvCase& cfg,
                           std::size_t H_out,
                           std::size_t W_out,
                           ProbeReport& report) {
    dnnl::engine engine = [&]() {
        try {
            return dnnl::engine(dnnl::engine::kind::cpu, 0);
        } catch (const dnnl::error& err) {
            throw NoEngineFailure(err.what());
        }
    }();
    report.engine_created = true;

    dnnl::stream stream(engine);
    const auto src_md = makeUserNchwMd(cfg, dnnl::memory::data_type::f32);
    const auto weights_md = makeUserOihwMd(cfg, dnnl::memory::data_type::f32);
    const auto bias_md = makeBiasMd(cfg, dnnl::memory::data_type::f32);
    const auto dst_md = makeUserDstMd(cfg, H_out, W_out, dnnl::memory::data_type::f32);
    const dnnl::memory::dims strides = {i64(cfg.stride), i64(cfg.stride)};
    const dnnl::memory::dims paddings = {i64(cfg.padding), i64(cfg.padding)};

    ProbeResult result{
        Matrix(cfg.N, cfg.O * H_out * W_out, 0.0),
        Matrix(cfg.N, cfg.C * cfg.H * cfg.W, 0.0),
        Matrix(cfg.O, cfg.C * cfg.kH * cfg.kW, 0.0),
        Matrix(1, cfg.O, 0.0),
        {},
        {},
        {}
    };

    report.attempted_execution = true;

    std::vector<float> input_f32 = toFloatBuffer(input);
    std::vector<float> weights_f32 = toFloatBuffer(weights);
    std::vector<float> bias_f32 = toFloatBuffer(bias);
    std::vector<float> grad_output_f32 = toFloatBuffer(grad_output);
    std::vector<float> output_f32(result.output.data.size(), 0.0f);
    std::vector<float> diff_src_f32(result.diff_src.data.size(), 0.0f);
    std::vector<float> diff_weights_f32(result.diff_weights.data.size(), 0.0f);
    std::vector<float> diff_bias_f32(result.diff_bias.data.size(), 0.0f);

    auto src_user = dnnl::memory(src_md, engine, input_f32.data());
    auto weights_user = dnnl::memory(weights_md, engine, weights_f32.data());
    auto bias_user = dnnl::memory(bias_md, engine, bias_f32.data());
    auto dst_user = dnnl::memory(dst_md, engine, output_f32.data());

    dnnl::convolution_forward::primitive_desc forward_pd = [&]() {
        try {
            return dnnl::convolution_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_training,
                dnnl::algorithm::convolution_direct,
                src_md,
                weights_md,
                bias_md,
                dst_md,
                strides,
                paddings,
                paddings);
        } catch (const dnnl::error& err) {
            throw PrimitiveRuntimeFailure(std::string("forward primitive_desc: ") + err.what());
        }
    }();

    result.forward.src = forward_pd.src_desc() != src_md;
    result.forward.weights = forward_pd.weights_desc() != weights_md;
    result.forward.bias = forward_pd.bias_desc() != bias_md;
    result.forward.dst = forward_pd.dst_desc() != dst_md;

    auto src_exec = result.forward.src ? dnnl::memory(forward_pd.src_desc(), engine) : src_user;
    auto weights_exec = result.forward.weights ? dnnl::memory(forward_pd.weights_desc(), engine) : weights_user;
    auto bias_exec = result.forward.bias ? dnnl::memory(forward_pd.bias_desc(), engine) : bias_user;
    auto dst_exec = result.forward.dst ? dnnl::memory(forward_pd.dst_desc(), engine) : dst_user;

    if (result.forward.src) {
        runPrimitiveStep("forward reorder src", [&] { dnnl::reorder(src_user, src_exec).execute(stream, src_user, src_exec); });
    }
    if (result.forward.weights) {
        runPrimitiveStep("forward reorder weights", [&] { dnnl::reorder(weights_user, weights_exec).execute(stream, weights_user, weights_exec); });
    }
    if (result.forward.bias) {
        runPrimitiveStep("forward reorder bias", [&] { dnnl::reorder(bias_user, bias_exec).execute(stream, bias_user, bias_exec); });
    }

    runPrimitiveStep("forward execute", [&] {
        dnnl::convolution_forward(forward_pd).execute(
            stream,
            {
                {DNNL_ARG_SRC, src_exec},
                {DNNL_ARG_WEIGHTS, weights_exec},
                {DNNL_ARG_BIAS, bias_exec},
                {DNNL_ARG_DST, dst_exec},
            });
    });

    if (result.forward.dst) {
        runPrimitiveStep("forward reorder dst", [&] { dnnl::reorder(dst_exec, dst_user).execute(stream, dst_exec, dst_user); });
    }

    auto diff_src_user = dnnl::memory(src_md, engine, diff_src_f32.data());
    auto diff_weights_user = dnnl::memory(weights_md, engine, diff_weights_f32.data());
    auto diff_bias_user = dnnl::memory(bias_md, engine, diff_bias_f32.data());
    auto diff_dst_user = dnnl::memory(dst_md, engine, grad_output_f32.data());

    dnnl::convolution_backward_data::primitive_desc backward_data_pd = [&]() {
        try {
            return dnnl::convolution_backward_data::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                src_md,
                weights_md,
                dst_md,
                strides,
                paddings,
                paddings,
                forward_pd);
        } catch (const dnnl::error& err) {
            throw PrimitiveRuntimeFailure(std::string("backward_data primitive_desc: ") + err.what());
        }
    }();

    result.backward_data.src = backward_data_pd.diff_src_desc() != src_md;
    result.backward_data.weights = backward_data_pd.weights_desc() != weights_md;
    result.backward_data.dst = backward_data_pd.diff_dst_desc() != dst_md;

    auto bd_weights_exec = result.backward_data.weights ? dnnl::memory(backward_data_pd.weights_desc(), engine) : weights_user;
    auto bd_diff_dst_exec = result.backward_data.dst ? dnnl::memory(backward_data_pd.diff_dst_desc(), engine) : diff_dst_user;
    auto bd_diff_src_exec = result.backward_data.src ? dnnl::memory(backward_data_pd.diff_src_desc(), engine) : diff_src_user;

    if (result.backward_data.weights) {
        runPrimitiveStep("backward_data reorder weights", [&] { dnnl::reorder(weights_user, bd_weights_exec).execute(stream, weights_user, bd_weights_exec); });
    }
    if (result.backward_data.dst) {
        runPrimitiveStep("backward_data reorder diff_dst", [&] { dnnl::reorder(diff_dst_user, bd_diff_dst_exec).execute(stream, diff_dst_user, bd_diff_dst_exec); });
    }

    runPrimitiveStep("backward_data execute", [&] {
        dnnl::convolution_backward_data(backward_data_pd).execute(
            stream,
            {
                {DNNL_ARG_DIFF_DST, bd_diff_dst_exec},
                {DNNL_ARG_WEIGHTS, bd_weights_exec},
                {DNNL_ARG_DIFF_SRC, bd_diff_src_exec},
            });
    });

    if (result.backward_data.src) {
        runPrimitiveStep("backward_data reorder diff_src", [&] { dnnl::reorder(bd_diff_src_exec, diff_src_user).execute(stream, bd_diff_src_exec, diff_src_user); });
    }

    dnnl::convolution_backward_weights::primitive_desc backward_weights_pd = [&]() {
        try {
            return dnnl::convolution_backward_weights::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                src_md,
                weights_md,
                bias_md,
                dst_md,
                strides,
                paddings,
                paddings,
                forward_pd);
        } catch (const dnnl::error& err) {
            throw PrimitiveRuntimeFailure(std::string("backward_weights primitive_desc: ") + err.what());
        }
    }();

    result.backward_weights.src = backward_weights_pd.src_desc() != src_md;
    result.backward_weights.weights = backward_weights_pd.diff_weights_desc() != weights_md;
    result.backward_weights.bias = backward_weights_pd.diff_bias_desc() != bias_md;
    result.backward_weights.dst = backward_weights_pd.diff_dst_desc() != dst_md;

    auto bw_src_exec = result.backward_weights.src ? dnnl::memory(backward_weights_pd.src_desc(), engine) : src_user;
    auto bw_diff_dst_exec = result.backward_weights.dst ? dnnl::memory(backward_weights_pd.diff_dst_desc(), engine) : diff_dst_user;
    auto bw_diff_weights_exec = result.backward_weights.weights ? dnnl::memory(backward_weights_pd.diff_weights_desc(), engine) : diff_weights_user;
    auto bw_diff_bias_exec = result.backward_weights.bias ? dnnl::memory(backward_weights_pd.diff_bias_desc(), engine) : diff_bias_user;

    if (result.backward_weights.src) {
        runPrimitiveStep("backward_weights reorder src", [&] { dnnl::reorder(src_user, bw_src_exec).execute(stream, src_user, bw_src_exec); });
    }
    if (result.backward_weights.dst) {
        runPrimitiveStep("backward_weights reorder diff_dst", [&] { dnnl::reorder(diff_dst_user, bw_diff_dst_exec).execute(stream, diff_dst_user, bw_diff_dst_exec); });
    }

    runPrimitiveStep("backward_weights execute", [&] {
        dnnl::convolution_backward_weights(backward_weights_pd).execute(
            stream,
            {
                {DNNL_ARG_SRC, bw_src_exec},
                {DNNL_ARG_DIFF_DST, bw_diff_dst_exec},
                {DNNL_ARG_DIFF_WEIGHTS, bw_diff_weights_exec},
                {DNNL_ARG_DIFF_BIAS, bw_diff_bias_exec},
            });
    });

    if (result.backward_weights.weights) {
        runPrimitiveStep("backward_weights reorder diff_weights", [&] { dnnl::reorder(bw_diff_weights_exec, diff_weights_user).execute(stream, bw_diff_weights_exec, diff_weights_user); });
    }
    if (result.backward_weights.bias) {
        runPrimitiveStep("backward_weights reorder diff_bias", [&] { dnnl::reorder(bw_diff_bias_exec, diff_bias_user).execute(stream, bw_diff_bias_exec, diff_bias_user); });
    }

    runPrimitiveStep("stream wait", [&] { stream.wait(); });
    copyFloatBufferToMatrix(output_f32, result.output);
    copyFloatBufferToMatrix(diff_src_f32, result.diff_src);
    copyFloatBufferToMatrix(diff_weights_f32, result.diff_weights);
    copyFloatBufferToMatrix(diff_bias_f32, result.diff_bias);
    report.execution_completed = true;
    return result;
}

#endif

ProbeReport runProbeGate() {
    ProbeReport report;
    report.compiled_with_onednn =
#if ONEDNN_PARITY_PROBE_HAVE_DNNL
        true;
#else
        false;
#endif

    const ConvCase cfg{2, 2, 4, 5, 3, 3, 2, 1, 1};
    const Matrix input = makeInput(cfg);
    const Matrix weights = makeWeights(cfg);
    const Matrix bias = makeBias(cfg);

    try {
        verifyMatrixContractAgainstNchwOihwView(cfg, input, weights);
        report.contract_verified = true;

        Conv2DLayer shape_probe(cfg.C, cfg.O, cfg.kH, cfg.kW, cfg.stride, cfg.padding);
        {
            auto params = shape_probe.parameters();
            Matrix& kernels_value = const_cast<Matrix&>(params[0]->value());
            Matrix& bias_value = const_cast<Matrix&>(params[1]->value());
            kernels_value = weights;
            bias_value = bias;
        }
        const auto [H_out, W_out] = shape_probe.outputShape(cfg.H, cfg.W);
        report.H_out = H_out;
        report.W_out = W_out;

        const Matrix manual = manualForwardNchwOihw(input, weights, bias, cfg, H_out, W_out);
        const Matrix reference_forward = shape_probe.forward(
            std::make_shared<Node>(input), cfg.N, cfg.C, cfg.H, cfg.W)->value();
        const ComparisonSummary contract_forward = summarizeComparison(
            reference_forward, manual, "reference_vs_manual_forward", kContractTol);
        report.comparisons.push_back(contract_forward);
        if (!contract_forward.pass) {
            throw ContractFailure("Reference Conv2D forward does not match the external NCHW/OIHW manual formula.");
        }

        const Matrix grad_output = makeGradOutput(cfg.N, cfg.O, H_out, W_out);
        const ReferenceResult reference = runReferenceAutograd(input, weights, bias, grad_output, cfg);

#if !ONEDNN_PARITY_PROBE_HAVE_DNNL
        report.status = ProbeStatus::SkipCompileNoOneDnn;
        report.message = "Probe target was built without oneDNN headers/libraries.";
        return report;
#else
        const ProbeResult probe = runOneDnnProbe(input, weights, bias, grad_output, cfg, H_out, W_out, report);
        report.forward = probe.forward;
        report.backward_data = probe.backward_data;
        report.backward_weights = probe.backward_weights;

        report.comparisons.push_back(summarizeComparison(probe.output, reference.output, "forward_output", kParityTol));
        report.comparisons.push_back(summarizeComparison(probe.diff_src, reference.diff_src, "diff_src", kParityTol));
        report.comparisons.push_back(summarizeComparison(probe.diff_weights, reference.diff_weights, "diff_weights", kParityTol));
        report.comparisons.push_back(summarizeComparison(probe.diff_bias, reference.diff_bias, "diff_bias", kParityTol));

        for (const auto& summary : report.comparisons) {
            if (summary.name == "reference_vs_manual_forward") {
                continue;
            }
            if (!summary.pass) {
                throw ParityFailure(summary.name + " exceeded tolerance.");
            }
        }

        report.status = ProbeStatus::Pass;
        report.message = "All oneDNN conv parity checks passed within tolerance.";
        return report;
#endif
    } catch (const ContractFailure& err) {
        report.status = ProbeStatus::FailContract;
        report.message = err.what();
        return report;
    }
#if ONEDNN_PARITY_PROBE_HAVE_DNNL
    catch (const NoEngineFailure& err) {
        report.status = ProbeStatus::SkipNoEngine;
        report.message = err.what();
        return report;
    } catch (const PrimitiveRuntimeFailure& err) {
        report.status = ProbeStatus::FailRuntime;
        report.message = err.what();
        return report;
    } catch (const ParityFailure& err) {
        report.status = ProbeStatus::FailParity;
        report.message = err.what();
        return report;
    }
#endif
}

int exitCodeForStatus(ProbeStatus status) {
    switch (status) {
    case ProbeStatus::Pass:
    case ProbeStatus::SkipCompileNoOneDnn:
    case ProbeStatus::SkipNoEngine:
        return 0;
    case ProbeStatus::FailRuntime:
        return 2;
    case ProbeStatus::FailParity:
        return 3;
    case ProbeStatus::FailContract:
        return 4;
    }
    return 5;
}

} // namespace

int main() {
    printExternalContractPolicy();
    const ProbeReport report = runProbeGate();

    printEnvironmentSummary(report);
    for (const auto& summary : report.comparisons) {
        printComparisonSummary(summary);
    }

#if ONEDNN_PARITY_PROBE_HAVE_DNNL
    if (report.engine_created || report.attempted_execution || report.execution_completed) {
        printBoundarySummary(report.forward, "forward");
        printBoundarySummary(report.backward_data, "backward_data");
        printBoundarySummary(report.backward_weights, "backward_weights");
    }
#endif

    std::cout << "RESULT " << toString(report.status) << " message=\"" << report.message << '"' << '\n';
    return exitCodeForStatus(report.status);
}
