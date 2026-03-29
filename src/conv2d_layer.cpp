/**
 * @file conv2d_layer.cpp
 * @brief 2D convolution layer implementation using im2col + matmul.
 */
#include "conv2d_layer.hpp"
#include "autograd/backward_context.hpp"
#include "autograd/grad_fn.hpp"
#include "math_ops.hpp"
#include "operation_node.hpp"
#include "profiling.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <chrono>

#if PPN_HAVE_ONEDNN_CONV_BACKEND
#include <dnnl.hpp>
#endif

class Conv2DLayer::Backend {
public:
    struct Config {
        std::size_t in_channels = 0;
        std::size_t out_channels = 0;
        std::size_t kernel_h = 0;
        std::size_t kernel_w = 0;
        std::size_t stride = 0;
        std::size_t padding = 0;
    };

    struct Parameters {
        const Node::Ptr& kernels;
        const Node::Ptr& bias;
    };

    struct InputShape {
        std::size_t N = 0;
        std::size_t C = 0;
        std::size_t H = 0;
        std::size_t W = 0;
        std::size_t H_out = 0;
        std::size_t W_out = 0;
    };

    struct ForwardResult {
        Matrix output;
        std::shared_ptr<BackwardContext> backward_context;
        std::shared_ptr<GradFn> grad_fn;
    };

    virtual ~Backend() = default;

    virtual ForwardResult forward(const Matrix& input,
                                  const InputShape& shape,
                                  const Config& config,
                                  const Parameters& params) const = 0;
};

namespace {
bool cnn_parallel_enabled() {
    static int mode = -1; // -1: uninit, 0: naive, 1: parallel
    if (mode == -1) {
        const char* v = std::getenv("CNN_PARALLEL");
        mode = (v && std::strcmp(v, "1") == 0) ? 1 : 0;
    }
    return mode == 1;
}

bool inferRequiresGrad(const Node::Ptr& input,
                       const Node::Ptr& kernels,
                       const Node::Ptr& bias) {
    return (input && input->requiresGrad()) ||
           (kernels && kernels->requiresGrad()) ||
           (bias && bias->requiresGrad());
}

Matrix col2imFromContext(const Matrix& cols,
                         std::size_t N,
                         std::size_t C,
                         std::size_t H,
                         std::size_t W,
                         std::size_t H_out,
                         std::size_t W_out,
                         std::size_t kernel_h,
                         std::size_t kernel_w,
                         std::size_t stride,
                         std::size_t padding) {
    Matrix input_grad(N, C * H * W, 0.0);
    const double* cols_data = cols.data.data();
    double* input_grad_data = input_grad.data.data();
    const std::size_t cols_stride = cols.cols;
    const std::size_t input_grad_stride = input_grad.cols;
    const std::size_t HW = H * W;
    const std::size_t NHW_out = H_out * W_out;

    #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t in_n_base = n * input_grad_stride;
        const std::size_t row_n_base = n * NHW_out;
        for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
                const std::size_t row = row_n_base + oh * W_out + ow;
                const std::size_t row_base = row * cols_stride;
                std::size_t col_idx = 0;
                for (std::size_t c = 0; c < C; ++c) {
                    const std::size_t in_c_base = c * HW;
                    for (std::size_t kh = 0; kh < kernel_h; ++kh) {
                        for (std::size_t kw = 0; kw < kernel_w; ++kw) {
                            const int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                            const int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                            if (ih >= 0 && ih < static_cast<int>(H) &&
                                iw >= 0 && iw < static_cast<int>(W)) {
                                const std::size_t input_idx =
                                    in_c_base + static_cast<std::size_t>(ih) * W + static_cast<std::size_t>(iw);
                                input_grad_data[in_n_base + input_idx] += cols_data[row_base + col_idx];
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }
    return input_grad;
}

Matrix referenceIm2col(const Matrix& input,
                       const Conv2DLayer::Backend::InputShape& shape,
                       const Conv2DLayer::Backend::Config& config) {
    const std::size_t col_h = config.kernel_h * config.kernel_w * shape.C;
    Matrix cols(shape.N * shape.H_out * shape.W_out, col_h);
    const double* input_data = input.data.data();
    double* cols_data = cols.data.data();
    const std::size_t input_stride = input.cols;
    const std::size_t HW = shape.H * shape.W;
    const std::size_t NHW_out = shape.H_out * shape.W_out;

    #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)
    for (std::size_t n = 0; n < shape.N; ++n) {
        const std::size_t in_n_base = n * input_stride;
        const std::size_t out_n_row_base = n * NHW_out;
        for (std::size_t oh = 0; oh < shape.H_out; ++oh) {
            for (std::size_t ow = 0; ow < shape.W_out; ++ow) {
                const std::size_t row = out_n_row_base + oh * shape.W_out + ow;
                const std::size_t row_base = row * col_h;
                std::size_t col_idx = 0;
                for (std::size_t c = 0; c < shape.C; ++c) {
                    const std::size_t in_c_base = c * HW;
                    for (std::size_t kh = 0; kh < config.kernel_h; ++kh) {
                        for (std::size_t kw = 0; kw < config.kernel_w; ++kw) {
                            const int ih = static_cast<int>(oh * config.stride + kh) - static_cast<int>(config.padding);
                            const int iw = static_cast<int>(ow * config.stride + kw) - static_cast<int>(config.padding);
                            if (ih >= 0 && ih < static_cast<int>(shape.H) &&
                                iw >= 0 && iw < static_cast<int>(shape.W)) {
                                const std::size_t input_idx =
                                    in_c_base + static_cast<std::size_t>(ih) * shape.W + static_cast<std::size_t>(iw);
                                cols_data[row_base + col_idx] = input_data[in_n_base + input_idx];
                            } else {
                                cols_data[row_base + col_idx] = 0.0;
                            }
                            ++col_idx;
                        }
                    }
                }
            }
        }
    }
    return cols;
}

#if PPN_HAVE_ONEDNN_CONV_BACKEND

#ifdef PROFILE_OPS
using ProfileClock = std::chrono::steady_clock;
#endif

std::vector<float> toFloatBuffer(const Matrix& matrix) {
    std::vector<float> buffer(matrix.data.size());
    for (std::size_t i = 0; i < matrix.data.size(); ++i) {
        buffer[i] = static_cast<float>(matrix.data[i]);
    }
    return buffer;
}

void copyFloatBufferToMatrix(const std::vector<float>& buffer, Matrix& matrix) {
    if (buffer.size() != matrix.data.size()) {
        throw std::logic_error("oneDNN Conv2D bridge buffer size mismatch.");
    }
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        matrix.data[i] = static_cast<double>(buffer[i]);
    }
}

dnnl::memory::desc makeOneDnnSrcMd(const Conv2DLayer::Backend::InputShape& shape) {
    return dnnl::memory::desc(
        {static_cast<dnnl::memory::dim>(shape.N),
         static_cast<dnnl::memory::dim>(shape.C),
         static_cast<dnnl::memory::dim>(shape.H),
         static_cast<dnnl::memory::dim>(shape.W)},
        dnnl::memory::data_type::f32,
        {static_cast<dnnl::memory::dim>(shape.C * shape.H * shape.W),
         static_cast<dnnl::memory::dim>(shape.H * shape.W),
         static_cast<dnnl::memory::dim>(shape.W),
         1});
}

dnnl::memory::desc makeOneDnnWeightsMd(const Conv2DLayer::Backend::InputShape& shape,
                                       const Conv2DLayer::Backend::Config& config) {
    return dnnl::memory::desc(
        {static_cast<dnnl::memory::dim>(config.out_channels),
         static_cast<dnnl::memory::dim>(shape.C),
         static_cast<dnnl::memory::dim>(config.kernel_h),
         static_cast<dnnl::memory::dim>(config.kernel_w)},
        dnnl::memory::data_type::f32,
        {static_cast<dnnl::memory::dim>(shape.C * config.kernel_h * config.kernel_w),
         static_cast<dnnl::memory::dim>(config.kernel_h * config.kernel_w),
         static_cast<dnnl::memory::dim>(config.kernel_w),
         1});
}

dnnl::memory::desc makeOneDnnBiasMd(const Conv2DLayer::Backend::Config& config) {
    return dnnl::memory::desc(
        {static_cast<dnnl::memory::dim>(config.out_channels)},
        dnnl::memory::data_type::f32,
        dnnl::memory::format_tag::x);
}

dnnl::memory::desc makeOneDnnDstMd(const Conv2DLayer::Backend::InputShape& shape,
                                   const Conv2DLayer::Backend::Config& config) {
    return dnnl::memory::desc(
        {static_cast<dnnl::memory::dim>(shape.N),
         static_cast<dnnl::memory::dim>(config.out_channels),
         static_cast<dnnl::memory::dim>(shape.H_out),
         static_cast<dnnl::memory::dim>(shape.W_out)},
        dnnl::memory::data_type::f32,
        {static_cast<dnnl::memory::dim>(config.out_channels * shape.H_out * shape.W_out),
         static_cast<dnnl::memory::dim>(shape.H_out * shape.W_out),
         static_cast<dnnl::memory::dim>(shape.W_out),
         1});
}

template <typename Fn>
void runOneDnnStep(const char* label, Fn&& fn) {
    try {
        fn();
    } catch (const dnnl::error& err) {
        throw std::runtime_error(std::string("Conv2DLayer oneDNN ") + label + ": " + err.what());
    }
}

dnnl::convolution_forward::primitive_desc makeOneDnnForwardPd(
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& weights_md,
    const dnnl::memory::desc& bias_md,
    const dnnl::memory::desc& dst_md,
    const Conv2DLayer::Backend::Config& config) {
    try {
        return dnnl::convolution_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct,
            src_md,
            weights_md,
            bias_md,
            dst_md,
            {static_cast<dnnl::memory::dim>(config.stride), static_cast<dnnl::memory::dim>(config.stride)},
            {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)},
            {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)});
    } catch (const dnnl::error& err) {
        throw std::runtime_error(std::string("Conv2DLayer oneDNN forward primitive_desc: ") + err.what());
    }
}

Matrix runOneDnnForwardValue(const Matrix& input,
                             const Matrix& weights,
                             const Matrix& bias,
                             const Conv2DLayer::Backend::InputShape& shape,
                             const Conv2DLayer::Backend::Config& config) {
#ifdef PROFILE_OPS
    const auto setup_start = ProfileClock::now();
#endif
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    const auto src_md = makeOneDnnSrcMd(shape);
    const auto weights_md = makeOneDnnWeightsMd(shape, config);
    const auto bias_md = makeOneDnnBiasMd(config);
    const auto dst_md = makeOneDnnDstMd(shape, config);

    const auto forward_pd = makeOneDnnForwardPd(engine, src_md, weights_md, bias_md, dst_md, config);
#ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_forward_setup",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - setup_start).count());
    const auto bridge_start = ProfileClock::now();
#endif
    std::vector<float> src_f32 = toFloatBuffer(input);
    std::vector<float> weights_f32 = toFloatBuffer(weights);
    std::vector<float> bias_f32 = toFloatBuffer(bias);
    Matrix output(shape.N, config.out_channels * shape.H_out * shape.W_out, 0.0);
    std::vector<float> dst_f32(output.data.size(), 0.0f);

    auto src_user = dnnl::memory(src_md, engine, src_f32.data());
    auto weights_user = dnnl::memory(weights_md, engine, weights_f32.data());
    auto bias_user = dnnl::memory(bias_md, engine, bias_f32.data());
    auto dst_user = dnnl::memory(dst_md, engine, dst_f32.data());
    auto src_exec = forward_pd.src_desc() == src_md ? src_user : dnnl::memory(forward_pd.src_desc(), engine);
    auto weights_exec = forward_pd.weights_desc() == weights_md ? weights_user : dnnl::memory(forward_pd.weights_desc(), engine);
    auto bias_exec = forward_pd.bias_desc() == bias_md ? bias_user : dnnl::memory(forward_pd.bias_desc(), engine);
    auto dst_exec = forward_pd.dst_desc() == dst_md ? dst_user : dnnl::memory(forward_pd.dst_desc(), engine);
#ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_forward_bridge",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - bridge_start).count());
    long long reorder_total_us = 0;
    const auto execute_start = ProfileClock::now();
#endif

    if (forward_pd.src_desc() != src_md) {
#ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
#endif
        runOneDnnStep("forward reorder src", [&] { dnnl::reorder(src_user, src_exec).execute(stream, src_user, src_exec); });
#ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
#endif
    }
    if (forward_pd.weights_desc() != weights_md) {
#ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
#endif
        runOneDnnStep("forward reorder weights", [&] { dnnl::reorder(weights_user, weights_exec).execute(stream, weights_user, weights_exec); });
#ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
#endif
    }
    if (forward_pd.bias_desc() != bias_md) {
#ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
#endif
        runOneDnnStep("forward reorder bias", [&] { dnnl::reorder(bias_user, bias_exec).execute(stream, bias_user, bias_exec); });
#ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
#endif
    }

    runOneDnnStep("forward execute", [&] {
        dnnl::convolution_forward(forward_pd).execute(
            stream,
            {
                {DNNL_ARG_SRC, src_exec},
                {DNNL_ARG_WEIGHTS, weights_exec},
                {DNNL_ARG_BIAS, bias_exec},
                {DNNL_ARG_DST, dst_exec},
            });
    });

    if (forward_pd.dst_desc() != dst_md) {
#ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
#endif
        runOneDnnStep("forward reorder dst", [&] { dnnl::reorder(dst_exec, dst_user).execute(stream, dst_exec, dst_user); });
#ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
#endif
    }
    runOneDnnStep("forward stream wait", [&] { stream.wait(); });
#ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_forward_execute",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - execute_start).count());
    if (reorder_total_us > 0) {
        opProfileRecord("conv2d_onednn_forward_reorder", reorder_total_us);
    }
    const auto bridge_out_start = ProfileClock::now();
#endif
    copyFloatBufferToMatrix(dst_f32, output);
#ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_forward_bridge",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - bridge_out_start).count());
#endif
    return output;
}

struct OneDnnBackwardResult {
    Matrix diff_src;
    Matrix diff_weights;
    Matrix diff_bias;
};

OneDnnBackwardResult runOneDnnBackwardValue(const Matrix& input,
                                            const Matrix& weights,
                                            const Matrix& bias,
                                            const Matrix& grad_output,
                                            const Conv2DLayer::Backend::InputShape& shape,
                                            const Conv2DLayer::Backend::Config& config) {
    #ifdef PROFILE_OPS
    const auto setup_start = ProfileClock::now();
    #endif
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream stream(engine);

    const auto src_md = makeOneDnnSrcMd(shape);
    const auto weights_md = makeOneDnnWeightsMd(shape, config);
    const auto bias_md = makeOneDnnBiasMd(config);
    const auto dst_md = makeOneDnnDstMd(shape, config);

    OneDnnBackwardResult result{
        Matrix(shape.N, shape.C * shape.H * shape.W, 0.0),
        Matrix(config.out_channels, shape.C * config.kernel_h * config.kernel_w, 0.0),
        Matrix(1, config.out_channels, 0.0)
    };

    const auto forward_pd = makeOneDnnForwardPd(engine, src_md, weights_md, bias_md, dst_md, config);

    dnnl::convolution_backward_data::primitive_desc backward_data_pd = [&]() {
        try {
            return dnnl::convolution_backward_data::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                src_md,
                weights_md,
                dst_md,
                {static_cast<dnnl::memory::dim>(config.stride), static_cast<dnnl::memory::dim>(config.stride)},
                {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)},
                {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)},
                forward_pd);
        } catch (const dnnl::error& err) {
            throw std::runtime_error(std::string("Conv2DLayer oneDNN backward_data primitive_desc: ") + err.what());
        }
    }();

    dnnl::convolution_backward_weights::primitive_desc backward_weights_pd = [&]() {
        try {
            return dnnl::convolution_backward_weights::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                src_md,
                weights_md,
                bias_md,
                dst_md,
                {static_cast<dnnl::memory::dim>(config.stride), static_cast<dnnl::memory::dim>(config.stride)},
                {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)},
                {static_cast<dnnl::memory::dim>(config.padding), static_cast<dnnl::memory::dim>(config.padding)},
                forward_pd);
        } catch (const dnnl::error& err) {
            throw std::runtime_error(std::string("Conv2DLayer oneDNN backward_weights primitive_desc: ") + err.what());
        }
    }();

    #ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_backward_setup",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - setup_start).count());
    const auto bridge_start = ProfileClock::now();
    #endif
    std::vector<float> src_f32 = toFloatBuffer(input);
    std::vector<float> weights_f32 = toFloatBuffer(weights);
    std::vector<float> bias_f32 = toFloatBuffer(bias);
    std::vector<float> diff_dst_f32 = toFloatBuffer(grad_output);
    std::vector<float> diff_src_f32(result.diff_src.data.size(), 0.0f);
    std::vector<float> diff_weights_f32(result.diff_weights.data.size(), 0.0f);
    std::vector<float> diff_bias_f32(result.diff_bias.data.size(), 0.0f);

    auto src_user = dnnl::memory(src_md, engine, src_f32.data());
    auto weights_user = dnnl::memory(weights_md, engine, weights_f32.data());
    auto bias_user = dnnl::memory(bias_md, engine, bias_f32.data());
    auto diff_dst_user = dnnl::memory(dst_md, engine, diff_dst_f32.data());
    auto diff_src_user = dnnl::memory(src_md, engine, diff_src_f32.data());
    auto diff_weights_user = dnnl::memory(weights_md, engine, diff_weights_f32.data());
    auto diff_bias_user = dnnl::memory(bias_md, engine, diff_bias_f32.data());

    auto bd_weights_exec = backward_data_pd.weights_desc() == weights_md ? weights_user : dnnl::memory(backward_data_pd.weights_desc(), engine);
    auto bd_diff_dst_exec = backward_data_pd.diff_dst_desc() == dst_md ? diff_dst_user : dnnl::memory(backward_data_pd.diff_dst_desc(), engine);
    auto bd_diff_src_exec = backward_data_pd.diff_src_desc() == src_md ? diff_src_user : dnnl::memory(backward_data_pd.diff_src_desc(), engine);
    auto bw_src_exec = backward_weights_pd.src_desc() == src_md ? src_user : dnnl::memory(backward_weights_pd.src_desc(), engine);
    auto bw_diff_dst_exec = backward_weights_pd.diff_dst_desc() == dst_md ? diff_dst_user : dnnl::memory(backward_weights_pd.diff_dst_desc(), engine);
    auto bw_diff_weights_exec = backward_weights_pd.diff_weights_desc() == weights_md ? diff_weights_user : dnnl::memory(backward_weights_pd.diff_weights_desc(), engine);
    auto bw_diff_bias_exec = backward_weights_pd.diff_bias_desc() == bias_md ? diff_bias_user : dnnl::memory(backward_weights_pd.diff_bias_desc(), engine);
    #ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_backward_bridge",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - bridge_start).count());
    long long reorder_total_us = 0;
    #endif

    if (backward_data_pd.weights_desc() != weights_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_data reorder weights", [&] { dnnl::reorder(weights_user, bd_weights_exec).execute(stream, weights_user, bd_weights_exec); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }
    if (backward_data_pd.diff_dst_desc() != dst_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_data reorder diff_dst", [&] { dnnl::reorder(diff_dst_user, bd_diff_dst_exec).execute(stream, diff_dst_user, bd_diff_dst_exec); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }
    #ifdef PROFILE_OPS
    const auto backward_data_execute_start = ProfileClock::now();
    #endif
    runOneDnnStep("backward_data execute", [&] {
        dnnl::convolution_backward_data(backward_data_pd).execute(
            stream,
            {
                {DNNL_ARG_DIFF_DST, bd_diff_dst_exec},
                {DNNL_ARG_WEIGHTS, bd_weights_exec},
                {DNNL_ARG_DIFF_SRC, bd_diff_src_exec},
            });
    });
    #ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_backward_data_execute",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - backward_data_execute_start).count());
    #endif
    if (backward_data_pd.diff_src_desc() != src_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_data reorder diff_src", [&] { dnnl::reorder(bd_diff_src_exec, diff_src_user).execute(stream, bd_diff_src_exec, diff_src_user); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }

    if (backward_weights_pd.src_desc() != src_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_weights reorder src", [&] { dnnl::reorder(src_user, bw_src_exec).execute(stream, src_user, bw_src_exec); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }
    if (backward_weights_pd.diff_dst_desc() != dst_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_weights reorder diff_dst", [&] { dnnl::reorder(diff_dst_user, bw_diff_dst_exec).execute(stream, diff_dst_user, bw_diff_dst_exec); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }
    #ifdef PROFILE_OPS
    const auto backward_weights_execute_start = ProfileClock::now();
    #endif
    runOneDnnStep("backward_weights execute", [&] {
        dnnl::convolution_backward_weights(backward_weights_pd).execute(
            stream,
            {
                {DNNL_ARG_SRC, bw_src_exec},
                {DNNL_ARG_DIFF_DST, bw_diff_dst_exec},
                {DNNL_ARG_DIFF_WEIGHTS, bw_diff_weights_exec},
                {DNNL_ARG_DIFF_BIAS, bw_diff_bias_exec},
            });
    });
    #ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_backward_weights_execute",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - backward_weights_execute_start).count());
    #endif
    if (backward_weights_pd.diff_weights_desc() != weights_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_weights reorder diff_weights", [&] { dnnl::reorder(bw_diff_weights_exec, diff_weights_user).execute(stream, bw_diff_weights_exec, diff_weights_user); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }
    if (backward_weights_pd.diff_bias_desc() != bias_md) {
        #ifdef PROFILE_OPS
        const auto reorder_start = ProfileClock::now();
        #endif
        runOneDnnStep("backward_weights reorder diff_bias", [&] { dnnl::reorder(bw_diff_bias_exec, diff_bias_user).execute(stream, bw_diff_bias_exec, diff_bias_user); });
        #ifdef PROFILE_OPS
        reorder_total_us += std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - reorder_start).count();
        #endif
    }

    runOneDnnStep("backward stream wait", [&] { stream.wait(); });
    #ifdef PROFILE_OPS
    if (reorder_total_us > 0) {
        opProfileRecord("conv2d_onednn_backward_reorder", reorder_total_us);
    }
    const auto bridge_out_start = ProfileClock::now();
    #endif
    copyFloatBufferToMatrix(diff_src_f32, result.diff_src);
    copyFloatBufferToMatrix(diff_weights_f32, result.diff_weights);
    copyFloatBufferToMatrix(diff_bias_f32, result.diff_bias);
    #ifdef PROFILE_OPS
    opProfileRecord(
        "conv2d_onednn_backward_bridge",
        std::chrono::duration_cast<std::chrono::microseconds>(ProfileClock::now() - bridge_out_start).count());
    #endif
    return result;
}

class OneDnnConv2DGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (!ctx || input_indices.size() != inputs.size() || inputs.size() < 2 ||
            ctx->sizes.size() < 12 || ctx->flags.empty()) {
            throw std::logic_error("OneDnnConv2DGradFn: invalid node state.");
        }

        const Node::Ptr& input = inputs[0];
        const Node::Ptr& kernels = inputs[1];
        const bool has_bias = ctx->flags[0] != 0;
        const Node::Ptr bias = has_bias && inputs.size() >= 3 ? inputs[2] : nullptr;

        const Conv2DLayer::Backend::InputShape shape{
            ctx->sizes[0], ctx->sizes[1], ctx->sizes[2], ctx->sizes[3], ctx->sizes[4], ctx->sizes[5]
        };
        const Conv2DLayer::Backend::Config config{
            ctx->sizes[7], ctx->sizes[6], ctx->sizes[10], ctx->sizes[11], ctx->sizes[8], ctx->sizes[9]
        };

        const OneDnnBackwardResult gradients = runOneDnnBackwardValue(
            input->value(), kernels->value(), bias->value(), grad_output, shape, config);

        ContributionList contributions;
        if (input && input->requiresGrad() && input_indices[0] != kInvalidNodeIndex) {
            contributions.push_back({input_indices[0], gradients.diff_src});
        }
        if (kernels && kernels->requiresGrad() && input_indices[1] != kInvalidNodeIndex) {
            contributions.push_back({input_indices[1], gradients.diff_weights});
        }
        if (bias && bias->requiresGrad() && input_indices[2] != kInvalidNodeIndex) {
            contributions.push_back({input_indices[2], gradients.diff_bias});
        }
        return contributions;
    }
};

class OneDnnConv2DBackend final : public Conv2DLayer::Backend {
public:
    ForwardResult forward(const Matrix& input,
                          const InputShape& shape,
                          const Config& config,
                          const Parameters& params) const override {
        Matrix result = runOneDnnForwardValue(input, params.kernels->value(), params.bias->value(), shape, config);

        auto context = std::make_shared<BackwardContext>();
        context->sizes = {
            shape.N, shape.C, shape.H, shape.W,
            shape.H_out, shape.W_out,
            config.out_channels, config.in_channels,
            config.stride, config.padding,
            config.kernel_h, config.kernel_w
        };
        context->flags.push_back(1);

        return ForwardResult{
            std::move(result),
            std::move(context),
            std::make_shared<OneDnnConv2DGradFn>()
        };
    }
};

#endif

class ReferenceConv2DGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (!ctx || input_indices.size() != inputs.size() || inputs.size() < 2 ||
            ctx->matrices.empty() || ctx->sizes.size() < 12 || ctx->flags.empty()) {
            throw std::logic_error("Conv2DGradFn: invalid node state.");
        }

        const Node::Ptr& input = inputs[0];
        const Node::Ptr& kernels = inputs[1];
        const bool has_bias = ctx->flags[0] != 0;
        const Node::Ptr bias = has_bias && inputs.size() >= 3 ? inputs[2] : nullptr;

        const std::size_t N = ctx->sizes[0];
        const std::size_t H = ctx->sizes[2];
        const std::size_t W = ctx->sizes[3];
        const std::size_t H_out = ctx->sizes[4];
        const std::size_t W_out = ctx->sizes[5];
        const std::size_t out_channels = ctx->sizes[6];
        const std::size_t in_channels = ctx->sizes[7];
        const std::size_t stride = ctx->sizes[8];
        const std::size_t padding = ctx->sizes[9];
        const std::size_t kernel_h = ctx->sizes[10];
        const std::size_t kernel_w = ctx->sizes[11];

        Matrix dout(N * H_out * W_out, out_channels);
#ifdef PROFILE_OPS
        auto grad_reshape_start = Clock::now();
#endif
        double* dout_data = dout.data.data();
        const double* grad_data = grad_output.data.data();
        const std::size_t grad_stride = grad_output.cols;
        const std::size_t dout_stride = dout.cols;
        const std::size_t HWW = H_out * W_out;

        #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)
        for (std::size_t n = 0; n < N; ++n) {
            const std::size_t grad_n_base = n * grad_stride;
            const std::size_t dout_n_row_base = n * HWW;
            for (std::size_t oc = 0; oc < out_channels; ++oc) {
                const std::size_t grad_oc_base = oc * HWW;
                for (std::size_t oh = 0; oh < H_out; ++oh) {
                    for (std::size_t ow = 0; ow < W_out; ++ow) {
                        const std::size_t hw = oh * W_out + ow;
                        const std::size_t src_col = grad_oc_base + hw;
                        const std::size_t dst_row = dout_n_row_base + hw;
                        dout_data[dst_row * dout_stride + oc] = grad_data[grad_n_base + src_col];
                    }
                }
            }
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "conv2d_backward_grad_reshape",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - grad_reshape_start).count());
#endif

        const Matrix& cols = ctx->matrices[0];
        const std::size_t col_w = cols.cols;
        ContributionList contributions;

        if (kernels && kernels->requiresGrad() && input_indices[1] != kInvalidNodeIndex) {
            Matrix dW(out_channels, col_w);
#ifdef PROFILE_OPS
            auto dw_start = Clock::now();
#endif
            dout.matmul_into(cols, dW, true, false);
#ifdef PROFILE_OPS
            opProfileRecord(
                "conv2d_backward_dW_gemm",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - dw_start).count());
#endif
            contributions.push_back({input_indices[1], std::move(dW)});
        }

        if (bias && bias->requiresGrad() && inputs.size() >= 3 && input_indices[2] != kInvalidNodeIndex) {
            Matrix db(1, out_channels, 0.0);
#ifdef PROFILE_OPS
            auto db_start = Clock::now();
#endif
            const double* dout_const_data = dout.data.data();
            double* db_data = db.data.data();
            for (std::size_t i = 0; i < dout.rows; ++i) {
                const double* row_ptr = dout_const_data + i * dout_stride;
                for (std::size_t j = 0; j < out_channels; ++j) {
                    db_data[j] += row_ptr[j];
                }
            }
#ifdef PROFILE_OPS
            opProfileRecord(
                "conv2d_backward_db_reduce",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - db_start).count());
#endif
            contributions.push_back({input_indices[2], std::move(db)});
        }

        if (input && input->requiresGrad() && input_indices[0] != kInvalidNodeIndex) {
            Matrix dcols(N * H_out * W_out, col_w);
#ifdef PROFILE_OPS
            auto dx_gemm_start = Clock::now();
#endif
            dout.matmul_into(kernels->value(), dcols, false, false);
#ifdef PROFILE_OPS
            opProfileRecord(
                "conv2d_backward_dX_gemm",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - dx_gemm_start).count());
            auto col2im_start = Clock::now();
#endif
            Matrix dX = col2imFromContext(
                dcols, N, in_channels, H, W, H_out, W_out, kernel_h, kernel_w, stride, padding);
#ifdef PROFILE_OPS
            opProfileRecord(
                "conv2d_backward_col2im",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - col2im_start).count());
#endif
            contributions.push_back({input_indices[0], std::move(dX)});
        }

        return contributions;
    }
};

class ReferenceConv2DBackend final : public Conv2DLayer::Backend {
public:
    ForwardResult forward(const Matrix& input,
                          const InputShape& shape,
                          const Config& config,
                          const Parameters& params) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
#endif
        // im2col: (N*H_out*W_out, C*kH*kW)
#ifdef PROFILE_OPS
        auto im2col_start = Clock::now();
#endif
        auto cols_ptr = std::make_shared<Matrix>(referenceIm2col(input, shape, config));
#ifdef PROFILE_OPS
        opProfileRecord(
            "conv2d_forward_im2col",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - im2col_start).count());
#endif

        const Matrix& kv = params.kernels->value();
        Matrix out(cols_ptr->rows, config.out_channels);
#ifdef PROFILE_OPS
        auto gemm_start = Clock::now();
#endif
        cols_ptr->matmul_into(kv, out, false, true);
#ifdef PROFILE_OPS
        opProfileRecord(
            "conv2d_forward_gemm",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - gemm_start).count());
#endif

        const Matrix& bv = params.bias->value();
        const std::size_t out_stride = out.cols;
        Matrix result(shape.N, config.out_channels * shape.H_out * shape.W_out);
        double* result_data = result.data.data();
        const double* out_const_data = out.data.data();
        const std::size_t result_stride = result.cols;
        const std::size_t HWW = shape.H_out * shape.W_out;

#ifdef PROFILE_OPS
        auto post_start = Clock::now();
#endif
        #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)
        for (std::size_t n = 0; n < shape.N; ++n) {
            const std::size_t src_n_row_base = n * HWW;
            const std::size_t dst_n_base = n * result_stride;
            for (std::size_t oc = 0; oc < config.out_channels; ++oc) {
                const std::size_t dst_oc_base = oc * HWW;
                const double b = bv(0, oc);
                for (std::size_t oh = 0; oh < shape.H_out; ++oh) {
                    for (std::size_t ow = 0; ow < shape.W_out; ++ow) {
                       const std::size_t hw = oh * shape.W_out + ow;
                       const std::size_t src_row = src_n_row_base + hw;
                       const std::size_t dst_col = dst_oc_base + hw;
                       result_data[dst_n_base + dst_col] =
                           out_const_data[src_row * out_stride + oc] + b;
                    }
                }
            }
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "conv2d_forward_bias_reshape",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - post_start).count());
#endif

        auto context = std::make_shared<BackwardContext>();
        context->matrices.push_back(*cols_ptr);
        context->sizes = {
            shape.N, shape.C, shape.H, shape.W,
            shape.H_out, shape.W_out,
            config.out_channels, config.in_channels,
            config.stride, config.padding,
            config.kernel_h, config.kernel_w
        };
        context->flags.push_back(1);

        return ForwardResult{
            std::move(result),
            std::move(context),
            std::make_shared<ReferenceConv2DGradFn>()
        };
    }
};

std::shared_ptr<const Conv2DLayer::Backend> makeReferenceBackend() {
    return std::make_shared<ReferenceConv2DBackend>();
}

std::shared_ptr<const Conv2DLayer::Backend> makeBackend(ConvBackend backend) {
    switch (backend) {
    case ConvBackend::Reference:
        return makeReferenceBackend();
    case ConvBackend::OneDnn:
#if PPN_HAVE_ONEDNN_CONV_BACKEND
        return std::make_shared<OneDnnConv2DBackend>();
#else
        throw std::invalid_argument("Conv2DLayer: onednn backend requested but this binary was built without oneDNN support.");
#endif
    }
    throw std::invalid_argument("Conv2DLayer: unsupported conv backend.");
}
}

std::string toString(ConvBackend backend) {
    switch (backend) {
    case ConvBackend::Reference: return "reference";
    case ConvBackend::OneDnn: return "onednn";
    }
    throw std::invalid_argument("toString(ConvBackend): unsupported backend.");
}

ConvBackend parseConvBackend(const std::string& backend_name) {
    if (backend_name == "reference") {
        return ConvBackend::Reference;
    }
    if (backend_name == "onednn") {
        return ConvBackend::OneDnn;
    }
    throw std::invalid_argument(
        "unsupported conv backend '" + backend_name + "'. Expected 'reference' or 'onednn'.");
}

Conv2DLayer::Conv2DLayer(size_t in_ch, size_t out_ch,
                         size_t kH, size_t kW,
                         size_t stride, size_t padding,
                         ConvBackend backend)
    : in_channels_(in_ch), out_channels_(out_ch),
      kernel_h_(kH), kernel_w_(kW),
      stride_(stride), padding_(padding),
      kernels_(std::make_shared<Node>(Matrix(out_ch, in_ch * kH * kW))),
      bias_(std::make_shared<Node>(Matrix(1, out_ch, 0.0))),
      backend_kind_(backend),
      backend_(makeBackend(backend))
{
    kernels_->setIsParameter(true);
    bias_->setIsParameter(true);
    if (in_channels_ == 0 || out_channels_ == 0) {
        throw std::invalid_argument("Conv2DLayer: in/out channels must be > 0.");
    }
    if (kernel_h_ == 0 || kernel_w_ == 0) {
        throw std::invalid_argument("Conv2DLayer: kernel size must be > 0.");
    }
    if (stride_ == 0) {
        throw std::invalid_argument("Conv2DLayer: stride must be > 0.");
    }
}

std::pair<size_t, size_t> Conv2DLayer::outputShape(size_t H, size_t W) const {
    const size_t maxv = std::numeric_limits<size_t>::max();
    if (padding_ > (maxv - H) / 2 || padding_ > (maxv - W) / 2) {
        throw std::overflow_error("Conv2DLayer::outputShape: padding causes size_t overflow.");
    }

    const size_t H_pad = H + 2 * padding_;
    const size_t W_pad = W + 2 * padding_;
    if (H_pad < kernel_h_ || W_pad < kernel_w_) {
        throw std::invalid_argument("Conv2DLayer::outputShape: kernel does not fit input.");
    }

    size_t H_out = (H_pad - kernel_h_) / stride_ + 1;
    size_t W_out = (W_pad - kernel_w_) / stride_ + 1;
    return {H_out, W_out};
}

void Conv2DLayer::randomInit(unsigned int seed) {
    // He initialization: stddev = sqrt(2 / fan_in)
    size_t fan_in = in_channels_ * kernel_h_ * kernel_w_;
    double stddev = std::sqrt(2.0 / static_cast<double>(fan_in));
    kernels_->grad() = Matrix(kernels_->value().rows, kernels_->value().cols, 0.0);

    Matrix& kv = const_cast<Matrix&>(kernels_->value());
    kv.randomInit(0.0, stddev, /*use_normal=*/true, seed);
}

std::vector<Node::Ptr> Conv2DLayer::parameters() const {
    return {kernels_, bias_};
}

// =====================================================================
// Forward: out = im2col(input) @ kernels^T + bias
// =====================================================================
Node::Ptr Conv2DLayer::forward(const Node::Ptr& input,
                               size_t N, size_t C, size_t H, size_t W) const {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
#endif
    if (!input) {
        throw std::invalid_argument("Conv2DLayer::forward: input node is null.");
    }
    if (C != in_channels_) {
        throw std::invalid_argument("Conv2DLayer::forward: input channels mismatch.");
    }

    const Matrix& x = input->value();
    if (x.rows != N) {
        throw std::invalid_argument("Conv2DLayer::forward: N does not match input rows.");
    }

    const size_t maxv = std::numeric_limits<size_t>::max();
    if (C != 0 && H > maxv / C) {
        throw std::overflow_error("Conv2DLayer::forward: C*H overflows.");
    }
    const size_t CH = C * H;
    if (CH != 0 && W > maxv / CH) {
        throw std::overflow_error("Conv2DLayer::forward: C*H*W overflows.");
    }
    const size_t expected_cols = CH * W;
    if (x.cols != expected_cols) {
        throw std::invalid_argument("Conv2DLayer::forward: input cols mismatch with C*H*W.");
    }

    auto [H_out, W_out] = outputShape(H, W);
    Backend::Config config{
        in_channels_, out_channels_,
        kernel_h_, kernel_w_,
        stride_, padding_
    };
    Backend::InputShape shape{N, C, H, W, H_out, W_out};
    Backend::Parameters params{kernels_, bias_};
    auto backend_result = backend_->forward(x, shape, config, params);

    const bool requires_grad = inferRequiresGrad(input, kernels_, bias_);
    auto node = std::make_shared<Node>(backend_result.output, requires_grad);
    if (!requires_grad) {
        return node;
    }

    node->setInputs({input, kernels_, bias_});
    node->setBackwardContext(std::move(backend_result.backward_context));
    node->setGradFn(std::move(backend_result.grad_fn));
    return node;
}
