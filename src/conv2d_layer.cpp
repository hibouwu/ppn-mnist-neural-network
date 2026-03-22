/**
 * @file conv2d_layer.cpp
 * @brief 2D convolution layer implementation using im2col + matmul.
 */
#include "conv2d_layer.hpp"
#include "autograd/backward_context.hpp"
#include "autograd/grad_fn.hpp"
#include "math_ops.hpp"
#include "operation_node.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

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

class Conv2DGradFn final : public GradFn {
public:
    std::vector<GradientContribution> apply(const Node& output,
                                            const Matrix& grad_output) const override {
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (!ctx || inputs.size() < 2 || ctx->matrices.empty() || ctx->sizes.size() < 12 || ctx->flags.empty()) {
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

        const Matrix& cols = ctx->matrices[0];
        const std::size_t col_w = cols.cols;
        std::vector<GradientContribution> contributions;
        contributions.reserve(3);

        if (kernels && kernels->requiresGrad()) {
            Matrix dW(out_channels, col_w);
            dout.matmul_into(cols, dW, true, false);
            contributions.push_back({kernels, std::move(dW)});
        }

        if (bias && bias->requiresGrad()) {
            Matrix db(1, out_channels, 0.0);
            const double* dout_const_data = dout.data.data();
            double* db_data = db.data.data();
            for (std::size_t i = 0; i < dout.rows; ++i) {
                const double* row_ptr = dout_const_data + i * dout_stride;
                for (std::size_t j = 0; j < out_channels; ++j) {
                    db_data[j] += row_ptr[j];
                }
            }
            contributions.push_back({bias, std::move(db)});
        }

        if (input && input->requiresGrad()) {
            Matrix dcols(N * H_out * W_out, col_w);
            dout.matmul_into(kernels->value(), dcols, false, false);
            Matrix dX = col2imFromContext(
                dcols, N, in_channels, H, W, H_out, W_out, kernel_h, kernel_w, stride, padding);
            contributions.push_back({input, std::move(dX)});
        }

        return contributions;
    }
};
}

Conv2DLayer::Conv2DLayer(size_t in_ch, size_t out_ch,
                         size_t kH, size_t kW,
                         size_t stride, size_t padding)
    : in_channels_(in_ch), out_channels_(out_ch),
      kernel_h_(kH), kernel_w_(kW),
      stride_(stride), padding_(padding),
      kernels_(std::make_shared<Node>(Matrix(out_ch, in_ch * kH * kW))),
      bias_(std::make_shared<Node>(Matrix(1, out_ch, 0.0)))
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
// im2col: (N, C*H*W) → (N*H_out*W_out, C*kH*kW)
// =====================================================================
Matrix Conv2DLayer::im2col(const Matrix& input,
                           size_t N, size_t C, size_t H, size_t W,
                           size_t H_out, size_t W_out) const {
    const size_t col_h = kernel_h_ * kernel_w_ * C;
    Matrix cols(N * H_out * W_out, col_h);
    const double* input_data = input.data.data();
    double* cols_data = cols.data.data();
    const size_t input_stride = input.cols;
    const size_t HW = H * W;
    const size_t NHW_out = H_out * W_out;


    #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)

    for (size_t n = 0; n < N; ++n) {
        const size_t in_n_base = n * input_stride;
        const size_t out_n_row_base = n * NHW_out;
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                const size_t row = out_n_row_base + oh * W_out + ow;
                const size_t row_base = row * col_h;
                size_t col_idx = 0;
                for (size_t c = 0; c < C; ++c) {
                    const size_t in_c_base = c * HW;
                    for (size_t kh = 0; kh < kernel_h_; ++kh) {
                        for (size_t kw = 0; kw < kernel_w_; ++kw) {
                            int ih = static_cast<int>(oh * stride_ + kh) - static_cast<int>(padding_);
                            int iw = static_cast<int>(ow * stride_ + kw) - static_cast<int>(padding_);
                            if (ih >= 0 && ih < static_cast<int>(H) &&
                                iw >= 0 && iw < static_cast<int>(W)) {
                                // input(n, c*H*W + ih*W + iw)
                                const size_t input_idx = in_c_base + static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
                                cols_data[row_base + col_idx] = input_data[in_n_base + input_idx];
                            } else {
                                // Explicit for zero-padding semantics.
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

// =====================================================================
// col2im: (N*H_out*W_out, C*kH*kW) → (N, C*H*W)
// =====================================================================
Matrix Conv2DLayer::col2im(const Matrix& cols,
                           size_t N, size_t C, size_t H, size_t W,
                           size_t H_out, size_t W_out) const {
    Matrix input_grad(N, C * H * W, 0.0);
    const double* cols_data = cols.data.data();
    double* input_grad_data = input_grad.data.data();
    const size_t cols_stride = cols.cols;
    const size_t input_grad_stride = input_grad.cols;
    const size_t HW = H * W;
    const size_t NHW_out = H_out * W_out;


   #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)

    for (size_t n = 0; n < N; ++n) {
        const size_t in_n_base = n * input_grad_stride;
        const size_t row_n_base = n * NHW_out;
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                const size_t row = row_n_base + oh * W_out + ow;
                const size_t row_base = row * cols_stride;
                size_t col_idx = 0;
                for (size_t c = 0; c < C; ++c) {
                    const size_t in_c_base = c * HW;
                    for (size_t kh = 0; kh < kernel_h_; ++kh) {
                        for (size_t kw = 0; kw < kernel_w_; ++kw) {
                            int ih = static_cast<int>(oh * stride_ + kh) - static_cast<int>(padding_);
                            int iw = static_cast<int>(ow * stride_ + kw) - static_cast<int>(padding_);
                            if (ih >= 0 && ih < static_cast<int>(H) &&
                                iw >= 0 && iw < static_cast<int>(W)) {
                                const size_t input_idx = in_c_base + static_cast<size_t>(ih) * W + static_cast<size_t>(iw);
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

// =====================================================================
// Forward: out = im2col(input) @ kernels^T + bias
// =====================================================================
Node::Ptr Conv2DLayer::forward(const Node::Ptr& input,
                               size_t N, size_t C, size_t H, size_t W) const {
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

    // im2col: (N*H_out*W_out, C*kH*kW)
    auto cols_ptr = std::make_shared<Matrix>(im2col(x, N, C, H, W, H_out, W_out));

    // kernels_: (out_ch, col_w)
    // out = cols @ kernels^T → (N*H_out*W_out, out_ch)
    const Matrix& kv = kernels_->value();
    Matrix out(cols_ptr->rows, out_channels_);
    cols_ptr->matmul_into(kv, out, false, true);

    // Reshape output to (N, out_ch * H_out * W_out)
    // While reshaping, apply bias to avoid an extra full pass over "out".
    const Matrix& bv = bias_->value();
    const size_t out_stride = out.cols;
    Matrix result(N, out_channels_ * H_out * W_out);
    double* result_data = result.data.data();
    const double* out_const_data = out.data.data();
    const size_t result_stride = result.cols;
    const size_t HWW = H_out * W_out;

    #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)

    for (size_t n = 0; n < N; ++n) {
        const size_t src_n_row_base = n * HWW;
        const size_t dst_n_base = n * result_stride;
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            const size_t dst_oc_base = oc * HWW;
            const double b = bv(0, oc);
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                   const size_t hw = oh * W_out + ow;
                   const size_t src_row = src_n_row_base + hw;
                   const size_t dst_col = dst_oc_base + hw;
                   result_data[dst_n_base + dst_col] =
                   out_const_data[src_row * out_stride + oc] + b;

                }
            }
        }
    }

    const bool requires_grad = inferRequiresGrad(input, kernels_, bias_);
    auto node = std::make_shared<Node>(result, requires_grad);
    if (!requires_grad) {
        return node;
    }

    auto context = std::make_shared<BackwardContext>();
    context->matrices.push_back(*cols_ptr);
    context->sizes = {
        N, C, H, W, H_out, W_out, out_channels_, in_channels_, stride_, padding_, kernel_h_, kernel_w_
    };
    context->flags.push_back(1);
    node->setInputs({input, kernels_, bias_});
    node->setBackwardContext(context);
    node->setGradFn(std::make_shared<Conv2DGradFn>());
    return node;
}
