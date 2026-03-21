/**
 * @file conv2d_layer.cpp
 * @brief 2D convolution layer implementation using im2col + matmul.
 */
#include "conv2d_layer.hpp"
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

    auto node = std::make_shared<Node>(result);
    node->addParent(input);
    node->addParent(kernels_);
    node->addParent(bias_);

    // Capture necessary state for backward
    auto input_ptr = input;
    auto kernels_ptr = kernels_;
    auto bias_ptr = bias_;
    auto cols_bw_ptr = cols_ptr;
    size_t out_ch = out_channels_;
    size_t in_ch = in_channels_;
    auto self = this;

    node->setBackwardFn([input_ptr, kernels_ptr, bias_ptr, cols_bw_ptr,
                         N, H_out, W_out, out_ch, in_ch, H, W, self](const Matrix& grad) {
        // grad shape: (N, out_ch * H_out * W_out)
        // Reshape to (N*H_out*W_out, out_ch)
        Matrix dout(N * H_out * W_out, out_ch);
        double* dout_data = dout.data.data();
        const double* grad_data = grad.data.data();
        const size_t grad_stride = grad.cols;
        const size_t dout_stride = dout.cols;
        const size_t HWW = H_out * W_out;

        #pragma omp parallel for if(cnn_parallel_enabled()) schedule(static)

        for (size_t n = 0; n < N; ++n) {
            const size_t grad_n_base = n * grad_stride;
            const size_t dout_n_row_base = n * HWW;
            for (size_t oc = 0; oc < out_ch; ++oc) {
                const size_t grad_oc_base = oc * HWW;
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        const size_t hw = oh * W_out + ow;
                        const size_t src_col = grad_oc_base + hw;
                        const size_t dst_row = dout_n_row_base + hw;
                        dout_data[dst_row * dout_stride + oc] = grad_data[grad_n_base + src_col];

                    }
                }
            }
        }

        // --- dW: kernels gradient ---
        // Reuse cached im2col from forward to avoid recomputing it in backward.
        const Matrix& cols_bw = *cols_bw_ptr;
        const size_t col_w_local = cols_bw.cols;
        // dW = dout^T @ cols → (out_ch, col_w)
        Matrix dW(out_ch, col_w_local);
        dout.matmul_into(cols_bw, dW, true, false);
        kernels_ptr->addGrad(dW);

        // --- db: bias gradient ---
        // db = sum of dout along rows → (1, out_ch)
        Matrix db(1, out_ch, 0.0);
        const double* dout_const_data = dout.data.data();
        double* db_data = db.data.data();
        for (size_t i = 0; i < dout.rows; ++i) {
            const double* row_ptr = dout_const_data + i * dout_stride;
            for (size_t j = 0; j < out_ch; ++j) {
                db_data[j] += row_ptr[j];
            }
        }
        bias_ptr->addGrad(db);

        // --- dX: input gradient ---
        // dcols = dout @ W → (N*H_out*W_out, col_w)
        Matrix dcols(N * H_out * W_out, col_w_local);
        dout.matmul_into(kernels_ptr->value(), dcols, false, false);

        // col2im: (N*H_out*W_out, col_w) → (N, C*H*W)
        Matrix dX = self->col2im(dcols, N, in_ch, H, W, H_out, W_out);
        input_ptr->addGrad(dX);
    });

    return node;
}
