/**
 * @file maxpool2d_layer.cpp
 * @brief 2D max pooling layer implementation.
 */
#include "maxpool2d_layer.hpp"
#include <limits>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <cstring>


namespace {
bool cnn_parallel_enabled() {
    static int mode = -1;
    if (mode == -1) {
        const char* v = std::getenv("CNN_PARALLEL");
        mode = (v && std::strcmp(v, "1") == 0) ? 1 : 0;
    }
    return mode == 1;
}
}


MaxPool2DLayer::MaxPool2DLayer(size_t pH, size_t pW, size_t stride)
    : pool_h_(pH), pool_w_(pW), stride_(stride)
{
    if (pool_h_ == 0 || pool_w_ == 0) {
        throw std::invalid_argument("MaxPool2DLayer: pool size must be > 0.");
    }
    if (stride_ == 0) {
        throw std::invalid_argument("MaxPool2DLayer: stride must be > 0.");
    }
}

std::pair<size_t, size_t> MaxPool2DLayer::outputShape(size_t H, size_t W) const {
    if (H < pool_h_ || W < pool_w_) {
        throw std::invalid_argument("MaxPool2DLayer::outputShape: pool window does not fit input.");
    }
    size_t H_out = (H - pool_h_) / stride_ + 1;
    size_t W_out = (W - pool_w_) / stride_ + 1;
    return {H_out, W_out};
}

Node::Ptr MaxPool2DLayer::forward(const Node::Ptr& input,
                                  size_t N, size_t C, size_t H, size_t W) const {
    if (!input) {
        throw std::invalid_argument("MaxPool2DLayer::forward: input node is null.");
    }

    const Matrix& x = input->value();
    if (x.rows != N) {
        throw std::invalid_argument("MaxPool2DLayer::forward: N does not match input rows.");
    }

    const size_t maxv = std::numeric_limits<size_t>::max();
    if (C != 0 && H > maxv / C) {
        throw std::overflow_error("MaxPool2DLayer::forward: C*H overflows.");
    }
    const size_t CH = C * H;
    if (CH != 0 && W > maxv / CH) {
        throw std::overflow_error("MaxPool2DLayer::forward: C*H*W overflows.");
    }
    const size_t expected_cols = CH * W;
    if (x.cols != expected_cols) {
        throw std::invalid_argument("MaxPool2DLayer::forward: input cols mismatch with C*H*W.");
    }

    auto [H_out, W_out] = outputShape(H, W);

    Matrix out(N, C * H_out * W_out);
    const double* in_data = x.data.data();
    double* out_data = out.data.data();
    const size_t in_stride = x.cols;
    const size_t out_stride = out.cols;
    // Store max indices for backward: one index per output element
    // index = position within the (H, W) spatial grid
    auto max_indices = std::make_shared<std::vector<size_t>>(N * C * H_out * W_out);
    #pragma omp parallel for collapse(2) if(cnn_parallel_enabled()) schedule(static)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    double max_val = std::numeric_limits<double>::lowest();
                    size_t max_idx = 0;

                    for (size_t ph = 0; ph < pool_h_; ++ph) {
                        for (size_t pw = 0; pw < pool_w_; ++pw) {
                            size_t ih = oh * stride_ + ph;
                            size_t iw = ow * stride_ + pw;
                            // input(n, c*H*W + ih*W + iw)
                            size_t input_col = c * H * W + ih * W + iw;
                            double val = in_data[n * in_stride + input_col];
                            if (val > max_val) {
                                max_val = val;
                                max_idx = ih * W + iw;
                            }
                        }
                    }

                    size_t out_col = c * H_out * W_out + oh * W_out + ow;
                    out_data[n * out_stride + out_col] = max_val;
                    (*max_indices)[n * C * H_out * W_out + c * H_out * W_out + oh * W_out + ow] = max_idx;
                }
            }
        }
    }

    auto node = std::make_shared<Node>(out);
    node->addParent(input);

    auto input_ptr = input;

    node->setBackwardFn([input_ptr, max_indices, N, C, H, W, H_out, W_out](const Matrix& grad) {
        // grad shape: (N, C*H_out*W_out)
        Matrix dX(N, C * H * W, 0.0);
        double* dX_data = dX.data.data();
        const double* grad_data = grad.data.data();
        const size_t dX_stride = dX.cols;
        const size_t grad_stride = grad.cols;
        #pragma omp parallel for collapse(2) if(cnn_parallel_enabled()) schedule(static)
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                const size_t dx_nc_base = n * dX_stride + c * H * W;
                const size_t out_nc_base = n * C * H_out * W_out + c * H_out * W_out;
                const size_t grad_n_base = n * grad_stride;
                const size_t out_c_col_base = c * H_out * W_out;
                for (size_t oh = 0; oh < H_out; ++oh) {
                    for (size_t ow = 0; ow < W_out; ++ow) {
                        const size_t hw = oh * W_out + ow;
                        const size_t out_col = out_c_col_base + hw; 
                        const size_t idx = (*max_indices)[out_nc_base + hw];

                        // Scatter gradient to the max position
                 
                        dX_data[dx_nc_base + idx] += grad_data[grad_n_base + out_col];
                    }
                }
            }
        }

        input_ptr->addGrad(dX);
    });

    return node;
}
