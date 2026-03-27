#include "math_ops.hpp"
#include "autograd/backward_context.hpp"
#include "autograd/grad_fn.hpp"
#include "profiling.hpp"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace MathOps {

using NodePtr = std::shared_ptr<Node>;
constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kGeluCubicCoeff = 0.044715;

namespace {

bool inferRequiresGrad(const std::vector<NodePtr>& inputs) {
    for (const auto& input : inputs) {
        if (input && input->requiresGrad()) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<BackwardContext> makeContext() {
    return std::make_shared<BackwardContext>();
}

inline void pushIfValid(ContributionList& contributions,
                        std::uint32_t target_index,
                        Matrix grad) {
    if (target_index == kInvalidNodeIndex) {
        return;
    }
    contributions.push_back({target_index, std::move(grad)});
}

class AddGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 2 || input_indices.size() != inputs.size() || !ctx) {
            throw std::logic_error("AddGradFn: invalid node state.");
        }

        const bool broadcast_b = !ctx->flags.empty() && ctx->flags[0] != 0;
        ContributionList contributions;

#ifdef PROFILE_OPS
        auto da_start = Clock::now();
#endif
        pushIfValid(contributions, input_indices[0], grad_output);
#ifdef PROFILE_OPS
        opProfileRecord(
            "add_backward_a",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - da_start).count());
#endif

        if (broadcast_b) {
#ifdef PROFILE_OPS
            auto db_start = Clock::now();
#endif
            Matrix grad_b(1, grad_output.cols, 0.0);
            for (size_t j = 0; j < grad_output.cols; ++j) {
                double sum = 0.0;
                for (size_t i = 0; i < grad_output.rows; ++i) {
                    sum += grad_output(i, j);
                }
                grad_b(0, j) = sum;
            }
            pushIfValid(contributions, input_indices[1], std::move(grad_b));
#ifdef PROFILE_OPS
            opProfileRecord(
                "add_backward_b_broadcast_reduce",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - db_start).count());
#endif
        } else {
#ifdef PROFILE_OPS
            auto db_start = Clock::now();
#endif
            pushIfValid(contributions, input_indices[1], grad_output);
#ifdef PROFILE_OPS
            opProfileRecord(
                "add_backward_b",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - db_start).count());
#endif
        }
        return contributions;
    }
};

class MulGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
#endif
        const auto& inputs = output.inputs();
        if (inputs.size() != 2 || input_indices.size() != inputs.size()) {
            throw std::logic_error("MulGradFn: invalid node inputs.");
        }
#ifdef PROFILE_OPS
        auto da_start = Clock::now();
#endif
        Matrix da = grad_output.mul(inputs[1]->value());
#ifdef PROFILE_OPS
        opProfileRecord(
            "mul_backward_a",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - da_start).count());
        auto db_start = Clock::now();
#endif
        Matrix db = grad_output.mul(inputs[0]->value());
#ifdef PROFILE_OPS
        opProfileRecord(
            "mul_backward_b",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - db_start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(da));
        pushIfValid(contributions, input_indices[1], std::move(db));
        return contributions;
    }
};

class MatMulGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
#endif
        const auto& inputs = output.inputs();
        if (inputs.size() != 2 || input_indices.size() != inputs.size()) {
            throw std::logic_error("MatMulGradFn: invalid node inputs.");
        }

        Matrix dA(grad_output.rows, inputs[1]->value().rows);
#ifdef PROFILE_OPS
        auto da_start = Clock::now();
#endif
        grad_output.matmul_into(inputs[1]->value(), dA, false, true);
#ifdef PROFILE_OPS
        opProfileRecord(
            "matmul_backward_dA",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - da_start).count());
#endif

        Matrix dB(inputs[0]->value().cols, grad_output.cols);
#ifdef PROFILE_OPS
        auto db_start = Clock::now();
#endif
        inputs[0]->value().matmul_into(grad_output, dB, true, false);
#ifdef PROFILE_OPS
        opProfileRecord(
            "matmul_backward_dB",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - db_start).count());
#endif

        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(dA));
        pushIfValid(contributions, input_indices[1], std::move(dB));
        return contributions;
    }
};

class SumGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        if (inputs.size() != 1 || input_indices.size() != inputs.size()) {
            throw std::logic_error("SumGradFn: invalid node inputs.");
        }
        Matrix gx(inputs[0]->value().rows, inputs[0]->value().cols, grad_output.data[0]);
#ifdef PROFILE_OPS
        opProfileRecord(
            "sum_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class MeanGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->sizes.empty()) {
            throw std::logic_error("MeanGradFn: invalid node state.");
        }
        const std::size_t count = ctx->sizes[0];
        const double coeff = count > 0 ? grad_output.data[0] / static_cast<double>(count) : 0.0;
        Matrix gx(inputs[0]->value().rows, inputs[0]->value().cols, coeff);
#ifdef PROFILE_OPS
        opProfileRecord(
            "mean_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class ReLUGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty()) {
            throw std::logic_error("ReLUGradFn: invalid node state.");
        }
        const Matrix& relu_out = ctx->matrices[0];
        Matrix gx(grad_output.rows, grad_output.cols, 0.0);
        for (size_t i = 0; i < relu_out.data.size(); ++i) {
            gx.data[i] = (relu_out.data[i] > 0.0) ? grad_output.data[i] : 0.0;
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "relu_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class LeakyReLUGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty() || ctx->scalars.empty()) {
            throw std::logic_error("LeakyReLUGradFn: invalid node state.");
        }
        const Matrix& in = ctx->matrices[0];
        const double alpha = ctx->scalars[0];
        Matrix gx(grad_output.rows, grad_output.cols, 0.0);
        for (size_t i = 0; i < in.data.size(); ++i) {
            gx.data[i] = grad_output.data[i] * ((in.data[i] > 0.0) ? 1.0 : alpha);
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "leaky_relu_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class GELUGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty()) {
            throw std::logic_error("GELUGradFn: invalid node state.");
        }
        const Matrix& in = ctx->matrices[0];
        Matrix gx(grad_output.rows, grad_output.cols, 0.0);
        for (size_t i = 0; i < in.data.size(); ++i) {
            const double xi = in.data[i];
            const double xi2 = xi * xi;
            const double u = kSqrt2OverPi * (xi + kGeluCubicCoeff * xi * xi2);
            const double t = std::tanh(u);
            const double sech2 = 1.0 - t * t;
            const double du_dx = kSqrt2OverPi * (1.0 + 3.0 * kGeluCubicCoeff * xi2);
            const double dy_dx = 0.5 * (1.0 + t) + 0.5 * xi * sech2 * du_dx;
            gx.data[i] = grad_output.data[i] * dy_dx;
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "gelu_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class SigmoidGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty()) {
            throw std::logic_error("SigmoidGradFn: invalid node state.");
        }
        const Matrix& sig = ctx->matrices[0];
        Matrix gx(grad_output.rows, grad_output.cols, 0.0);
        for (size_t i = 0; i < sig.data.size(); ++i) {
            gx.data[i] = grad_output.data[i] * sig.data[i] * (1.0 - sig.data[i]);
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "sigmoid_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

class TanhGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
#ifdef PROFILE_OPS
        using Clock = std::chrono::steady_clock;
        auto start = Clock::now();
#endif
        const auto& inputs = output.inputs();
        const auto& ctx = output.backwardContext();
        if (inputs.size() != 1 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty()) {
            throw std::logic_error("TanhGradFn: invalid node state.");
        }
        const Matrix& out = ctx->matrices[0];
        Matrix gx(grad_output.rows, grad_output.cols, 0.0);
        for (size_t i = 0; i < out.data.size(); ++i) {
            gx.data[i] = grad_output.data[i] * (1.0 - out.data[i] * out.data[i]);
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "tanh_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        ContributionList contributions;
        pushIfValid(contributions, input_indices[0], std::move(gx));
        return contributions;
    }
};

template <typename GradFnT>
Node::Ptr makeUnaryNode(OpKind kind,
                        const Node::Ptr& input,
                        Matrix out,
                        std::shared_ptr<BackwardContext> context = nullptr) {
    const bool requires_grad = inferRequiresGrad({input});
    auto node = std::make_shared<OperationNode>(kind, out, std::vector<Node::Ptr>{input}, requires_grad);
    if (!requires_grad) {
        return node;
    }
    node->setInputs({input});
    node->setBackwardContext(std::move(context));
    node->setGradFn(std::make_shared<GradFnT>());
    return node;
}

}

Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    const Matrix& val_a = a->value();
    const Matrix& val_b = b->value();

    const bool broadcast_b = (val_b.rows == 1 && val_b.cols == val_a.cols && val_a.rows > 1);
    Matrix out(val_a.rows, val_a.cols);

    if (broadcast_b) {
        for (size_t i = 0; i < val_a.rows; ++i) {
            for (size_t j = 0; j < val_a.cols; ++j) {
                out(i, j) = val_a(i, j) + val_b(0, j);
            }
        }
    } else {
        out = val_a.add(val_b);
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "add_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif

    const bool requires_grad = inferRequiresGrad({a, b});
    auto node = std::make_shared<OperationNode>(OpKind::ADD, out, std::vector<Node::Ptr>{a, b}, requires_grad);
    if (!requires_grad) {
        return node;
    }

    auto context = makeContext();
    context->flags.push_back(static_cast<std::uint8_t>(broadcast_b));
    node->setInputs({a, b});
    node->setBackwardContext(context);
    node->setGradFn(std::make_shared<AddGradFn>());
    return node;
}

Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix out = a->value().mul(b->value());
#ifdef PROFILE_OPS
    opProfileRecord(
        "mul_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    const bool requires_grad = inferRequiresGrad({a, b});
    auto node = std::make_shared<OperationNode>(OpKind::MUL, out, std::vector<Node::Ptr>{a, b}, requires_grad);
    if (!requires_grad) {
        return node;
    }
    node->setInputs({a, b});
    node->setGradFn(std::make_shared<MulGradFn>());
    node->setBackwardContext(makeContext());
    return node;
}

Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix out = a->value().matmul(b->value());
#ifdef PROFILE_OPS
    opProfileRecord(
        "matmul_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    const bool requires_grad = inferRequiresGrad({a, b});
    auto node = std::make_shared<OperationNode>(OpKind::MATMUL, out, std::vector<Node::Ptr>{a, b}, requires_grad);
    if (!requires_grad) {
        return node;
    }
    node->setInputs({a, b});
    node->setGradFn(std::make_shared<MatMulGradFn>());
    node->setBackwardContext(makeContext());
    return node;
}

Node::Ptr sum(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    const Matrix& xv = x->value();
    double s = 0.0;
    for (double v : xv.data) {
        s += v;
    }

    Matrix out(1, 1);
    out.data[0] = s;
#ifdef PROFILE_OPS
    opProfileRecord(
        "sum_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif

    const bool requires_grad = inferRequiresGrad({x});
    auto node = std::make_shared<Node>(out, requires_grad);
    if (!requires_grad) {
        return node;
    }

    node->setInputs({x});
    node->setBackwardContext(makeContext());
    node->setGradFn(std::make_shared<SumGradFn>());
    return node;
}

Node::Ptr mean(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    const Matrix& xv = x->value();
    const std::size_t count = xv.data.size();
    double s = 0.0;
    for (double v : xv.data) {
        s += v;
    }

    Matrix out(1, 1);
    out.data[0] = (count > 0) ? s / static_cast<double>(count) : 0.0;
#ifdef PROFILE_OPS
    opProfileRecord(
        "mean_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif

    const bool requires_grad = inferRequiresGrad({x});
    auto node = std::make_shared<Node>(out, requires_grad);
    if (!requires_grad) {
        return node;
    }

    auto context = makeContext();
    context->sizes.push_back(count);
    node->setInputs({x});
    node->setBackwardContext(context);
    node->setGradFn(std::make_shared<MeanGradFn>());
    return node;
}

Node::Ptr relu(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix out = x->value();
    for (double& v : out.data) {
        v = std::max(0.0, v);
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "relu_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    auto context = makeContext();
    context->matrices.push_back(out);
    return makeUnaryNode<ReLUGradFn>(OpKind::RELU, x, std::move(out), std::move(context));
}

Node::Ptr leaky_relu(const Node::Ptr& x, double alpha) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix in = x->value();
    Matrix out = in;
    for (double& v : out.data) {
        v = (v > 0.0) ? v : (alpha * v);
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "leaky_relu_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    auto context = makeContext();
    context->matrices.push_back(in);
    context->scalars.push_back(alpha);
    return makeUnaryNode<LeakyReLUGradFn>(OpKind::LEAKY_RELU, x, std::move(out), std::move(context));
}

Node::Ptr gelu(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix in = x->value();
    Matrix out = in;
    for (double& v : out.data) {
        const double xi = v;
        const double u = kSqrt2OverPi * (xi + kGeluCubicCoeff * xi * xi * xi);
        v = 0.5 * xi * (1.0 + std::tanh(u));
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "gelu_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    auto context = makeContext();
    context->matrices.push_back(in);
    return makeUnaryNode<GELUGradFn>(OpKind::GELU, x, std::move(out), std::move(context));
}

Node::Ptr sigmoid(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix out = x->value();
    for (double& v : out.data) {
        v = 1.0 / (1.0 + std::exp(-v));
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "sigmoid_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    auto context = makeContext();
    context->matrices.push_back(out);
    return makeUnaryNode<SigmoidGradFn>(OpKind::SIGMOID, x, std::move(out), std::move(context));
}

Node::Ptr tanh(const Node::Ptr& x) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    Matrix out = x->value();
    for (double& v : out.data) {
        v = std::tanh(v);
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "tanh_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    auto context = makeContext();
    context->matrices.push_back(out);
    return makeUnaryNode<TanhGradFn>(OpKind::TANH, x, std::move(out), std::move(context));
}

} // namespace MathOps
