#include "loss.hpp"
#include "autograd/backward_context.hpp"
#include "autograd/grad_fn.hpp"
#include "profiling.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <chrono>

namespace {

class MSELossGradFn final : public GradFn {
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
        if (inputs.size() != 2 || input_indices.size() != inputs.size() || !ctx || ctx->sizes.empty()) {
            throw std::logic_error("MSELossGradFn: invalid node state.");
        }

        const std::size_t channels = ctx->sizes[0];
        const Matrix& target = inputs[1]->value();
        const Matrix& pred = inputs[0]->value();
        Matrix gp(pred.rows, pred.cols, 0.0);

        const double coeff = (channels > 0)
            ? (2.0 * grad_output.data[0] / static_cast<double>(channels))
            : 0.0;
        for (std::size_t i = 0; i < pred.data.size(); ++i) {
            gp.data[i] = coeff * (pred.data[i] - target.data[i]);
        }

        ContributionList out;
        if (input_indices[0] != kInvalidNodeIndex) {
            out.push_back({input_indices[0], std::move(gp)});
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "mse_loss_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        return out;
    }
};

class CrossEntropyLossGradFn final : public GradFn {
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
        if (inputs.size() != 2 || input_indices.size() != inputs.size() || !ctx || ctx->matrices.empty()) {
            throw std::logic_error("CrossEntropyLossGradFn: invalid node state.");
        }

        const Matrix& probs = ctx->matrices[0];
        const Matrix& target = inputs[1]->value();
        Matrix g_logits(probs.rows, probs.cols, 0.0);

        const double g = grad_output.data[0];
        for (std::size_t i = 0; i < probs.data.size(); ++i) {
            g_logits.data[i] = g * (probs.data[i] - target.data[i]);
        }

        ContributionList out;
        if (input_indices[0] != kInvalidNodeIndex) {
            out.push_back({input_indices[0], std::move(g_logits)});
        }
#ifdef PROFILE_OPS
        opProfileRecord(
            "cross_entropy_backward",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
        return out;
    }
};

bool inferRequiresGrad(const LossFunction::NodePtr& pred,
                       const LossFunction::NodePtr& target) {
    return (pred && pred->requiresGrad()) || (target && target->requiresGrad());
}

}

LossFunction::NodePtr MSELoss::forward(const NodePtr& pred, const NodePtr& target) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    const Matrix& p = pred->value();
    const Matrix& t = target->value();

    if (p.rows != t.rows || p.cols != t.cols) {
        throw std::invalid_argument("MSELoss::forward : pred/target shape mismatch");
    }

    const std::size_t channels = p.cols;
    double sum = 0.0;
    for (std::size_t i = 0; i < p.data.size(); ++i) {
        const double d = p.data[i] - t.data[i];
        sum += d * d;
    }

    Matrix out(1, 1);
    out.data[0] = (channels > 0) ? sum / static_cast<double>(channels) : 0.0;

    const bool requires_grad = inferRequiresGrad(pred, target);
    auto loss = std::make_shared<Node>(out, requires_grad);
    if (!requires_grad) {
        return loss;
    }

    auto context = std::make_shared<BackwardContext>();
    context->sizes.push_back(channels);
    loss->setInputs({pred, target});
    loss->setBackwardContext(context);
    loss->setGradFn(std::make_shared<MSELossGradFn>());
#ifdef PROFILE_OPS
    opProfileRecord(
        "mse_loss_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    return loss;
}

LossFunction::NodePtr CrossEntropyLoss::forward(const NodePtr& pred, const NodePtr& target) {
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
#endif
    const Matrix& logits = pred->value();
    const Matrix& y = target->value();

    if (logits.rows != y.rows || logits.cols != y.cols) {
        throw std::invalid_argument("CrossEntropyLoss::forward : pred/target shape mismatch");
    }

    const std::size_t batch = logits.rows;
    const std::size_t classes = logits.cols;
    Matrix probs(batch, classes, 0.0);

    double total = 0.0;
    for (std::size_t i = 0; i < batch; ++i) {
        Scalar max_logit = logits.data[i * classes];
        for (std::size_t c = 1; c < classes; ++c) {
            max_logit = std::max(max_logit, logits.data[i * classes + c]);
        }

        double denom = 0.0;
        for (std::size_t c = 0; c < classes; ++c) {
            const double e = std::exp(logits.data[i * classes + c] - max_logit);
            probs.data[i * classes + c] = e;
            denom += e;
        }

        for (std::size_t c = 0; c < classes; ++c) {
            probs.data[i * classes + c] /= (denom + eps_);
            const double yi = y.data[i * classes + c];
            if (yi != 0.0) {
                total += -yi * std::log(probs.data[i * classes + c] + eps_);
            }
        }
    }

    Matrix out(1, 1);
    out.data[0] = total;

    const bool requires_grad = inferRequiresGrad(pred, target);
    auto loss = std::make_shared<Node>(out, requires_grad);
    if (!requires_grad) {
        return loss;
    }

    auto context = std::make_shared<BackwardContext>();
    context->matrices.push_back(probs);
    loss->setInputs({pred, target});
    loss->setBackwardContext(context);
    loss->setGradFn(std::make_shared<CrossEntropyLossGradFn>());
#ifdef PROFILE_OPS
    opProfileRecord(
        "cross_entropy_forward",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
#endif
    return loss;
}
