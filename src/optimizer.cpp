#include "optimizer.hpp"
#include <cmath>

namespace {
constexpr float kSparseEps = 1e-12f;
}

Optimizer::Optimizer(std::vector<Node::Ptr> params, double lr)
    : parameters_(std::move(params)), lr_(lr) {}

void Optimizer::zeroGrad() {
    for (auto& p : parameters_) {
        p->zeroGrad();
    }
}

SGDOptimizer::SGDOptimizer(std::vector<Node::Ptr> params, double lr)
    : Optimizer(std::move(params), lr) {}

void SGDOptimizer::step(double gradScale) {
    for (auto& p : parameters_) {
        // W = W - lr * grad; const_cast used because Node lacks a mutable accessor.
        Matrix& val = const_cast<Matrix&>(p->value());
        const Matrix& grad = p->grad();

        size_t n = val.data.size();
        for (size_t i = 0; i < n; ++i) {
            const float gi = grad.data[i];
            if (std::fabs(gi) <= kSparseEps) {
                continue;
            }
            val.data[i] -= lr_ * gradScale * gi;
        }
    }
}

MomentumSGDOptimizer::MomentumSGDOptimizer(std::vector<Node::Ptr> params,
                                           double lr,
                                           double momentum,
                                           bool nesterov,
                                           double weightDecay)
    : Optimizer(std::move(params), lr),
      momentum_(momentum),
      nesterov_(nesterov),
      weight_decay_(weightDecay) {
    velocity_.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        const Matrix& v = p->value();
        velocity_.emplace_back(v.rows, v.cols, 0.0);
    }
}

void MomentumSGDOptimizer::step(double gradScale) {
    for (std::size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
        auto& p = parameters_[param_idx];
        Matrix& val = const_cast<Matrix&>(p->value());
        const Matrix& grad = p->grad();
        Matrix& vel = velocity_[param_idx];

        const std::size_t n = val.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            const double g = gradScale * grad.data[i] + weight_decay_ * val.data[i];
            vel.data[i] = momentum_ * vel.data[i] + g;
            const double update = nesterov_ ? (g + momentum_ * vel.data[i]) : vel.data[i];
            val.data[i] -= lr_ * update;
        }
    }
}

AdamWOptimizer::AdamWOptimizer(std::vector<Node::Ptr> params,
                               double lr,
                               double beta1,
                               double beta2,
                               double eps,
                               double weightDecay)
    : Optimizer(std::move(params), lr),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weightDecay),
      step_count_(0) {
    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        const Matrix& v = p->value();
        first_moment_.emplace_back(v.rows, v.cols, 0.0);
        second_moment_.emplace_back(v.rows, v.cols, 0.0);
    }
}

void AdamWOptimizer::step(double gradScale) {
    ++step_count_;
    const double bias_correction1 = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
    const double bias_correction2 = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

    for (std::size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
        auto& p = parameters_[param_idx];
        Matrix& val = const_cast<Matrix&>(p->value());
        const Matrix& grad = p->grad();
        Matrix& m = first_moment_[param_idx];
        Matrix& v = second_moment_[param_idx];

        const std::size_t n = val.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            const double g = gradScale * grad.data[i];
            m.data[i] = beta1_ * m.data[i] + (1.0 - beta1_) * g;
            v.data[i] = beta2_ * v.data[i] + (1.0 - beta2_) * g * g;

            const double m_hat = m.data[i] / bias_correction1;
            const double v_hat = v.data[i] / bias_correction2;

            // Decoupled weight decay (AdamW)
            val.data[i] *= (1.0 - lr_ * weight_decay_);
            val.data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
        }
    }
}
