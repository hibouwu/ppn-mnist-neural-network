#include "optimizer.hpp"
#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace {

template <typename T>
void writeExact(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("Failed to write optimizer state.");
    }
}

template <typename T>
void readExact(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error("Failed to read optimizer state.");
    }
}

void writeMatrix(std::ostream& os, const Matrix& matrix) {
    const std::uint64_t rows = static_cast<std::uint64_t>(matrix.rows);
    const std::uint64_t cols = static_cast<std::uint64_t>(matrix.cols);
    const std::uint64_t size = static_cast<std::uint64_t>(matrix.data.size());
    writeExact(os, rows);
    writeExact(os, cols);
    writeExact(os, size);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(matrix.data.data()),
                 static_cast<std::streamsize>(size * sizeof(Scalar)));
        if (!os) {
            throw std::runtime_error("Failed to write optimizer matrix state.");
        }
    }
}

Matrix readMatrix(std::istream& is) {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t size = 0;
    readExact(is, rows);
    readExact(is, cols);
    readExact(is, size);

    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Invalid optimizer matrix shape in checkpoint.");
    }
    if (size != rows * cols) {
        throw std::runtime_error("Corrupted optimizer matrix size in checkpoint.");
    }

    Matrix matrix(static_cast<std::size_t>(rows), static_cast<std::size_t>(cols));
    if (size > 0) {
        is.read(reinterpret_cast<char*>(matrix.data.data()),
                static_cast<std::streamsize>(size * sizeof(Scalar)));
        if (!is) {
            throw std::runtime_error("Failed to read optimizer matrix state.");
        }
    }
    return matrix;
}

void validateStateShape(const Matrix& actual,
                        const Matrix& expected,
                        const char* state_name,
                        std::size_t index) {
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        throw std::runtime_error(
            std::string("Checkpoint ") + state_name +
            " shape mismatch for parameter #" + std::to_string(index) + ".");
    }
}

}  // namespace

namespace {
constexpr float kSparseEps = 1e-12f;
}

Optimizer::Optimizer(std::vector<Node::Ptr> params, double lr)
    : parameters_(std::move(params)), lr_(lr) {}

void Optimizer::saveState(std::ostream&) const {}

void Optimizer::loadState(std::istream&) {}

void Optimizer::zeroGrad() {
    for (auto& p : parameters_) {
        p->zeroGrad();
    }
}

SGDOptimizer::SGDOptimizer(std::vector<Node::Ptr> params, double lr)
    : Optimizer(std::move(params), lr) {}

std::string SGDOptimizer::typeName() const {
    return "sgd";
}

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

std::string MomentumSGDOptimizer::typeName() const {
    return "momentum_sgd";
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

void MomentumSGDOptimizer::saveState(std::ostream& os) const {
    const std::uint64_t velocity_count = static_cast<std::uint64_t>(velocity_.size());
    writeExact(os, velocity_count);
    for (const Matrix& velocity : velocity_) {
        writeMatrix(os, velocity);
    }
}

void MomentumSGDOptimizer::loadState(std::istream& is) {
    std::uint64_t velocity_count = 0;
    readExact(is, velocity_count);
    if (velocity_count != velocity_.size()) {
        throw std::runtime_error("Momentum checkpoint parameter count mismatch.");
    }

    for (std::size_t i = 0; i < velocity_.size(); ++i) {
        Matrix loaded = readMatrix(is);
        validateStateShape(loaded, velocity_[i], "momentum velocity", i);
        velocity_[i] = std::move(loaded);
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

std::string AdamWOptimizer::typeName() const {
    return "adamw";
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

void AdamWOptimizer::saveState(std::ostream& os) const {
    writeExact(os, static_cast<std::uint64_t>(step_count_));

    const std::uint64_t first_count = static_cast<std::uint64_t>(first_moment_.size());
    writeExact(os, first_count);
    for (const Matrix& moment : first_moment_) {
        writeMatrix(os, moment);
    }

    const std::uint64_t second_count = static_cast<std::uint64_t>(second_moment_.size());
    writeExact(os, second_count);
    for (const Matrix& moment : second_moment_) {
        writeMatrix(os, moment);
    }
}

void AdamWOptimizer::loadState(std::istream& is) {
    std::uint64_t saved_step_count = 0;
    readExact(is, saved_step_count);
    step_count_ = static_cast<std::size_t>(saved_step_count);

    std::uint64_t first_count = 0;
    readExact(is, first_count);
    if (first_count != first_moment_.size()) {
        throw std::runtime_error("AdamW checkpoint first-moment parameter count mismatch.");
    }
    for (std::size_t i = 0; i < first_moment_.size(); ++i) {
        Matrix loaded = readMatrix(is);
        validateStateShape(loaded, first_moment_[i], "AdamW first moment", i);
        first_moment_[i] = std::move(loaded);
    }

    std::uint64_t second_count = 0;
    readExact(is, second_count);
    if (second_count != second_moment_.size()) {
        throw std::runtime_error("AdamW checkpoint second-moment parameter count mismatch.");
    }
    for (std::size_t i = 0; i < second_moment_.size(); ++i) {
        Matrix loaded = readMatrix(is);
        validateStateShape(loaded, second_moment_[i], "AdamW second moment", i);
        second_moment_[i] = std::move(loaded);
    }
}
