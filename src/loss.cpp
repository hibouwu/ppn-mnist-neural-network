#include "loss.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

LossFunction::NodePtr MSELoss::forward(const NodePtr& pred, const NodePtr& target) {
    const Matrix& p = pred->value();
    const Matrix& t = target->value();

    if (p.rows != t.rows || p.cols != t.cols) {
        throw std::invalid_argument("MSELoss::forward : pred/target shape mismatch");
    }

    const std::size_t B = p.rows;
    const std::size_t C = p.cols;
    const std::size_t n = p.data.size();

    // L = (1/C) * sum (p - t)^2   (donc somme sur batch, normalisée par nb de classes)
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double d = p.data[i] - t.data[i];
        sum += d * d;
    }

    double loss_val = (C > 0) ? sum / static_cast<double>(C) : 0.0;

    Matrix out(1, 1);
    out.data[0] = loss_val;

    auto loss = std::make_shared<Node>(out);
    loss->addParent(pred);

    // dL/dp_ij = (2/C) * (p_ij - t_ij) * grad_out
    loss->setBackwardFn([pred, target, C](const Matrix& grad_out) {
        const Matrix& p = pred->value();
        const Matrix& t = target->value();

        Matrix gp(p.rows, p.cols, 0.0);

        double g = grad_out.data[0];
        double coeff = (C > 0) ? (2.0 * g / static_cast<double>(C)) : 0.0;

        const std::size_t n = p.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            gp.data[i] = coeff * (p.data[i] - t.data[i]);
        }

        pred->addGrad(gp);
        // target traité comme constante
    });

    return loss;
}

LossFunction::NodePtr CrossEntropyLoss::forward(const NodePtr& pred, const NodePtr& target) {
    const Matrix& logits = pred->value();
    const Matrix& y      = target->value();

    if (logits.rows != y.rows || logits.cols != y.cols) {
        throw std::invalid_argument("CrossEntropyLoss::forward : pred/target shape mismatch");
    }

    const std::size_t B = logits.rows;
    const std::size_t C = logits.cols;

    // On calcule softmax(logits) de manière stable, et loss = sum_i - sum_c y_ic log(p_ic)
    // IMPORTANT: on renvoie la somme sur le batch (pas la moyenne) pour coller à Trainer::avg_loss = total_loss/total_samples
    Matrix probs(B, C, 0.0);

    double total = 0.0;
    for (std::size_t i = 0; i < B; ++i) {
        // max pour stabilité
        double m = logits.data[i * C + 0];
        for (std::size_t c = 1; c < C; ++c) {
            m = std::max(m, logits.data[i * C + c]);
        }

        // exp et somme
        double s = 0.0;
        for (std::size_t c = 0; c < C; ++c) {
            double e = std::exp(logits.data[i * C + c] - m);
            probs.data[i * C + c] = e;
            s += e;
        }

        // normalisation + loss
        for (std::size_t c = 0; c < C; ++c) {
            probs.data[i * C + c] = probs.data[i * C + c] / (s + eps_);
            double yi = y.data[i * C + c];
            if (yi != 0.0) {
                total += -yi * std::log(probs.data[i * C + c] + eps_);
            }
        }
    }

    Matrix out(1, 1);
    out.data[0] = total;

    auto loss = std::make_shared<Node>(out);
    loss->addParent(pred);

    // Gradient connu et stable pour CE(softmax(logits), y):
    // dL/dlogits = (probs - y) * grad_out  (ici loss est somme batch, donc pas /B)
    loss->setBackwardFn([pred, target, probs](const Matrix& grad_out) {
        const Matrix& y = target->value();
        Matrix g_logits(probs.rows, probs.cols, 0.0);

        double g = grad_out.data[0];

        const std::size_t n = probs.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            g_logits.data[i] = g * (probs.data[i] - y.data[i]);
        }

        pred->addGrad(g_logits);
    });

    return loss;
}

