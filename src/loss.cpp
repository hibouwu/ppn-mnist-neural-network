#include "loss.hpp"
#include <stdexcept>
#include <cmath>

namespace Loss {

using NodePtr = std::shared_ptr<Node>;

// MSELoss : mean( (pred - target)^2 )
NodePtr MSELoss(const NodePtr& pred, const NodePtr& target) {
    const Matrix& p = pred->value();
    const Matrix& t = target->value();

    if (p.rows != t.rows || p.cols != t.cols) {
        throw std::invalid_argument("MSELoss : pred et target n'ont pas la même forme.");
    }

    std::size_t n = p.data.size();
    double sum = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        double diff = p.data[i] - t.data[i];
        sum += diff * diff;
    }

    double mean = (n > 0) ? sum / static_cast<double>(n) : 0.0;
    Matrix out(1,1);
    out.data[0] = mean;

    NodePtr loss = std::make_shared<Node>(out);
    loss->addParent(pred);

    // backward : dL/dpred = 2/N * (pred - target) * grad_out
    loss->setBackwardFn([pred, target, n](const Matrix& grad_out) {
        double g = grad_out.data[0];          // scalaire
        const Matrix& p = pred->value();
        const Matrix& t = target->value();

        Matrix gp(p.rows, p.cols);
        double coeff = (n > 0) ? (2.0 * g / static_cast<double>(n)) : 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            gp.data[i] = coeff * (p.data[i] - t.data[i]);
        }

        pred->addGrad(gp);
        // target est traité comme constante : pas de gradient
    });

    return loss;
}

// CrossEntropyLoss
// L = - (1 / batch_size) * sum_i sum_j target_ij * log(pred_ij + eps)
NodePtr CrossEntropyLoss(const NodePtr& pred,
                         const NodePtr& target,
                         double eps) {
    const Matrix& p = pred->value();
    const Matrix& t = target->value();

    if (p.rows != t.rows || p.cols != t.cols) {
        throw std::invalid_argument("CrossEntropyLoss : pred et target n'ont pas la même forme.");
    }

    std::size_t n = p.data.size();
    std::size_t batch_size = p.rows;

    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ti = t.data[i];
        if (ti != 0.0) {
            double pi = p.data[i];
            sum += -ti * std::log(pi + eps);
        }
    }

    double mean = (batch_size > 0) ? sum / static_cast<double>(batch_size) : 0.0;
    Matrix out(1,1);
    out.data[0] = mean;

    NodePtr loss = std::make_shared<Node>(out);
    loss->addParent(pred);

    // backward : dL/dpred_ij = - (1 / batch_size) * target_ij / (pred_ij + eps) * grad_out
    loss->setBackwardFn([pred, target, eps, batch_size](const Matrix& grad_out) {
        double g = grad_out.data[0];
        const Matrix& p = pred->value();
        const Matrix& t = target->value();

        Matrix gp(p.rows, p.cols);
        std::size_t n = p.data.size();
        double scale = (batch_size > 0) ? (g / static_cast<double>(batch_size)) : 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            double ti = t.data[i];
            if (ti != 0.0) {
                gp.data[i] = -scale * ti / (p.data[i] + eps);
            } else {
                gp.data[i] = 0.0;
            }
        }

        pred->addGrad(gp);
    });

    return loss;
}

} // namespace Loss
