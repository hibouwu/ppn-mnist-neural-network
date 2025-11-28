#include "trainer.hpp"
#include "node.hpp"      // Node::constant
#include <limits>
#include <iostream>

// Trainer keeps references to all components.
Trainer::Trainer(MLPNetwork& model,
                 LossFunction& lossFn,
                 Optimizer& optimizer,
                 DataLoader& dataLoader)
    : model_(model),
      lossFn_(lossFn),
      optimizer_(optimizer),
      dataLoader_(dataLoader) {}

Metrics Trainer::trainEpoch() {
    Metrics m = runEpoch(/*training=*/true);
    std::cout << "[Train] loss = " << m.avg_loss
              << ", acc = " << m.accuracy << std::endl;
    return m;
}

Metrics Trainer::evaluate() {
    Metrics m = runEpoch(/*training=*/false);
    std::cout << "[Eval ] loss = " << m.avg_loss
              << ", acc = " << m.accuracy << std::endl;
    return m;
}

// Shared implementation for training and evaluation.
Metrics Trainer::runEpoch(bool training) {
    double total_loss = 0.0;
    std::size_t total_samples = 0;
    std::size_t total_correct = 0;

    // Assumed DataLoader API: reset() and hasNext().
    dataLoader_.reset();

    while (dataLoader_.hasNext()) {
        // Each batch is (inputs, targets).
        auto batch = dataLoader_.nextBatch();
        const Matrix& inputs  = batch.first;
        const Matrix& targets = batch.second;

        std::size_t batch_size = inputs.rows;
        total_samples += batch_size;

        // Wrap raw matrices into computation graph nodes.
        auto x = Node::constant(inputs);
        auto y = Node::constant(targets);

        // Forward pass through the model.
        auto preds = model_.forward(x);

        // Loss node (usually a scalar).
        auto loss_node = lossFn_.forward(preds, y);

        // Accumulate loss value (assume 1x1 matrix).
        const Matrix& loss_val = loss_node->value();
        total_loss += loss_val(0, 0);

        // Accuracy for this batch.
        total_correct += countCorrect(preds->value(), targets);

        // Backward + parameter update only in training mode.
        if (training) {
            optimizer_.zeroGrad();
            loss_node->backward();
            optimizer_.step();
        }
    }

    Metrics m;
    if (total_samples > 0) {
        m.avg_loss = total_loss / static_cast<double>(total_samples);
        m.accuracy = static_cast<double>(total_correct) /
                     static_cast<double>(total_samples);
    }
    return m;
}

// Count correct predictions assuming:
//  - logits: (batch_size x num_classes)
//  - targets: same shape, one-hot encoded.
std::size_t Trainer::countCorrect(const Matrix& logits,
                                  const Matrix& targets) const {
    std::size_t batch_size  = logits.rows;
    std::size_t num_classes = logits.cols;
    std::size_t correct = 0;

    for (std::size_t i = 0; i < batch_size; ++i) {
        // argmax over logits
        std::size_t pred_idx = 0;
        double max_logit = std::numeric_limits<double>::lowest();
        for (std::size_t c = 0; c < num_classes; ++c) {
            double v = logits(i, c);
            if (v > max_logit) {
                max_logit = v;
                pred_idx = c;
            }
        }

        // argmax over targets (one-hot)
        std::size_t target_idx = 0;
        double max_target = std::numeric_limits<double>::lowest();
        for (std::size_t c = 0; c < num_classes; ++c) {
            double v = targets(i, c);
            if (v > max_target) {
                max_target = v;
                target_idx = c;
            }
        }

        if (pred_idx == target_idx) {
            ++correct;
        }
    }

    return correct;
}
