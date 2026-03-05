#include "trainer.hpp"
#include "node.hpp"      // Node::constant
#include <limits>

// Trainer keeps references to all components.
Trainer::Trainer(NeuralNetwork& model,
                 LossFunction& lossFn,
                 Optimizer& optimizer,
                 DataLoader& dataLoader,
                 GradSyncFn gradSyncFn)
    : model_(model),
      lossFn_(lossFn),
      optimizer_(optimizer),
      dataLoader_(dataLoader),
      trainable_params_(model.getParameters()),
      grad_sync_fn_(std::move(gradSyncFn)) {}

Metrics Trainer::trainEpoch() {
    Metrics m = runEpoch(/*training=*/true);
    return m;
}

Metrics Trainer::evaluate() {
    Metrics m = runEpoch(/*training=*/false);
    return m;
}

// Shared implementation for training and evaluation.
Metrics Trainer::runEpoch(bool training) {
    double total_loss = 0.0;
    std::uint64_t total_samples = 0;
    std::uint64_t total_correct = 0;

    // Assumed DataLoader API: reset() and hasNext().
    dataLoader_.reset();
    

    Matrix batch_x(dataLoader_.batchSize(), dataLoader_.inputCols());
    Matrix batch_y(dataLoader_.batchSize(), dataLoader_.targetCols());

    while (dataLoader_.hasNext()) {

        size_t actual = dataLoader_.nextBatchInto(batch_x, batch_y);
        if (actual == 0) break;
        
        // Each batch is (inputs, targets).
        const Matrix& inputs  = batch_x;
        const Matrix& targets = batch_y;

        std::uint64_t batch_size = static_cast<std::uint64_t>(actual);
        total_samples += batch_size;

        // Wrap raw matrices into computation graph nodes.
        auto x = constant(inputs);
        auto y = constant(targets);

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
            std::uint64_t global_batch_size = batch_size;
            if (grad_sync_fn_) {
                global_batch_size = grad_sync_fn_(trainable_params_, batch_size);
            }
            if (global_batch_size > 0) {
                optimizer_.step(1.0 / static_cast<double>(global_batch_size));
            }
        }
    }

    Metrics m;
    m.loss_sum = total_loss;
    m.sample_count = total_samples;
    m.correct_count = total_correct;
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
