#include "trainer.hpp"
#include "node.hpp"      // Node::constant
#include <chrono>
#include <algorithm>
#include <limits>

// Trainer keeps references to all components.
Trainer::Trainer(NeuralNetwork& model,
                 LossFunction& lossFn,
                 Optimizer& optimizer,
                 DataLoader& dataLoader,
                 GradSyncFn gradSyncFn,
                 ProgressFn progressFn)
    : model_(model),
      lossFn_(lossFn),
      optimizer_(optimizer),
      dataLoader_(dataLoader),
      trainable_params_(model.getParameters()),
      grad_sync_fn_(std::move(gradSyncFn)),
      progress_fn_(std::move(progressFn)) {}

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
    using Clock = std::chrono::steady_clock;
    const auto epoch_start = Clock::now();

    double total_loss = 0.0;
    std::uint64_t total_samples = 0;
    std::uint64_t total_correct = 0;
    std::uint64_t processed_batches = 0;
    EpochProfile profile;
    const std::uint64_t total_epoch_samples =
        static_cast<std::uint64_t>(dataLoader_.totalRows());
    const std::uint64_t total_epoch_batches =
        static_cast<std::uint64_t>(dataLoader_.totalBatches());

    // Assumed DataLoader API: reset() and hasNext().
    dataLoader_.reset();
    

    Matrix batch_x(dataLoader_.batchSize(), dataLoader_.inputCols());
    Matrix batch_y(dataLoader_.batchSize(), dataLoader_.targetCols());

    while (dataLoader_.hasNext()) {
        const auto step_start = Clock::now();

        const auto data_start = Clock::now();
        size_t actual = dataLoader_.nextBatchInto(batch_x, batch_y);
        const auto data_end = Clock::now();
        profile.data_time_s += std::chrono::duration<double>(data_end - data_start).count();
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
        const auto fwd_bwd_start = Clock::now();
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
            const auto fwd_bwd_end = Clock::now();
            profile.fwd_bwd_time_s += std::chrono::duration<double>(fwd_bwd_end - fwd_bwd_start).count();

            std::uint64_t global_batch_size = batch_size;
            if (grad_sync_fn_) {
                const auto sync_start = Clock::now();
                global_batch_size = grad_sync_fn_(trainable_params_, batch_size);
                const auto sync_end = Clock::now();
                const double sync_time_s =
                    std::chrono::duration<double>(sync_end - sync_start).count();
                // Current path is fully blocking, so total communication time equals exposed wait time.
                profile.sync_total_time_s += sync_time_s;
                profile.sync_wait_time_s += sync_time_s;
            }
            if (global_batch_size > 0) {
                const auto opt_start = Clock::now();
                optimizer_.step(1.0 / static_cast<double>(global_batch_size));
                const auto opt_end = Clock::now();
                profile.opt_time_s += std::chrono::duration<double>(opt_end - opt_start).count();
            }
        } else {
            const auto fwd_bwd_end = Clock::now();
            profile.fwd_bwd_time_s += std::chrono::duration<double>(fwd_bwd_end - fwd_bwd_start).count();
        }

        const auto step_end = Clock::now();
        const double step_time_s =
            std::chrono::duration<double>(step_end - step_start).count();
        profile.step_time_s_sum += step_time_s;
        profile.max_step_time_s = std::max(profile.max_step_time_s, step_time_s);
        profile.step_count += 1;
        processed_batches += 1;

        if (progress_fn_) {
            progress_fn_(training,
                         processed_batches,
                         total_epoch_batches,
                         total_samples,
                         total_epoch_samples);
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
    profile.epoch_time_s = std::chrono::duration<double>(Clock::now() - epoch_start).count();
    m.profile = profile;
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
