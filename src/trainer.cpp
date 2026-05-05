#include "trainer.hpp"
#include "autograd/engine.hpp"
#include "distributed/gradient_sync_runtime.hpp"
#include "node.hpp"      // Node::constant
#include "profiling.hpp"
#include <chrono>
#include <algorithm>
#include <limits>
#include <optional>
#include <cmath>

namespace {

Scalar quantizeToFp16Like(Scalar x) {
    if (!std::isfinite(x) || x == 0.0f) {
        return x;
    }

    int exp = 0;
    const float mantissa = std::frexp(x, &exp);

    // Half precision exponent range (approx): [-14, 15] for normal numbers.
    if (exp > 16) {
        return std::copysign(std::numeric_limits<Scalar>::infinity(), x);
    }
    if (exp < -24) {
        return std::copysign(0.0f, x);
    }

    // Keep about 10 mantissa bits to mimic fp16 rounding.
    const float rounded_mantissa =
        std::ldexp(std::nearbyint(std::ldexp(mantissa, 10)), -10);

    return std::ldexp(rounded_mantissa, exp);
}

void keepTopKAbsInPlace(Matrix& g, double ratio) {
    const std::size_t n = g.data.size();
    if (n == 0) {
        return;
    }
    if (ratio >= 1.0) {
        return;
    }
    if (ratio <= 0.0) {
        std::fill(g.data.begin(), g.data.end(), 0.0f);
        return;
    }

    const std::size_t k = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(ratio * static_cast<double>(n))));
    const std::size_t split = n - k;

    std::vector<Scalar> mags(n);
    for (std::size_t i = 0; i < n; ++i) {
        mags[i] = std::fabs(g.data[i]);
    }

    std::nth_element(mags.begin(), mags.begin() + split, mags.end());
    const Scalar threshold = mags[split];

    for (Scalar& v : g.data) {
        if (std::fabs(v) < threshold) {
            v = 0.0f;
        }
    }
}

void compressGradientsIfEnabled(const std::vector<Node::Ptr>& params,
                                const GradCompressionConfig& cfg) {
    if (!cfg.enabled) {
        return;
    }

    if (cfg.mode == "fp16") {
        for (const auto& p : params) {
            if (!p || !p->hasAllocatedGrad()) {
                continue;
            }
            Matrix& g = p->grad();
            for (Scalar& v : g.data) {
                v = quantizeToFp16Like(v);
            }
        }
        return;
    }

    if (cfg.mode == "topk") {
        for (const auto& p : params) {
            if (!p || !p->hasAllocatedGrad()) {
                continue;
            }
            Matrix& g = p->grad();
            keepTopKAbsInPlace(g, cfg.topk_ratio);
        }
        return;
    }
}

void updateResidualIfEnabled(const std::vector<Node::Ptr>& params,
                             const GradCompressionConfig& cfg) {
    // Placeholder for future error-feedback residual update.
    (void)params;
    (void)cfg;
}



void accumulateSyncProfile(EpochProfile& profile,
                           const SyncStepProfile& sync_profile,
                           std::optional<double> sync_total_time_s = std::nullopt) {
    profile.sync_pack_time_s += sync_profile.pack_time_s;
    profile.sync_launch_time_s += sync_profile.launch_time_s;
    profile.sync_unpack_time_s += sync_profile.unpack_time_s;
    profile.sync_encode_time_s += sync_profile.encode_time_s;
    profile.sync_comm_time_s += sync_profile.comm_time_s;
    profile.sync_decode_time_s += sync_profile.decode_time_s;
    profile.sync_bucket_count += sync_profile.bucket_count;
    profile.sync_bucket_bytes += sync_profile.bucket_bytes;
    profile.sync_launched_bucket_count += sync_profile.launched_bucket_count;
    profile.sync_effective_overlap = std::max<std::uint64_t>(
        profile.sync_effective_overlap,
        sync_profile.effective_overlap ? 1ULL : 0ULL);

    if (sync_total_time_s.has_value()) {
        profile.sync_total_time_s += *sync_total_time_s;
        return;
    }

    profile.sync_total_time_s +=
        sync_profile.pack_time_s +
        sync_profile.launch_time_s +
        sync_profile.wait_time_s +
        sync_profile.unpack_time_s +
        sync_profile.batch_reduce_time_s;
}

bool istEnabled(const IstTrainingConfig& cfg) {
    return cfg.enabled && cfg.local_steps > 0 && cfg.ownership_masks != nullptr;
}

void applyIstOwnershipMasksToGradients(const std::vector<Node::Ptr>& params,
                                       const std::vector<Matrix>& masks) {
    const std::size_t n = std::min(params.size(), masks.size());
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = params[i];
        if (!p || !p->hasAllocatedGrad()) {
            continue;
        }
        Matrix& g = p->grad();
        const Matrix& m = masks[i];
        if (m.rows != g.rows || m.cols != g.cols) {
            continue;
        }
        for (std::size_t j = 0; j < g.data.size(); ++j) {
            g.data[j] *= m.data[j];
        }
    }
}

void ensureIstSnapshotsInitialized(const std::vector<Node::Ptr>& params,
                                   std::vector<Matrix>& snapshots,
                                   bool& initialized) {
    if (initialized) {
        return;
    }
    snapshots.clear();
    snapshots.reserve(params.size());
    for (const auto& p : params) {
        if (!p) {
            snapshots.emplace_back(1, 1, 0.0);
            continue;
        }
        snapshots.push_back(p->value());
    }
    initialized = true;
}

void runIstParameterSync(const std::vector<Node::Ptr>& params,
                         const std::vector<Matrix>& masks,
                         std::vector<Matrix>& snapshots,
                         const Trainer::ParamAllReduceFn& all_reduce_fn) {
    const std::size_t n = std::min({params.size(), masks.size(), snapshots.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = params[i];
        if (!p) {
            continue;
        }
        Matrix& current = const_cast<Matrix&>(p->value());
        Matrix& snapshot = snapshots[i];
        const Matrix& mask = masks[i];
        if (snapshot.rows != current.rows || snapshot.cols != current.cols ||
            mask.rows != current.rows || mask.cols != current.cols) {
            continue;
        }

        Matrix delta(current.rows, current.cols, MatrixInit::Uninitialized);
        for (std::size_t j = 0; j < current.data.size(); ++j) {
            delta.data[j] = (current.data[j] - snapshot.data[j]) * mask.data[j];
        }

        if (all_reduce_fn) {
            all_reduce_fn(delta.data.data(), delta.data.size());
        }

        for (std::size_t j = 0; j < current.data.size(); ++j) {
            current.data[j] = snapshot.data[j] + delta.data[j];
        }
        snapshot = current;
    }
}

}

// Trainer keeps references to all components.
Trainer::Trainer(NeuralNetwork& model,
                 LossFunction& lossFn,
                 Optimizer& optimizer,
                 DataLoader& dataLoader,
                 GradSyncFn gradSyncFn,
                 ProgressFn progressFn,
                 GradientSyncRuntime* gradientSyncRuntime,
                 SyncProfileProviderFn syncProfileProvider,
                 StepObserverFn stepObserver,
                 GradCompressionConfig grad_compression_cfg,
                 bool grad_compression_handled_in_runtime,
                 IstTrainingConfig ist_training_cfg,
                 ParamAllReduceFn paramAllReduceFn)
    : model_(model),
      lossFn_(lossFn),
      optimizer_(optimizer),
      dataLoader_(dataLoader),
      trainable_params_(model.getParameters()),
      grad_sync_fn_(std::move(gradSyncFn)),
      progress_fn_(std::move(progressFn)),
      gradient_sync_runtime_(gradientSyncRuntime),
      sync_profile_provider_(std::move(syncProfileProvider)),
      step_observer_(std::move(stepObserver)),
      grad_compression_cfg_(std::move(grad_compression_cfg)),
      grad_compression_handled_in_runtime_(grad_compression_handled_in_runtime),
      ist_training_cfg_(std::move(ist_training_cfg)),
      param_all_reduce_fn_(std::move(paramAllReduceFn)) {}


Metrics Trainer::trainEpoch() {
    Metrics m = runEpoch(/*training=*/true);
    return m;
}

Metrics Trainer::evaluate() {
    Metrics m = runEpoch(/*training=*/false);
    return m;
}

void Trainer::onIstOwnershipMasksUpdated() {
    ist_local_step_counter_ = 0;
    ist_snapshots_.clear();
    ist_snapshots_initialized_ = false;
}

// Shared implementation for training and evaluation.
Metrics Trainer::runEpoch(bool training) {
    using Clock = std::chrono::steady_clock;
    const auto epoch_start = Clock::now();
    const ScopedProfileTask epoch_scope(training ? "train_epoch" : "eval_epoch");

    double total_loss = 0.0;
    std::uint64_t total_samples = 0;
    std::uint64_t total_correct = 0;
    std::uint64_t processed_batches = 0;
    EpochProfile profile;
    const std::uint64_t total_epoch_samples =
        static_cast<std::uint64_t>(dataLoader_.totalRows());
    const std::uint64_t total_epoch_batches =
        static_cast<std::uint64_t>(dataLoader_.totalBatches());
    const bool ist_enabled = training && istEnabled(ist_training_cfg_);
    if (ist_enabled) {
        ensureIstSnapshotsInitialized(trainable_params_, ist_snapshots_, ist_snapshots_initialized_);
    }

    // Assumed DataLoader API: reset() and hasNext().
    dataLoader_.reset();
    

    Matrix batch_x(dataLoader_.batchSize(), dataLoader_.inputCols());
    Matrix batch_y(dataLoader_.batchSize(), dataLoader_.targetCols());

    while (dataLoader_.hasNext()) {
        const ScopedProfileTask batch_scope(training ? "train_batch" : "eval_batch");
        const auto step_start = Clock::now();

        const auto data_start = Clock::now();
        size_t actual = 0;
        {
            const ScopedProfileTask data_scope("data_loader");
            actual = dataLoader_.nextBatchInto(batch_x, batch_y);
        }
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
        Node::Ptr preds;
        Node::Ptr loss_node;
        {
            const ScopedProfileTask forward_scope(training ? "forward_loss" : "eval_forward_loss");
            preds = model_.forward(x);

            // Loss node (usually a scalar).
            loss_node = lossFn_.forward(preds, y);
        }

        // Accumulate loss value (assume 1x1 matrix).
        const Matrix& loss_val = loss_node->value();
        total_loss += loss_val(0, 0);

        // Accuracy for this batch.
        total_correct += countCorrect(preds->value(), targets);

        // Backward + parameter update only in training mode.
        if (training) {
            if (ist_enabled) {
                {
                    const ScopedProfileTask backward_scope("backward");
                    optimizer_.zeroGrad();
                    loss_node->backward();
                }
                const auto fwd_bwd_end = Clock::now();
                profile.fwd_bwd_time_s += std::chrono::duration<double>(fwd_bwd_end - fwd_bwd_start).count();

                applyIstOwnershipMasksToGradients(
                    trainable_params_, *ist_training_cfg_.ownership_masks);

                const auto opt_start = Clock::now();
                {
                    const ScopedProfileTask optimizer_scope("optimizer_step");
                    optimizer_.step(1.0 / static_cast<double>(batch_size));
                }
                const auto opt_end = Clock::now();
                profile.opt_time_s += std::chrono::duration<double>(opt_end - opt_start).count();

                ++ist_local_step_counter_;
                if ((ist_local_step_counter_ %
                     static_cast<std::uint64_t>(std::max(1, ist_training_cfg_.local_steps))) == 0) {
                    const auto sync_start = Clock::now();
                    {
                        const ScopedProfileTask sync_scope("gradient_sync");
                        runIstParameterSync(trainable_params_,
                                            *ist_training_cfg_.ownership_masks,
                                            ist_snapshots_,
                                            param_all_reduce_fn_);
                    }
                    const auto sync_end = Clock::now();
                    const double sync_time_s =
                        std::chrono::duration<double>(sync_end - sync_start).count();
                    profile.sync_total_time_s += sync_time_s;
                    profile.sync_wait_time_s += sync_time_s;
                }
                if (step_observer_) {
                    step_observer_(processed_batches + 1, nullptr);
                }
            } else {
                std::uint64_t global_batch_size = batch_size;
                std::optional<SyncStepProfile> step_sync_profile;
                {
                    const ScopedProfileTask backward_scope("backward");
                    optimizer_.zeroGrad();
                    if (gradient_sync_runtime_) {
                        gradient_sync_runtime_->beginStep(batch_size);
                        AutogradEngine engine;
                        engine.setReachableLeafHook([this](const std::vector<Node::Ptr>& reachable_leaf_params) {
                            gradient_sync_runtime_->planStep(reachable_leaf_params);
                        });
                        engine.setParameterReadyHook([this](Node& param) {
                            gradient_sync_runtime_->onParameterGradReady(param);
                        });
                        engine.setBackwardCompleteHook([this]() {
                            gradient_sync_runtime_->onBackwardComplete();
                        });
                        engine.backward(loss_node);
                    } else {
                        loss_node->backward();
                    }
                }
                const auto fwd_bwd_end = Clock::now();
                profile.fwd_bwd_time_s += std::chrono::duration<double>(fwd_bwd_end - fwd_bwd_start).count();

                if (gradient_sync_runtime_) {
                    const auto sync_start = Clock::now();
                    {
                        const ScopedProfileTask sync_scope("gradient_sync");
                        global_batch_size = gradient_sync_runtime_->finalizeAndGetGlobalBatch();
                    }
                    const auto sync_end = Clock::now();
                    profile.sync_wait_time_s += std::chrono::duration<double>(sync_end - sync_start).count();
                    if (sync_profile_provider_) {
                        step_sync_profile = sync_profile_provider_();
                        accumulateSyncProfile(profile, *step_sync_profile);
                    }
                } else if (grad_sync_fn_) {
                    const auto sync_start = Clock::now();
                    {
                        const ScopedProfileTask sync_scope("gradient_sync");
                        global_batch_size = grad_sync_fn_(trainable_params_, batch_size);
                    }
                    const auto sync_end = Clock::now();
                    const double sync_time_s =
                        std::chrono::duration<double>(sync_end - sync_start).count();
                    profile.sync_total_time_s += sync_time_s;
                    profile.sync_wait_time_s += sync_time_s;
                    if (sync_profile_provider_) {
                        step_sync_profile = sync_profile_provider_();
                        accumulateSyncProfile(profile, *step_sync_profile, sync_time_s);
                    }
                }
                if (global_batch_size > 0) {
                    const int interval = std::max(1, grad_compression_cfg_.interval);
                    const bool do_compress_this_step =
                        (processed_batches % static_cast<std::uint64_t>(interval) == 0);
                    if (!grad_compression_handled_in_runtime_ && do_compress_this_step) {
                        compressGradientsIfEnabled(trainable_params_, grad_compression_cfg_);
                    }
                    const auto opt_start = Clock::now();
                    {
                        const ScopedProfileTask optimizer_scope("optimizer_step");
                        optimizer_.step(1.0 / static_cast<double>(global_batch_size));
                    }
                    const auto opt_end = Clock::now();
                    profile.opt_time_s += std::chrono::duration<double>(opt_end - opt_start).count();
                    updateResidualIfEnabled(trainable_params_, grad_compression_cfg_);
                }
                if (step_observer_) {
                    step_observer_(processed_batches + 1,
                                   step_sync_profile.has_value() ? &*step_sync_profile : nullptr);
                }
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

    if (ist_enabled &&
        (ist_local_step_counter_ %
         static_cast<std::uint64_t>(std::max(1, ist_training_cfg_.local_steps))) != 0) {
        const auto sync_start = Clock::now();
        {
            const ScopedProfileTask sync_scope("gradient_sync");
            runIstParameterSync(trainable_params_,
                                *ist_training_cfg_.ownership_masks,
                                ist_snapshots_,
                                param_all_reduce_fn_);
        }
        const auto sync_end = Clock::now();
        const double sync_time_s =
            std::chrono::duration<double>(sync_end - sync_start).count();
        profile.sync_total_time_s += sync_time_s;
        profile.sync_wait_time_s += sync_time_s;
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
