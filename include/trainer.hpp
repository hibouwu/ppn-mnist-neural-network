#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "distributed/sync_profile.hpp"
#include "neural_network.hpp"  // NeuralNetwork
#include "loss.hpp"       // LossFunction
#include "optimizer.hpp"  // Optimizer
#include "dataloader.hpp" // DataLoader

class GradientSyncRuntime;

struct EpochProfile {
    double epoch_time_s = 0.0;
    double data_time_s = 0.0;
    double fwd_bwd_time_s = 0.0;
    double sync_total_time_s = 0.0;
    double sync_wait_time_s = 0.0;
    double sync_pack_time_s = 0.0;
    double sync_launch_time_s = 0.0;
    double sync_unpack_time_s = 0.0;
    double opt_time_s = 0.0;
    double step_time_s_sum = 0.0;
    double max_step_time_s = 0.0;
    std::uint64_t step_count = 0;
    std::uint64_t sync_bucket_count = 0;
    std::uint64_t sync_bucket_bytes = 0;
    std::uint64_t sync_launched_bucket_count = 0;
    std::uint64_t sync_effective_overlap = 0;
};

// Basic statistics for one epoch
struct Metrics {
    double loss_sum = 0.0;   // summed loss over all processed samples
    std::uint64_t sample_count = 0;
    std::uint64_t correct_count = 0;
    double avg_loss = 0.0;   // mean loss over all samples
    double accuracy = 0.0;   // classification accuracy in [0, 1]
    EpochProfile profile;
};

// Trainer: orchestrates model, loss, optimizer and dataloader.
// It implements the high-level loop:
//   forward -> loss -> backward -> parameter update
class Trainer {
public:
    using GradSyncFn = std::function<std::uint64_t(
        const std::vector<Node::Ptr>& params, std::uint64_t local_batch)>;
    using SyncProfileProviderFn = std::function<SyncStepProfile()>;
    using ProgressFn = std::function<void(
        bool training,
        std::uint64_t processed_batches,
        std::uint64_t total_batches,
        std::uint64_t processed_samples,
        std::uint64_t total_samples)>;

    // Trainer keeps references; it does not own these objects.
    Trainer(NeuralNetwork& model,
            LossFunction& lossFn,
            Optimizer& optimizer,
            DataLoader& dataLoader,
            GradSyncFn gradSyncFn = nullptr,
            ProgressFn progressFn = nullptr,
            GradientSyncRuntime* gradientSyncRuntime = nullptr,
            SyncProfileProviderFn syncProfileProvider = nullptr);

    // One training epoch (with backward + parameter updates)
    Metrics trainEpoch();

    // One evaluation epoch (no parameter updates)
    Metrics evaluate();

private:
    NeuralNetwork&   model_;
    LossFunction& lossFn_;
    Optimizer&    optimizer_;
    DataLoader&   dataLoader_;
    std::vector<Node::Ptr> trainable_params_;
    GradSyncFn grad_sync_fn_;
    ProgressFn progress_fn_;
    GradientSyncRuntime* gradient_sync_runtime_ = nullptr;
    SyncProfileProviderFn sync_profile_provider_;

    // Shared implementation used by trainEpoch() and evaluate()
    Metrics runEpoch(bool training);

    // Count correctly classified samples in a batch
    std::size_t countCorrect(const Matrix& logits,
                             const Matrix& targets) const;
};
