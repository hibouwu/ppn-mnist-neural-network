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
#include <string>

class GradientSyncRuntime;

struct GradCompressionConfig {
    bool enabled = false;
    std::string mode = "none";
    double topk_ratio = 0.1;
    bool error_feedback = false;
    int interval = 1;
};

struct IstTrainingConfig {
    bool enabled = false;
    int local_steps = 1;
    const std::vector<Matrix>* ownership_masks = nullptr;
};


struct EpochProfile {
    double epoch_time_s = 0.0;
    double data_time_s = 0.0;
    double fwd_bwd_time_s = 0.0;
    double sync_total_time_s = 0.0;
    double sync_wait_time_s = 0.0;
    double sync_pack_time_s = 0.0;
    double sync_launch_time_s = 0.0;
    double sync_unpack_time_s = 0.0;
    double sync_encode_time_s = 0.0;
    double sync_comm_time_s = 0.0;
    double sync_decode_time_s = 0.0;
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
    using StepObserverFn = std::function<void(
        std::uint64_t step_index,
        const SyncStepProfile* sync_profile)>;
    using ParamAllReduceFn = std::function<void(Scalar* data, std::size_t n)>;

    // Trainer keeps references; it does not own these objects.
    Trainer(NeuralNetwork& model,
            LossFunction& lossFn,
            Optimizer& optimizer,
            DataLoader& dataLoader,
            GradSyncFn gradSyncFn = nullptr,
            ProgressFn progressFn = nullptr,
            GradientSyncRuntime* gradientSyncRuntime = nullptr,
            SyncProfileProviderFn syncProfileProvider = nullptr,
            StepObserverFn stepObserver = nullptr,
            GradCompressionConfig grad_compression_cfg = {},
            bool grad_compression_handled_in_runtime = false,
            IstTrainingConfig ist_training_cfg = {},
            ParamAllReduceFn paramAllReduceFn = nullptr);

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
    StepObserverFn step_observer_;
    GradCompressionConfig grad_compression_cfg_;
    bool grad_compression_handled_in_runtime_ = false;
    IstTrainingConfig ist_training_cfg_;
    ParamAllReduceFn param_all_reduce_fn_;
    std::vector<Matrix> ist_snapshots_;
    bool ist_snapshots_initialized_ = false;
    std::uint64_t ist_local_step_counter_ = 0;


    // Shared implementation used by trainEpoch() and evaluate()
    Metrics runEpoch(bool training);

    // Count correctly classified samples in a batch
    std::size_t countCorrect(const Matrix& logits,
                             const Matrix& targets) const;
};
