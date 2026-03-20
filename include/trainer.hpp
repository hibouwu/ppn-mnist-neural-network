#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "neural_network.hpp"  // NeuralNetwork
#include "loss.hpp"       // LossFunction
#include "optimizer.hpp"  // Optimizer
#include "dataloader.hpp" // DataLoader

struct EpochProfile {
    double epoch_time_s = 0.0;
    double data_time_s = 0.0;
    double fwd_bwd_time_s = 0.0;
    double sync_total_time_s = 0.0;
    double sync_wait_time_s = 0.0;
    double opt_time_s = 0.0;
    double step_time_s_sum = 0.0;
    double max_step_time_s = 0.0;
    std::uint64_t step_count = 0;
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

    // Trainer keeps references; it does not own these objects.
    Trainer(NeuralNetwork& model,
            LossFunction& lossFn,
            Optimizer& optimizer,
            DataLoader& dataLoader,
            GradSyncFn gradSyncFn = nullptr);

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

    // Shared implementation used by trainEpoch() and evaluate()
    Metrics runEpoch(bool training);

    // Count correctly classified samples in a batch
    std::size_t countCorrect(const Matrix& logits,
                             const Matrix& targets) const;
};
