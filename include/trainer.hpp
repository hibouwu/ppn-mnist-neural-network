#pragma once

#include <cstddef>
#include "network.hpp"    // MLPNetwork
#include "loss.hpp"       // LossFunction
#include "optimizer.hpp"  // Optimizer
#include "dataloader.hpp" // DataLoader

// Basic statistics for one epoch
struct Metrics {
    double avg_loss = 0.0;   // mean loss over all samples
    double accuracy = 0.0;   // classification accuracy in [0, 1]
};

// Trainer: orchestrates model, loss, optimizer and dataloader.
// It implements the high-level loop:
//   forward -> loss -> backward -> parameter update
class Trainer {
public:
    // Trainer keeps references; it does not own these objects.
    Trainer(MLPNetwork& model,
            LossFunction& lossFn,
            Optimizer& optimizer,
            DataLoader& dataLoader);

    // One training epoch (with backward + parameter updates)
    Metrics trainEpoch();

    // One evaluation epoch (no parameter updates)
    Metrics evaluate();

private:
    MLPNetwork&   model_;
    LossFunction& lossFn_;
    Optimizer&    optimizer_;
    DataLoader&   dataLoader_;

    // Shared implementation used by trainEpoch() and evaluate()
    Metrics runEpoch(bool training);

    // Count correctly classified samples in a batch
    std::size_t countCorrect(const Matrix& logits,
                             const Matrix& targets) const;
};

