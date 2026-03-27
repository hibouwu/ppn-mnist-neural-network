#pragma once

#include "dataset.hpp"
#include "io/derived_training_stats.hpp"
#include "io/run_artifacts_writer.hpp"
#include "runtime/grad_sync_mode_info.hpp"
#include "trainer.hpp"

#include <cstddef>
#include <string>

namespace io {

struct TrainingStartupSummaryData {
    std::string dataset;
    std::string data_dir;
    DatasetInfo dataset_info;
    int epochs = 0;
    int batch_size = 0;
    double learning_rate = 0.0;
    std::string hidden_sizes;
    unsigned int seed = 0;
    std::string activation;
    std::string init;
    std::string model;
    std::string optimizer;
    runtime::GradSyncModeInfo grad_sync_mode;
    std::size_t bucket_size_bytes = 0;
    double momentum = 0.0;
    bool nesterov = false;
    double weight_decay = 0.0;
    double beta1 = 0.0;
    double beta2 = 0.0;
    double eps = 0.0;
    int world_size = 1;
};

std::string formatTrainingStartupSummary(const TrainingStartupSummaryData& data);
std::string formatGradSyncWarning(const runtime::GradSyncModeInfo& mode_info);
std::string formatEpochSummary(int epoch,
                               int total_epochs,
                               const Metrics& train_metrics,
                               const Metrics& test_metrics,
                               const DerivedTrainingStats& derived_stats);
std::string formatProfileRunTotalSummary(const RunProfileSummary& summary, double uncovered_us);
std::string formatProfileRunOpRow(std::size_t rank,
                                  const RunOpStat& stat,
                                  long long profiled_total_us,
                                  double run_fwd_bwd_us);

}
