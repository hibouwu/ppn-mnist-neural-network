#pragma once

#include "distributed/sync_profile.hpp"
#include "io/derived_training_stats.hpp"
#include "node.hpp"
#include "profiling.hpp"
#include "runtime/grad_sync_mode_info.hpp"
#include "trainer.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace io {

struct RunProfileSummary {
    std::uint64_t total_epochs = 0;
    double fwd_bwd_s = 0.0;
    double sync_total_s = 0.0;
    double sync_wait_s = 0.0;
    double opt_s = 0.0;
    std::uint64_t sync_effective_overlap = 0;
    runtime::GradSyncModeInfo grad_sync_mode;
    long long profiled_total_us = 0;
};

struct RunOpStat {
    std::string scope;
    std::string name;
    std::size_t calls = 0;
    long long total_us = 0;
};

struct RunArtifactsPaths {
    std::string metrics_csv_path;
    std::string profile_epoch_summary_csv_path;
    std::string profile_run_summary_csv_path;
    std::string profile_epoch_ops_csv_path;
    std::string profile_run_ops_csv_path;
    std::string qualification_dir;
    std::string qualification_rank_dir;
    std::string sync_trace_csv_path;
    std::string parameter_layout_csv_path;
    std::string parameter_snapshot_index_csv_path;
    std::string parameter_snapshot_dir;
    std::string run_id;
};

RunArtifactsPaths makeRunArtifactsPaths(const std::string& out_dir, int rank);

class RunArtifactsWriter {
public:
    RunArtifactsWriter(bool enabled,
                       bool qualification_enabled,
                       RunArtifactsPaths paths);

    const RunArtifactsPaths& paths() const { return paths_; }

    void initialize();
    void initializeQualificationArtifacts(const std::vector<Node::Ptr>& synchronizable_params);
    void appendProfileEpochSummary(int epoch,
                                   const runtime::GradSyncModeInfo& mode_info,
                                   const EpochProfile& profile,
                                   long long profiled_total_us);
    void appendProfileEpochOps(int epoch, const std::vector<OpTimingStat>& stats);
    void appendMetricsRow(int epoch,
                          const Metrics& train_metrics,
                          const Metrics& test_metrics,
                          const DerivedTrainingStats& derived_stats,
                          const runtime::GradSyncModeInfo& mode_info,
                          int world_size,
                          int batch_size);
    void appendSyncTraceStep(int epoch,
                             std::uint64_t step_index,
                             const SyncStepProfile& profile);
    void writeParameterSnapshot(int epoch,
                                std::uint64_t step_index,
                                const std::vector<Node::Ptr>& synchronizable_params);
    void writeRunSummary(const RunProfileSummary& summary);
    void writeRunOps(const std::vector<RunOpStat>& run_stats);

private:
    bool enabled_ = false;
    bool qualification_enabled_ = false;
    RunArtifactsPaths paths_;
    std::ofstream metrics_file_;
};

}
