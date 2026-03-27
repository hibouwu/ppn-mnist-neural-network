#include "io/run_artifacts_writer.hpp"

#include "io/run_op_stats.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace io {

RunArtifactsPaths makeRunArtifactsPaths(const std::string& out_dir) {
    RunArtifactsPaths paths;
    paths.metrics_csv_path = out_dir + "/metrics.csv";
    paths.profile_epoch_summary_csv_path = out_dir + "/profile_epoch_summary.csv";
    paths.profile_run_summary_csv_path = out_dir + "/profile_run_summary.csv";
    paths.profile_epoch_ops_csv_path = out_dir + "/profile_epoch_ops.csv";
    paths.profile_run_ops_csv_path = out_dir + "/profile_run_ops.csv";
    paths.run_id = fs::path(out_dir).filename().empty()
        ? std::string("default_run")
        : fs::path(out_dir).filename().string();
    return paths;
}

RunArtifactsWriter::RunArtifactsWriter(bool enabled, RunArtifactsPaths paths)
    : enabled_(enabled),
      paths_(std::move(paths)) {}

void RunArtifactsWriter::initialize() {
    if (!enabled_) {
        return;
    }

    metrics_file_.open(paths_.metrics_csv_path, std::ios::trunc);
    if (metrics_file_.is_open()) {
        metrics_file_
            << "epoch,train_loss,train_acc,test_loss,test_acc,train_samples,"
            << "epoch_time_s,data_time_s,fwd_bwd_time_s,sync_total_time_s,sync_wait_time_s,"
            << "sync_pack_time_s,sync_launch_time_s,sync_unpack_time_s,"
            << "sync_bucket_count,sync_bucket_bytes,sync_launched_bucket_count,sync_effective_overlap,opt_time_s,"
            << "avg_step_time_ms,max_step_time_ms,samples_per_s,allreduce_wait_ratio,"
            << "grad_sync_mode_label,grad_sync_semantics,grad_sync_correctness_only,"
            << "world_size,batch_size\n";
    }

    std::ofstream epoch_summary_out(paths_.profile_epoch_summary_csv_path, std::ios::trunc);
    epoch_summary_out
        << "epoch,grad_sync_mode_label,grad_sync_semantics,grad_sync_correctness_only,"
           "fwd_bwd_s,sync_total_s,sync_wait_s,sync_effective_overlap,opt_s,profiled_total_us\n";

    std::ofstream epoch_ops_out(paths_.profile_epoch_ops_csv_path, std::ios::trunc);
    epoch_ops_out << "epoch,scope,name,calls,total_us,avg_us\n";
}

void RunArtifactsWriter::appendProfileEpochSummary(int epoch,
                                                   const runtime::GradSyncModeInfo& mode_info,
                                                   const EpochProfile& profile,
                                                   long long profiled_total_us) {
    if (!enabled_) {
        return;
    }
    std::ofstream out(paths_.profile_epoch_summary_csv_path, std::ios::app);
    out << epoch << ","
        << mode_info.label << ","
        << mode_info.semantics << ","
        << mode_info.correctness_only << ","
        << profile.fwd_bwd_time_s << ","
        << profile.sync_total_time_s << ","
        << profile.sync_wait_time_s << ","
        << profile.sync_effective_overlap << ","
        << profile.opt_time_s << ","
        << profiled_total_us << "\n";
}

void RunArtifactsWriter::appendProfileEpochOps(int epoch, const std::vector<OpTimingStat>& stats) {
    if (!enabled_) {
        return;
    }
    std::ofstream out(paths_.profile_epoch_ops_csv_path, std::ios::app);
    for (const auto& stat : stats) {
        const double avg_us =
            (stat.calls > 0)
                ? static_cast<double>(stat.total_us) / static_cast<double>(stat.calls)
                : 0.0;
        out << epoch << ","
            << classifyProfileScope(stat.name) << ","
            << stat.name << ","
            << stat.calls << ","
            << stat.total_us << ","
            << avg_us << "\n";
    }
}

void RunArtifactsWriter::appendMetricsRow(int epoch,
                                          const Metrics& train_metrics,
                                          const Metrics& test_metrics,
                                          const DerivedTrainingStats& derived_stats,
                                          const runtime::GradSyncModeInfo& mode_info,
                                          int world_size,
                                          int batch_size) {
    if (!enabled_ || !metrics_file_.is_open()) {
        return;
    }
    metrics_file_ << epoch << ","
                  << train_metrics.avg_loss << "," << train_metrics.accuracy << ","
                  << test_metrics.avg_loss << "," << test_metrics.accuracy << ","
                  << train_metrics.sample_count << ","
                  << train_metrics.profile.epoch_time_s << ","
                  << train_metrics.profile.data_time_s << ","
                  << train_metrics.profile.fwd_bwd_time_s << ","
                  << train_metrics.profile.sync_total_time_s << ","
                  << train_metrics.profile.sync_wait_time_s << ","
                  << train_metrics.profile.sync_pack_time_s << ","
                  << train_metrics.profile.sync_launch_time_s << ","
                  << train_metrics.profile.sync_unpack_time_s << ","
                  << train_metrics.profile.sync_bucket_count << ","
                  << train_metrics.profile.sync_bucket_bytes << ","
                  << train_metrics.profile.sync_launched_bucket_count << ","
                  << train_metrics.profile.sync_effective_overlap << ","
                  << train_metrics.profile.opt_time_s << ","
                  << derived_stats.avg_step_time_ms << ","
                  << derived_stats.max_step_time_ms << ","
                  << derived_stats.samples_per_s << ","
                  << derived_stats.allreduce_wait_ratio << ","
                  << mode_info.label << ","
                  << mode_info.semantics << ","
                  << mode_info.correctness_only << ","
                  << world_size << ","
                  << batch_size << "\n";
}

void RunArtifactsWriter::writeRunSummary(const RunProfileSummary& summary) {
    if (!enabled_) {
        return;
    }
    std::ofstream out(paths_.profile_run_summary_csv_path, std::ios::trunc);
    out << "run_id,grad_sync_mode_label,grad_sync_semantics,grad_sync_correctness_only,"
           "total_epochs,fwd_bwd_s,sync_total_s,sync_wait_s,sync_effective_overlap,opt_s,profiled_total_us\n";
    out << paths_.run_id << ","
        << summary.grad_sync_mode.label << ","
        << summary.grad_sync_mode.semantics << ","
        << summary.grad_sync_mode.correctness_only << ","
        << summary.total_epochs << ","
        << summary.fwd_bwd_s << ","
        << summary.sync_total_s << ","
        << summary.sync_wait_s << ","
        << summary.sync_effective_overlap << ","
        << summary.opt_s << ","
        << summary.profiled_total_us << "\n";
}

void RunArtifactsWriter::writeRunOps(const std::vector<RunOpStat>& run_stats) {
    if (!enabled_) {
        return;
    }
    std::ofstream out(paths_.profile_run_ops_csv_path, std::ios::trunc);
    out << "scope,name,calls,total_us,avg_us\n";
    for (const auto& stat : run_stats) {
        const double avg_us =
            (stat.calls > 0)
                ? static_cast<double>(stat.total_us) / static_cast<double>(stat.calls)
                : 0.0;
        out << stat.scope << ","
            << stat.name << ","
            << stat.calls << ","
            << stat.total_us << ","
            << avg_us << "\n";
    }
}

}
