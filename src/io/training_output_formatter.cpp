#include "io/training_output_formatter.hpp"

#include <iomanip>
#include <sstream>

namespace io {

std::string formatTrainingStartupSummary(const TrainingStartupSummaryData& data) {
    std::ostringstream out;
    out << "Starting training with config:" << "\n"
        << "  Dataset: " << data.dataset << "\n"
        << "  Data Dir: " << data.data_dir << "\n"
        << "  Input Shape: " << data.dataset_info.input_channels << "x"
        << data.dataset_info.input_height << "x" << data.dataset_info.input_width << "\n"
        << "  Classes: " << data.dataset_info.num_classes << "\n"
        << "  Epochs: " << data.epochs << "\n"
        << "  Batch Size: " << data.batch_size << "\n"
        << "  Learning Rate: " << data.learning_rate << "\n"
        << "  Hidden (sizes): " << data.hidden_sizes << "\n"
        << "  Seed: " << data.seed << "\n"
        << "  Activation: " << data.activation << "\n"
        << "  Init: " << data.init << "\n"
        << "  Model: " << data.model << "\n"
        << "  Optimizer: " << data.optimizer << "\n"
        << "  Grad Sync Mode: " << data.grad_sync_mode.label << "\n"
        << "  Grad Sync Semantics: " << data.grad_sync_mode.semantics << "\n"
        << "  Grad Sync Correctness Only: " << data.grad_sync_mode.correctness_only << "\n"
        << "  Bucket Size (bytes): " << data.bucket_size_bytes << "\n"
        << "  Momentum: " << data.momentum << "\n"
        << "  Nesterov: " << (data.nesterov ? 1 : 0) << "\n"
        << "  Weight Decay: " << data.weight_decay << "\n"
        << "  Beta1: " << data.beta1 << "\n"
        << "  Beta2: " << data.beta2 << "\n"
        << "  Eps: " << data.eps << "\n";
    if (data.world_size > 1) {
        out << "  MPI World Size: " << data.world_size << "\n";
    }
    out << "\n";
    return out.str();
}

std::string formatGradSyncWarning(const runtime::GradSyncModeInfo& mode_info) {
    if (mode_info.parsed_mode == runtime::GradSyncMode::BucketedBlocking) {
        return "[INFO] bucketed grad sync currently uses the conservative distributed correctness baseline: "
               "fixed full-bucket launch order, zero-fill packing, and full-registry unpack.";
    }
    if (mode_info.parsed_mode == runtime::GradSyncMode::BucketedOverlap) {
        return "[WARNING] overlap_bucketed is correctness-only at this stage. "
               "Do not interpret its timing or sync_effective_overlap as overlap evidence until new multi-rank proof exists.";
    }
    return "";
}

std::string formatEpochSummary(int epoch,
                               int total_epochs,
                               const Metrics& train_metrics,
                               const Metrics& test_metrics,
                               const DerivedTrainingStats& derived_stats) {
    std::ostringstream out;
    out << "Epoch " << epoch << "/" << total_epochs
        << ": [Train] loss = " << std::fixed << std::setprecision(4) << train_metrics.avg_loss
        << ", acc = " << std::fixed << std::setprecision(2) << (train_metrics.accuracy * 100.0) << "%"
        << " | [Test] loss = " << std::fixed << std::setprecision(4) << test_metrics.avg_loss
        << ", acc = " << std::fixed << std::setprecision(2) << (test_metrics.accuracy * 100.0) << "%"
        << " | epoch = " << std::fixed << std::setprecision(3) << train_metrics.profile.epoch_time_s << "s"
        << ", data = " << train_metrics.profile.data_time_s << "s"
        << ", fwd_bwd = " << train_metrics.profile.fwd_bwd_time_s << "s"
        << ", sync_total = " << train_metrics.profile.sync_total_time_s << "s"
        << ", sync_wait = " << train_metrics.profile.sync_wait_time_s << "s"
        << ", sync_pack = " << train_metrics.profile.sync_pack_time_s << "s"
        << ", sync_launch = " << train_metrics.profile.sync_launch_time_s << "s"
        << ", sync_unpack = " << train_metrics.profile.sync_unpack_time_s << "s"
        << ", effective_overlap = " << train_metrics.profile.sync_effective_overlap
        << ", opt = " << train_metrics.profile.opt_time_s << "s"
        << ", avg_step = " << derived_stats.avg_step_time_ms << "ms"
        << ", samples/s = " << derived_stats.samples_per_s;
    return out.str();
}

std::string formatProfileRunTotalSummary(const RunProfileSummary& summary, double uncovered_us) {
    std::ostringstream out;
    out << "[PROFILE_RUN_TOTAL] "
        << "grad_sync_mode_label=" << summary.grad_sync_mode.label
        << ", grad_sync_semantics=" << summary.grad_sync_mode.semantics
        << ", grad_sync_correctness_only=" << summary.grad_sync_mode.correctness_only
        << ", fwd_bwd_s=" << std::fixed << std::setprecision(3) << summary.fwd_bwd_s
        << ", sync_total_s=" << summary.sync_total_s
        << ", sync_wait_s=" << summary.sync_wait_s
        << ", sync_effective_overlap=" << summary.sync_effective_overlap
        << ", opt_s=" << summary.opt_s
        << ", profiled_total_us=" << summary.profiled_total_us
        << ", uncovered_us=" << std::fixed << std::setprecision(0) << uncovered_us;
    return out.str();
}

std::string formatProfileRunOpRow(std::size_t rank,
                                  const RunOpStat& stat,
                                  long long profiled_total_us,
                                  double run_fwd_bwd_us) {
    const double avg_us =
        (stat.calls > 0)
            ? static_cast<double>(stat.total_us) / static_cast<double>(stat.calls)
            : 0.0;
    const double share_of_profiled_total =
        (profiled_total_us > 0)
            ? static_cast<double>(stat.total_us) / static_cast<double>(profiled_total_us)
            : 0.0;
    const double share_of_fwd_bwd =
        (run_fwd_bwd_us > 0.0)
            ? static_cast<double>(stat.total_us) / run_fwd_bwd_us
            : 0.0;

    std::ostringstream out;
    out << "  " << rank
        << ". [" << stat.scope << "] " << stat.name
        << ": calls=" << stat.calls
        << ", total_us=" << stat.total_us
        << ", avg_us=" << std::fixed << std::setprecision(2) << avg_us
        << ", share_of_profiled_total=" << std::setprecision(4) << share_of_profiled_total
        << ", share_of_fwd_bwd=" << share_of_fwd_bwd;
    return out.str();
}

}
