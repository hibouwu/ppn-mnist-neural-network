#include "io/derived_training_stats.hpp"

namespace io {

DerivedTrainingStats deriveTrainingStats(const Metrics& train_metrics) {
    DerivedTrainingStats stats;
    if (train_metrics.profile.step_count > 0) {
        stats.avg_step_time_ms =
            (train_metrics.profile.step_time_s_sum /
             static_cast<double>(train_metrics.profile.step_count)) * 1000.0;
    }
    stats.max_step_time_ms = train_metrics.profile.max_step_time_s * 1000.0;
    if (train_metrics.profile.epoch_time_s > 0.0) {
        stats.samples_per_s =
            static_cast<double>(train_metrics.sample_count) / train_metrics.profile.epoch_time_s;
        stats.allreduce_wait_ratio =
            train_metrics.profile.sync_wait_time_s / train_metrics.profile.epoch_time_s;
    }
    return stats;
}

}
