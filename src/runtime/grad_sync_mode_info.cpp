#include "runtime/grad_sync_mode_info.hpp"

#include <stdexcept>

namespace runtime {

GradSyncModeInfo parseGradSyncModeInfo(const std::string& mode) {
    if (mode == "per_param") {
        return {GradSyncMode::PerParamBlocking, "per_param", "per_param_blocking", 0};
    }
    if (mode == "bucketed") {
        return {GradSyncMode::BucketedBlocking, "bucketed_baseline", "baseline_full_bucket_sequence", 0};
    }
    if (mode == "overlap_bucketed") {
        return {GradSyncMode::BucketedOverlap,
                "overlap_bucketed_correctness_only",
                "correctness_only_full_bucket_sequence",
                1};
    }
    throw std::invalid_argument(
        "unsupported --grad_sync_mode '" + mode +
        "'. Expected 'per_param', 'bucketed', or 'overlap_bucketed'.");
}

}
