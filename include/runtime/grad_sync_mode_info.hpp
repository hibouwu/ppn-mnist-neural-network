#pragma once

#include <cstdint>
#include <string>

namespace runtime {

enum class GradSyncMode {
    PerParamBlocking,
    BucketedBlocking,
    BucketedOverlap
};

struct GradSyncModeInfo {
    GradSyncMode parsed_mode = GradSyncMode::PerParamBlocking;
    std::string label;
    std::string semantics;
    std::uint64_t correctness_only = 0;
};

GradSyncModeInfo parseGradSyncModeInfo(const std::string& mode);

}
