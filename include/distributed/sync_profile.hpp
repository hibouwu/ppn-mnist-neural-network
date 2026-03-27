#pragma once

#include <cstdint>

struct SyncStepProfile {
    double pack_time_s = 0.0;
    double launch_time_s = 0.0;
    double wait_time_s = 0.0;
    double unpack_time_s = 0.0;
    double batch_reduce_time_s = 0.0;
    std::uint64_t bucket_count = 0;
    std::uint64_t bucket_bytes = 0;
    std::uint64_t launched_bucket_count = 0;
    bool effective_overlap = false;
};
