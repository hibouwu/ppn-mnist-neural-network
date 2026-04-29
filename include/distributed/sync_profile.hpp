#pragma once

#include <cstdint>
#include <vector>

enum class SyncTraceChannel {
    Simulated,
    RealComm
};

enum class SyncLaunchReason {
    None,
    RealEarlyLaunch,
    StructuralZeroExpectedLaunch,
    TailFlushLaunch
};

enum class SyncRequestLifecycle {
    NotLaunched,
    SimulatedOnly,
    InFlight,
    Completed
};

struct SyncLaunchEvent {
    std::uint64_t bucket_idx = 0;
    std::uint64_t expected_count = 0;
    std::uint64_t ready_count = 0;
    std::uint64_t outstanding_requests = 0;
    double seconds_since_begin_step = 0.0;
    double seconds_since_backward_complete = -1.0;
    SyncTraceChannel channel = SyncTraceChannel::RealComm;
    SyncLaunchReason reason = SyncLaunchReason::None;
    SyncRequestLifecycle request_lifecycle = SyncRequestLifecycle::NotLaunched;
};

struct SyncStepProfile {
    double encode_time_s = 0.0;
    double comm_time_s = 0.0;
    double decode_time_s = 0.0;
    double pack_time_s = 0.0;
    double launch_time_s = 0.0;
    double wait_time_s = 0.0;
    double unpack_time_s = 0.0;
    double batch_reduce_time_s = 0.0;
    std::uint64_t bucket_count = 0;
    std::uint64_t bucket_bytes = 0;
    std::uint64_t launched_bucket_count = 0;
    std::uint64_t simulated_launch_bucket_count = 0;
    std::uint64_t outstanding_requests_before_finalize = 0;
    bool effective_overlap = false;
    std::vector<SyncLaunchEvent> launch_events;
};
