#pragma once

#include "distributed/bucket_layout.hpp"
#include "distributed/distributed.hpp"
#include "distributed/param_registry.hpp"
#include "distributed/sync_profile.hpp"

#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct CommCompressionConfig {
    bool enabled = false;
    std::string mode = "none";
    double topk_ratio = 0.1;
    bool error_feedback = false;
    int interval = 1;
};

class GradientSyncRuntime {
public:
    virtual ~GradientSyncRuntime() = default;

    virtual void beginStep(std::uint64_t local_batch) = 0;
    virtual void planStep(const std::vector<Node::Ptr>& reachable_leaf_params) = 0;
    virtual void onParameterGradReady(Node& param) = 0;
    virtual void onBackwardComplete() = 0;
    virtual std::uint64_t finalizeAndGetGlobalBatch() = 0;
    virtual SyncStepProfile lastStepProfile() const = 0;
};

class StepBoundaryBucketedSync {
public:
    StepBoundaryBucketedSync(const DistributedContext& dist,
                            const std::vector<Node::Ptr>& params,
                            std::size_t bucket_size_bytes,
                            const CommCompressionConfig& grad_compression_cfg = CommCompressionConfig{});

    std::uint64_t sync(std::uint64_t local_batch);
    SyncStepProfile lastStepProfile() const { return last_profile_; }

private:
    void encodeBeforeComm(std::size_t bucket_idx, std::vector<Scalar>& buffer) const;
    void decodeAfterComm(std::vector<Scalar>& buffer) const;

    const DistributedContext& dist_;
    CommCompressionConfig grad_compression_cfg_;
    ParamRegistry registry_;
    BucketLayout bucket_layout_;
    SyncStepProfile last_profile_;
    std::uint64_t step_index_ = 0;
};

class BucketedOverlapRuntime : public GradientSyncRuntime {
public:
    BucketedOverlapRuntime(const DistributedContext& dist,
                           const std::vector<Node::Ptr>& params,
                           std::size_t bucket_size_bytes,
                           const CommCompressionConfig& grad_compression_cfg = CommCompressionConfig{});

    void beginStep(std::uint64_t local_batch) override;
    void planStep(const std::vector<Node::Ptr>& reachable_leaf_params) override;
    void onParameterGradReady(Node& param) override;
    void onBackwardComplete() override;
    std::uint64_t finalizeAndGetGlobalBatch() override;
    SyncStepProfile lastStepProfile() const override { return last_profile_; }

private:
    void encodeBeforeComm(std::size_t bucket_idx, std::vector<Scalar>& buffer) const;
    void decodeAfterComm(std::vector<Scalar>& buffer) const;

    void validateLayoutAgreement() const;
    void validateStableLayout() const;
    void recordLaunchEvent(std::size_t bucket_idx,
                           SyncTraceChannel channel,
                           SyncLaunchReason reason,
                           SyncRequestLifecycle lifecycle);
    bool bucketPackReady(std::size_t bucket_idx) const;
    bool bucketCanRealLaunch(std::size_t bucket_idx) const;
    bool bucketCanSimulateLaunch(std::size_t bucket_idx) const;
    void simulateLaunchBucket(std::size_t bucket_idx, SyncLaunchReason reason);
    void launchBucket(std::size_t bucket_idx, SyncLaunchReason reason);
    void drainSimulatedLaunchablePrefix();
    void drainRealLaunchablePrefix(bool tail_flush_only);
    void resetStepState();
    std::uint64_t plannedBucketBytes() const;

    struct BucketStepState {
        std::size_t expected_count = 0;
        std::size_t ready_count = 0;
        bool touched = false;
        bool launched = false;
        bool simulated_launched = false;
        bool completed = false;
        SyncLaunchReason launch_reason = SyncLaunchReason::None;
        SyncRequestLifecycle request_lifecycle = SyncRequestLifecycle::NotLaunched;
        DistributedRequest request;
    };

    struct StepContext {
        std::unordered_set<const Node*> touched_params;
        std::unordered_set<const Node*> ready_params;
        std::vector<BucketStepState> buckets;
        std::uint64_t local_batch = 0;
        std::uint64_t global_batch_size = 0;
        std::size_t real_launch_cursor = 0;
        std::size_t simulated_launch_cursor = 0;
        bool global_batch_reduced = false;
        bool backward_complete = false;
        bool finalized = false;
        bool step_active = false;
        bool saw_ready_event = false;
        bool planning_completed = false;
        std::uint64_t step_index = 0;
        std::chrono::steady_clock::time_point step_begin_time {};
        std::chrono::steady_clock::time_point backward_complete_time {};
        bool backward_complete_time_recorded = false;
    };

    const DistributedContext& dist_;
    CommCompressionConfig grad_compression_cfg_;
    ParamRegistry registry_;
    BucketLayout bucket_layout_;
    const std::string layout_descriptor_;
    std::uint64_t step_index_ = 0;
    StepContext step_;
    SyncStepProfile last_profile_;
};
