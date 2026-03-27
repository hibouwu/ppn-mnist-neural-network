#include "distributed/gradient_sync_runtime.hpp"
#include "synchronizable_param.hpp"

#include <chrono>
#include <stdexcept>

namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

std::unordered_set<const Node*> allRegisteredParams(const ParamRegistry& registry) {
    std::unordered_set<const Node*> out;
    for (const auto& param : registry.params()) {
        out.insert(param.get());
    }
    return out;
}

}

StepBoundaryBucketedSync::StepBoundaryBucketedSync(const DistributedContext& dist,
                                                   const std::vector<Node::Ptr>& params,
                                                   std::size_t bucket_size_bytes)
    : dist_(dist),
      registry_(params),
      bucket_layout_(registry_, bucket_size_bytes) {}

std::uint64_t StepBoundaryBucketedSync::sync(std::uint64_t local_batch) {
    last_profile_ = {};
    last_profile_.bucket_count = static_cast<std::uint64_t>(bucket_layout_.bucketCount());
    for (std::size_t bucket_idx = 0; bucket_idx < bucket_layout_.bucketCount(); ++bucket_idx) {
        last_profile_.bucket_bytes += bucket_layout_.bucketBytes(bucket_idx);
    }

    const auto touched_params = allRegisteredParams(registry_);
    auto pack_start = Clock::now();
    for (std::size_t bucket_idx = 0; bucket_idx < bucket_layout_.bucketCount(); ++bucket_idx) {
        bucket_layout_.packBucket(bucket_idx, touched_params);
    }
    auto pack_end = Clock::now();
    last_profile_.pack_time_s = elapsedSeconds(pack_start, pack_end);

    auto launch_start = Clock::now();
    for (std::size_t bucket_idx = 0; bucket_idx < bucket_layout_.bucketCount(); ++bucket_idx) {
        auto& bucket = bucket_layout_.bucket(bucket_idx);
        dist_.allReduceSum(bucket.buffer.data(), bucket.buffer.size());
        last_profile_.launched_bucket_count += 1;
    }
    auto launch_end = Clock::now();
    last_profile_.launch_time_s = elapsedSeconds(launch_start, launch_end);

    auto unpack_start = Clock::now();
    for (std::size_t bucket_idx = 0; bucket_idx < bucket_layout_.bucketCount(); ++bucket_idx) {
        bucket_layout_.unpackBucket(bucket_idx, touched_params);
    }
    auto unpack_end = Clock::now();
    last_profile_.unpack_time_s = elapsedSeconds(unpack_start, unpack_end);

    auto batch_start = Clock::now();
    const auto global_batch = dist_.allReduceSumU64(local_batch);
    auto batch_end = Clock::now();
    last_profile_.batch_reduce_time_s = elapsedSeconds(batch_start, batch_end);
    last_profile_.wait_time_s =
        last_profile_.launch_time_s + last_profile_.batch_reduce_time_s;
    last_profile_.effective_overlap = false;
    return global_batch;
}

BucketedOverlapRuntime::BucketedOverlapRuntime(const DistributedContext& dist,
                                               const std::vector<Node::Ptr>& params,
                                               std::size_t bucket_size_bytes)
    : dist_(dist),
      registry_(params),
      bucket_layout_(registry_, bucket_size_bytes) {
    resetStepState();
}

void BucketedOverlapRuntime::resetStepState() {
    step_ = {};
    step_.buckets.assign(bucket_layout_.bucketCount(), BucketStepState{});
}

std::uint64_t BucketedOverlapRuntime::plannedBucketBytes() const {
    std::uint64_t total = 0;
    for (std::size_t bucket_idx = 0; bucket_idx < step_.buckets.size(); ++bucket_idx) {
        total += bucket_layout_.bucketBytes(bucket_idx);
    }
    return total;
}

void BucketedOverlapRuntime::beginStep(std::uint64_t local_batch) {
    if (step_.step_active && !step_.finalized) {
        throw std::logic_error("BucketedOverlapRuntime::beginStep called before previous step finalized.");
    }
    resetStepState();
    step_.step_active = true;
    step_.local_batch = local_batch;
    last_profile_ = {};
}

void BucketedOverlapRuntime::planStep(const std::vector<Node::Ptr>& reachable_leaf_params) {
    if (!step_.step_active) {
        throw std::logic_error("BucketedOverlapRuntime::planStep called before beginStep.");
    }
    if (step_.finalized || step_.backward_complete) {
        throw std::logic_error("BucketedOverlapRuntime::planStep called after step completion.");
    }
    if (step_.saw_ready_event) {
        throw std::logic_error("BucketedOverlapRuntime::planStep must run before any ready event.");
    }
    if (step_.planning_completed) {
        throw std::logic_error("BucketedOverlapRuntime::planStep called more than once in the same step.");
    }

    step_.planning_completed = true;

    std::unordered_set<const Node*> seen;
    for (const auto& param : reachable_leaf_params) {
        if (!param) {
            throw std::invalid_argument("BucketedOverlapRuntime::planStep received null parameter.");
        }
        if (!isSynchronizableLeafParameter(*param)) {
            throw std::logic_error("BucketedOverlapRuntime::planStep received non-synchronizable leaf parameter.");
        }
        if (!registry_.contains(*param)) {
            throw std::logic_error("BucketedOverlapRuntime::planStep received parameter missing from registry.");
        }
        const bool inserted = seen.insert(param.get()).second;
        if (!inserted) {
            throw std::logic_error("BucketedOverlapRuntime::planStep received duplicate parameter.");
        }
        const auto bucket_idx = bucket_layout_.bucketIndexFor(*param);
        if (!bucket_idx.has_value()) {
            throw std::logic_error("BucketedOverlapRuntime::planStep could not resolve bucket for reachable parameter.");
        }
        step_.buckets[*bucket_idx].expected_count += 1;
    }

    for (const auto& bucket_state : step_.buckets) {
        if (bucket_state.expected_count > 0) {
            last_profile_.bucket_count += 1;
        }
    }
    last_profile_.bucket_bytes = plannedBucketBytes();
}

void BucketedOverlapRuntime::launchBucket(std::size_t bucket_idx) {
    auto& bucket_state = step_.buckets.at(bucket_idx);
    if (bucket_state.launched) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket attempted duplicate launch.");
    }
    if (!step_.planning_completed) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket encountered planning state mismatch.");
    }
    auto pack_start = Clock::now();
    bucket_layout_.packBucket(bucket_idx, step_.touched_params);
    auto pack_end = Clock::now();
    last_profile_.pack_time_s += elapsedSeconds(pack_start, pack_end);

    auto launch_start = Clock::now();
    auto& bucket = bucket_layout_.bucket(bucket_idx);
    bucket_state.request = dist_.iallReduceSum(bucket.buffer.data(), bucket.buffer.size());
    bucket_state.launched = true;
    bucket_state.completed = dist_.isNullRequest(bucket_state.request);
    last_profile_.launched_bucket_count += 1;
    auto launch_end = Clock::now();
    last_profile_.launch_time_s += elapsedSeconds(launch_start, launch_end);
}

void BucketedOverlapRuntime::onParameterGradReady(Node& param) {
    if (!step_.step_active) {
        throw std::logic_error("BucketedOverlapRuntime::onParameterGradReady called before beginStep.");
    }
    if (step_.finalized || step_.backward_complete) {
        throw std::logic_error("BucketedOverlapRuntime::onParameterGradReady called after backward completion.");
    }
    if (!isSynchronizableLeafParameter(param) || !registry_.contains(param)) {
        throw std::logic_error(
            "BucketedOverlapRuntime::onParameterGradReady received non-registered synchronizable parameter.");
    }

    step_.saw_ready_event = true;
    if (!step_.planning_completed) {
        throw std::logic_error(
            "BucketedOverlapRuntime::onParameterGradReady requires planStep before any ready event.");
    }
    const bool inserted = step_.ready_params.insert(&param).second;
    if (!inserted) {
        throw std::logic_error("BucketedOverlapRuntime::onParameterGradReady received duplicate ready event.");
    }
    step_.touched_params.insert(&param);

    const auto bucket_idx = bucket_layout_.bucketIndexFor(param);
    if (!bucket_idx.has_value()) {
        throw std::logic_error("BucketedOverlapRuntime::onParameterGradReady could not resolve bucket.");
    }
    auto& bucket_state = step_.buckets[*bucket_idx];
    bucket_state.touched = true;
    bucket_state.ready_count += 1;

    if (bucket_state.expected_count == 0) {
        throw std::logic_error(
            "BucketedOverlapRuntime::onParameterGradReady received parameter outside planned reachable set.");
    }
    if (bucket_state.ready_count > bucket_state.expected_count) {
        throw std::logic_error("BucketedOverlapRuntime::onParameterGradReady exceeded bucket expected count.");
    }
}

void BucketedOverlapRuntime::onBackwardComplete() {
    if (!step_.step_active) {
        throw std::logic_error("BucketedOverlapRuntime::onBackwardComplete called before beginStep.");
    }
    if (step_.finalized) {
        throw std::logic_error("BucketedOverlapRuntime::onBackwardComplete called after finalize.");
    }
    if (step_.backward_complete) {
        throw std::logic_error("BucketedOverlapRuntime::onBackwardComplete called more than once.");
    }
    step_.backward_complete = true;

    if (!step_.planning_completed) {
        throw std::logic_error(
            "BucketedOverlapRuntime::onBackwardComplete requires planStep before backward completion.");
    }
    last_profile_.bucket_count = static_cast<std::uint64_t>(step_.buckets.size());
    for (std::size_t bucket_idx = 0; bucket_idx < step_.buckets.size(); ++bucket_idx) {
        auto& bucket_state = step_.buckets[bucket_idx];
        if (bucket_state.expected_count > 0 &&
            (!bucket_state.touched || bucket_state.ready_count != bucket_state.expected_count)) {
            throw std::logic_error(
                "BucketedOverlapRuntime::onBackwardComplete detected reachable bucket with missing ready parameters.");
        }
        if (!bucket_state.launched) {
            launchBucket(bucket_idx);
        }
    }
    last_profile_.bucket_bytes = plannedBucketBytes();
}

std::uint64_t BucketedOverlapRuntime::finalizeAndGetGlobalBatch() {
    if (!step_.step_active) {
        throw std::logic_error("BucketedOverlapRuntime::finalizeAndGetGlobalBatch called before beginStep.");
    }
    if (step_.finalized) {
        throw std::logic_error("BucketedOverlapRuntime::finalizeAndGetGlobalBatch called more than once.");
    }
    if (!step_.backward_complete) {
        throw std::logic_error("BucketedOverlapRuntime::finalizeAndGetGlobalBatch called before backward completion.");
    }

    if (!step_.global_batch_reduced) {
        auto batch_start = Clock::now();
        step_.global_batch_size = dist_.allReduceSumU64(step_.local_batch);
        auto batch_end = Clock::now();
        last_profile_.batch_reduce_time_s += elapsedSeconds(batch_start, batch_end);
        step_.global_batch_reduced = true;
    }

    auto wait_start = Clock::now();
    for (auto& bucket_state : step_.buckets) {
        if (!bucket_state.launched || bucket_state.completed) {
            continue;
        }
        dist_.wait(bucket_state.request);
        bucket_state.completed = true;
    }
    auto wait_end = Clock::now();
    last_profile_.wait_time_s += elapsedSeconds(wait_start, wait_end);

    auto unpack_start = Clock::now();
    const auto registered_params = allRegisteredParams(registry_);
    for (std::size_t bucket_idx = 0; bucket_idx < step_.buckets.size(); ++bucket_idx) {
        if (!step_.buckets[bucket_idx].launched) {
            continue;
        }
        bucket_layout_.unpackBucket(bucket_idx, registered_params);
    }
    auto unpack_end = Clock::now();
    last_profile_.unpack_time_s += elapsedSeconds(unpack_start, unpack_end);

    step_.finalized = true;
    return step_.global_batch_size;
}
