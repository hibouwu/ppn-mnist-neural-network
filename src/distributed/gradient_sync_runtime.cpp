#include "distributed/gradient_sync_runtime.hpp"
#include "synchronizable_param.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <cstring>
#include <sstream>
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

Scalar quantizeToFp16Like(Scalar x) {
    // Fast fp16-like quantization on float32:
    // - clamp exponent to fp16 normal range
    // - round mantissa to 10 bits using bit operations
    // This avoids expensive frexp/ldexp paths in hot loops.
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));

    const std::uint32_t sign = bits & 0x80000000u;
    const std::uint32_t exp = bits & 0x7f800000u;

    // Keep NaN/Inf as-is.
    if (exp == 0x7f800000u) {
        return x;
    }

    // Zero/subnormal -> signed zero.
    if (exp == 0u || exp < 0x33800000u) { // unbiased exp < -24
        const std::uint32_t z = sign;
        Scalar out = 0.0f;
        std::memcpy(&out, &z, sizeof(out));
        return out;
    }

    // Overflow wrt fp16 range -> signed inf.
    if (exp > 0x47000000u) { // unbiased exp > 15
        const std::uint32_t inf_bits = sign | 0x7f800000u;
        Scalar out = 0.0f;
        std::memcpy(&out, &inf_bits, sizeof(out));
        return out;
    }

    // Round to 10 mantissa bits (drop lower 13 bits).
    std::uint32_t rounded = bits + 0x00001000u;
    if ((rounded & 0x7f800000u) > 0x47000000u) {
        const std::uint32_t inf_bits = sign | 0x7f800000u;
        Scalar out = 0.0f;
        std::memcpy(&out, &inf_bits, sizeof(out));
        return out;
    }
    rounded &= 0xffffe000u;

    Scalar out = 0.0f;
    std::memcpy(&out, &rounded, sizeof(out));
    return out;
}

void applyFp16LikeQuantInPlace(std::vector<Scalar>& buffer) {
    for (Scalar& v : buffer) {
        v = quantizeToFp16Like(v);
    }
}

}

StepBoundaryBucketedSync::StepBoundaryBucketedSync(const DistributedContext& dist,
                                                   const std::vector<Node::Ptr>& params,
                                                   std::size_t bucket_size_bytes,
                                                   const CommCompressionConfig& grad_compression_cfg)
    : dist_(dist),
      grad_compression_cfg_(grad_compression_cfg),
      registry_(params),
      bucket_layout_(registry_, bucket_size_bytes) {}

void StepBoundaryBucketedSync::encodeBeforeComm(std::size_t bucket_idx, std::vector<Scalar>& buffer) const {
    if (!grad_compression_cfg_.enabled) {
        return;
    }
    (void)bucket_idx;
    const int interval = std::max(1, grad_compression_cfg_.interval);
    if ((step_index_ % static_cast<std::uint64_t>(interval)) != 0) {
        return;
    }
    if (grad_compression_cfg_.mode == "fp16") {
        applyFp16LikeQuantInPlace(buffer);
    }
}

void StepBoundaryBucketedSync::decodeAfterComm(std::vector<Scalar>& buffer) const {
    // fp16-like mode keeps data in float buffer after quantization + allreduce.
    (void)buffer;
}

std::uint64_t StepBoundaryBucketedSync::sync(std::uint64_t local_batch) {
    ++step_index_;
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
        auto encode_start = Clock::now();
        encodeBeforeComm(bucket_idx, bucket.buffer);
        auto encode_end = Clock::now();
        last_profile_.encode_time_s += elapsedSeconds(encode_start, encode_end);

        auto comm_start = Clock::now();
        dist_.allReduceSum(bucket.buffer.data(), bucket.buffer.size());
        auto comm_end = Clock::now();
        last_profile_.comm_time_s += elapsedSeconds(comm_start, comm_end);

        auto decode_start = Clock::now();
        decodeAfterComm(bucket.buffer);
        auto decode_end = Clock::now();
        last_profile_.decode_time_s += elapsedSeconds(decode_start, decode_end);
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
                                               std::size_t bucket_size_bytes,
                                               const CommCompressionConfig& grad_compression_cfg)
    : dist_(dist),
      grad_compression_cfg_(grad_compression_cfg),
      registry_(params),
      bucket_layout_(registry_, bucket_size_bytes),
      layout_descriptor_(bucket_layout_.serializedDescriptor()) {
    validateLayoutAgreement();
    resetStepState();
}

void BucketedOverlapRuntime::encodeBeforeComm(std::size_t bucket_idx, std::vector<Scalar>& buffer) const {
    if (!grad_compression_cfg_.enabled) {
        return;
    }
    (void)bucket_idx;
    const int interval = std::max(1, grad_compression_cfg_.interval);
    if ((step_.step_index % static_cast<std::uint64_t>(interval)) != 0) {
        return;
    }
    if (grad_compression_cfg_.mode == "fp16") {
        applyFp16LikeQuantInPlace(buffer);
    }
}

void BucketedOverlapRuntime::decodeAfterComm(std::vector<Scalar>& buffer) const {
    // fp16-like mode keeps data in float buffer after quantization + allreduce.
    (void)buffer;
}

void BucketedOverlapRuntime::validateLayoutAgreement() const {
    const auto gathered = dist_.allGatherStrings(layout_descriptor_);
    if (gathered.empty()) {
        throw std::logic_error("BucketedOverlapRuntime: empty layout agreement result.");
    }
    for (std::size_t rank = 1; rank < gathered.size(); ++rank) {
        if (gathered[rank] != gathered[0]) {
            std::ostringstream out;
            out << "BucketedOverlapRuntime: rank-wide layout agreement failed between rank 0 and rank "
                << rank << ".";
            throw std::logic_error(out.str());
        }
    }
}

void BucketedOverlapRuntime::validateStableLayout() const {
    if (bucket_layout_.serializedDescriptor() != layout_descriptor_) {
        throw std::logic_error(
            "BucketedOverlapRuntime: layout drift detected; rebuild runtime and rerun agreement check.");
    }
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

void BucketedOverlapRuntime::recordLaunchEvent(std::size_t bucket_idx,
                                               SyncTraceChannel channel,
                                               SyncLaunchReason reason,
                                               SyncRequestLifecycle lifecycle) {
    const auto outstanding_requests = [this]() {
        std::uint64_t outstanding = 0;
        for (const auto& bucket : step_.buckets) {
            if (bucket.launched && !bucket.completed) {
                outstanding += 1;
            }
        }
        return outstanding;
    };
    SyncLaunchEvent event;
    const auto& bucket_state = step_.buckets.at(bucket_idx);
    event.bucket_idx = static_cast<std::uint64_t>(bucket_idx);
    event.expected_count = static_cast<std::uint64_t>(bucket_state.expected_count);
    event.ready_count = static_cast<std::uint64_t>(bucket_state.ready_count);
    event.outstanding_requests = outstanding_requests();
    event.seconds_since_begin_step =
        elapsedSeconds(step_.step_begin_time, Clock::now());
    event.channel = channel;
    event.reason = reason;
    event.request_lifecycle = lifecycle;
    if (step_.backward_complete_time_recorded) {
        event.seconds_since_backward_complete =
            elapsedSeconds(step_.backward_complete_time, Clock::now());
    }
    last_profile_.launch_events.push_back(event);
}

bool BucketedOverlapRuntime::bucketPackReady(std::size_t bucket_idx) const {
    const auto& bucket = step_.buckets.at(bucket_idx);
    if (bucket.expected_count == 0) {
        return true;
    }
    return bucket.ready_count == bucket.expected_count;
}

bool BucketedOverlapRuntime::bucketCanRealLaunch(std::size_t bucket_idx) const {
    if (!step_.planning_completed || bucket_idx != step_.real_launch_cursor) {
        return false;
    }
    const auto& bucket = step_.buckets.at(bucket_idx);
    if (bucket.launched) {
        return false;
    }
    return bucketPackReady(bucket_idx);
}

bool BucketedOverlapRuntime::bucketCanSimulateLaunch(std::size_t bucket_idx) const {
    if (!step_.planning_completed || bucket_idx != step_.simulated_launch_cursor) {
        return false;
    }
    const auto& bucket = step_.buckets.at(bucket_idx);
    if (bucket.simulated_launched) {
        return false;
    }
    return bucketPackReady(bucket_idx);
}

void BucketedOverlapRuntime::simulateLaunchBucket(std::size_t bucket_idx, SyncLaunchReason reason) {
    auto& bucket = step_.buckets.at(bucket_idx);
    if (bucket.simulated_launched) {
        throw std::logic_error("BucketedOverlapRuntime: duplicate simulated launch.");
    }
    bucket.simulated_launched = true;
    last_profile_.simulated_launch_bucket_count += 1;
    recordLaunchEvent(bucket_idx, SyncTraceChannel::Simulated, reason, SyncRequestLifecycle::SimulatedOnly);
}

void BucketedOverlapRuntime::launchBucket(std::size_t bucket_idx, SyncLaunchReason reason) {
    if (!step_.planning_completed) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket encountered planning state mismatch.");
    }
    if (bucket_idx != step_.real_launch_cursor) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket attempted out-of-order launch.");
    }

    auto& bucket_state = step_.buckets.at(bucket_idx);
    if (bucket_state.launched) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket attempted duplicate launch.");
    }
    if (!bucketPackReady(bucket_idx)) {
        throw std::logic_error("BucketedOverlapRuntime::launchBucket attempted launch before pack-safe readiness.");
    }

    auto pack_start = Clock::now();
    bucket_layout_.packBucket(bucket_idx, step_.touched_params);
    auto pack_end = Clock::now();
    last_profile_.pack_time_s += elapsedSeconds(pack_start, pack_end);

    auto launch_start = Clock::now();
    auto& bucket = bucket_layout_.bucket(bucket_idx);
    auto encode_start = Clock::now();
    encodeBeforeComm(bucket_idx, bucket.buffer);
    auto encode_end = Clock::now();
    last_profile_.encode_time_s += elapsedSeconds(encode_start, encode_end);

    auto comm_start = Clock::now();
    bucket_state.request = dist_.iallReduceSum(bucket.buffer.data(), bucket.buffer.size());
    auto comm_end = Clock::now();
    last_profile_.comm_time_s += elapsedSeconds(comm_start, comm_end);
    bucket_state.launched = true;
    bucket_state.completed = dist_.isNullRequest(bucket_state.request);
    bucket_state.launch_reason = reason;
    bucket_state.request_lifecycle =
        bucket_state.completed ? SyncRequestLifecycle::Completed : SyncRequestLifecycle::InFlight;
    last_profile_.launched_bucket_count += 1;
    auto launch_end = Clock::now();
    last_profile_.launch_time_s += elapsedSeconds(launch_start, launch_end);
    recordLaunchEvent(bucket_idx,
                      SyncTraceChannel::RealComm,
                      reason,
                      bucket_state.request_lifecycle);
}

void BucketedOverlapRuntime::drainSimulatedLaunchablePrefix() {
    while (step_.simulated_launch_cursor < step_.buckets.size() &&
           bucketCanSimulateLaunch(step_.simulated_launch_cursor)) {
        const auto& bucket = step_.buckets[step_.simulated_launch_cursor];
        const SyncLaunchReason reason =
            (bucket.expected_count == 0)
                ? SyncLaunchReason::StructuralZeroExpectedLaunch
                : (step_.backward_complete
                       ? SyncLaunchReason::TailFlushLaunch
                       : SyncLaunchReason::RealEarlyLaunch);
        simulateLaunchBucket(step_.simulated_launch_cursor, reason);
        step_.simulated_launch_cursor += 1;
    }
}

void BucketedOverlapRuntime::drainRealLaunchablePrefix(bool tail_flush_only) {
    while (step_.real_launch_cursor < step_.buckets.size() &&
           bucketCanRealLaunch(step_.real_launch_cursor)) {
        const auto& bucket = step_.buckets[step_.real_launch_cursor];
        const SyncLaunchReason reason =
            (bucket.expected_count == 0)
                ? SyncLaunchReason::StructuralZeroExpectedLaunch
                : (step_.backward_complete
                       ? SyncLaunchReason::TailFlushLaunch
                       : SyncLaunchReason::RealEarlyLaunch);
        if (tail_flush_only && reason == SyncLaunchReason::RealEarlyLaunch) {
            break;
        }
        launchBucket(step_.real_launch_cursor, reason);
        step_.real_launch_cursor += 1;
    }
}

void BucketedOverlapRuntime::beginStep(std::uint64_t local_batch) {
    if (step_.step_active && !step_.finalized) {
        throw std::logic_error("BucketedOverlapRuntime::beginStep called before previous step finalized.");
    }
    validateStableLayout();
    resetStepState();
    step_.step_active = true;
    step_.step_index = ++step_index_;
    step_.local_batch = local_batch;
    step_.step_begin_time = Clock::now();
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

    validateStableLayout();
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

    last_profile_.bucket_count = static_cast<std::uint64_t>(step_.buckets.size());
    last_profile_.bucket_bytes = plannedBucketBytes();
    drainSimulatedLaunchablePrefix();
    drainRealLaunchablePrefix(/*tail_flush_only=*/false);
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

    drainSimulatedLaunchablePrefix();
    drainRealLaunchablePrefix(/*tail_flush_only=*/false);
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
    step_.backward_complete_time = Clock::now();
    step_.backward_complete_time_recorded = true;

    if (!step_.planning_completed) {
        throw std::logic_error(
            "BucketedOverlapRuntime::onBackwardComplete requires planStep before backward completion.");
    }

    for (std::size_t bucket_idx = 0; bucket_idx < step_.buckets.size(); ++bucket_idx) {
        const auto& bucket_state = step_.buckets[bucket_idx];
        if (bucket_state.expected_count > 0 && bucket_state.ready_count != bucket_state.expected_count) {
            throw std::logic_error(
                "BucketedOverlapRuntime::onBackwardComplete detected reachable bucket with missing ready parameters.");
        }
    }

    drainSimulatedLaunchablePrefix();
    drainRealLaunchablePrefix(/*tail_flush_only=*/true);
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
    if (step_.real_launch_cursor != step_.buckets.size()) {
        throw std::logic_error(
            "BucketedOverlapRuntime::finalizeAndGetGlobalBatch called before terminal bucket state was reached.");
    }

    for (const auto& bucket : step_.buckets) {
        if (!bucket.launched) {
            throw std::logic_error("BucketedOverlapRuntime::finalizeAndGetGlobalBatch found non-launched bucket.");
        }
    }

    if (!step_.global_batch_reduced) {
        auto batch_start = Clock::now();
        step_.global_batch_size = dist_.allReduceSumU64(step_.local_batch);
        auto batch_end = Clock::now();
        last_profile_.batch_reduce_time_s += elapsedSeconds(batch_start, batch_end);
        step_.global_batch_reduced = true;
    }

    last_profile_.outstanding_requests_before_finalize = 0;
    for (const auto& bucket : step_.buckets) {
        if (bucket.launched && !bucket.completed) {
            last_profile_.outstanding_requests_before_finalize += 1;
        }
    }

    auto wait_start = Clock::now();
    for (auto& bucket_state : step_.buckets) {
        if (bucket_state.completed) {
            continue;
        }
        if (!bucket_state.launched) {
            throw std::logic_error("BucketedOverlapRuntime::finalizeAndGetGlobalBatch found terminal-state mismatch.");
        }
        dist_.wait(bucket_state.request);
        bucket_state.completed = true;
        bucket_state.request_lifecycle = SyncRequestLifecycle::Completed;
    }
    auto wait_end = Clock::now();
    last_profile_.wait_time_s += elapsedSeconds(wait_start, wait_end);

    auto unpack_start = Clock::now();
    const auto registered_params = allRegisteredParams(registry_);
    for (std::size_t bucket_idx = 0; bucket_idx < step_.buckets.size(); ++bucket_idx) {
        auto& bucket = bucket_layout_.bucket(bucket_idx);
        auto decode_start = Clock::now();
        decodeAfterComm(bucket.buffer);
        auto decode_end = Clock::now();
        last_profile_.decode_time_s += elapsedSeconds(decode_start, decode_end);
        bucket_layout_.unpackBucket(bucket_idx, registered_params);
    }
    auto unpack_end = Clock::now();
    last_profile_.unpack_time_s += elapsedSeconds(unpack_start, unpack_end);

    step_.finalized = true;
    return step_.global_batch_size;
}
