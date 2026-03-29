#include "autograd/engine.hpp"
#include "distributed/bucket_layout.hpp"
#include "distributed/distributed.hpp"
#include "distributed/gradient_sync_runtime.hpp"
#include "distributed/param_registry.hpp"
#include "math_ops.hpp"
#include "node.hpp"
#include "runtime/synchronizable_params.hpp"
#include "synchronizable_param.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::size_t kBucketElemBytes = sizeof(Scalar);

bool almostEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

Node::Ptr makeParam(std::size_t rows, std::size_t cols, double base) {
    Matrix value(rows, cols);
    for (std::size_t i = 0; i < value.data.size(); ++i) {
        value.data[i] = base + static_cast<double>(i);
    }
    auto param = std::make_shared<Node>(value);
    param->setIsParameter(true);
    return param;
}

void seedGrad(const Node::Ptr& param, double scale) {
    Matrix grad(param->value().rows, param->value().cols);
    for (std::size_t i = 0; i < grad.data.size(); ++i) {
        grad.data[i] = scale * static_cast<double>(i + 1);
    }
    param->zeroGrad();
    param->addGrad(grad);
}

void runBackwardWithRuntime(GradientSyncRuntime& runtime,
                            const Node::Ptr& loss) {
    AutogradEngine engine;
    engine.setReachableLeafHook([&runtime](const std::vector<Node::Ptr>& reachable_leaf_params) {
        runtime.planStep(reachable_leaf_params);
    });
    engine.setParameterReadyHook([&runtime](Node& param) {
        runtime.onParameterGradReady(param);
    });
    engine.setBackwardCompleteHook([&runtime]() {
        runtime.onBackwardComplete();
    });
    engine.backward(loss);
}

std::string encodeLaunchSequence(const SyncStepProfile& profile, SyncTraceChannel channel) {
    std::ostringstream out;
    bool first = true;
    for (const auto& event : profile.launch_events) {
        if (event.channel != channel) {
            continue;
        }
        if (!first) {
            out << ",";
        }
        out << event.bucket_idx;
        first = false;
    }
    return out.str();
}

std::string expectedPrefixSequence(std::size_t bucket_count) {
    std::ostringstream out;
    for (std::size_t bucket_idx = 0; bucket_idx < bucket_count; ++bucket_idx) {
        if (bucket_idx > 0) {
            out << ",";
        }
        out << bucket_idx;
    }
    return out.str();
}

std::size_t countLaunchEvents(const SyncStepProfile& profile,
                              SyncTraceChannel channel,
                              SyncLaunchReason reason) {
    std::size_t count = 0;
    for (const auto& event : profile.launch_events) {
        if (event.channel == channel && event.reason == reason) {
            count += 1;
        }
    }
    return count;
}

void assertTraceLifecycleSeparation(const SyncStepProfile& profile) {
    for (const auto& event : profile.launch_events) {
        if (event.channel == SyncTraceChannel::Simulated) {
            assert(event.request_lifecycle == SyncRequestLifecycle::SimulatedOnly);
        } else {
            assert(event.request_lifecycle != SyncRequestLifecycle::SimulatedOnly);
        }
    }
}

void test_bucket_layout_mapping_and_zero_fill() {
    auto p1 = makeParam(1, 2, 1.0);
    auto p2 = makeParam(1, 3, 10.0);
    auto p3 = makeParam(1, 5, 20.0);
    ParamRegistry registry({p1, p2, p3});
    BucketLayout layout(registry, 5 * kBucketElemBytes);

    assert(layout.bucketCount() == 2);
    assert(layout.bucketIndexFor(*p1).has_value());
    assert(layout.bucketIndexFor(*p2).has_value());
    assert(layout.bucketIndexFor(*p3).has_value());
    assert(*layout.bucketIndexFor(*p1) == *layout.bucketIndexFor(*p2));
    assert(*layout.bucketIndexFor(*p3) != *layout.bucketIndexFor(*p1));
    assert(layout.bucketBytes(*layout.bucketIndexFor(*p1)) == 5 * kBucketElemBytes);
    const std::string descriptor = layout.serializedDescriptor();
    assert(descriptor.find("|float32|") != std::string::npos);
    assert(descriptor.find("|double|") == std::string::npos);

    seedGrad(p1, 1.0);
    seedGrad(p2, 2.0);
    seedGrad(p3, 3.0);

    std::unordered_set<const Node*> touched = {p1.get(), p3.get()};
    layout.packBucket(*layout.bucketIndexFor(*p1), touched);
    const auto& bucket0 = layout.bucket(*layout.bucketIndexFor(*p1));
    assert(almostEqual(bucket0.buffer[0], 1.0));
    assert(almostEqual(bucket0.buffer[1], 2.0));
    assert(almostEqual(bucket0.buffer[2], 0.0));
    assert(almostEqual(bucket0.buffer[3], 0.0));
    assert(almostEqual(bucket0.buffer[4], 0.0));

    auto& bucket1 = layout.bucket(*layout.bucketIndexFor(*p3));
    layout.packBucket(*layout.bucketIndexFor(*p3), touched);
    for (Scalar& v : bucket1.buffer) {
        v += 5.0;
    }
    layout.unpackBucket(*layout.bucketIndexFor(*p3), touched);
    assert(almostEqual(p3->grad().data[0], 8.0));
    assert(almostEqual(p2->grad().data[0], 2.0));
}

void test_step_boundary_bucketed_sync_world_size_one() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);
    auto p1 = makeParam(1, 2, 1.0);
    auto p2 = makeParam(1, 3, 10.0);
    seedGrad(p1, 1.0);
    seedGrad(p2, 2.0);

    StepBoundaryBucketedSync sync(dist, {p1, p2}, 4 * kBucketElemBytes);
    const std::uint64_t global_batch = sync.sync(7);
    assert(global_batch == 7);

    const auto profile = sync.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(profile.bucket_bytes >= static_cast<std::uint64_t>(5 * kBucketElemBytes));
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[2], 6.0));
}

void test_bucketed_runtime_planned_overlap_and_shared_bucket() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto p3 = makeParam(1, 1, 4.0);
    auto loss = MathOps::sum(p1);

    BucketedOverlapRuntime runtime(dist, {p1, p2, p3}, 2 * kBucketElemBytes);
    runtime.beginStep(4);
    runBackwardWithRuntime(runtime, loss);

    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 4);
    const auto profile = runtime.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(p2->hasAllocatedGrad());
    assert(almostEqual(p2->grad().data[0], 0.0));
    assert(p3->hasAllocatedGrad());
    assert(almostEqual(p3->grad().data[0], 0.0));
}

void test_bucketed_runtime_rejects_unknown_parameter_event() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto outsider = makeParam(1, 1, 5.0);
    BucketedOverlapRuntime runtime(dist, {p1}, kBucketElemBytes);
    runtime.beginStep(1);

    bool threw = false;
    try {
        runtime.onParameterGradReady(*outsider);
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}

void test_bucketed_runtime_plan_must_precede_ready() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    seedGrad(p1, 1.0);
    BucketedOverlapRuntime runtime(dist, {p1}, kBucketElemBytes);
    runtime.beginStep(1);

    bool threw = false;
    try {
        runtime.onParameterGradReady(*p1);
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}

void test_bucketed_runtime_empty_step_is_legal() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    BucketedOverlapRuntime runtime(dist, {p1}, kBucketElemBytes);
    runtime.beginStep(9);
    runtime.planStep({});
    runtime.onBackwardComplete();
    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();

    assert(global_batch == 9);
    const auto profile = runtime.lastStepProfile();
    assert(profile.bucket_count == 1);
    assert(profile.launched_bucket_count == 1);
    assert(!profile.effective_overlap);
    assert(p1->hasAllocatedGrad());
    assert(almostEqual(p1->grad().data[0], 0.0));
}

void test_bucketed_runtime_detects_missing_ready_in_planned_mode() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    BucketedOverlapRuntime runtime(dist, {p1, p2}, 2 * kBucketElemBytes);
    runtime.beginStep(3);
    runtime.planStep({p1, p2});
    seedGrad(p1, 1.0);
    runtime.onParameterGradReady(*p1);

    bool threw = false;
    try {
        runtime.onBackwardComplete();
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}

void test_bucketed_runtime_emits_separate_simulated_and_real_traces() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto loss = MathOps::sum(p1);

    BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
    runtime.beginStep(4);
    runBackwardWithRuntime(runtime, loss);
    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 4);

    const auto profile = runtime.lastStepProfile();
    assert(!profile.launch_events.empty());
    assert(!encodeLaunchSequence(profile, SyncTraceChannel::Simulated).empty());
    assert(!encodeLaunchSequence(profile, SyncTraceChannel::RealComm).empty());
    assert(profile.simulated_launch_bucket_count == profile.launched_bucket_count);
    assert(countLaunchEvents(profile,
                             SyncTraceChannel::RealComm,
                             SyncLaunchReason::StructuralZeroExpectedLaunch) >= 1);
    assertTraceLifecycleSeparation(profile);
}

void test_bucketed_runtime_pack_reads_final_ready_value() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p = makeParam(1, 1, 2.0);
    auto c1 = constant(Matrix(1, 1, 3.0));
    auto c2 = constant(Matrix(1, 1, 5.0));
    auto loss = MathOps::add(MathOps::mul(p, c1), MathOps::mul(p, c2));

    BucketedOverlapRuntime runtime(dist, {p}, kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);
    const auto global_batch = runtime.finalizeAndGetGlobalBatch();

    assert(global_batch == 1);
    assert(almostEqual(p->grad().data[0], 8.0));
    const auto profile = runtime.lastStepProfile();
    assert(countLaunchEvents(profile,
                             SyncTraceChannel::RealComm,
                             SyncLaunchReason::RealEarlyLaunch) == 1);
}

void test_bucketed_runtime_later_bucket_ready_but_blocked_by_cursor() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
    runtime.beginStep(1);
    runtime.planStep({p1, p2});

    seedGrad(p2, 7.0);
    runtime.onParameterGradReady(*p2);
    auto profile = runtime.lastStepProfile();
    assert(encodeLaunchSequence(profile, SyncTraceChannel::RealComm).empty());

    seedGrad(p1, 11.0);
    runtime.onParameterGradReady(*p1);
    profile = runtime.lastStepProfile();
    assert(encodeLaunchSequence(profile, SyncTraceChannel::RealComm) == "0,1");

    runtime.onBackwardComplete();
    const auto global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 1);
    assert(almostEqual(p1->grad().data[0], 11.0));
    assert(almostEqual(p2->grad().data[0], 7.0));
}

void test_bucketed_runtime_finalize_negative_paths() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);

    {
        BucketedOverlapRuntime runtime(dist, {p1}, kBucketElemBytes);
        runtime.beginStep(1);
        runtime.planStep({p1});

        bool threw = false;
        try {
            (void)runtime.finalizeAndGetGlobalBatch();
        } catch (const std::logic_error&) {
            threw = true;
        }
        assert(threw);
    }

    {
        BucketedOverlapRuntime runtime(dist, {p1}, kBucketElemBytes);
        runtime.beginStep(1);
        runtime.planStep({p1});
        seedGrad(p1, 1.0);
        runtime.onParameterGradReady(*p1);
        runtime.onBackwardComplete();
        (void)runtime.finalizeAndGetGlobalBatch();

        bool threw = false;
        try {
            (void)runtime.finalizeAndGetGlobalBatch();
        } catch (const std::logic_error&) {
            threw = true;
        }
        assert(threw);
    }

    {
        BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
        runtime.beginStep(1);
        runtime.planStep({p1, p2});
        seedGrad(p2, 1.0);
        runtime.onParameterGradReady(*p2);

        bool backward_threw = false;
        try {
            runtime.onBackwardComplete();
        } catch (const std::logic_error&) {
            backward_threw = true;
        }
        assert(backward_threw);

        bool finalize_threw = false;
        try {
            (void)runtime.finalizeAndGetGlobalBatch();
        } catch (const std::logic_error&) {
            finalize_threw = true;
        }
        assert(finalize_threw);
    }
}

void test_bucketed_runtime_shared_bucket_pack_safe_multiple_parameters() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto c3 = constant(Matrix(1, 1, 3.0));
    auto c5 = constant(Matrix(1, 1, 5.0));
    auto c7 = constant(Matrix(1, 1, 7.0));
    auto loss = MathOps::add(
        MathOps::add(MathOps::mul(p1, c3), MathOps::mul(p1, c5)),
        MathOps::mul(p2, c7));

    BucketedOverlapRuntime runtime(dist, {p1, p2}, 2 * kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);
    const auto global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 1);

    assert(almostEqual(p1->grad().data[0], 8.0));
    assert(almostEqual(p2->grad().data[0], 7.0));
    const auto profile = runtime.lastStepProfile();
    assert(encodeLaunchSequence(profile, SyncTraceChannel::RealComm) == "0");
}

void test_bucketed_baseline_and_overlap_match_parameter_grads() {
    int argc = 0;
    char** argv = nullptr;
    DistributedContext dist(argc, argv);

    auto baseline_p1 = makeParam(1, 1, 2.0);
    auto baseline_p2 = makeParam(1, 1, 3.0);
    auto overlap_p1 = makeParam(1, 1, 2.0);
    auto overlap_p2 = makeParam(1, 1, 3.0);
    auto c3 = constant(Matrix(1, 1, 3.0));
    auto c5 = constant(Matrix(1, 1, 5.0));
    auto c7 = constant(Matrix(1, 1, 7.0));

    auto baseline_loss = MathOps::add(
        MathOps::add(MathOps::mul(baseline_p1, c3), MathOps::mul(baseline_p1, c5)),
        MathOps::mul(baseline_p2, c7));
    baseline_loss->backward();

    StepBoundaryBucketedSync baseline_sync(dist, {baseline_p1, baseline_p2}, 2 * kBucketElemBytes);
    const auto baseline_global_batch = baseline_sync.sync(1);
    assert(baseline_global_batch == 1);

    auto overlap_loss = MathOps::add(
        MathOps::add(MathOps::mul(overlap_p1, c3), MathOps::mul(overlap_p1, c5)),
        MathOps::mul(overlap_p2, c7));
    BucketedOverlapRuntime overlap_runtime(dist, {overlap_p1, overlap_p2}, 2 * kBucketElemBytes);
    overlap_runtime.beginStep(1);
    runBackwardWithRuntime(overlap_runtime, overlap_loss);
    const auto overlap_global_batch = overlap_runtime.finalizeAndGetGlobalBatch();
    assert(overlap_global_batch == 1);

    assert(almostEqual(baseline_p1->grad().data[0], overlap_p1->grad().data[0]));
    assert(almostEqual(baseline_p2->grad().data[0], overlap_p2->grad().data[0]));
}

void test_bucketed_runtime_layout_mismatch_fails_fast(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI layout mismatch test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 30.0);

    bool threw = false;
    try {
        if (dist.rank() == 0) {
            BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
            (void)runtime;
        } else {
            BucketedOverlapRuntime runtime(dist, {p2, p1}, kBucketElemBytes);
            (void)runtime;
        }
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}

void test_bucketed_runtime_real_launch_sequence_rank_consistent(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI real launch sequence test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto p3 = makeParam(1, 1, 4.0);
    Node::Ptr loss = (dist.rank() == 0)
        ? MathOps::add(MathOps::sum(p1), MathOps::sum(p3))
        : MathOps::add(MathOps::sum(p2), MathOps::sum(p3));

    BucketedOverlapRuntime runtime(dist, {p1, p2, p3}, 2 * kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);
    (void)runtime.finalizeAndGetGlobalBatch();

    const auto profile = runtime.lastStepProfile();
    const auto sequences = dist.allGatherStrings(encodeLaunchSequence(profile, SyncTraceChannel::RealComm));
    for (std::size_t i = 1; i < sequences.size(); ++i) {
        assert(sequences[i] == sequences[0]);
    }
    assert(sequences[0] == expectedPrefixSequence(profile.launched_bucket_count));
}

void test_bucketed_runtime_tail_flush_trace_reason(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI tail flush trace test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    Node::Ptr loss = (dist.rank() == 0) ? MathOps::sum(p1) : MathOps::sum(p2);

    BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);
    (void)runtime.finalizeAndGetGlobalBatch();

    const auto profile = runtime.lastStepProfile();
    assert(countLaunchEvents(profile,
                             SyncTraceChannel::RealComm,
                             SyncLaunchReason::TailFlushLaunch) >= 1);
}

void test_step_boundary_bucketed_sync_rank_divergent_active_sets(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI baseline divergent-active-set test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    p1->zeroGrad();
    p2->zeroGrad();
    if (dist.rank() == 0) {
        seedGrad(p1, 1.0);
    } else {
        seedGrad(p2, 1.0);
    }

    StepBoundaryBucketedSync sync(dist, {p1, p2}, kBucketElemBytes);
    const std::uint64_t global_batch = sync.sync(1);

    assert(global_batch == 2);
    const auto profile = sync.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[0], 1.0));
}

void test_step_boundary_bucketed_sync_shared_bucket_partial_reachable(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI baseline shared-bucket test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto p3 = makeParam(1, 1, 4.0);
    p1->zeroGrad();
    p2->zeroGrad();
    p3->zeroGrad();

    if (dist.rank() == 0) {
        seedGrad(p1, 1.0);
        seedGrad(p3, 1.0);
    } else {
        seedGrad(p2, 1.0);
        seedGrad(p3, 1.0);
    }

    StepBoundaryBucketedSync sync(dist, {p1, p2, p3}, 2 * kBucketElemBytes);
    const std::uint64_t global_batch = sync.sync(1);

    assert(global_batch == 2);
    const auto profile = sync.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[0], 1.0));
    assert(almostEqual(p3->grad().data[0], 2.0));
}

void test_step_boundary_bucketed_sync_filters_frozen_and_non_parameter_leaves(
    DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI baseline mixed-parameter test requires world_size == 2.");
    }

    auto sync_param = makeParam(1, 1, 2.0);
    auto frozen_param = makeParam(1, 1, 5.0);
    frozen_param->setRequiresGrad(false);
    auto non_param_leaf = std::make_shared<Node>(Matrix(1, 1, 7.0));

    const auto filtered =
        runtime::collectSynchronizableParams({sync_param, frozen_param, non_param_leaf});
    assert(filtered.size() == 1);
    assert(filtered[0].get() == sync_param.get());

    sync_param->zeroGrad();
    if (dist.rank() == 0) {
        seedGrad(sync_param, 1.0);
    } else {
        sync_param->zeroGrad();
    }

    StepBoundaryBucketedSync sync(dist, filtered, kBucketElemBytes);
    const std::uint64_t global_batch = sync.sync(1);

    assert(global_batch == 2);
    const auto profile = sync.lastStepProfile();
    assert(profile.bucket_count == 1);
    assert(profile.launched_bucket_count == 1);
    assert(!profile.effective_overlap);
    assert(almostEqual(sync_param->grad().data[0], 1.0));
    assert(!frozen_param->hasAllocatedGrad());
    assert(!non_param_leaf->hasAllocatedGrad());
}

void test_bucketed_runtime_rank_divergent_active_sets_planned(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI planned test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto loss = (dist.rank() == 0) ? MathOps::sum(p1) : MathOps::sum(p2);

    BucketedOverlapRuntime runtime(dist, {p1, p2}, kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);

    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 2);
    const auto profile = runtime.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[0], 1.0));
}

void test_bucketed_runtime_shared_bucket_partial_reachable(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI shared-bucket test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    auto p3 = makeParam(1, 1, 4.0);

    Node::Ptr loss = (dist.rank() == 0)
        ? MathOps::add(MathOps::sum(p1), MathOps::sum(p3))
        : MathOps::add(MathOps::sum(p2), MathOps::sum(p3));

    BucketedOverlapRuntime runtime(dist, {p1, p2, p3}, 2 * kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);

    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 2);
    const auto profile = runtime.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(!profile.effective_overlap);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[0], 1.0));
    assert(almostEqual(p3->grad().data[0], 2.0));
}

void test_bucketed_runtime_no_cross_step_leak(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI cross-step test requires world_size == 2.");
    }

    auto p1 = makeParam(1, 1, 2.0);
    auto p2 = makeParam(1, 1, 3.0);
    BucketedOverlapRuntime runtime(dist, {p1, p2}, 2 * kBucketElemBytes);

    runtime.beginStep(1);
    runBackwardWithRuntime(
        runtime,
        (dist.rank() == 0) ? MathOps::sum(p1) : MathOps::sum(p2));
    const std::uint64_t first_global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(first_global_batch == 2);
    assert(almostEqual(p1->grad().data[0], 1.0));
    assert(almostEqual(p2->grad().data[0], 1.0));

    p1->zeroGrad();
    p2->zeroGrad();

    auto two = constant(Matrix(1, 1, 2.0));
    auto three = constant(Matrix(1, 1, 3.0));
    runtime.beginStep(1);
    runBackwardWithRuntime(
        runtime,
        (dist.rank() == 0)
            ? MathOps::sum(MathOps::mul(p2, two))
            : MathOps::sum(MathOps::mul(p1, three)));
    const std::uint64_t second_global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(second_global_batch == 2);
    assert(almostEqual(p1->grad().data[0], 3.0));
    assert(almostEqual(p2->grad().data[0], 2.0));
}

void test_bucketed_runtime_filters_frozen_and_non_parameter_leaves(DistributedContext& dist) {
    if (dist.worldSize() != 2) {
        throw std::runtime_error("MPI mixed-parameter test requires world_size == 2.");
    }

    auto sync_param = makeParam(1, 1, 2.0);
    auto frozen_param = makeParam(1, 1, 5.0);
    frozen_param->setRequiresGrad(false);
    auto non_param_leaf = std::make_shared<Node>(Matrix(1, 1, 7.0));

    const auto filtered =
        runtime::collectSynchronizableParams({sync_param, frozen_param, non_param_leaf});
    assert(filtered.size() == 1);
    assert(filtered[0].get() == sync_param.get());

    auto loss = MathOps::add(MathOps::sum(sync_param), MathOps::sum(non_param_leaf));
    BucketedOverlapRuntime runtime(dist, filtered, kBucketElemBytes);
    runtime.beginStep(1);
    runBackwardWithRuntime(runtime, loss);

    const std::uint64_t global_batch = runtime.finalizeAndGetGlobalBatch();
    assert(global_batch == 2);
    const auto profile = runtime.lastStepProfile();
    assert(profile.bucket_count == 1);
    assert(profile.launched_bucket_count == 1);
    assert(!profile.effective_overlap);
    assert(almostEqual(sync_param->grad().data[0], 2.0));
    assert(almostEqual(non_param_leaf->grad().data[0], 1.0));
    assert(!frozen_param->hasAllocatedGrad());
}

}

int runMpiContractTests(int argc, char** argv) {
    DistributedContext dist(argc, argv);
    test_bucketed_runtime_layout_mismatch_fails_fast(dist);
    test_step_boundary_bucketed_sync_rank_divergent_active_sets(dist);
    test_step_boundary_bucketed_sync_shared_bucket_partial_reachable(dist);
    test_step_boundary_bucketed_sync_filters_frozen_and_non_parameter_leaves(dist);
    test_bucketed_runtime_rank_divergent_active_sets_planned(dist);
    test_bucketed_runtime_shared_bucket_partial_reachable(dist);
    test_bucketed_runtime_real_launch_sequence_rank_consistent(dist);
    test_bucketed_runtime_tail_flush_trace_reason(dist);
    test_bucketed_runtime_no_cross_step_leak(dist);
    test_bucketed_runtime_filters_frozen_and_non_parameter_leaves(dist);
    std::cout << "Distributed MPI sync tests passed!" << std::endl;
    return 0;
}

int runLocalDistributedSyncTests() {
    test_bucket_layout_mapping_and_zero_fill();
    test_step_boundary_bucketed_sync_world_size_one();
    test_bucketed_runtime_planned_overlap_and_shared_bucket();
    test_bucketed_runtime_rejects_unknown_parameter_event();
    test_bucketed_runtime_plan_must_precede_ready();
    test_bucketed_runtime_empty_step_is_legal();
    test_bucketed_runtime_detects_missing_ready_in_planned_mode();
    test_bucketed_runtime_emits_separate_simulated_and_real_traces();
    test_bucketed_runtime_pack_reads_final_ready_value();
    test_bucketed_runtime_later_bucket_ready_but_blocked_by_cursor();
    test_bucketed_runtime_finalize_negative_paths();
    test_bucketed_runtime_shared_bucket_pack_safe_multiple_parameters();
    test_bucketed_baseline_and_overlap_match_parameter_grads();
    std::cout << "Distributed sync tests passed!" << std::endl;
    return 0;
}

int main(int argc, char** argv) {
#ifdef DISTRIBUTED_MPI_CONTRACTS_ONLY
    return runMpiContractTests(argc, argv);
#else
    bool run_mpi_world_size_2 = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mpi-world-size-2") {
            run_mpi_world_size_2 = true;
        }
    }

    if (run_mpi_world_size_2) {
        return runMpiContractTests(argc, argv);
    }

    return runLocalDistributedSyncTests();
#endif
}
