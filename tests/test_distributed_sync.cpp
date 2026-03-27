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
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

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

void test_bucket_layout_mapping_and_zero_fill() {
    auto p1 = makeParam(1, 2, 1.0);
    auto p2 = makeParam(1, 3, 10.0);
    auto p3 = makeParam(1, 5, 20.0);
    ParamRegistry registry({p1, p2, p3});
    BucketLayout layout(registry, 5 * sizeof(double));

    assert(layout.bucketCount() == 2);
    assert(layout.bucketIndexFor(*p1).has_value());
    assert(layout.bucketIndexFor(*p2).has_value());
    assert(layout.bucketIndexFor(*p3).has_value());
    assert(*layout.bucketIndexFor(*p1) == *layout.bucketIndexFor(*p2));
    assert(*layout.bucketIndexFor(*p3) != *layout.bucketIndexFor(*p1));

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
    for (double& v : bucket1.buffer) {
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

    StepBoundaryBucketedSync sync(dist, {p1, p2}, 4 * sizeof(double));
    const std::uint64_t global_batch = sync.sync(7);
    assert(global_batch == 7);

    const auto profile = sync.lastStepProfile();
    assert(profile.bucket_count == 2);
    assert(profile.launched_bucket_count == 2);
    assert(profile.bucket_bytes >= static_cast<std::uint64_t>(5 * sizeof(double)));
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

    BucketedOverlapRuntime runtime(dist, {p1, p2, p3}, 2 * sizeof(double));
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
    BucketedOverlapRuntime runtime(dist, {p1}, sizeof(double));
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
    BucketedOverlapRuntime runtime(dist, {p1}, sizeof(double));
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
    BucketedOverlapRuntime runtime(dist, {p1}, sizeof(double));
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
    BucketedOverlapRuntime runtime(dist, {p1, p2}, 2 * sizeof(double));
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

    StepBoundaryBucketedSync sync(dist, {p1, p2}, sizeof(double));
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

    StepBoundaryBucketedSync sync(dist, {p1, p2, p3}, 2 * sizeof(double));
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

    StepBoundaryBucketedSync sync(dist, filtered, sizeof(double));
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

    BucketedOverlapRuntime runtime(dist, {p1, p2}, sizeof(double));
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

    BucketedOverlapRuntime runtime(dist, {p1, p2, p3}, 2 * sizeof(double));
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
    BucketedOverlapRuntime runtime(dist, {p1, p2}, 2 * sizeof(double));

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
    BucketedOverlapRuntime runtime(dist, filtered, sizeof(double));
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
    test_step_boundary_bucketed_sync_rank_divergent_active_sets(dist);
    test_step_boundary_bucketed_sync_shared_bucket_partial_reachable(dist);
    test_step_boundary_bucketed_sync_filters_frozen_and_non_parameter_leaves(dist);
    test_bucketed_runtime_rank_divergent_active_sets_planned(dist);
    test_bucketed_runtime_shared_bucket_partial_reachable(dist);
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
