#include "runtime/grad_sync_setup_factory.hpp"

#include "distributed/distributed.hpp"
#include "runtime/synchronizable_params.hpp"
#include "synchronizable_param.hpp"

namespace runtime {

GradSyncSetup buildGradSyncSetup(const DistributedContext& dist,
                                 const std::vector<Node::Ptr>& model_params,
                                 const GradSyncModeInfo& mode_info,
                                 std::size_t bucket_size_bytes,
                                 const GradCompressionConfig& grad_compression_cfg) {
    GradSyncSetup setup;
    const auto syncable_params = collectSynchronizableParams(model_params);
    const CommCompressionConfig comm_cfg {
        grad_compression_cfg.enabled,
        grad_compression_cfg.mode,
        grad_compression_cfg.topk_ratio,
        grad_compression_cfg.error_feedback,
        grad_compression_cfg.interval
    };

    if (mode_info.parsed_mode == GradSyncMode::PerParamBlocking) {
        if (dist.worldSize() > 1) {
            setup.grad_sync_fn = [&dist](const std::vector<Node::Ptr>& params_to_sync,
                                         std::uint64_t local_batch) -> std::uint64_t {
                for (const auto& param : params_to_sync) {
                    if (!param || !isSynchronizableLeafParameter(*param)) {
                        continue;
                    }
                    Matrix& grad = param->grad();
                    if (!grad.data.empty()) {
                        dist.allReduceSum(grad.data.data(), grad.data.size());
                    }
                }
                return dist.allReduceSumU64(local_batch);
            };
        }
        return setup;
    }

    if (mode_info.parsed_mode == GradSyncMode::BucketedBlocking) {
        auto bucketed_sync = std::make_shared<StepBoundaryBucketedSync>(
            dist,
            syncable_params,
            bucket_size_bytes,
            comm_cfg);
        setup.grad_sync_fn = [bucketed_sync](const std::vector<Node::Ptr>&,
                                             std::uint64_t local_batch) -> std::uint64_t {
            return bucketed_sync->sync(local_batch);
        };
        setup.sync_profile_provider = [bucketed_sync]() {
            return bucketed_sync->lastStepProfile();
        };
        setup.grad_compression_handled_in_runtime = grad_compression_cfg.enabled;
        return setup;
    }

    setup.gradient_sync_runtime = std::make_unique<BucketedOverlapRuntime>(
        dist,
        syncable_params,
        bucket_size_bytes,
        comm_cfg);
    GradientSyncRuntime* runtime_ptr = setup.gradient_sync_runtime.get();
    setup.sync_profile_provider = [runtime_ptr]() {
        return runtime_ptr->lastStepProfile();
    };
    setup.grad_compression_handled_in_runtime = grad_compression_cfg.enabled;
    return setup;
}

}
