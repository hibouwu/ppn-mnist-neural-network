#pragma once

#include "distributed/gradient_sync_runtime.hpp"
#include "runtime/grad_sync_mode_info.hpp"
#include "trainer.hpp"

#include <memory>
#include <vector>

class DistributedContext;

namespace runtime {

struct GradSyncSetup {
    Trainer::GradSyncFn grad_sync_fn = nullptr;
    Trainer::SyncProfileProviderFn sync_profile_provider = nullptr;
    std::unique_ptr<GradientSyncRuntime> gradient_sync_runtime;
};

GradSyncSetup buildGradSyncSetup(const DistributedContext& dist,
                                 const std::vector<Node::Ptr>& model_params,
                                 const GradSyncModeInfo& mode_info,
                                 std::size_t bucket_size_bytes);

}
