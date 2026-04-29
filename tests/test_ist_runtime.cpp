#include "ist/ist_runtime.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

Node::Ptr makeParam(std::size_t rows, std::size_t cols) {
    Matrix v(rows, cols, 0.0);
    auto p = std::make_shared<Node>(v);
    p->setIsParameter(true);
    return p;
}

std::vector<Node::Ptr> makeSyntheticCnnLikeParams() {
    // conv_w(8, 27), conv_b(1,8), fc_w(32,10), fc_b(1,10)
    return {
        makeParam(8, 27),
        makeParam(1, 8),
        makeParam(32, 10),
        makeParam(1, 10),
    };
}

void testAxisInferenceForCnnLikeLayout() {
    const auto params = makeSyntheticCnnLikeParams();
    const auto plan = ist::buildOwnershipPlan(params, 0, 2, 42);
    assert(plan.axes.size() == params.size());
    assert(plan.axes[0] == ist::PartitionAxis::Rows);
    assert(plan.axes[1] == ist::PartitionAxis::Cols);
    assert(plan.axes[2] == ist::PartitionAxis::Cols);
    assert(plan.axes[3] == ist::PartitionAxis::Cols);
}

void testCoverageAndDisjointAcrossRanks() {
    const auto params = makeSyntheticCnnLikeParams();
    constexpr int world = 4;
    std::vector<ist::OwnershipPlan> plans;
    plans.reserve(world);
    for (int rank = 0; rank < world; ++rank) {
        plans.push_back(ist::buildOwnershipPlan(params, rank, world, 123));
    }

    for (std::size_t p = 0; p < params.size(); ++p) {
        const Matrix& ref = params[p]->value();
        const std::size_t n = ref.data.size();
        for (std::size_t i = 0; i < n; ++i) {
            int owners = 0;
            for (int r = 0; r < world; ++r) {
                if (plans[r].masks[p].data[i] > 0.5f) {
                    owners += 1;
                }
            }
            assert(owners == 1);
        }
    }
}

void testDeterministicBySeed() {
    const auto params = makeSyntheticCnnLikeParams();
    const auto a = ist::buildOwnershipPlan(params, 1, 3, 777);
    const auto b = ist::buildOwnershipPlan(params, 1, 3, 777);
    assert(a.masks.size() == b.masks.size());
    for (std::size_t p = 0; p < a.masks.size(); ++p) {
        assert(a.masks[p].rows == b.masks[p].rows);
        assert(a.masks[p].cols == b.masks[p].cols);
        for (std::size_t i = 0; i < a.masks[p].data.size(); ++i) {
            assert(std::fabs(a.masks[p].data[i] - b.masks[p].data[i]) < 1e-12);
        }
    }
}

}  // namespace

int main() {
    std::cout << "Running IST runtime tests..." << std::endl;
    testAxisInferenceForCnnLikeLayout();
    testCoverageAndDisjointAcrossRanks();
    testDeterministicBySeed();
    std::cout << "All IST runtime tests PASSED!" << std::endl;
    return 0;
}

