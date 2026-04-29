#pragma once

#include "node.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ist {

enum class PartitionAxis {
    Rows,
    Cols,
    Elements,
};

struct OwnershipPlan {
    std::vector<Matrix> masks;
    std::vector<PartitionAxis> axes;
    std::vector<std::size_t> owned_elements_per_param;
    std::size_t owned_elements = 0;
    std::size_t total_elements = 0;
    int rank = 0;
    int world_size = 1;
    unsigned int partition_seed = 0;
};

OwnershipPlan buildOwnershipPlan(const std::vector<Node::Ptr>& params,
                                 int rank,
                                 int world_size,
                                 unsigned int partition_seed);

double ownershipRatio(const OwnershipPlan& plan);
const char* toString(PartitionAxis axis);

}  // namespace ist

