#pragma once

#include "tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

struct BackwardContext {
    std::vector<Matrix> matrices;
    std::vector<double> scalars;
    std::vector<std::size_t> sizes;
    std::vector<std::uint8_t> flags;
    std::vector<std::vector<std::size_t>> index_vectors;
};
