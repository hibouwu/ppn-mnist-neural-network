#include "ist/ist_runtime.hpp"

#include "synchronizable_param.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ist {
namespace {

std::uint32_t mixSeed(unsigned int base_seed, std::size_t param_idx) {
    std::uint32_t x = static_cast<std::uint32_t>(base_seed);
    x ^= static_cast<std::uint32_t>(param_idx + 0x9e3779b9u + (x << 6) + (x >> 2));
    return x;
}

std::vector<std::size_t> shuffledIndices(std::size_t n, std::uint32_t seed) {
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    return indices;
}

PartitionAxis inferPartitionAxis(const std::vector<Node::Ptr>& params, std::size_t idx) {
    if (idx + 1 < params.size() && params[idx] && params[idx + 1]) {
        const Matrix& w = params[idx]->value();
        const Matrix& b = params[idx + 1]->value();
        if (b.rows == 1 && b.cols > 0) {
            if (w.rows == b.cols) {
                return PartitionAxis::Rows;  // conv kernel: (out_ch, in_ch*k*k)
            }
            if (w.cols == b.cols) {
                return PartitionAxis::Cols;  // linear weight: (in, out)
            }
        }
    }

    const Matrix& m = params[idx]->value();
    if (m.rows == 1 && m.cols > 1) {
        return PartitionAxis::Cols;
    }
    if (m.rows > 1) {
        return PartitionAxis::Rows;
    }
    return PartitionAxis::Elements;
}

std::size_t fillMaskForRows(Matrix& mask,
                            std::size_t rows,
                            std::size_t cols,
                            int rank,
                            int world_size,
                            std::uint32_t seed) {
    const auto shuffled_rows = shuffledIndices(rows, seed);
    std::size_t owned = 0;
    for (std::size_t pos = 0; pos < shuffled_rows.size(); ++pos) {
        if (static_cast<int>(pos % static_cast<std::size_t>(world_size)) != rank) {
            continue;
        }
        const std::size_t r = shuffled_rows[pos];
        const std::size_t row_base = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            mask.data[row_base + c] = 1.0f;
        }
        owned += cols;
    }
    return owned;
}

std::size_t fillMaskForCols(Matrix& mask,
                            std::size_t rows,
                            std::size_t cols,
                            int rank,
                            int world_size,
                            std::uint32_t seed) {
    const auto shuffled_cols = shuffledIndices(cols, seed);
    std::size_t owned = 0;
    for (std::size_t pos = 0; pos < shuffled_cols.size(); ++pos) {
        if (static_cast<int>(pos % static_cast<std::size_t>(world_size)) != rank) {
            continue;
        }
        const std::size_t c = shuffled_cols[pos];
        for (std::size_t r = 0; r < rows; ++r) {
            mask.data[r * cols + c] = 1.0f;
        }
        owned += rows;
    }
    return owned;
}

std::size_t fillMaskForElements(Matrix& mask,
                                std::size_t total_elements,
                                int rank,
                                int world_size,
                                std::uint32_t seed) {
    const auto shuffled = shuffledIndices(total_elements, seed);
    std::size_t owned = 0;
    for (std::size_t pos = 0; pos < shuffled.size(); ++pos) {
        if (static_cast<int>(pos % static_cast<std::size_t>(world_size)) != rank) {
            continue;
        }
        mask.data[shuffled[pos]] = 1.0f;
        owned += 1;
    }
    return owned;
}

}  // namespace

OwnershipPlan buildOwnershipPlan(const std::vector<Node::Ptr>& params,
                                 int rank,
                                 int world_size,
                                 unsigned int partition_seed) {
    if (world_size <= 0) {
        throw std::invalid_argument("buildOwnershipPlan: world_size must be > 0.");
    }
    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument("buildOwnershipPlan: rank out of range.");
    }

    OwnershipPlan plan;
    plan.rank = rank;
    plan.world_size = world_size;
    plan.partition_seed = partition_seed;
    plan.masks.reserve(params.size());
    plan.axes.reserve(params.size());
    plan.owned_elements_per_param.reserve(params.size());

    for (std::size_t idx = 0; idx < params.size(); ++idx) {
        if (!params[idx] || !isSynchronizableLeafParameter(*params[idx])) {
            plan.masks.emplace_back(1, 1, 0.0);
            plan.axes.push_back(PartitionAxis::Elements);
            plan.owned_elements_per_param.push_back(0);
            continue;
        }

        const Matrix& value = params[idx]->value();
        const std::size_t rows = value.rows;
        const std::size_t cols = value.cols;
        const std::size_t n = value.data.size();
        Matrix mask(rows, cols, 0.0);

        const PartitionAxis axis = inferPartitionAxis(params, idx);
        const std::uint32_t seed = mixSeed(partition_seed, idx);
        std::size_t owned = 0;

        if (axis == PartitionAxis::Rows) {
            owned = fillMaskForRows(mask, rows, cols, rank, world_size, seed);
        } else if (axis == PartitionAxis::Cols) {
            owned = fillMaskForCols(mask, rows, cols, rank, world_size, seed);
        } else {
            owned = fillMaskForElements(mask, n, rank, world_size, seed);
        }

        plan.total_elements += n;
        plan.owned_elements += owned;
        plan.masks.push_back(std::move(mask));
        plan.axes.push_back(axis);
        plan.owned_elements_per_param.push_back(owned);
    }

    return plan;
}

double ownershipRatio(const OwnershipPlan& plan) {
    if (plan.total_elements == 0) {
        return 0.0;
    }
    return static_cast<double>(plan.owned_elements) /
           static_cast<double>(plan.total_elements);
}

const char* toString(PartitionAxis axis) {
    switch (axis) {
        case PartitionAxis::Rows:
            return "rows";
        case PartitionAxis::Cols:
            return "cols";
        case PartitionAxis::Elements:
            return "elements";
    }
    return "unknown";
}

}  // namespace ist
