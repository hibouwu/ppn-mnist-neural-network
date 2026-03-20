#pragma once

#include "tensor.hpp"
#include <cstddef>
#include <vector>

class BatchSource {
public:
    virtual ~BatchSource() = default;

    virtual std::size_t rowCount() const = 0;
    virtual std::size_t inputCols() const = 0;
    virtual std::size_t targetCols() const = 0;
    virtual void loadRows(const std::vector<std::size_t>& indices,
                          Matrix& inputs,
                          Matrix& targets) const = 0;
};
