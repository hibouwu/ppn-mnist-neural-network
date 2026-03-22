#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>

class Node;

struct GradientContribution {
    std::shared_ptr<Node> target;
    Matrix grad;
};

class GradFn {
public:
    virtual ~GradFn() = default;
    virtual std::vector<GradientContribution> apply(const Node& output,
                                                    const Matrix& grad_output) const = 0;
};
