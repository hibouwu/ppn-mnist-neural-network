#pragma once

#include "node.hpp"

#include <cstddef>
#include <vector>

class ParamRegistry {
public:
    explicit ParamRegistry(const std::vector<Node::Ptr>& params);

    std::size_t size() const { return params_.size(); }
    bool contains(const Node& param) const;
    const std::vector<Node::Ptr>& params() const { return params_; }

private:
    std::vector<Node::Ptr> params_;
};
