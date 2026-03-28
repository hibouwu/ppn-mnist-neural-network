#pragma once

#include "node.hpp"

#include <cstddef>
#include <string>
#include <vector>

class ParamRegistry {
public:
    explicit ParamRegistry(const std::vector<Node::Ptr>& params);

    std::size_t size() const { return params_.size(); }
    bool contains(const Node& param) const;
    const std::vector<Node::Ptr>& params() const { return params_; }
    std::size_t ordinalFor(const Node& param) const;
    const std::string& logicalKeyFor(const Node& param) const;
    const std::vector<std::string>& logicalKeys() const { return logical_keys_; }

private:
    std::vector<Node::Ptr> params_;
    std::vector<std::string> logical_keys_;
};
