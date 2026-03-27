#include "distributed/param_registry.hpp"
#include "synchronizable_param.hpp"

#include <stdexcept>
#include <unordered_set>

ParamRegistry::ParamRegistry(const std::vector<Node::Ptr>& params) {
    params_.reserve(params.size());
    std::unordered_set<const Node*> seen;
    for (const auto& param : params) {
        if (!param) {
            throw std::invalid_argument("ParamRegistry: null parameter in parameter list.");
        }
        if (!isSynchronizableLeafParameter(*param)) {
            throw std::invalid_argument(
                "ParamRegistry: expected synchronizable leaf parameters only.");
        }
        const bool inserted = seen.insert(param.get()).second;
        if (!inserted) {
            throw std::logic_error("ParamRegistry: duplicate parameter identity detected.");
        }
        params_.push_back(param);
    }
}

bool ParamRegistry::contains(const Node& param) const {
    for (const auto& registered : params_) {
        if (registered.get() == &param) {
            return true;
        }
    }
    return false;
}
