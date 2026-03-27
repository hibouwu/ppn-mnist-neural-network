#include "runtime/synchronizable_params.hpp"

#include "synchronizable_param.hpp"

namespace runtime {

std::vector<Node::Ptr> collectSynchronizableParams(const std::vector<Node::Ptr>& params) {
    std::vector<Node::Ptr> out;
    out.reserve(params.size());
    for (const auto& param : params) {
        if (param && isSynchronizableLeafParameter(*param)) {
            out.push_back(param);
        }
    }
    return out;
}

}
