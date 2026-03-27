#pragma once

#include "node.hpp"

#include <vector>

namespace runtime {

std::vector<Node::Ptr> collectSynchronizableParams(const std::vector<Node::Ptr>& params);

}
