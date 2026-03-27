#pragma once

#include "node.hpp"

inline bool isSynchronizableLeafParameter(const Node& node) {
    return node.isParameter() && node.isLeaf() && node.inputs().empty() && node.requiresGrad();
}
