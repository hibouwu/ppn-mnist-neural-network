#pragma once

#include "operation_node.hpp"

namespace MathOps {

    Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b);

    Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b);

    Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b);

}
