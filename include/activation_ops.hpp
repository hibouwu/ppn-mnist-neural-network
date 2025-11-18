#pragma once

#include "operation_node.hpp"

namespace ActivationOps {

    Node::Ptr relu(const Node::Ptr& x);
    Node::Ptr sigmoid(const Node::Ptr& x);
    Node::Ptr tanh(const Node::Ptr& x);

}
