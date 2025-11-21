#pragma once

#include "operation_node.hpp"

namespace MathOps {

    Node::Ptr add(const Node::Ptr& a, const Node::Ptr& b);

    Node::Ptr mul(const Node::Ptr& a, const Node::Ptr& b);

    Node::Ptr matmul(const Node::Ptr& a, const Node::Ptr& b);
    
    Node::Ptr sum(const Node::Ptr& x); 
    
    Node::Ptr mean(const Node::Ptr& x);

    Node::Ptr relu(const Node::Ptr& x);
    Node::Ptr sigmoid(const Node::Ptr& x);
    Node::Ptr tanh(const Node::Ptr& x);

}
