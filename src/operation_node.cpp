#include "operation_node.hpp"

OperationNode::OperationNode(OpKind type,
                             const Matrix& value,
                             const std::vector<Node::Ptr>& parents)
    : Node(value), opType_(type)
{
    for (const auto& p : parents)
        addParent(p);
}
