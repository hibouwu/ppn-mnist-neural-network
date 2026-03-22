#include "operation_node.hpp"

/**
 * @brief OperationNode constructor.
 *
 * It forwards the value to the base Node constructor, stores the operation type
 * and registers all inputs in the underlying computation graph.
 *
 * @param type    The operation kind that produced this node.
 * @param value   The forward value associated with this node.
 * @param parents The list of parent nodes (inputs of the operation).
 */
OperationNode::OperationNode(OpKind type,
                             const Matrix& value,
                             const std::vector<Node::Ptr>& parents,
                             bool requiresGrad)
    : Node(value, requiresGrad), opType_(type)
{
    setInputs(parents);
}
