#pragma once

#include "node.hpp"

/**
 * @enum OpKind
 * @brief Enumerates the different operation types supported by OperationNode.
 *
 * This is mainly used for debugging, logging or visualization of the
 * computation graph (e.g. to distinguish between Add, Mul, MatMul, etc.).
 */
enum class OpKind {
    ADD,     ///< Element-wise addition
    MUL,     ///< Element-wise multiplication (Hadamard product)
    MATMUL,  ///< Matrix multiplication
    RELU,    ///< ReLU activation
    SIGMOID, ///< Sigmoid activation
    TANH     ///< Tanh activation
};

/**
 * @class OperationNode
 * @brief Specialized Node that stores the type of operation used to produce it.
 *
 * OperationNode extends the base Node class by adding an OpKind field.
 * This field is not required for automatic differentiation itself, but
 * it is useful to inspect or visualize the computation graph.
 */
class OperationNode : public Node {
public:
    using Ptr = std::shared_ptr<OperationNode>;

    /**
     * @brief Constructs an OperationNode with a given operation type, value and parents.
     *
     * @param type    The operation kind that produced this node.
     * @param value   The forward value associated with this node.
     * @param parents The list of parent nodes (inputs of the operation).
     */
    OperationNode(OpKind type,
                  const Matrix& value,
                  const std::vector<Node::Ptr>& parents);

    /**
     * @brief Returns the operation type associated with this node.
     */
    OpKind opType() const { return opType_; }

private:
    OpKind opType_; ///< Operation type that created this node.
};
