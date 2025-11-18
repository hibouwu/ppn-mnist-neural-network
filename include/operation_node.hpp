#pragma once

#include "node.hpp"

enum class OpKind {
    ADD,
    MUL,
    MATMUL,
    RELU,
    SIGMOID,
    TANH
};

class OperationNode : public Node {
public:
    using Ptr = std::shared_ptr<OperationNode>;

    OperationNode(OpKind type,
                  const Matrix& value,
                  const std::vector<Node::Ptr>& parents);

    OpKind opType() const { return opType_; }

private:
    OpKind opType_;
};
