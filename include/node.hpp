#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>

class Node : public std::enable_shared_from_this<Node> {
public:
    using Ptr = std::shared_ptr<Node>;

    explicit Node(const Matrix& v);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    const Matrix& value() const { return value_; }
    const Matrix& grad()  const { return grad_;  }
    Matrix&       grad()        { return grad_;  }

    void addParent(const Ptr& p) { parents_.push_back(p); }

    // 反向传播函数：输入是当前节点的梯度
    void setBackwardFn(std::function<void(const Matrix&)> fn) {
        backwardFn_ = std::move(fn);
    }

    // 梯度操作
    void addGrad(const Matrix& g);
    void zeroGrad();
    void backward();

    // 计算图拓扑排序（父在前，子在后）
    static std::vector<Ptr> topoSort(const Ptr& root);

private:
    Matrix value_;
    Matrix grad_;
    std::vector<std::weak_ptr<Node>> parents_;
    std::function<void(const Matrix&)> backwardFn_;
};

// 便捷工厂
inline std::shared_ptr<Node> constant(const Matrix& t){
    return std::make_shared<Node>(t);
}
