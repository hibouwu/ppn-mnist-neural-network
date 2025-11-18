#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>

class Node : public std::enable_shared_from_this<Node> {
public:
    using Ptr = std::shared_ptr<Node>;

    explicit Node(const Tensor& v);      // 叶子/常量节点
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // API
    const Tensor& value() const { return value_; }
    const Tensor& grad () const { return grad_;  }
    Tensor&       grad ()       { return grad_;  }

    void addParent(const Ptr& p) { parents_.push_back(p); }
    void setBackwardFn(std::function<void(const Tensor&)> fn) { backwardFn_ = std::move(fn); }
    void addGrad(const Tensor& g);                // grad += g
    void zeroGrad();                              // grad = 0
    void backward();                              // 触发反向传播（从本节点出发）

    // 辅助：拓扑排序
    static std::vector<Ptr> topoSort(const Ptr& root);

private:
    Tensor value_;
    Tensor grad_;
    std::vector<std::weak_ptr<Node>> parents_;
    std::function<void(const Tensor&)> backwardFn_;
};

// 便捷工厂
inline std::shared_ptr<Node> constant(const Tensor& t){
    return std::make_shared<Node>(t);
}