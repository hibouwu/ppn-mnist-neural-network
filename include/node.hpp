#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>
#include <functional>

// 在当前项目中，用 Matrix 作为张量类型
using Tensor = Matrix;

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

    // 记录父节点（本节点依赖哪些父节点的输出）
    void addParent(const Ptr& p) { parents_.push_back(p); }

    // 设置本节点的反向传播函数：
    // 传入当前节点关于输出的梯度 g，函数内部负责把梯度分配到父节点
    void setBackwardFn(std::function<void(const Tensor&)> fn) { backwardFn_ = std::move(fn); }

    // 梯度操作
    void addGrad(const Tensor& g);   // grad_ += g
    void zeroGrad();                 // grad_ = 0
    void backward();                 // 从当前节点触发整张图的反向传播

    // 辅助：对计算图做拓扑排序（父节点在前，子节点在后）
    static std::vector<Ptr> topoSort(const Ptr& root);

private:
    Tensor value_;   // 正向计算的值
    Tensor grad_;    // 关于某个标量 loss 的梯度
    std::vector<std::weak_ptr<Node>> parents_;
    std::function<void(const Tensor&)> backwardFn_;
};

// 便捷工厂：创建常量/叶子节点
inline std::shared_ptr<Node> constant(const Tensor& t){
    return std::make_shared<Node>(t);
}