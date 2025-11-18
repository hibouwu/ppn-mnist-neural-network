#include "node.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

Node::Node(const Matrix& v)
    : value_(v), grad_(v.rows, v.cols, 0.0) {}

void Node::addGrad(const Matrix& g) {
    if (g.rows != grad_.rows || g.cols != grad_.cols) {
        throw std::invalid_argument("Gradient shape mismatch in Node::addGrad");
    }
    const std::size_t n = grad_.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_.data[i] += g.data[i];
    }
}

void Node::zeroGrad() {
    std::fill(grad_.data.begin(), grad_.data.end(), 0.0);
}

std::vector<Node::Ptr> Node::topoSort(const Ptr& root) {
    std::vector<Ptr> order;
    std::unordered_set<const Node*> vis;

    std::function<void(const Ptr&)> dfs = [&](const Ptr& u) {
        if (!u || vis.count(u.get())) return;
        vis.insert(u.get());

        // 先走父节点，再把自己 push 进去
        for (auto &wp : u->parents_) {
            if (auto p = wp.lock()) dfs(p);
        }
        order.push_back(u);
    };

    dfs(root);
    // 由于我们是“父先递归，后 push 自己”，
    // 得到的顺序本身就是：所有父节点在前，子节点在后，不需要 reverse。
    return order;
}

void Node::backward() {
    auto self  = shared_from_this();
    auto order = topoSort(self);

    // 1) 清零所有梯度
    for (auto &n : order) n->zeroGrad();

    // 2) 根节点注入种子梯度：
    //    标量 -> 1；否则 -> 全 1
    if (value_.rows == 1 && value_.cols == 1)
        order.back()->grad_ = Matrix(1, 1, 1.0);
    else
        order.back()->grad_ = Matrix(value_.rows, value_.cols, 1.0);

    // 3) 按拓扑顺序执行每个节点自己的 backward 函数
    for (auto &n : order) {
        if (n->backwardFn_) n->backwardFn_(n->grad_);
    }
}


