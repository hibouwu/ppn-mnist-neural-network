#include "node.hpp"
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

Node::Node(const Tensor& v)
    : value_(v), grad_(v.rows, v.cols, 0.0) {}

// grad_ += g
void Node::addGrad(const Tensor& g) {
    if (g.rows != grad_.rows || g.cols != grad_.cols) {
        throw std::invalid_argument("Gradient shape mismatch in Node::addGrad");
    }

    const std::size_t n = grad_.data.size();
    for (std::size_t i = 0; i < n; ++i) {
        grad_.data[i] += g.data[i];
    }
}

// grad_ = 0
void Node::zeroGrad() {
    std::fill(grad_.data.begin(), grad_.data.end(), 0.0);
}

std::vector<Node::Ptr> Node::topoSort(const Ptr& root) {
    std::vector<Ptr> order;
    std::unordered_set<const Node*> vis;

    std::function<void(const Ptr&)> dfs = [&](const Ptr& u) {
        if (!u || vis.count(u.get())) return;
        vis.insert(u.get());

        // 先遍历所有父节点
        for (auto &wp : u->parents_) {
            if (auto p = wp.lock()) {
                dfs(p);
            }
        }
        // 后序：父节点在前，子节点在后
        order.push_back(u);
    };

    dfs(root);
    // dfs 是「父 -> 子」后序，所以这里 reverse 保证 parents 在前
    std::reverse(order.begin(), order.end());
    return order;
}

void Node::backward() {
    auto self  = shared_from_this();
    auto order = topoSort(self);

    // 1) 清零所有节点的梯度
    for (auto &n : order) {
        n->zeroGrad();
    }

    // 2) 对根节点注入种子梯度：
    //    如果是标量（1x1），种子梯度为 1
    //    否则就注入全 1 的张量
    if (value_.rows == 1 && value_.cols == 1) {
        order.back()->grad_ = Tensor(1, 1, 1.0);
    } else {
        order.back()->grad_ = Tensor(value_.rows, value_.cols, 1.0);
    }

    // 3) 按拓扑顺序执行每个节点的 backwardFn
    for (auto &n : order) {
        if (n->backwardFn_) {
            n->backwardFn_(n->grad_);
        }
    }
<<<<<<< HEAD
}
=======
}

>>>>>>> abc8f7a32d989b621489b44e985e04ef80e680d5
