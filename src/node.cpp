#include "node.hpp"
#include <unordered_set>
#include <algorithm>

Node::Node(const Tensor& v) : value_(v), grad_(Tensor::zerosLike(v)) {}

void Node::addGrad(const Tensor& g){ grad_ += g; }
void Node::zeroGrad(){ grad_ = Tensor::zerosLike(value_); }

std::vector<Node::Ptr> Node::topoSort(const Ptr& root){
    std::vector<Ptr> order;
    std::unordered_set<const Node*> vis;

    std::function<void(const Ptr&)> dfs = [&](const Ptr& u){
        if (!u || vis.count(u.get())) return;
        vis.insert(u.get());
        for (auto& wp : u->parents_) {
            if (auto p = wp.lock()) dfs(p);
        }
        order.push_back(u); // post-order
    };

    dfs(root);
    std::reverse(order.begin(), order.end()); // parents first
    return order;
}

void Node::backward(){
    auto self = shared_from_this();
    auto order = topoSort(self);

    // 1) 清零所有梯度
    for (auto& n : order) n->zeroGrad();

    // 2) 根节点注入种子梯度（标量 -> 1；否则 -> 全1）
    if (value_.size()==1) order.back()->grad_ = Tensor(1,1,1.f);
    else                  order.back()->grad_ = Tensor::onesLike(value_);

    // 3) 按拓扑顺序执行每个节点的 backwardFn
    for (auto& n : order) {
        if (n->backwardFn_) n->backwardFn_(n->grad_);
    }
}