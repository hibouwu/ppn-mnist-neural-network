#include "autograd/engine.hpp"
#include "autograd/grad_fn.hpp"

#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

struct NodeState {
    std::size_t pending_incoming_grads = 0;
    bool enqueued = false;
    bool parameter_hook_fired = false;
};

void collectReachable(const Node::Ptr& node,
                      std::unordered_set<const Node*>& visited,
                      std::vector<Node::Ptr>& order) {
    if (!node || visited.count(node.get()) != 0) {
        return;
    }
    visited.insert(node.get());
    order.push_back(node);
    for (const auto& input : node->inputs()) {
        collectReachable(input, visited, order);
    }
}

bool isScalar(const Matrix& m) {
    return m.rows == 1 && m.cols == 1;
}

}

void AutogradEngine::backward(const Node::Ptr& root) const {
    if (!root) {
        throw std::invalid_argument("AutogradEngine::backward: root is null.");
    }
    if (!root->requiresGrad()) {
        return;
    }

    std::unordered_set<const Node*> visited;
    std::vector<Node::Ptr> nodes;
    collectReachable(root, visited, nodes);

    std::unordered_map<const Node*, NodeState> states;
    states.reserve(nodes.size());
    for (const auto& node : nodes) {
        if (node->requiresGrad()) {
            states.emplace(node.get(), NodeState{});
        }
    }

    for (const auto& child : nodes) {
        if (!child->requiresGrad()) {
            continue;
        }
        for (const auto& input : child->inputs()) {
            if (!input || !input->requiresGrad()) {
                continue;
            }
            states[input.get()].pending_incoming_grads += 1;
        }
    }

    if (!root->hasExplicitGradSeed()) {
        if (!isScalar(root->value())) {
            throw std::runtime_error(
                "AutogradEngine::backward: non-scalar root requires an explicit seed gradient.");
        }
        Matrix seed(root->value().rows, root->value().cols, 1.0);
        root->addGrad(seed);
    }

    auto maybeFireParameterHook = [&](Node& node) {
        if (!parameter_ready_hook_) {
            return;
        }
        auto it = states.find(&node);
        if (it == states.end() || it->second.parameter_hook_fired) {
            return;
        }
        if (node.isParameter() && node.isLeaf() && node.inputs().empty()) {
            it->second.parameter_hook_fired = true;
            parameter_ready_hook_(node);
        }
    };

    std::queue<Node::Ptr> ready;
    ready.push(root);
    states[root.get()].enqueued = true;
    maybeFireParameterHook(*root);

    while (!ready.empty()) {
        Node::Ptr node = ready.front();
        ready.pop();

        const auto& gradFn = node->gradFn();
        if (!gradFn) {
            continue;
        }

        const auto contributions = gradFn->apply(*node, node->grad());
        for (const auto& contribution : contributions) {
            if (!contribution.target || !contribution.target->requiresGrad()) {
                continue;
            }
            contribution.target->addGrad(contribution.grad);
            auto& parentState = states[contribution.target.get()];
            if (parentState.pending_incoming_grads == 0) {
                throw std::logic_error("AutogradEngine::backward: pending contribution underflow.");
            }
            parentState.pending_incoming_grads -= 1;
            if (parentState.pending_incoming_grads == 0 && !parentState.enqueued) {
                parentState.enqueued = true;
                ready.push(contribution.target);
                maybeFireParameterHook(*contribution.target);
            }
        }
    }

    if (backward_complete_hook_) {
        backward_complete_hook_();
    }
}
