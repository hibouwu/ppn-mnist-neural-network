#include "autograd/engine.hpp"
#include "autograd/grad_fn.hpp"
#include "profiling.hpp"
#include "synchronizable_param.hpp"

#include <cassert>
#include <chrono>
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
#ifdef PROFILE_OPS
    using Clock = std::chrono::steady_clock;
    auto total_start = Clock::now();
#endif
    if (!root) {
        throw std::invalid_argument("AutogradEngine::backward: root is null.");
    }
    if (!root->requiresGrad()) {
        return;
    }

    std::unordered_set<const Node*> visited;
    std::vector<Node::Ptr> reachable;
#ifdef PROFILE_OPS
    auto collect_start = Clock::now();
#endif
    collectReachable(root, visited, reachable);
#ifdef PROFILE_OPS
    opProfileRecord(
        "engine_collect_reachable",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - collect_start).count());
#endif

#ifdef PROFILE_OPS
    auto state_start = Clock::now();
#endif
    std::vector<Node::Ptr> nodes;
    nodes.reserve(reachable.size());
    for (const auto& node : reachable) {
        if (node->requiresGrad()) {
            nodes.push_back(node);
        }
    }
    std::unordered_map<const Node*, std::uint32_t> node_to_index;
    node_to_index.reserve(nodes.size());
    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        node_to_index.emplace(nodes[i].get(), i);
    }
    std::vector<NodeState> states(nodes.size());
    std::vector<std::vector<std::uint32_t>> input_indices_per_node(nodes.size());
    std::vector<Node::Ptr> reachable_leaf_params;

    for (const auto& child : nodes) {
        const auto child_it = node_to_index.find(child.get());
        if (child_it == node_to_index.end()) {
            throw std::logic_error("AutogradEngine::backward: child node missing dense index.");
        }
        if (isSynchronizableLeafParameter(*child)) {
            reachable_leaf_params.push_back(child);
        }
        auto& input_indices = input_indices_per_node[child_it->second];
        input_indices.reserve(child->inputs().size());
        for (const auto& input : child->inputs()) {
            if (!input || !input->requiresGrad()) {
                input_indices.push_back(kInvalidNodeIndex);
                continue;
            }
            const auto it = node_to_index.find(input.get());
            if (it == node_to_index.end()) {
                input_indices.push_back(kInvalidNodeIndex);
                continue;
            }
            input_indices.push_back(it->second);
            states[it->second].pending_incoming_grads += 1;
        }
    }
#ifndef NDEBUG
    for (std::uint32_t i = 0; i < nodes.size(); ++i) {
        const auto it = node_to_index.find(nodes[i].get());
        assert(it != node_to_index.end());
        assert(it->second == i);
    }
#endif
#ifdef PROFILE_OPS
    opProfileRecord(
        "engine_build_state",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - state_start).count());
#endif

    if (reachable_leaf_hook_) {
        reachable_leaf_hook_(reachable_leaf_params);
    }

    if (!root->hasExplicitGradSeed()) {
        if (!isScalar(root->value())) {
            throw std::runtime_error(
                "AutogradEngine::backward: non-scalar root requires an explicit seed gradient.");
        }
        Matrix seed(root->value().rows, root->value().cols, 1.0);
        root->addGrad(seed);
    }

    const auto root_it = node_to_index.find(root.get());
    if (root_it == node_to_index.end()) {
        throw std::logic_error("AutogradEngine::backward: root missing dense index.");
    }
    const std::uint32_t root_index = root_it->second;

    auto maybeFireParameterHook = [&](std::uint32_t index) {
        if (!parameter_ready_hook_) {
            return;
        }
        if (index >= states.size()) {
            throw std::logic_error("AutogradEngine::backward: parameter hook index out of range.");
        }
        auto& state = states[index];
        if (state.parameter_hook_fired) {
            return;
        }
        Node& node = *nodes[index];
        if (isSynchronizableLeafParameter(node)) {
            state.parameter_hook_fired = true;
            parameter_ready_hook_(node);
        }
    };

    std::queue<std::uint32_t> ready;
    ready.push(root_index);
    states[root_index].enqueued = true;
    maybeFireParameterHook(root_index);

    while (!ready.empty()) {
        const std::uint32_t node_index = ready.front();
        ready.pop();
        const Node::Ptr& node = nodes[node_index];

        const auto& gradFn = node->gradFn();
        if (!gradFn) {
            continue;
        }

#ifdef PROFILE_OPS
        auto apply_start = Clock::now();
#endif
        const auto contributions = gradFn->apply(
            *node,
            node->grad(),
            InputIndexView(input_indices_per_node[node_index].data(), input_indices_per_node[node_index].size()));
#ifdef PROFILE_OPS
        opProfileRecord(
            "engine_apply_gradfn_total",
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - apply_start).count());
#endif
        for (std::size_t i = 0; i < contributions.size(); ++i) {
            const auto& contribution = contributions[i];
#ifdef PROFILE_OPS
            auto target_access_start = Clock::now();
#endif
            const std::uint32_t target_index = contribution.target_index;
            if (target_index == kInvalidNodeIndex) {
                throw std::logic_error("AutogradEngine::backward: invalid target index contribution.");
            }
            if (target_index >= states.size()) {
                throw std::logic_error("AutogradEngine::backward: contribution target index out of range.");
            }
            Node& target = *nodes[target_index];
            auto& target_state = states[target_index];
#ifndef NDEBUG
            assert(target_index < nodes.size());
#endif
#ifdef PROFILE_OPS
            opProfileRecord(
                "engine_target_access",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - target_access_start).count());
            const bool had_grad = target.hasAllocatedGrad();
            auto merge_start = Clock::now();
#endif
            target.addGrad(contribution.grad);
#ifdef PROFILE_OPS
            opProfileRecord(
                had_grad ? "engine_grad_merge_accumulate" : "engine_grad_merge_first_write",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - merge_start).count());
            auto ready_update_start = Clock::now();
#endif
            if (target_state.pending_incoming_grads == 0) {
                throw std::logic_error("AutogradEngine::backward: pending contribution underflow.");
            }
            target_state.pending_incoming_grads -= 1;
            const bool became_ready = (target_state.pending_incoming_grads == 0);
            if (became_ready) {
                maybeFireParameterHook(target_index);
                if (!target_state.enqueued) {
#ifdef PROFILE_OPS
                    auto enqueue_start = Clock::now();
#endif
                    target_state.enqueued = true;
                    ready.push(target_index);
#ifdef PROFILE_OPS
                    opProfileRecord(
                        "engine_enqueue_ready",
                        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - enqueue_start).count());
#endif
                }
            }
#ifdef PROFILE_OPS
            opProfileRecord(
                "engine_ready_update",
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ready_update_start).count());
#endif
        }
    }

    if (backward_complete_hook_) {
        backward_complete_hook_();
    }
#ifdef PROFILE_OPS
    opProfileRecord(
        "engine_backward_total",
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - total_start).count());
#endif
}
