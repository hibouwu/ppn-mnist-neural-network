#pragma once

#include "node.hpp"
#include <functional>
#include <vector>

class AutogradEngine {
public:
    using ParameterReadyHook = std::function<void(Node&)>;
    using BackwardCompleteHook = std::function<void()>;
    using ReachableLeafHook = std::function<void(const std::vector<Node::Ptr>&)>;

    void setParameterReadyHook(ParameterReadyHook hook) { parameter_ready_hook_ = std::move(hook); }
    void setBackwardCompleteHook(BackwardCompleteHook hook) { backward_complete_hook_ = std::move(hook); }
    void setReachableLeafHook(ReachableLeafHook hook) { reachable_leaf_hook_ = std::move(hook); }

    void backward(const Node::Ptr& root) const;

private:
    ParameterReadyHook parameter_ready_hook_;
    BackwardCompleteHook backward_complete_hook_;
    ReachableLeafHook reachable_leaf_hook_;
};
