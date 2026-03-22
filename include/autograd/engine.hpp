#pragma once

#include "node.hpp"
#include <functional>

class AutogradEngine {
public:
    using ParameterReadyHook = std::function<void(Node&)>;
    using BackwardCompleteHook = std::function<void()>;

    void setParameterReadyHook(ParameterReadyHook hook) { parameter_ready_hook_ = std::move(hook); }
    void setBackwardCompleteHook(BackwardCompleteHook hook) { backward_complete_hook_ = std::move(hook); }

    void backward(const Node::Ptr& root) const;

private:
    ParameterReadyHook parameter_ready_hook_;
    BackwardCompleteHook backward_complete_hook_;
};
