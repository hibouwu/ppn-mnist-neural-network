#pragma once
#include "node.hpp"
#include <vector>

class Optimizer {
public:
    Optimizer(std::vector<Node::Ptr> params, double lr);
    virtual ~Optimizer() = default;
    
    // Updates parameters based on gradients
    virtual void step() = 0;
    
    // Clears gradients
    virtual void zeroGrad();

protected:
    std::vector<Node::Ptr> parameters_;
    double lr_;
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(std::vector<Node::Ptr> params, double lr);
    
    void step() override;
};
