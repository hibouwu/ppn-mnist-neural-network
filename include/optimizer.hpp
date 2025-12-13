#pragma once
#include "node.hpp"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // Updates parameters based on gradients
    virtual void step() = 0;
    
    // Clears gradients
    virtual void zeroGrad() = 0;
};
