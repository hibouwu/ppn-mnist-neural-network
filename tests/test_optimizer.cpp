#include "optimizer.hpp"
#include "node.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    // 1. Create a parameter node
    Matrix w_data(1, 1);
    w_data.data[0] = 10.0;
    auto w = std::make_shared<Node>(w_data);
    
    // 2. Set a gradient manually
    Matrix grad(1, 1);
    grad.data[0] = 2.0;
    w->addGrad(grad);
    
    // 3. Create Optimizer with lr = 0.1
    double lr = 0.1;
    SGDOptimizer opt({w}, lr);
    
    // 4. Step
    opt.step();
    
    // 5. Verify
    // Expected: 10.0 - 0.1 * 2.0 = 9.8
    double val = w->value().data[0];
    std::cout << "Value after step: " << val << " (Expected: 9.8)" << std::endl;
    
    if (std::abs(val - 9.8) > 1e-6) {
        std::cerr << "Optimizer step failed!" << std::endl;
        return 1;
    }
    
    // 6. Test zeroGrad
    opt.zeroGrad();
    double g_val = w->grad().data[0];
    if (g_val != 0.0) {
        std::cerr << "zeroGrad failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Optimizer Test PASSED." << std::endl;
    return 0;
}
