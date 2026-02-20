#pragma once
#include "node.hpp"
#include <vector>
#include <cstddef>

class Optimizer {
public:
    Optimizer(std::vector<Node::Ptr> params, double lr);
    virtual ~Optimizer() = default;
    
    // Updates parameters based on gradients
    virtual void step(double gradScale = 1.0) = 0;
    
    // Clears gradients
    virtual void zeroGrad();

protected:
    std::vector<Node::Ptr> parameters_;
    double lr_;
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(std::vector<Node::Ptr> params, double lr);
    
    void step(double gradScale = 1.0) override;
};

class MomentumSGDOptimizer : public Optimizer {
public:
    MomentumSGDOptimizer(std::vector<Node::Ptr> params,
                         double lr,
                         double momentum = 0.9,
                         bool nesterov = false,
                         double weightDecay = 0.0);

    void step(double gradScale = 1.0) override;

private:
    double momentum_;
    bool nesterov_;
    double weight_decay_;
    std::vector<Matrix> velocity_;
};

class AdamWOptimizer : public Optimizer {
public:
    AdamWOptimizer(std::vector<Node::Ptr> params,
                   double lr,
                   double beta1 = 0.9,
                   double beta2 = 0.999,
                   double eps = 1e-8,
                   double weightDecay = 0.0);

    void step(double gradScale = 1.0) override;

private:
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    std::size_t step_count_;
    std::vector<Matrix> first_moment_;
    std::vector<Matrix> second_moment_;
};
