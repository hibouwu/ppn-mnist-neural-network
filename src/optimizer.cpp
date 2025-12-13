#include "optimizer.hpp"

Optimizer::Optimizer(std::vector<Node::Ptr> params, double lr)
    : parameters_(std::move(params)), lr_(lr) {}

void Optimizer::zeroGrad() {
    for (auto& p : parameters_) {
        p->zeroGrad();
    }
}

SGDOptimizer::SGDOptimizer(std::vector<Node::Ptr> params, double lr)
    : Optimizer(std::move(params), lr) {}

void SGDOptimizer::step() {
    for (auto& p : parameters_) {
        // Param = Param - lr * Grad
        // Note: We need to access mutable data references.
        // Node implementation stores Matrix value_ and grad_.
        // We assume Node::value() returns const ref, but we need to modify it.
        // Let's check Node interface. Node has private value_.
        // BUT Node implementation is usually friendly or has a way to update?
        // Wait, Node.hpp usually exposes value() as const.
        // How do we update weights?
        // Let's look at Node.hpp again.
        
        // Actually typically in simple implementations we might need a method "update(matrix)" or make friend.
        // Or maybe const_cast (ugly but common in simple autodiff if no setter).
        // Let's check if we can add a method or friend.
        // For now, let's assume we might need to add `update(delta)` to Node or similiar.
        
        // Checking Node.hpp (from memory/previous steps):
        // It had `const Matrix& value() const`.
        // It does NOT have a setter.
        
        // FIX: We need to modify Node.hpp to allow parameter updates.
        // I'll implement logic here assuming I can access it, and then I will update Node.hpp in next step.
        // Or I can use a const_cast for now if I don't want to change Node header yet, but changing header is better.
        
        // Let's assume we will add `Matrix& mutable_value()` to Node.
        
        // Logic: W = W - lr * grad
        Matrix& val = const_cast<Matrix&>(p->value()); // Temporary hack until we add API
        const Matrix& grad = p->grad();
        
        // val = val - lr * grad
        // Matrix doesn't have operator-. It has add, mul.
        // val = val.add(grad.mul(scalar(-lr)))
        // For efficiency we might want in-place operations, but Matrix class shown earlier returns new Matrix.
        // Let's do: W_new = W.add( grad.mul(-lr) )
        // And assign back.
        
        // Create a scalar matrix for -lr? Matrix::mul takes Matrix. 
        // Does Matrix support scalar mul?
        // Docs/conception_detaillee had: mul(other: Matrix).
        // Check tensor.cpp/hpp capabilities.
        // If not supported, we must do element-wise loop here manually to be safe and fast.
        
        size_t n = val.data.size();
        for(size_t i=0; i<n; ++i) {
            val.data[i] -= lr_ * grad.data[i];
        }
    }
}
