#include "autograd/engine.hpp"
#include "math_ops.hpp"
#include "node.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

bool almostEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

void test_shared_parent_accumulation() {
    Matrix x_val(1, 1, 2.0);
    Matrix a_val(1, 1, 3.0);
    Matrix b_val(1, 1, 5.0);

    auto x = std::make_shared<Node>(x_val);
    auto a = std::make_shared<Node>(a_val);
    auto b = std::make_shared<Node>(b_val);

    auto z = MathOps::add(MathOps::mul(x, a), MathOps::mul(x, b));
    z->backward();

    assert(almostEqual(x->grad().data[0], 8.0));
    assert(almostEqual(a->grad().data[0], 2.0));
    assert(almostEqual(b->grad().data[0], 2.0));
}

void test_requires_grad_propagation() {
    Matrix a_val(1, 2);
    a_val.data = {1.0, 2.0};
    Matrix b_val(1, 2);
    b_val.data = {3.0, 4.0};

    auto a = constant(a_val);
    auto b = constant(b_val);
    auto c = MathOps::add(a, b);
    assert(!c->requiresGrad());
    assert(!c->gradFn());
    assert(!c->backwardContext());
    assert(c->inputs().empty());

    auto x = std::make_shared<Node>(a_val);
    auto d = MathOps::add(x, b);
    assert(d->requiresGrad());
    assert(static_cast<bool>(d->gradFn()));
    assert(static_cast<bool>(d->backwardContext()));
    assert(d->inputs().size() == 2);
}

void test_lazy_grad_allocation() {
    Matrix a_val(1, 2);
    a_val.data = {7.0, -2.0};
    auto a = constant(a_val);
    assert(!a->hasAllocatedGrad());
    Matrix& grad = a->grad();
    assert(a->hasAllocatedGrad());
    assert(grad.rows == 1 && grad.cols == 2);
    assert(almostEqual(grad.data[0], 0.0));
    assert(almostEqual(grad.data[1], 0.0));
}

void test_root_seed_rules() {
    Matrix x_val(1, 3);
    x_val.data = {1.0, -2.0, 3.0};
    auto x = std::make_shared<Node>(x_val);
    auto y = MathOps::relu(x);

    bool threw = false;
    try {
        y->backward();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    x->zeroGrad();
    y->zeroGrad();
    y->addGrad(Matrix(1, 3, 0.0));
    y->backward();
    for (double v : x->grad().data) {
        assert(almostEqual(v, 0.0));
    }
}

void test_parameter_ready_hook() {
    Matrix p_val(1, 1, 2.0);
    auto p = std::make_shared<Node>(p_val);
    p->setIsParameter(true);

    Matrix c_val(1, 1, 4.0);
    auto c = constant(c_val);
    auto loss = MathOps::sum(MathOps::mul(p, c));

    int fire_count = 0;
    AutogradEngine engine;
    engine.setParameterReadyHook([&fire_count](Node& node) {
        (void)node;
        fire_count += 1;
    });

    engine.backward(loss);

    assert(fire_count == 1);
    assert(almostEqual(p->grad().data[0], 4.0));
}

}

int main() {
    test_shared_parent_accumulation();
    test_requires_grad_propagation();
    test_lazy_grad_allocation();
    test_root_seed_rules();
    test_parameter_ready_hook();
    std::cout << "Autograd engine tests passed!" << std::endl;
    return 0;
}
