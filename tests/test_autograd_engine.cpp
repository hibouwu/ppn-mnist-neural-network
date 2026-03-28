#include "autograd/engine.hpp"
#include "autograd/grad_fn.hpp"
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

void test_repeated_input_not_deduplicated() {
    auto x = std::make_shared<Node>(Matrix(1, 1, 2.0));

    auto sum_twice = MathOps::add(x, x);
    sum_twice->backward();
    assert(almostEqual(x->grad().data[0], 2.0));

    x->zeroGrad();
    auto square = MathOps::mul(x, x);
    square->backward();
    assert(almostEqual(x->grad().data[0], 4.0));
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

void test_parameter_ready_hook_fires_once_after_full_accumulation() {
    auto p = std::make_shared<Node>(Matrix(1, 1, 2.0));
    p->setIsParameter(true);
    auto c1 = constant(Matrix(1, 1, 3.0));
    auto c2 = constant(Matrix(1, 1, 5.0));
    auto loss = MathOps::add(MathOps::mul(p, c1), MathOps::mul(p, c2));

    int fire_count = 0;
    double grad_seen_at_hook = 0.0;
    AutogradEngine engine;
    engine.setParameterReadyHook([&](Node& node) {
        fire_count += 1;
        grad_seen_at_hook = node.grad().data[0];
    });
    engine.backward(loss);

    assert(fire_count == 1);
    assert(almostEqual(grad_seen_at_hook, 8.0));
    assert(almostEqual(p->grad().data[0], 8.0));
}

void test_parameter_ready_hook_matches_final_accumulated_tensor() {
    Matrix p_val(1, 2);
    p_val.data = {2.0, -1.0};
    auto p = std::make_shared<Node>(p_val);
    p->setIsParameter(true);
    auto c1 = constant(Matrix(1, 2, 3.0));
    auto c2 = constant(Matrix(1, 2, 5.0));
    auto left = MathOps::sum(MathOps::mul(p, c1));
    auto right = MathOps::sum(MathOps::mul(p, c2));
    auto loss = MathOps::add(left, right);

    Matrix grad_seen_at_hook(1, 2, 0.0);
    int fire_count = 0;
    AutogradEngine engine;
    engine.setParameterReadyHook([&](Node& node) {
        fire_count += 1;
        grad_seen_at_hook = node.grad();
    });
    engine.backward(loss);

    assert(fire_count == 1);
    assert(grad_seen_at_hook.data.size() == p->grad().data.size());
    for (std::size_t i = 0; i < p->grad().data.size(); ++i) {
        assert(almostEqual(grad_seen_at_hook.data[i], p->grad().data[i]));
    }
}

void test_parameter_ready_hook_ignores_non_parameter_leaves() {
    auto x = std::make_shared<Node>(Matrix(1, 1, 2.0));
    auto c = constant(Matrix(1, 1, 4.0));
    auto loss = MathOps::sum(MathOps::mul(x, c));

    int fire_count = 0;
    AutogradEngine engine;
    engine.setParameterReadyHook([&](Node&) {
        fire_count += 1;
    });
    engine.backward(loss);

    assert(fire_count == 0);
}

class InvalidTargetGradFn final : public GradFn {
public:
    ContributionList apply(const Node& output,
                           const Matrix& grad_output,
                           InputIndexView input_indices) const override {
        (void)output;
        (void)grad_output;
        (void)input_indices;
        ContributionList out;
        Matrix bogus(1, 1, 1.0);
        out.push_back({kInvalidNodeIndex, std::move(bogus)});
        return out;
    }
};

void test_invalid_target_rejected() {
    auto x = std::make_shared<Node>(Matrix(1, 1, 2.0));
    auto y = std::make_shared<Node>(Matrix(1, 1, 3.0));
    y->setInputs({x});
    y->setGradFn(std::make_shared<InvalidTargetGradFn>());

    bool threw = false;
    try {
        y->backward();
    } catch (const std::logic_error&) {
        threw = true;
    }
    assert(threw);
}

}

int main() {
    test_shared_parent_accumulation();
    test_repeated_input_not_deduplicated();
    test_requires_grad_propagation();
    test_lazy_grad_allocation();
    test_root_seed_rules();
    test_parameter_ready_hook();
    test_parameter_ready_hook_fires_once_after_full_accumulation();
    test_parameter_ready_hook_matches_final_accumulated_tensor();
    test_parameter_ready_hook_ignores_non_parameter_leaves();
    test_invalid_target_rejected();
    std::cout << "Autograd engine tests passed!" << std::endl;
    return 0;
}
