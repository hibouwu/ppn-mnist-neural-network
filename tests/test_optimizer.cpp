#include "optimizer.hpp"
#include "node.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>

namespace {

constexpr double kScalarTol = 1e-6;

Node::Ptr makeScalarNode(double v) {
    Matrix m(1, 1);
    m.data[0] = v;
    return std::make_shared<Node>(m);
}

void setScalarGrad(const Node::Ptr& p, double g) {
    p->zeroGrad();
    Matrix grad(1, 1);
    grad.data[0] = g;
    p->addGrad(grad);
}

void assertNear(double got, double expected, double tol, const std::string& name) {
    if (std::abs(got - expected) > tol) {
        std::cerr << std::setprecision(17)
                  << name << " failed: got=" << got
                  << ", expected=" << expected << std::endl;
        std::exit(1);
    }
}

void testSGDBasicStepAndZeroGrad() {
    auto w = makeScalarNode(10.0);
    setScalarGrad(w, 2.0);

    SGDOptimizer opt({w}, 0.1);
    opt.step();
    assertNear(w->value().data[0], 9.8, kScalarTol, "SGD step");

    opt.zeroGrad();
    assertNear(w->grad().data[0], 0.0, kScalarTol, "SGD zeroGrad");
}

void testMomentumNoNesterovTwoSteps() {
    auto w = makeScalarNode(10.0);
    MomentumSGDOptimizer opt({w}, 0.1, 0.9, false, 0.0);

    setScalarGrad(w, 2.0);
    opt.step();
    assertNear(w->value().data[0], 9.8, kScalarTol, "Momentum step1");

    setScalarGrad(w, 2.0);
    opt.step();
    assertNear(w->value().data[0], 9.42, kScalarTol, "Momentum step2");
}

void testMomentumNesterovTwoSteps() {
    auto w = makeScalarNode(10.0);
    MomentumSGDOptimizer opt({w}, 0.1, 0.9, true, 0.0);

    setScalarGrad(w, 2.0);
    opt.step();
    assertNear(w->value().data[0], 9.62, kScalarTol, "Nesterov step1");

    setScalarGrad(w, 2.0);
    opt.step();
    assertNear(w->value().data[0], 9.078, kScalarTol, "Nesterov step2");
}

void testAdamWOneStepWithBiasCorrection() {
    auto w = makeScalarNode(1.0);
    AdamWOptimizer opt({w}, 0.1, 0.9, 0.999, 1e-8, 0.01);

    setScalarGrad(w, 0.5);
    opt.step();

    const double shrinked = 1.0 * (1.0 - 0.1 * 0.01);
    const double m = (1.0 - 0.9) * 0.5;
    const double v = (1.0 - 0.999) * (0.5 * 0.5);
    const double m_hat = m / (1.0 - 0.9);
    const double v_hat = v / (1.0 - 0.999);
    const double expected = shrinked - 0.1 * m_hat / (std::sqrt(v_hat) + 1e-8);

    assertNear(w->value().data[0], expected, kScalarTol, "AdamW step");
}

void testGradScaleAcrossOptimizers() {
    {
        auto w = makeScalarNode(1.0);
        SGDOptimizer opt({w}, 0.1);
        setScalarGrad(w, 2.0);
        opt.step(0.5);
        assertNear(w->value().data[0], 0.9, kScalarTol, "SGD gradScale");
    }
    {
        auto w = makeScalarNode(1.0);
        MomentumSGDOptimizer opt({w}, 0.1, 0.0, false, 0.0);
        setScalarGrad(w, 2.0);
        opt.step(0.5);
        assertNear(w->value().data[0], 0.9, kScalarTol, "Momentum gradScale");
    }
    {
        auto w = makeScalarNode(1.0);
        AdamWOptimizer opt({w}, 0.1, 0.0, 0.0, 1.0, 0.0);
        setScalarGrad(w, 2.0);
        opt.step(0.5);
        const double expected = 1.0 - 0.1 * (1.0 / (1.0 + 1.0));
        assertNear(w->value().data[0], expected, kScalarTol, "AdamW gradScale");
    }
}

} // namespace

int main() {
    testSGDBasicStepAndZeroGrad();
    testMomentumNoNesterovTwoSteps();
    testMomentumNesterovTwoSteps();
    testAdamWOneStepWithBiasCorrection();
    testGradScaleAcrossOptimizers();

    std::cout << "Optimizer tests PASSED." << std::endl;
    return 0;
}
