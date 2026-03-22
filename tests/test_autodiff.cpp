#include <iostream>
#include <memory>
#include <functional>
#include <cmath>

#include "node.hpp"
#include "math_ops.hpp"
#include "loss.hpp"


using NodePtr = std::shared_ptr<Node>;

// Affiche une matrice
void printMatrix(const Matrix& m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++)
            std::cout << m.data[i * m.cols + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Test de l’opération ADD

void test_add() {
    std::cout << "===== TEST ADD =====\n";

    Matrix a(1,1); a.data[0] = 2.0;
    Matrix b(1,1); b.data[0] = 3.0;

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);

    NodePtr Z = MathOps::add(A, B);

    Z->zeroGrad();
    Z->addGrad(Matrix(1,1,1.0));
    Z->backward();

    std::cout << "A.grad = " << A->grad().data[0] << "\n";
    std::cout << "B.grad = " << B->grad().data[0] << "\n";
}


// Test de l’opération MUL élément par élément

void test_mul() {
    std::cout << "===== TEST MUL =====\n";

    Matrix a(1,1); a.data[0] = 2.0;
    Matrix b(1,1); b.data[0] = 3.0;

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);

    NodePtr Z = MathOps::mul(A, B);

    Z->zeroGrad();
    Z->addGrad(Matrix(1,1,1.0));
    Z->backward();

    std::cout << "A.grad = " << A->grad().data[0] << "\n";
    std::cout << "B.grad = " << B->grad().data[0] << "\n";
}


// Test de la multiplication matricielle

void test_matmul() {
    std::cout << "===== TEST MATMUL =====\n";

    Matrix a(1,2); a.data = {2, 3};
    Matrix b(2,1); b.data = {4, 5};

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);

    NodePtr Z = MathOps::matmul(A, B);

    Z->zeroGrad();
    Z->addGrad(Matrix(1,1,1.0));
    Z->backward();

    std::cout << "A.grad = ";
    printMatrix(A->grad());

    std::cout << "B.grad = ";
    printMatrix(B->grad());
}

// Test de l’activation ReLU
void test_relu() {
    std::cout << "===== TEST RELU =====\n";

    Matrix x(1,3); x.data = {-1, 2, -3};
    NodePtr X = std::make_shared<Node>(x);

    NodePtr Y = MathOps::relu(X);

    Y->zeroGrad();
    Y->addGrad(Matrix(1,3,1.0));
    Y->backward();

    std::cout << "X.grad = ";
    printMatrix(X->grad());
}

// Test de la règle de chaîne : ReLU(A*B + C)

void test_chain() {
    std::cout << "===== TEST CHAIN RULE =====\n";

    Matrix a(1,2); a.data = {1, 2};
    Matrix b(2,1); b.data = {3, 4};
    Matrix c(1,1); c.data = {5};

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);
    NodePtr C = std::make_shared<Node>(c);

    auto M = MathOps::matmul(A, B);
    auto T = MathOps::add(M, C);
    auto Z = MathOps::relu(T);

    Z->zeroGrad();
    Z->addGrad(Matrix(1,1,1.0));
    Z->backward();

    std::cout << "A.grad: ";
    printMatrix(A->grad());

    std::cout << "B.grad: ";
    printMatrix(B->grad());

    std::cout << "C.grad: ";
    printMatrix(C->grad());
}


// Test de SUM

void test_sum() {
    std::cout << "===== TEST SUM =====\n";

    Matrix x(1,3); x.data = {1,2,3};
    NodePtr X = std::make_shared<Node>(x);

    NodePtr S = MathOps::sum(X);

    S->zeroGrad();
    S->addGrad(Matrix(1,1,1.0));
    S->backward();

    std::cout << "X.grad = ";
    printMatrix(X->grad());
}


// Test de MEAN

void test_mean() {
    std::cout << "===== TEST MEAN =====\n";

    Matrix x(1,3); x.data = {1,2,3};
    NodePtr X = std::make_shared<Node>(x);

    NodePtr M = MathOps::mean(X);

    M->zeroGrad();
    M->addGrad(Matrix(1,1,1.0));
    M->backward();

    std::cout << "X.grad = ";
    printMatrix(X->grad());
}


// Outil : gradient numérique pour f(x) -> scalaire

Matrix numerical_grad(
    const Matrix& x0,
    const std::function<NodePtr(const Matrix&)>& build,
    double eps = 1e-4)
{
    Matrix grad(x0.rows, x0.cols, 0.0);

    for (size_t k = 0; k < x0.data.size(); ++k) {
        Matrix xp = x0, xn = x0;
        xp.data[k] += eps;
        xn.data[k] -= eps;

        double fp = build(xp)->value().data[0];
        double fn = build(xn)->value().data[0];

        grad.data[k] = (fp - fn) / (2.0 * eps);
    }
    return grad;
}

// Compare gradients auto vs numérique
void compare(const Matrix& a, const Matrix& b, const std::string& name) {
    std::cout << "=== AUTO : " << name << " ===\n";
    printMatrix(a);
    std::cout << "=== NUM  : " << name << " ===\n";
    printMatrix(b);
}


// Test numérique : SIGMOID

void test_num_sigmoid() {
    std::cout << "===== NUM GRAD SIGMOID =====\n";

    Matrix x(1,3); x.data = {-1, 0.5, 2};

    auto build = [](const Matrix& m) {
        NodePtr X = std::make_shared<Node>(m);
        NodePtr Y = MathOps::sigmoid(X);
        return MathOps::sum(Y);
    };

    NodePtr X = std::make_shared<Node>(x);
    NodePtr Y = MathOps::sigmoid(X);
    NodePtr S = MathOps::sum(Y);

    S->zeroGrad();
    S->addGrad(Matrix(1,1,1.0));
    S->backward();

    Matrix auto_grad = X->grad();
    Matrix num_grad = numerical_grad(x, build);

    compare(auto_grad, num_grad, "sigmoid");
}


// Test numérique : TANH

void test_num_tanh() {
    std::cout << "===== NUM GRAD TANH =====\n";

    Matrix x(1,3); x.data = {-1, 0.5, 2};

    auto build = [](const Matrix& m) {
        NodePtr X = std::make_shared<Node>(m);
        NodePtr Y = MathOps::tanh(X);
        return MathOps::sum(Y);
    };

    NodePtr X = std::make_shared<Node>(x);
    NodePtr Y = MathOps::tanh(X);
    NodePtr S = MathOps::sum(Y);

    S->zeroGrad();
    S->addGrad(Matrix(1,1,1.0));
    S->backward();

    Matrix auto_grad = X->grad();
    Matrix num_grad = numerical_grad(x, build);

    compare(auto_grad, num_grad, "tanh");
}


// Test numérique : SUM

void test_num_sum() {
    std::cout << "===== NUM GRAD SUM =====\n";

    Matrix x(1,3); x.data = {1,2,3};

    auto build = [](const Matrix& m) {
        NodePtr X = std::make_shared<Node>(m);
        return MathOps::sum(X);
    };

    NodePtr X = std::make_shared<Node>(x);
    NodePtr S = MathOps::sum(X);

    S->zeroGrad();
    S->addGrad(Matrix(1,1,1.0));
    S->backward();

    Matrix auto_grad = X->grad();
    Matrix num_grad = numerical_grad(x, build);

    compare(auto_grad, num_grad, "sum");
}


// Test numérique : MEAN

void test_num_mean() {
    std::cout << "===== NUM GRAD MEAN =====\n";

    Matrix x(1,3); x.data = {1,2,3};

    auto build = [](const Matrix& m) {
        NodePtr X = std::make_shared<Node>(m);
        return MathOps::mean(X);
    };

    NodePtr X = std::make_shared<Node>(x);
    NodePtr M = MathOps::mean(X);

    M->zeroGrad();
    M->addGrad(Matrix(1,1,1.0));
    M->backward();

    Matrix auto_grad = X->grad();
    Matrix num_grad = numerical_grad(x, build);

    compare(auto_grad, num_grad, "mean");
}

// Test de MSELoss via l’interface (classe)
void test_mse_loss() {
    std::cout << "===== TEST MSE LOSS (interface) =====\n";

    Matrix p(1,3); p.data = {1.0, 2.0, 3.0};
    Matrix t(1,3); t.data = {1.0, 2.0, 4.0};

    auto P = std::make_shared<Node>(p);
    auto T = constant(t);

    // Interface LossFunction
    MSELoss loss_fn;
    auto L = loss_fn.forward(P, T);

    // backward() injecte automatiquement grad=1 au nœud racine
    P->zeroGrad();
    L->backward();

    std::cout << "Loss value = " << L->value().data[0] << " (attendu : 0.333333 si version mean)\n";
    std::cout << "dL/dpred = ";
    printMatrix(P->grad());
}

// Test de CrossEntropyLoss sur logits (stable)
void test_ce_loss() {
    std::cout << "===== TEST CROSS ENTROPY LOSS (logits) =====\n";

    // logits = [0, 0] => softmax = [0.5, 0.5]
    // target = [1, 0]
    // loss = -log(0.5) = 0.693147...
    // grad = probs - target = [-0.5, 0.5]
    Matrix z(1,2); z.data = {0.0, 0.0};
    Matrix y(1,2); y.data = {1.0, 0.0};

    auto Z = std::make_shared<Node>(z);
    auto Y = constant(y);

    CrossEntropyLoss loss_fn;
    auto L = loss_fn.forward(Z, Y);

    Z->zeroGrad();
    L->backward();

    std::cout << "Loss value = " << L->value().data[0] << " (attendu ~ 0.693147)\n";
    std::cout << "dL/dlogits = ";
    printMatrix(Z->grad());
}



int main() {
    test_add();
    test_mul();
    test_matmul();
    test_relu();
    test_chain();

    test_sum();
    test_mean();

    test_num_sigmoid();
    test_num_tanh();
    test_num_sum();
    test_num_mean();
    
    test_mse_loss();    
    test_ce_loss();

    return 0;
}
