#include <iostream>
#include <memory>
#include "node.hpp"
#include "math_ops.hpp"
#include "activation_ops.hpp"

using NodePtr = std::shared_ptr<Node>;

void printMatrix(const Matrix& m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++)
            std::cout << m.data[i * m.cols + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

void test_add() {
    std::cout << "===== TEST ADD =====\n";

    Matrix a(1,1); a.data[0] = 2.0;
    Matrix b(1,1); b.data[0] = 3.0;

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);

    NodePtr Z = MathOps::add(A, B);

    Z->zeroGrad();
    Z->addGrad(Matrix(1, 1, 1.0));  // dZ/dZ = 1
    Z->backward();

    std::cout << "A.grad = " << A->grad().data[0] << "\n";  // expect 1
    std::cout << "B.grad = " << B->grad().data[0] << "\n";  // expect 1
}

void test_mul() {
    std::cout << "===== TEST MUL =====\n";

    Matrix a(1,1); a.data[0] = 2.0;
    Matrix b(1,1); b.data[0] = 3.0;

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);

    NodePtr Z = MathOps::mul(A, B);

    Z->zeroGrad();
    Z->addGrad(Matrix(1, 1, 1.0));
    Z->backward();

    std::cout << "A.grad = " << A->grad().data[0] << "\n";  // expect 3
    std::cout << "B.grad = " << B->grad().data[0] << "\n";  // expect 2
}

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

    std::cout << "A.grad = "; printMatrix(A->grad());
    std::cout << "B.grad = "; printMatrix(B->grad());
}

void test_relu() {
    std::cout << "===== TEST RELU =====\n";

    Matrix x(1,3); x.data = {-1, 2, -3};
    NodePtr X = std::make_shared<Node>(x);

    NodePtr Y = ActivationOps::relu(X);

    Y->zeroGrad();
    Y->addGrad(Matrix(1,3,1.0));
    Y->backward();

    std::cout << "X.grad = "; printMatrix(X->grad());
    // Expect: 0 1 0
}

void test_chain() {
    std::cout << "===== TEST CHAIN rule =====\n";

    // f = relu( (A * B) + C )

    Matrix a(1,2); a.data = {1, 2};
    Matrix b(2,1); b.data = {3, 4};
    Matrix c(1,1); c.data = {5};

    NodePtr A = std::make_shared<Node>(a);
    NodePtr B = std::make_shared<Node>(b);
    NodePtr C = std::make_shared<Node>(c);

    auto M = MathOps::matmul(A, B);
    auto T = MathOps::add(M, C);
    auto Z = ActivationOps::relu(T);

    Z->zeroGrad();
    Z->addGrad(Matrix(1,1,1.0));
    Z->backward();

    std::cout << "A.grad: "; printMatrix(A->grad());
    std::cout << "B.grad: "; printMatrix(B->grad());
    std::cout << "C.grad: "; printMatrix(C->grad());
}

int main() {
    test_add();
    test_mul();
    test_matmul();
    test_relu();
    test_chain();
    return 0;
}
