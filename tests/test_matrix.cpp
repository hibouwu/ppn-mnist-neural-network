#include "tensor.hpp"
#include <iostream>

int main() {
    // Test 1 : Construction
    Matrix A(2, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;

    std::cout << "Matrix A:\n";
    A.print();

    // Test 2 : Addition
    Matrix B(2, 3, 1.0);  // Rempli de 1.0
    Matrix C = A.add(B);
    std::cout << "A + B:\n";
    C.print();

    // Test 3 : Multiplication élément par élément
    Matrix D = A.mul(B);
    std::cout << "A .* B:\n";
    D.print();

    // Test 4 : Produit matriciel
    Matrix E(3, 2);
    E(0, 0) = 1.0; E(1, 0) = 2.0; E(2, 0) = 3.0;
    E(0, 1) = 4.0; E(1, 1) = 5.0; E(2, 1) = 6.0;

    Matrix F = A.matmul(E);
    std::cout << "A @ E:\n";
    F.print();

    // Test 5 : GEMM avec transpose flag (A^T @ A)
    Matrix AT_A(3, 3);
    A.matmul_into(A, AT_A, true, false);
    std::cout << "A^T @ A:\n";
    AT_A.print();

    const double expected[3][3] = {
        {17.0, 22.0, 27.0},
        {22.0, 29.0, 36.0},
        {27.0, 36.0, 45.0}
    };

    if (AT_A.rows != 3 || AT_A.cols != 3) {
        std::cerr << "Test FAILED: A^T @ A has wrong shape\n";
        return 1;
    }

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (AT_A(i, j) != expected[i][j]) {
                std::cerr << "Test FAILED: A^T @ A mismatch at (" << i << ", " << j
                          << "), got " << AT_A(i, j) << ", expected " << expected[i][j] << "\n";
                return 1;
            }
        }
    }

    // Test 6 : Random init
    Matrix H(2, 2);
    H.randomInit(-1.0, 1.0);
    std::cout << "Random matrix H:\n";
    H.print();

    std::cout << "All tests passed!\n";
    return 0;
}
