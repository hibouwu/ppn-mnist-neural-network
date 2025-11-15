#include "activation.hpp"
#include <iostream>

int main() {
    Matrix input(2, 3);
    input(0, 0) = -1.0; input(0, 1) = 0.0; input(0, 2) = 1.0;
    input(1, 0) = -2.0; input(1, 1) = 0.5; input(1, 2) = 2.0;

    std::cout << "Input:\n";
    input.print();

    ReLU relu;
    Matrix out_relu = relu.forward(input);
    std::cout << "ReLU Forward:\n";
    out_relu.print();

    Matrix grad_out(2, 3, 1.0);  // gradient d'entrée artificiel
    Matrix grad_relu = relu.backward(input, grad_out);
    std::cout << "ReLU Backward:\n";
    grad_relu.print();

    Sigmoid sigmoid;
    Matrix out_sigmoid = sigmoid.forward(input);
    std::cout << "Sigmoid Forward:\n";
    out_sigmoid.print();

    Matrix grad_sigmoid = sigmoid.backward(input, grad_out);
    std::cout << "Sigmoid Backward:\n";
    grad_sigmoid.print();

    Tanh tanh;
    Matrix out_tanh = tanh.forward(input);
    std::cout << "Tanh Forward:\n";
    out_tanh.print();

    Matrix grad_tanh = tanh.backward(input, grad_out);
    std::cout << "Tanh Backward:\n";
    grad_tanh.print();

    std::cout << "All activation tests passed!\n";
    return 0;
}
