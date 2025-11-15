#include "layer.hpp"
#include <iostream>

int main() {
    LinearLayer layer(3, 2);

    Matrix input(2, 3);  // Batch de 2 exemples, 3 features
    input(0, 0) = 1.0; input(0, 1) = 2.0; input(0, 2) = 3.0;
    input(1, 0) = 4.0; input(1, 1) = 5.0; input(1, 2) = 6.0;

    std::cout << "Input:\n";
    input.print();

    Matrix output = layer.forward(input);
    std::cout << "Output of LinearLayer:\n";
    output.print();

    std::cout << "LinearLayer test passed!\n";
    return 0;
}
