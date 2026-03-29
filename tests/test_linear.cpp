#include "layer.hpp"
#include "math_ops.hpp"
#include "node.hpp"
#include <iostream>

int main() {
    LinearLayer layer(3, 2);

    Matrix input_val(2, 3);  // Batch de 2 exemples, 3 features
    input_val(0, 0) = 1.0; input_val(0, 1) = 2.0; input_val(0, 2) = 3.0;
    input_val(1, 0) = 4.0; input_val(1, 1) = 5.0; input_val(1, 2) = 6.0;

    std::cout << "Input:\n";
    input_val.print();

    Node::Ptr input = std::make_shared<Node>(input_val);
    Node::Ptr output = layer.forward(input);
    
    std::cout << "Output of LinearLayer:\n";
    output->value().print();

    // Test Backward
    MathOps::sum(output)->backward();
    std::cout << "Input Gradients:\n";
    input->grad().print();

    std::cout << "LinearLayer test passed!\n";
    return 0;
}
