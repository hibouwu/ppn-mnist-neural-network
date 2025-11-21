#include "activation.hpp"
#include "node.hpp"
#include <iostream>

int main() {
    Matrix input_val(2, 3);
    input_val(0, 0) = -1.0; input_val(0, 1) = 0.0; input_val(0, 2) = 1.0;
    input_val(1, 0) = -2.0; input_val(1, 1) = 0.5; input_val(1, 2) = 2.0;

    std::cout << "Input:\n";
    input_val.print();

    // ReLU
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        ReLU relu;
        Node::Ptr out_relu = relu.forward(input);
        std::cout << "ReLU Forward:\n";
        out_relu->value().print();

        // Backward
        out_relu->backward();
        std::cout << "ReLU Backward (Input Grad):\n";
        input->grad().print();
    }

    // Sigmoid
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        Sigmoid sigmoid;
        Node::Ptr out_sigmoid = sigmoid.forward(input);
        std::cout << "Sigmoid Forward:\n";
        out_sigmoid->value().print();

        // Backward
        out_sigmoid->backward();
        std::cout << "Sigmoid Backward (Input Grad):\n";
        input->grad().print();
    }

    // Tanh
    {
        Node::Ptr input = std::make_shared<Node>(input_val);
        Tanh tanh;
        Node::Ptr out_tanh = tanh.forward(input);
        std::cout << "Tanh Forward:\n";
        out_tanh->value().print();

        // Backward
        out_tanh->backward();
        std::cout << "Tanh Backward (Input Grad):\n";
        input->grad().print();
    }

    std::cout << "All activation tests passed!\n";
    return 0;
}
