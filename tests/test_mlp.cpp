#include "network.hpp"
#include <iostream>

int main() {
    MLPNetwork net;

    // Ajouter une couche 3 -> 4 avec ReLU
    auto layer1 = std::make_unique<LinearLayer>(3, 4);
    auto relu1 = std::make_unique<ReLU>();
    net.addLayer(std::move(layer1), std::move(relu1));

    // Ajouter une couche 4 -> 2 avec Sigmoid
    auto layer2 = std::make_unique<LinearLayer>(4, 2);
    auto sigmoid = std::make_unique<Sigmoid>();
    net.addLayer(std::move(layer2), std::move(sigmoid));

    // Input: 2 exemples avec 3 features
    Matrix input(2, 3);
    input(0, 0) = 1.0; input(0, 1) = 2.0; input(0, 2) = 3.0;
    input(1, 0) = 4.0; input(1, 1) = 5.0; input(1, 2) = 6.0;

    std::cout << "Input:\n";
    input.print();

    Matrix output = net.forward(input);
    std::cout << "Output of MLPNetwork:\n";
    output.print();

    std::cout << "MLPNetwork test passed!\n";
    return 0;
}
