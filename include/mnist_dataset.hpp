#pragma once

#include "dataset.hpp"
#include <string>

/**
 * @brief Handles loading MNIST data from IDX files using mnist_reader.
 */
class MNISTDataset : public Dataset {
public:
    /**
     * @param data_dir Directory containing the 'mnist' folder or the files.
     */
    explicit MNISTDataset(const std::string& data_dir);

    /**
     * @brief Loads training and test data from files.
     * throws std::runtime_error if files missing or parse error.
     */
    void load() override;

private:
    std::string data_dir_;

    // Helper to call C function and convert to Matrix
    Matrix loadImages(const std::string& filename, int count, int size=784);
    Matrix loadLabels(const std::string& filename, int count, int num_classes=10);
};
